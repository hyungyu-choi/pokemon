#!/usr/bin/env node
/**
 * Export Pokemon Champions battle data from a built Pokemon Showdown checkout.
 *
 * Usage:
 *   node tools/export_showdown_data.js <path-to-built-pokemon-showdown> [outDir]
 *
 * The Showdown checkout must be built (`node build`) so that `dist/sim` exists.
 * Data is taken from the `champions` mod (which inherits from gen9) so that all
 * Champions-specific changes (move/ability/item/condition overrides, learnsets,
 * legality) are already applied.
 *
 * Output (JSON, written to pokechamp/data by default):
 *   species.json    - every species usable in Champions (+ all of their formes)
 *   moves.json      - every move usable in Champions (+ moves needed internally)
 *   abilities.json  - every ability usable in Champions
 *   items.json      - every item usable in Champions
 *   conditions.json - generic conditions (status, weather, volatiles, ...)
 *   typechart.json, natures.json, learnsets.json, formats.json
 *   legality.json   - per format: legal species and their legal abilities/moves/items
 *                     (computed with Showdown's TeamValidator)
 *   empties.json    - default fields of empty effects (dex.items.get('') etc.)
 *   callbacks.json  - inventory of all JS callbacks that the Python engine ports
 *   meta.json       - source commit / version information
 *
 * Function-valued fields are replaced with {"__fn__": true}; their behaviour is
 * implemented by hand in pokechamp/sim/effects/*.py and checked against
 * callbacks.json by the test-suite.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const child_process = require('child_process');

const psPath = path.resolve(process.argv[2] || '.');
const outDir = path.resolve(process.argv[3] || path.join(__dirname, '..', 'pokechamp', 'data'));
const sourcesFile = process.argv[4] ? path.resolve(process.argv[4]) : null;
const sources = [];
const { Dex } = require(path.join(psPath, 'dist', 'sim'));

const MOD = 'champions';
const dex = Dex.mod(MOD);
dex.includeData();
const FORMATS = [
	'gen9championsbssregmc', 'gen9championsvgc2026regmc', 'gen9championsou',
];

const callbacks = {};
function noteCallback(kind, id, pathName, fn) {
	const key = `${kind}:${id}`;
	if (!callbacks[key]) callbacks[key] = {};
	callbacks[key][pathName] = fn.toString().length;
	sources.push(`// ===== ${key} :: ${pathName}\n${fn.toString()}\n`);
}

/** Recursively convert an object to plain JSON, replacing functions. */
function plain(value, kind, id, pathName = '') {
	if (typeof value === 'function') {
		noteCallback(kind, id, pathName, value);
		return { __fn__: true };
	}
	if (value === null || typeof value !== 'object') return value;
	if (Array.isArray(value)) return value.map((v, i) => plain(v, kind, id, `${pathName}[${i}]`));
	const out = {};
	for (const k of Object.keys(value)) {
		const v = value[k];
		if (v === undefined) continue;
		out[k] = plain(v, kind, id, pathName ? `${pathName}.${k}` : k);
	}
	return out;
}

/** Copy all own + prototype-chain enumerable data fields of a dex object. */
function effectData(obj, skip = []) {
	const out = {};
	for (const k in obj) {
		if (skip.includes(k)) continue;
		if (['desc', 'shortDesc', 'tags', 'contestType', 'zMove', 'maxMove', 'gmaxPower'].includes(k)) continue;
		out[k] = obj[k];
	}
	return out;
}

const isStandard = e => e.exists && !e.isNonstandard;

// ---------------------------------------------------------------------------
// Species
const speciesIds = new Set();
for (const s of dex.species.all()) {
	if (!isStandard(s)) continue;
	speciesIds.add(s.id);
}
// include all formes / base species reachable from standard species
let changed = true;
while (changed) {
	changed = false;
	for (const id of [...speciesIds]) {
		const s = dex.species.get(id);
		const related = [s.baseSpecies, s.battleOnly, s.changesFrom, ...(s.otherFormes || []),
			...(s.cosmeticFormes || []), ...(s.formeOrder || [])].flat().filter(Boolean);
		for (const r of related) {
			const rs = dex.species.get(r);
			if (rs.exists && !speciesIds.has(rs.id) && rs.isNonstandard !== 'CAP' && rs.isNonstandard !== 'Custom') {
				speciesIds.add(rs.id);
				changed = true;
			}
		}
	}
}
const species = {};
for (const id of [...speciesIds].sort()) {
	const s = dex.species.get(id);
	const data = effectData(s, ['effectType', 'fullname']);
	data.legal = isStandard(s);
	species[id] = plain(data, 'species', id);
}

// ---------------------------------------------------------------------------
// Learnsets (only for exported species)
const learnsets = {};
for (const id of Object.keys(species)) {
	const ls = dex.data.Learnsets[id];
	if (ls && ls.learnset) learnsets[id] = Object.keys(ls.learnset).sort();
}

// ---------------------------------------------------------------------------
// Moves
const moveIds = new Set();
for (const m of dex.moves.all()) if (isStandard(m)) moveIds.add(m.id);
// moves used internally by the engine
for (const m of ['struggle', 'recharge']) moveIds.add(m);
// non-standard moves whose condition can still be applied by a standard move
// (e.g. Psychic Noise inflicts the Heal Block volatile)
{
	const refs = new Set();
	for (const id of moveIds) {
		const m = dex.moves.get(id);
		for (const k of ['volatileStatus', 'sideCondition', 'slotCondition', 'pseudoWeather', 'weather', 'terrain']) {
			if (m[k]) refs.add(m[k]);
		}
		for (const sec of (m.secondaries || [])) {
			if (sec.volatileStatus) refs.add(sec.volatileStatus);
			if (sec.self?.volatileStatus) refs.add(sec.self.volatileStatus);
		}
		if (m.self?.volatileStatus) refs.add(m.self.volatileStatus);
		if (m.self?.sideCondition) refs.add(m.self.sideCondition);
	}
	for (const m of dex.moves.all()) {
		if (m.isNonstandard && m.condition && refs.has(m.id)) moveIds.add(m.id);
	}
}
const moves = {};
for (const id of [...moveIds].sort()) {
	const m = dex.moves.get(id);
	const data = effectData(m, ['effectType', 'fullname']);
	moves[id] = plain(data, 'move', id);
}

// ---------------------------------------------------------------------------
// Abilities
const abilityIds = new Set();
// Only abilities that a Champions-legal Pokemon (or one of its formes) can have. Moves can only
// copy/swap abilities between Pokemon on the field, plus Worry Seed (Insomnia) / Simple Beam (Simple).
for (const a of ['insomnia', 'simple']) abilityIds.add(a);
for (const id of Object.keys(species)) {
	for (const a of Object.values(species[id].abilities || {})) abilityIds.add(dex.abilities.get(a).id);
}
abilityIds.add('noability');
const abilities = {};
for (const id of [...abilityIds].sort()) {
	const a = dex.abilities.get(id);
	if (!a.exists) continue;
	abilities[id] = plain(effectData(a, ['effectType', 'fullname']), 'ability', id);
}

// ---------------------------------------------------------------------------
// Items
const itemIds = new Set();
for (const i of dex.items.all()) if (isStandard(i)) itemIds.add(i.id);
for (const id of Object.keys(species)) {
	const s = species[id];
	for (const r of [s.requiredItem, ...(s.requiredItems || [])]) if (r && dex.items.get(r).exists) itemIds.add(dex.items.get(r).id);
}
const items = {};
for (const id of [...itemIds].sort()) {
	const i = dex.items.get(id);
	items[id] = plain(effectData(i, ['effectType', 'fullname']), 'item', id);
}

// ---------------------------------------------------------------------------
// Conditions (generic, i.e. data/conditions.ts after mod inheritance)
const conditions = {};
for (const id of Object.keys(dex.data.Conditions).sort()) {
	const c = dex.conditions.getByID(id);
	conditions[id] = plain(effectData(c, ['fullname']), 'condition', id);
}

// ---------------------------------------------------------------------------
// Type chart / natures
const typechart = {};
for (const t of dex.types.all()) {
	if (t.isNonstandard) continue;
	typechart[t.name] = { damageTaken: t.damageTaken, HPivs: t.HPivs || {}, HPdvs: t.HPdvs || {} };
}
const natures = {};
for (const n of dex.natures.all()) natures[n.id] = { name: n.name, plus: n.plus || null, minus: n.minus || null };

// ---------------------------------------------------------------------------
// Formats (the rules we care about)
const formats = {};
for (const fid of FORMATS) {
	const f = Dex.formats.get(fid);
	if (!f.exists) continue;
	const rt = Dex.formats.getRuleTable(f);
	const rules = {};
	for (const [k, v] of rt.entries()) rules[k] = v;
	formats[f.id] = {
		name: f.name, mod: f.mod, gameType: f.gameType || 'singles', debug: !!f.debug,
		ruleset: f.ruleset, banlist: f.banlist, rules, valueRules: Object.fromEntries(rt.valueRules),
		pickedTeamSize: rt.pickedTeamSize, minTeamSize: rt.minTeamSize, maxTeamSize: rt.maxTeamSize,
		adjustLevel: rt.adjustLevel, defaultLevel: rt.defaultLevel, maxLevel: rt.maxLevel,
		maxMoveCount: rt.maxMoveCount, evLimit: rt.evLimit,
	};
}

// ---------------------------------------------------------------------------
// Rulesets that carry battle event handlers (they're attached as pseudo-weather)
const rulesetCallbacks = {};
for (const fid of FORMATS) {
	const f = Dex.formats.get(fid);
	if (!f.exists) continue;
	const rt = Dex.formats.getRuleTable(f);
	for (const rule of rt.keys()) {
		if ('+*-!'.includes(rule.charAt(0))) continue;
		const sub = Dex.formats.get(rule);
		if (!sub.exists) continue;
		for (const k of Object.keys(sub)) {
			if (k.startsWith('on') && typeof sub[k] === 'function') noteCallback('ruleset', sub.id, k, sub[k]);
		}
	}
}

// ---------------------------------------------------------------------------
// Legality per format, computed with Showdown's own TeamValidator: for every species that can be
// on a team, the abilities / moves / items it may legally use.
const { TeamValidator } = require(path.join(psPath, 'dist', 'sim', 'team-validator'));
const legality = {};
for (const formatid of FORMATS) {
	const validator = TeamValidator.get(formatid);
	const out = { species: {}, items: [] };
	const allItems = Object.keys(items).map(id => dex.items.get(id)).filter(i => !i.isNonstandard);
	for (const id of Object.keys(species)) {
		const s = dex.species.get(id);
		if (!species[id].legal || s.battleOnly || s.isMega) continue;
		const learn = [...new Set([id, dex.toID(s.changesFrom), dex.toID(s.baseSpecies)]
			.flatMap(x => (x && learnsets[x]) || []))].filter(m => moves[m]);
		const baseSet = (extra) => Object.assign({
			name: s.baseSpecies, species: s.name, item: '', ability: Object.values(s.abilities)[0], moves: [],
			nature: 'Hardy', evs: { hp: 32, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, level: 50, gender: '',
		}, extra);
		const legalMoves = learn.filter(m => {
			const errors = validator.validateSet(baseSet({ moves: [dex.moves.get(m).name] }), {});
			return !errors;
		}).map(m => dex.moves.get(m).name);
		if (!legalMoves.length) continue;
		const probe = legalMoves[0];
		const legalAbilities = [...new Set(Object.values(s.abilities))].filter(a => (
			!validator.validateSet(baseSet({ ability: a, moves: [probe] }), {})
		));
		if (!legalAbilities.length) continue;
		const legalItems = allItems.filter(i => (
			!validator.validateSet(baseSet({ ability: legalAbilities[0], moves: [probe], item: i.name }), {})
		)).map(i => i.name);
		const required = s.requiredItem || (s.requiredItems || [])[0];
		if (required && !legalItems.includes(required)) continue;
		out.species[s.name] = { abilities: legalAbilities, moves: legalMoves, items: legalItems, num: s.num };
	}
	out.items = [...new Set(Object.values(out.species).flatMap(x => x.items))].sort();
	for (const sp of Object.values(out.species)) {
		// store per-species items only where they differ from the format-wide list (mega stones etc.)
		const own = new Set(sp.items);
		sp.itemsExcluded = out.items.filter(i => !own.has(i));
		delete sp.items;
	}
	out.evLimit = validator.ruleTable.evLimit;
	out.pickedTeamSize = validator.ruleTable.pickedTeamSize;
	out.minTeamSize = validator.ruleTable.minTeamSize;
	out.maxTeamSize = validator.ruleTable.maxTeamSize;
	legality[formatid] = out;
}

// Empty effects (what `dex.items.get('')` etc. return): the engine needs their exact default fields
// (e.g. an empty item has `isBerry: false`, which Stuff Cheeks' onTry returns).
const empties = {};
for (const [kind, table] of Object.entries({
	item: dex.items, ability: dex.abilities, move: dex.moves, species: dex.species, condition: dex.conditions,
})) {
	const e = table.get('');
	const o = {};
	for (const k in e) if (!['zMove', 'maxMove', 'tags'].includes(k)) o[k] = e[k];
	empties[kind] = plain(o, 'empty', kind);
}

// ---------------------------------------------------------------------------
let commit = '';
try {
	commit = child_process.execSync('git rev-parse HEAD', { cwd: psPath }).toString().trim();
} catch {}
const meta = {
	source: 'smogon/pokemon-showdown', mod: MOD, commit,
	exportedAt: new Date().toISOString(),
	counts: {
		species: Object.keys(species).length, legalSpecies: Object.values(species).filter(s => s.legal).length,
		moves: Object.keys(moves).length, abilities: Object.keys(abilities).length, items: Object.keys(items).length,
		conditions: Object.keys(conditions).length,
	},
};

fs.mkdirSync(outDir, { recursive: true });
const write = (name, obj) => fs.writeFileSync(path.join(outDir, name), JSON.stringify(obj, null, 0) + '\n');
write('species.json', species);
write('learnsets.json', learnsets);
write('moves.json', moves);
write('abilities.json', abilities);
write('items.json', items);
write('conditions.json', conditions);
write('typechart.json', typechart);
write('natures.json', natures);
write('formats.json', formats);
write('empties.json', empties);
write('legality.json', legality);
write('callbacks.json', callbacks);
write('meta.json', meta);
if (sourcesFile) fs.writeFileSync(sourcesFile, sources.join('\n'));
console.log(JSON.stringify(meta, null, 2));
