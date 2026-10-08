#!/usr/bin/env node
/**
 * Reference battle runner used for differential testing of the Python engine.
 *
 * Reads a JSON job list from stdin (one job per line) and writes one JSON
 * result per line to stdout. Each job is:
 *   {"formatid": "...", "seed": "1,2,3,4", "teams": [[set...], [set...]],
 *    "chooserSeed": "5,6,7,8", "maxTurns": 200, "switchChance": 0.15, "megaChance": 0.5}
 * Each result is:
 *   {"inputLog": [...], "log": [...], "winner": "...", "turns": n, "error": null}
 *
 * The battle is run with the real Pokemon Showdown simulator (champions mod);
 * players make random legal choices. The Python engine then replays the
 * recorded input log and must reproduce `log` exactly.
 *
 * Usage: node tools/showdown_runner.js <path-to-built-pokemon-showdown> < jobs.jsonl
 */
'use strict';
const path = require('path');
const readline = require('readline');

const psPath = path.resolve(process.argv[2] || '.');
const { Battle } = require(path.join(psPath, 'dist', 'sim', 'battle'));
const { PRNG } = require(path.join(psPath, 'dist', 'sim', 'prng'));

function range(a, b) { const r = []; for (let i = a; i <= b; i++) r.push(i); return r; }

function makeChooser(prng, opts) {
	const switchChance = opts.switchChance ?? 0.15;
	const megaChance = opts.megaChance ?? 0.5;

	function moveChoice(req, i, pokemon, chosen, state) {
		const active = req.active[i];
		if (pokemon[i].condition.endsWith(' fnt') || pokemon[i].commanding) return 'pass';
		const nActive = req.active.length;
		const hasAlly = nActive > 1 && pokemon[i ^ 1] && !pokemon[i ^ 1].condition.endsWith(' fnt');
		let moves = range(1, active.moves.length).filter(j => !active.moves[j - 1].disabled).map(j => {
			const m = active.moves[j - 1];
			let choice = `move ${j}`;
			if (nActive > 1) {
				if (['normal', 'any', 'adjacentFoe'].includes(m.target)) {
					if (m.target === 'any' && hasAlly && prng.random() < 0.1) choice += ` -${(i ^ 1) + 1}`;
					else choice += ` ${1 + prng.random(2)}`;
				} else if (m.target === 'adjacentAlly') {
					choice += ` -${(i ^ 1) + 1}`;
				} else if (m.target === 'adjacentAllyOrSelf') {
					choice += hasAlly ? ` -${1 + prng.random(2)}` : ` -${i + 1}`;
				}
			}
			return { choice, target: m.target };
		});
		const filtered = moves.filter(m => m.target !== 'adjacentAlly' || hasAlly);
		if (filtered.length) moves = filtered;
		const canSwitch = range(1, pokemon.length).filter(j => (
			!pokemon[j - 1].active && !chosen.includes(j) && !pokemon[j - 1].condition.endsWith(' fnt')
		));
		const switches = active.trapped ? [] : canSwitch;
		if (switches.length && (!moves.length || prng.random() < switchChance)) {
			const target = prng.sample(switches);
			chosen.push(target);
			return `switch ${target}`;
		}
		if (!moves.length) return 'move 1';
		let choice = prng.sample(moves).choice;
		if (active.canMegaEvo && !state.mega && prng.random() < megaChance) {
			state.mega = true;
			choice += ' mega';
		}
		return choice;
	}

	return function choose(req) {
		if (req.wait) return null;
		const pokemon = req.side.pokemon;
		if (req.teamPreview) {
			const order = range(1, pokemon.length);
			prng.shuffle(order);
			return 'team ' + order.join('');
		}
		if (req.forceSwitch) {
			const chosen = [];
			return req.forceSwitch.map((must, i) => {
				if (!must) return 'pass';
				const reviving = pokemon[i].reviving;
				const canSwitch = range(1, pokemon.length).filter(j => (
					(j > req.forceSwitch.length || reviving) && !chosen.includes(j) &&
					!pokemon[j - 1].condition.endsWith(' fnt') === !reviving
				));
				if (!canSwitch.length) return 'pass';
				const target = prng.sample(canSwitch);
				chosen.push(target);
				return `switch ${target}`;
			}).join(', ');
		}
		const chosen = [];
		const state = { mega: false };
		return req.active.map((a, i) => moveChoice(req, i, pokemon, chosen, state)).join(', ');
	};
}

function runJob(job) {
	const battle = new Battle({ formatid: job.formatid, seed: job.seed });
	battle.setPlayer('p1', { name: 'Player 1', team: job.teams[0] });
	battle.setPlayer('p2', { name: 'Player 2', team: job.teams[1] });
	const chooser = makeChooser(new PRNG(job.chooserSeed), job);
	const maxTurns = job.maxTurns || 300;
	// The original choice strings: Showdown's inputLog re-serialises choices, which is not always
	// replayable (e.g. locked moves keep their target location in doubles).
	const choices = [];
	let guard = 0;
	while (!battle.ended && battle.turn <= maxTurns) {
		if (++guard > 5000) throw new Error('runner guard exceeded');
		let progressed = false;
		for (const side of battle.sides) {
			if (!side.activeRequest || side.activeRequest.wait || side.isChoiceDone()) continue;
			let ok = false;
			for (let attempt = 0; attempt < 20 && !ok; attempt++) {
				const choice = chooser(side.activeRequest);
				if (choice === null) break;
				ok = battle.choose(side.id, choice);
				if (ok) choices.push(`>${side.id} ${choice}`);
				if (!ok && side.activeRequest && side.choice) side.clearChoice();
			}
			if (!ok && side.activeRequest && !side.activeRequest.wait && !side.isChoiceDone()) {
				if (battle.choose(side.id, 'default')) choices.push(`>${side.id} default`);
			}
			progressed = true;
			if (battle.ended) break;
		}
		if (!progressed) break;
	}
	return {
		inputLog: battle.inputLog,
		choices,
		log: battle.log,
		winner: battle.winner ?? null,
		turns: battle.turn,
		ended: battle.ended,
		error: null,
	};
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on('line', line => {
	if (!line.trim()) return;
	let result;
	try {
		result = runJob(JSON.parse(line));
	} catch (err) {
		result = { error: String(err && err.stack || err) };
	}
	process.stdout.write(JSON.stringify(result) + '\n');
});
