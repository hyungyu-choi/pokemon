/* pokechamp battle assistant - single page UI (vanilla JS, no build step).
 * Flow: 1) recommended team / my team  2) team preview (pick 3)  3) battle (record by selection, get advice)
 * All API requests use canonical English names; Korean names from /api/data are only for display.
 */
'use strict';
/* global Combobox */

// ---------------------------------------------------------------------------
// constants

const STORE_KEY = 'pokechamp-ui-v1';
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const STAT_KO = { hp: 'HP', atk: '공격', def: '방어', spa: '특공', spd: '특방', spe: '스피드' };
const STAT_ABBR = { hp: 'H', atk: 'A', def: 'B', spa: 'C', spd: 'D', spe: 'S' };
const TYPE_KO = {
  Normal: '노말', Fire: '불꽃', Water: '물', Electric: '전기', Grass: '풀', Ice: '얼음', Fighting: '격투',
  Poison: '독', Ground: '땅', Flying: '비행', Psychic: '에스퍼', Bug: '벌레', Rock: '바위', Ghost: '고스트',
  Dragon: '드래곤', Dark: '악', Steel: '강철', Fairy: '페어리', Stellar: '스텔라',
};
const NATURE_KO = {
  Adamant: '고집', Bashful: '수줍음', Bold: '대담', Brave: '용감', Calm: '차분', Careful: '신중', Docile: '온순',
  Gentle: '얌전', Hardy: '노력', Hasty: '성급', Impish: '장난꾸러기', Jolly: '명랑', Lax: '촐랑', Lonely: '외로움',
  Mild: '의젓', Modest: '조심', Naive: '천진난만', Naughty: '개구쟁이', Quiet: '냉정', Quirky: '변덕', Rash: '덜렁',
  Relaxed: '무사태평', Sassy: '건방', Serious: '성실', Timid: '겁쟁이',
};
const CAT_KO = { Physical: '물리', Special: '특수', Status: '변화' };
/**
 * "최선의 행동 계산" presets: samples = simulated battles with the opponent's hidden information (items, abilities,
 * moves, stats) filled in at random; depth = how many turns each action is simulated ahead before the neural network
 * judges the position. Times: 6 legal actions on a typical PC (more actions take longer).
 */
const PRESETS = [
  { id: 'fast', ko: '빠름', samples: 6, depth: 1, time: '2~5초' },
  { id: 'normal', ko: '보통', samples: 12, depth: 1, time: '4~10초' },
  { id: 'precise', ko: '정밀', samples: 24, depth: 2, time: '10~30초' },
];
const presetOf = (id) => PRESETS.find((p) => p.id === id) || PRESETS[1];
const presetText = (p) => `${p.ko} · 샘플 ${p.samples}개 · ${p.depth}턴 앞까지`;
// "교체 추천" runs one calculation per remaining Pokemon
const SWITCH_PRESET = { samples: 6, depth: 1 };
const TABS = [
  { id: 'team', n: '1', ko: '추천 파티 / 내 팀', short: '팀' },
  { id: 'preview', n: '2', ko: '선출', short: '선출' },
  { id: 'battle', n: '3', ko: '배틀', short: '배틀' },
  { id: 'help', n: '?', ko: '도움말', short: '도움말' },
];
const PSEUDO = ['trickroom', 'gravity', 'magicroom', 'wonderroom'];
const BOOST_RANGE = [6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6];

// ---------------------------------------------------------------------------
// small DOM helpers

const PROPS = new Set(['value', 'checked', 'disabled', 'selected', 'hidden', 'open', 'indeterminate', 'readOnly']);

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  const props = [];
  if (attrs) {
    for (const k of Object.keys(attrs)) {
      const v = attrs[k];
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (PROPS.has(k)) props.push([k, v]);
      else el.setAttribute(k, v === true ? '' : String(v));
    }
  }
  addKids(el, kids);
  for (const [k, v] of props) el[k] = v;
  return el;
}

function addKids(el, kids) {
  for (const c of kids) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) addKids(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
}

const $ = (sel, root) => (root || document).querySelector(sel);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const pct = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x) ? '–' : `${(x * 100).toFixed(d)}%`);
const uniq = (a) => [...new Set(a)];

function sel(id, opts, value, onChange, label) {
  const el = h('select', { id, 'aria-label': label || null },
    opts.map(([v, t]) => h('option', { value: String(v) }, t)));
  el.value = String(value);
  el.addEventListener('change', () => onChange(el.value, el));
  return el;
}

function fld(label, control, cls) {
  const target = control instanceof Node ? control : control.el;
  const id = control instanceof Node ? control.id : control.input.id;
  return h('div', { class: `fld ${cls || ''}` }, h('label', { for: id || null }, label), target);
}

function chk(id, label, checked, onChange, disabled, ariaLabel) {
  const input = h('input', { type: 'checkbox', id, checked: !!checked, disabled: !!disabled, 'aria-label': ariaLabel || null });
  input.addEventListener('change', () => onChange(input.checked));
  return h('label', { class: 'chk', for: id }, input, h('span', null, label));
}

function chipToggle(id, content, pressed, onToggle, title) {
  return h('button', {
    type: 'button', class: 'chip', id, 'aria-pressed': String(!!pressed), title: title || null,
    onclick: () => onToggle(!pressed),
  }, content);
}

function seg(id, label, options, value, onChange) {
  return h('div', { class: 'seg', role: 'radiogroup', 'aria-label': label, id },
    options.map(([v, t, title], i) => h('button', {
      type: 'button', role: 'radio', class: 'seg-btn', id: `${id}-${i}`, 'aria-checked': String(v === value),
      title: title || null, onclick: () => onChange(v),
    }, t)));
}

function card(title, opts, ...body) {
  const o = opts || {};
  return h('section', { class: `card ${o.cls || ''}`, id: o.id || null, 'aria-labelledby': o.id ? `${o.id}-t` : null },
    (title || o.actions) ? h('div', { class: 'card-h' },
      h('h2', { id: o.id ? `${o.id}-t` : null }, title),
      o.sub ? h('span', { class: 'card-sub' }, o.sub) : null,
      o.actions ? h('div', { class: 'card-actions' }, o.actions) : null) : null,
    h('div', { class: 'card-b' }, body));
}

function errBox(msg) {
  return msg ? h('div', { class: 'msg msg-err', role: 'alert' }, h('strong', null, '오류: '), msg) : null;
}

function spinner(text, cancel) {
  return h('div', { class: 'running', role: 'status', 'aria-live': 'polite' },
    h('span', { class: 'spin', 'aria-hidden': 'true' }), h('span', null, text, ' '),
    h('span', { class: 'elapsed', 'data-elapsed': '1' }, '0.0초'),
    cancel ? h('button', { type: 'button', class: 'btn btn-sm btn-ghost', onclick: cancelRun }, '취소') : null);
}

const narrowScreen = () => !!(window.matchMedia && window.matchMedia('(max-width: 640px)').matches);

/** A short message at the bottom; `action` = {label, fn} adds a button (e.g. 되돌리기). */
function toast(msg, kind, action) {
  const area = $('#toasts');
  if (!area) return;
  const el = h('div', { class: `toast toast-${kind || 'info'}`, role: kind === 'error' ? 'alert' : 'status' },
    h('span', { class: 'toast-msg' }, msg),
    action ? h('button', { type: 'button', class: 'toast-act', onclick: () => { el.remove(); action.fn(); } }, action.label) : null,
    h('button', { type: 'button', class: 'toast-x', 'aria-label': '알림 닫기', onclick: () => el.remove() }, '×'));
  area.appendChild(el);
  // a phone screen has room for one message only: the newest replaces the older one
  const max = narrowScreen() ? 1 : 2;
  while (area.children.length > max) area.firstChild.remove();
  setTimeout(() => el.remove(), action ? 8000 : kind === 'error' ? 6000 : 3000);
}

/** A button that asks for a second click when `needConfirm()` is true. */
function confirmBtn(attrs, text, confirmText, needConfirm, fn) {
  const b = h('button', Object.assign({ type: 'button' }, attrs), text);
  let armed = false;
  let timer = null;
  b.addEventListener('click', () => {
    if (armed || !needConfirm()) {
      clearTimeout(timer);
      armed = false;
      b.textContent = text;
      b.classList.remove('armed');
      fn();
      return;
    }
    armed = true;
    b.textContent = confirmText;
    b.classList.add('armed');
    timer = setTimeout(() => { armed = false; b.textContent = text; b.classList.remove('armed'); }, 4000);
  });
  return b;
}

async function copyText(text) {
  let ok = false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      ok = true;
    }
  } catch (e) { ok = false; }
  if (!ok) {
    const ta = h('textarea', { class: 'offscreen', 'aria-hidden': 'true', readonly: true });
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    ta.remove();
  }
  if (ok) toast('클립보드에 복사했습니다.');
  else toast('복사하지 못했습니다. "텍스트 보기"에서 직접 선택해 복사하세요.', 'error');
}

// ---------------------------------------------------------------------------
// API

async function api(path, body, signal) {
  let res;
  try {
    res = await fetch(path, body === undefined ? { signal } : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
    });
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    throw new Error('서버에 연결할 수 없습니다. "python -m pokechamp ui" 가 실행 중인지 확인하세요.');
  }
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  if (!res.ok || !data || data.error) {
    throw new Error((data && data.error) || `서버 오류 (HTTP ${res.status})`);
  }
  return data;
}

let RUN = null; // the AI request in flight: {kind, start, ctrl, quiet}
const ERR = {}; // last error per action (shown inline)
let scrollAfter = null;
let CANCEL_P = null; // a POST /api/cancel still on its way

const AI_KINDS = new Set(['advise', 'switch', 'preview']);

/** Stop the request in flight: abort the fetch and tell the server to stop computing (POST /api/cancel). */
function cancelRun(quiet) {
  if (!RUN) return;
  if (quiet === true) RUN.quiet = true;
  const aiCall = AI_KINDS.has(RUN.kind);
  RUN.ctrl.abort();
  if (aiCall) {
    CANCEL_P = fetch('/api/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
      .catch(() => null).then(() => { CANCEL_P = null; });
  }
}

async function runAI(kind, fn) {
  if (RUN) {
    toast('다른 AI 계산이 진행 중입니다. 끝난 뒤 다시 시도하세요.', 'error');
    return;
  }
  const ctrl = new AbortController();
  const run = { kind, start: performance.now(), ctrl, quiet: false };
  RUN = run;
  ERR[kind] = null;
  rerender();
  try {
    // a cancel sent just before must reach the server first, or it would stop this new request
    if (CANCEL_P) await CANCEL_P;
    await fn(ctrl.signal);
  } catch (e) {
    if (e.name === 'AbortError') {
      if (!run.quiet) toast('계산을 취소했습니다.');
    } else {
      ERR[kind] = e.message;
      toast(e.message, 'error');
    }
  } finally {
    RUN = null;
    rerender();
    if (scrollAfter) {
      const el = document.getElementById(scrollAfter);
      scrollAfter = null;
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }
}

function tickTimers() {
  if (!RUN) return;
  const t = ((performance.now() - RUN.start) / 1000).toFixed(1);
  document.querySelectorAll('[data-elapsed]').forEach((el) => { el.textContent = `${t}초`; });
}

// ---------------------------------------------------------------------------
// game data (names, option lists)

let D = null;

async function loadData() {
  const raw = await api('/api/data');
  D = { raw, species: new Map(), forme: new Map(), items: new Map(), natures: new Map(), cache: new Map(),
    moves: (n) => (n && Object.prototype.hasOwnProperty.call(raw.moves, n) ? raw.moves[n] : null) };
  for (const s of raw.species) {
    D.species.set(s.name, s);
    for (const m of s.megas || []) D.forme.set(m.forme, { ko: m.forme_ko, base: s.name, mega: m });
  }
  for (const it of raw.items) D.items.set(it.name, it);
  for (const n of raw.natures) D.natures.set(n.name, n);
}

function cached(key, fn) {
  if (!D.cache.has(key)) D.cache.set(key, fn());
  return D.cache.get(key);
}

function spKo(n) {
  const s = D.species.get(n);
  if (s) return s.ko || '';
  const f = D.forme.get(n);
  return f ? f.ko || '' : '';
}
const spMain = (n) => spKo(n) || n || '';
const spText = (n) => (spKo(n) ? `${spKo(n)} (${n})` : n);
const moveKo = (n) => (D.moves(n) ? D.moves(n).ko || '' : '');
const moveMain = (n) => moveKo(n) || n;
const itemKo = (n) => (D.items.get(n) ? D.items.get(n).ko || '' : '');
const itemMain = (n) => itemKo(n) || n;
const abilKo = (n) => (D.raw.abilities && D.raw.abilities[n]) || '';
const abilMain = (n) => abilKo(n) || n;
const natureKo = (n) => (D.natures.get(n) && D.natures.get(n).ko) || NATURE_KO[n] || '';
const typeKo = (t) => (D.raw.types && D.raw.types[t]) || TYPE_KO[t] || t;
const typeTypes = (n) => {
  const s = D.species.get(n);
  if (s) return s.types;
  const f = D.forme.get(n);
  return f ? f.mega.types : [];
};

/** "한카리아스 Garchomp" as two spans (Korean first, English small). */
function dual(ko, en, cls) {
  return h('span', { class: `dual ${cls || ''}` }, h('span', { class: 'd-main' }, ko || en),
    ko && ko !== en ? h('span', { class: 'd-en' }, en) : null);
}
const spDual = (n, cls) => dual(spKo(n), n, cls);
const moveDual = (n) => dual(moveKo(n), n);

function typeChip(t) {
  return h('span', { class: `type-chip t-${String(t).toLowerCase()}` }, typeKo(t));
}
const typeChips = (types) => h('span', { class: 'types' }, (types || []).map(typeChip));

function moveTypeOf(m) {
  const mv = D.moves(m);
  return mv ? mv.type : '';
}

function moveOpt(m, extraHint) {
  const mv = D.moves(m) || {};
  const hint = [CAT_KO[mv.category] || '', mv.basePower ? `위력 ${mv.basePower}` : '',
    mv.priority ? `우선도 ${mv.priority > 0 ? '+' : ''}${mv.priority}` : '', extraHint || ''].filter(Boolean).join(' · ');
  return {
    value: m, label: mv.ko || m, sub: mv.ko ? m : '', hint,
    badge: mv.type ? { text: typeKo(mv.type), cls: `type-chip t-${mv.type.toLowerCase()}` } : null,
  };
}

const optSpecies = () => cached('species', () => D.raw.species.map((s) => ({
  value: s.name, label: s.ko || s.name, sub: s.ko ? s.name : '', hint: s.types.map(typeKo).join('/'),
})));
const optMoves = (species) => cached(`mv:${species}`, () => {
  const s = D.species.get(species);
  return (s ? s.moves : []).map((m) => moveOpt(m));
});
const optItems = () => cached('items', () => D.raw.items.map((i) => ({
  value: i.name, label: i.ko || i.name, sub: i.ko ? i.name : '',
  hint: i.megaStone ? `메가스톤 · ${i.megaFor.map(spMain).join(', ')}` : '',
})));
const optItemsFoe = () => cached('itemsFoe', () => [
  { value: '?', label: '모름', sub: 'unknown', keys: ['unknown', '?'] },
  { value: '', label: '없음 / 소모됨', sub: 'none', keys: ['none', 'consumed', 'knocked off'] },
  ...optItems()]);
const optItemsReveal = () => cached('itemsReveal', () => [
  { value: '?', label: '변경 없음', sub: '', keys: ['-'] },
  { value: '', label: '없음 / 소모됨', sub: 'none', keys: ['none', 'consumed', 'knocked off'] },
  ...optItems()]);
const optAbilities = (species) => cached(`ab:${species}`, () => {
  const s = D.species.get(species);
  return (s ? s.abilities : []).map((a) => ({ value: a, label: abilKo(a) || a, sub: abilKo(a) ? a : '' }));
});
const optAbilitiesFoe = (species) => cached(`abf:${species}`, () => [
  { value: '?', label: '모름', sub: 'unknown', keys: ['unknown'] }, ...optAbilities(species)]);
const optAbilitiesReveal = (species) => cached(`abr:${species}`, () => [
  { value: '?', label: '변경 없음', sub: '' }, ...optAbilities(species)]);
const optNatures = () => cached('natures', () => D.raw.natures.map((n) => ({
  value: n.name, label: natureKo(n.name) || n.name, sub: natureKo(n.name) ? n.name : '',
  hint: n.plus ? `+${STAT_KO[n.plus]} −${STAT_KO[n.minus]}` : '무보정',
})));

function natureEffect(n) {
  const x = D.natures.get(n);
  if (!x) return '';
  return x.plus ? `+${STAT_KO[x.plus]} −${STAT_KO[x.minus]}` : '무보정';
}

function calcStats(set) {
  const sp = D.species.get(set.species);
  if (!sp) return null;
  const nat = D.natures.get(set.nature) || {};
  const out = {};
  for (const s of STATS) {
    const base = sp.baseStats[s];
    const ev = Number((set.evs || {})[s]) || 0;
    if (s === 'hp') { out.hp = base + ev + 75; continue; }
    let v = base + ev + 20;
    if (nat.plus === s) v = Math.floor((v * 110) / 100);
    else if (nat.minus === s) v = Math.floor((v * 90) / 100);
    out[s] = v;
  }
  return out;
}

const spLine = (evs) => STATS.filter((k) => Number((evs || {})[k])).map((k) => `${STAT_KO[k]} ${evs[k]}`).join(' / ') || '없음';
const statLine = (st) => (st ? STATS.map((k) => `${STAT_ABBR[k]}${st[k]}`).join(' ') : '');
const spTotal = (evs) => STATS.reduce((a, k) => a + (Number((evs || {})[k]) || 0), 0);

// ---------------------------------------------------------------------------
// persistent state

const emptySlot = () => ({ species: '', item: '', ability: '', nature: '', moves: ['', '', '', ''],
  evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } });

function defaultState() {
  return {
    v: 1, tab: 'team', team: null, builder: [0, 1, 2, 3, 4, 5].map(emptySlot),
    paste: { text: '', result: null }, foeTeam: ['', '', '', '', '', ''], preview: null,
    manual: { picked: [], lead: '' }, battle: null, preset: 'normal', advice: null, switchAdvice: null,
    autoAdvise: true, open: {},
  };
}

let S = defaultState();

function loadState() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    if (!s || s.v !== 1) return null;
    // saved before the presets had a depth: samples 6 / 12 / 24 were 빠름 / 보통 / 정밀
    if (s.preset === undefined && s.samples !== undefined) {
      s.preset = (PRESETS.find((p) => p.samples === Number(s.samples)) || PRESETS[1]).id;
    }
    delete s.samples;
    return s;
  } catch (e) {
    return null;
  }
}

let saveTimer = null;
function saveNow() {
  clearTimeout(saveTimer);
  try { localStorage.setItem(STORE_KEY, JSON.stringify(S)); } catch (e) { /* storage unavailable */ }
}
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveNow, 150);
}

// ---------------------------------------------------------------------------
// team helpers

function plainSet(s) {
  const evs = {};
  for (const k of STATS) evs[k] = Number((s.evs || {})[k]) || 0;
  return { species: s.species, name: s.name || s.species, item: s.item || '', ability: s.ability || '',
    moves: (s.moves || []).filter(Boolean), nature: s.nature || '', evs, level: 50 };
}
const teamSet = (name) => (S.team ? S.team.sets.find((s) => s.species === name) : null);
function maxHp(name) {
  const s = teamSet(name);
  if (!s) return 100;
  return (s.stats && s.stats.hp) || (calcStats(s) || {}).hp || 100;
}
function megaStoneOf(set) {
  if (!set) return null;
  const it = D.items.get(set.item);
  return it && it.megaStone && it.megaFor.includes(set.species) ? it : null;
}
const isChoice = (item) => /^Choice /.test(item || '');
/** "도구 사용/잃음" of my Pokemon (a Mega Stone can be neither lost nor consumed). */
const itemGone = (m, set) => !!(m.itemLost && set && set.item && !megaStoneOf(set));
/** Identity of a team: every field of every set, independent of the order of the Pokemon. */
const teamKey = (sets) => (sets || []).map((s) => {
  const p = plainSet(s);
  return [p.species, p.item, p.ability, p.nature, [...p.moves].sort().join(','), STATS.map((k) => p.evs[k]).join('.')].join('|');
}).sort().join('||');
const sameTeam = (sets) => !!(S.team && teamKey(sets) === teamKey(S.team.sets));
/** Using a different team (or starting over) resets the battle in progress: ask first. */
const resetsBattle = (sets) => !!(S.battle && !(sets && sameTeam(sets)));
const RESET_CONFIRM = '배틀이 초기화됩니다. 다시 누르면 확정';

// one-level undo for "use another team" / "새 배틀" (in memory: a reload forgets it)
let UNDO = null;
const UNDO_KEYS = ['team', 'foeTeam', 'preview', 'manual', 'battle', 'advice', 'switchAdvice'];

function snapshotForUndo() {
  if (!S.battle && !S.foeTeam.some(Boolean)) return null;
  return JSON.stringify(Object.fromEntries(UNDO_KEYS.map((k) => [k, S[k]])));
}

function offerUndo(snap, msg) {
  UNDO = snap;
  toast(msg, 'info', snap ? {
    label: '되돌리기',
    fn: () => {
      if (!UNDO) return;
      cancelRun(true);
      Object.assign(S, JSON.parse(UNDO));
      UNDO = null;
      saveNow();
      toast('이전 상태로 되돌렸습니다.', 'ok');
      go(S.battle ? 'battle' : S.team ? 'preview' : 'team');
    },
  } : null);
}

function useTeam(sets, text, label) {
  if (sameTeam(sets)) {
    // the team already in use: keep the battle in progress
    toast(S.battle ? '이미 사용 중인 팀입니다. 진행 중인 배틀로 돌아갑니다.' : '이미 사용 중인 팀입니다.');
    go(S.battle ? 'battle' : 'preview');
    return;
  }
  const snap = snapshotForUndo();
  cancelRun(true);
  S.team = { sets: sets.map((s) => Object.assign(plainSet(s), { stats: s.stats || calcStats(s) })), text, label };
  S.preview = null;
  S.battle = null;
  S.advice = null;
  S.switchAdvice = null;
  S.manual = { picked: [], lead: '' };
  saveNow();
  offerUndo(snap, `${label} 팀으로 설정했습니다. 이제 상대 엔트리를 입력하세요.`);
  go('preview');
}

// ---------------------------------------------------------------------------
// battle state helpers

const newMyMon = (fresh) => ({ hp: null, status: '', fainted: false, boosts: {}, mega: false, itemLost: false,
  locked: '', volatiles: [], sleep: 0, toxic: 0, fresh: !!fresh, subHp: 25 });
const newFoeMon = (fresh) => ({ hp: 100, status: '', fainted: false, boosts: {}, item: null, ability: null,
  moves: [], mega: false, volatiles: [], faster: [], slower: [], fresh: !!fresh, sleep: 0, toxic: 0, subHp: 25 });
const newField = () => ({ weather: '', weather_turns: 0, terrain: '', terrain_turns: 0, trickroom: 0, gravity: 0,
  magicroom: 0, wonderroom: 0 });
const newQuick = () => ({ foeMove: '', myMove: '', order: '', foeItem: '?', foeAbility: '?', foeSwitch: '',
  mySwitch: '', myHp: null, foeHp: null, myStatus: '?', foeStatus: '?', myMega: false, foeMega: false, tick: true });
// per-turn entries that describe one side's Pokemon on the field (cleared when that Pokemon changes)
const QUICK_MINE = { myMove: '', mySwitch: '', myHp: null, myStatus: '?', myMega: false, order: '' };
const QUICK_FOE = { foeMove: '', foeItem: '?', foeAbility: '?', foeSwitch: '', foeHp: null, foeStatus: '?', foeMega: false, order: '' };

/** Opponent species entered twice in the team preview (impossible under Species Clause). */
const foeDups = () => uniq(S.foeTeam.filter((x, i) => x && S.foeTeam.indexOf(x) !== i));
const foeEntryReady = () => S.foeTeam.every(Boolean) && !foeDups().length;

function myMon(name) {
  const b = S.battle;
  if (!b.me.mons[name]) b.me.mons[name] = newMyMon(false);
  const m = b.me.mons[name];
  if (m.hp === null || m.hp === undefined) m.hp = maxHp(name);
  return m;
}

function foeMon(name) {
  const b = S.battle;
  if (!b.foe.mons[name]) b.foe.mons[name] = newFoeMon(false);
  return b.foe.mons[name];
}

function startBattle(names, lead) {
  if (S.foeTeam.some((x) => !x)) {
    toast('상대 엔트리 6마리를 먼저 모두 입력하세요.', 'error');
    return;
  }
  if (foeDups().length) {
    toast(`상대 엔트리에 같은 포켓몬이 두 번 있습니다: ${foeDups().map(spMain).join(', ')}`, 'error');
    return;
  }
  cancelRun(true);
  const brought = [lead, ...names.filter((n) => n !== lead)];
  S.battle = {
    turn: 1,
    me: { brought, active: lead, mons: {}, side: {}, megaUsed: false },
    foe: { team: [...S.foeTeam], seen: [], active: '', mons: {}, side: {}, megaUsed: false },
    field: newField(), quick: newQuick(),
  };
  for (const n of brought) {
    S.battle.me.mons[n] = newMyMon(n === lead);
    S.battle.me.mons[n].hp = maxHp(n);
  }
  S.advice = null;
  S.switchAdvice = null;
  const fx = entryFieldEffects('me', lead);
  saveNow();
  toast(`배틀을 시작합니다. 상대 선봉을 선택하세요.${fx.length ? ` (${fieldNoticeText(fx)})` : ''}`);
  go('battle');
}

/**
 * Make `name` the Pokemon on the field. `fresh` = it switched in this turn; undefined = the user corrected the
 * active Pokemon by hand: a Pokemon taken off by mistake earlier in the same turn gets back what it had.
 */
function setActive(sideKey, name, fresh, dflt) {
  const b = S.battle;
  const side = b[sideKey];
  const prev = side.active;
  const m = sideKey === 'me' ? myMon(name) : foeMon(name);
  if (sideKey === 'foe' && !side.seen.includes(name)) side.seen.push(name);
  if (prev === name) {
    if (fresh !== undefined) m.fresh = fresh;
    return;
  }
  if (prev && side.mons[prev]) leaveField(side.mons[prev], b.turn);
  side.active = name;
  if (fresh === undefined && m.left && m.left.turn === b.turn) {
    Object.assign(m, { boosts: m.left.boosts || {}, volatiles: m.left.volatiles || [], locked: m.left.locked || '',
      fresh: !!m.left.fresh, toxic: m.left.toxic || 0 });
  } else {
    // default for a hand correction: on turn 1 the Pokemon on the field is the lead, which has just come in
    m.fresh = fresh !== undefined ? fresh : dflt !== undefined ? !!dflt : b.turn === 1;
    // a new stay on the field: its weather / terrain ability has not taken effect yet (see lateEntryEffects)
    delete m.abilityFx;
    delete m.fx;
    delete m.midTurn;
  }
  delete m.left;
  // what was entered in "이번 턴 기록" was about the previous Pokemon
  Object.assign(b.quick, sideKey === 'me' ? QUICK_MINE : QUICK_FOE);
}
const setFoeActive = (name, fresh, dflt) => setActive('foe', name, fresh, dflt);
const setMyActive = (name, fresh, dflt) => setActive('me', name, fresh, dflt);

/** The Pokemon leaves the field: boosts, volatiles, Choice lock and the toxic counter go away. */
function leaveField(m, turn) {
  if (turn !== undefined && !m.fainted) {
    m.left = { turn, boosts: m.boosts, volatiles: m.volatiles, locked: m.locked, fresh: m.fresh, toxic: m.toxic };
  }
  m.boosts = {};
  m.volatiles = [];
  m.locked = '';
  m.fresh = false;
  m.toxic = 0;
}

const nonZero = (o) => Object.fromEntries(Object.entries(o || {}).filter(([, v]) => Number(v)).map(([k, v]) => [k, Number(v)]));
const sideOut = (side) => Object.fromEntries(Object.entries(side || {}).filter(([, v]) => Number(v) > 0).map(([k, v]) => [k, Number(v)]));

/** The <state> object of API.md, built from the selections. */
function buildState() {
  const b = S.battle;
  const meP = {};
  for (const n of b.me.brought) {
    const m = myMon(n);
    const mx = maxHp(n);
    const set = teamSet(n);
    const o = { hp: `${m.fainted ? 0 : clamp(Math.round(Number(m.hp) || 0), 0, mx)}/${mx}`, status: m.fainted ? '' : m.status || '' };
    if (m.fainted) o.fainted = true;
    if (m.mega) o.mega = true;
    if (itemGone(m, set)) o.item = '';
    if (!m.fainted && m.status === 'slp') o.sleep_turns = Number(m.sleep) || 0;
    if (!m.fainted && m.status === 'tox') o.toxic_turns = Number(m.toxic) || 0;
    if (n === b.me.active) {
      const bo = nonZero(m.boosts);
      if (Object.keys(bo).length) o.boosts = bo;
      if (m.volatiles.length) o.volatiles = [...m.volatiles];
      if (m.volatiles.includes('substitute')) o.substitute_hp = (Number(m.subHp) || 25) / 100;
      if (m.locked && set && isChoice(set.item) && !itemGone(m, set)) o.locked_move = m.locked;
      if (m.fresh) o.fresh = true;
    }
    meP[n] = o;
  }
  const foeP = {};
  for (const n of b.foe.seen) {
    const m = foeMon(n);
    const o = { hp: `${m.fainted ? 0 : clamp(Math.round(Number(m.hp)), 0, 100)}%`, status: m.fainted ? '' : m.status || '' };
    if (m.fainted) o.fainted = true;
    if (m.item !== null && m.item !== undefined && m.item !== '?') o.item = m.item;
    if (m.ability && m.ability !== '?') o.ability = m.ability;
    const moves = uniq((m.moves || []).filter(Boolean));
    if (moves.length) o.moves = moves;
    if (m.mega) o.mega = true;
    if (!m.fainted && m.status === 'slp') o.sleep_turns = Number(m.sleep) || 0;
    if (!m.fainted && m.status === 'tox') o.toxic_turns = Number(m.toxic) || 0;
    if (n === b.foe.active) {
      const bo = nonZero(m.boosts);
      if (Object.keys(bo).length) o.boosts = bo;
      if (m.volatiles.length) o.volatiles = [...m.volatiles];
      if (m.volatiles.includes('substitute')) o.substitute_hp = (Number(m.subHp) || 25) / 100;
      if (m.fresh) o.fresh = true;
    }
    const ft = (m.faster || []).filter((x) => b.me.brought.includes(x));
    const st = (m.slower || []).filter((x) => b.me.brought.includes(x));
    if (ft.length) o.faster_than = ft;
    if (st.length) o.slower_than = st;
    foeP[n] = o;
  }
  const f = b.field;
  return {
    format: D.raw.format,
    turn: b.turn,
    me: { team: S.team.sets.map(plainSet), brought: [...b.me.brought], active: [b.me.active], pokemon: meP,
      side: sideOut(b.me.side), mega_used: !!b.me.megaUsed },
    foe: { team: [...b.foe.team], brought: [...b.foe.seen], active: b.foe.active ? [b.foe.active] : [], pokemon: foeP,
      side: sideOut(b.foe.side), mega_used: !!b.foe.megaUsed },
    field: { weather: f.weather || '', weather_turns: f.weather ? Number(f.weather_turns) || 0 : 0,
      terrain: f.terrain || '', terrain_turns: f.terrain ? Number(f.terrain_turns) || 0 : 0,
      trickroom: Number(f.trickroom) || 0, gravity: Number(f.gravity) || 0, magicroom: Number(f.magicroom) || 0,
      wonderroom: Number(f.wonderroom) || 0 },
  };
}

function stateKey() {
  try { return JSON.stringify(buildState()); } catch (e) { return ''; }
}

function adviseProblems() {
  const out = [];
  const b = S.battle;
  if (!S.team) return ['먼저 내 팀을 정하세요.'];
  if (!b) return ['먼저 선출 탭에서 배틀을 시작하세요.'];
  if (b.foe.team.some((x) => !x)) out.push('상대 엔트리 6마리가 모두 입력되지 않았습니다.');
  if (!b.me.active) out.push('내 출전 포켓몬을 선택하세요.');
  else if (myMon(b.me.active).fainted) out.push('내 출전 포켓몬이 쓰러졌습니다. "교체 추천"으로 다음 포켓몬을 고르세요.');
  if (!b.foe.active) out.push('상대 출전 포켓몬(선봉)을 선택하세요.');
  else if (foeMon(b.foe.active).fainted) out.push('상대 출전 포켓몬이 쓰러졌습니다. 상대가 다음에 내보낸 포켓몬을 선택하세요.');
  return out;
}

async function runAdvise() {
  const probs = adviseProblems();
  if (probs.length) {
    ERR.advise = probs.join(' ');
    ERR.adviseLocal = true;
    toast(probs[0], 'error');
    rerender();
    return;
  }
  const state = buildState();
  const key = JSON.stringify(state);
  const pr = presetOf(S.preset);
  const { samples, depth } = pr;
  const battle = S.battle;
  ERR.adviseLocal = false;
  await runAI('advise', async (signal) => {
    scrollAfter = 'results-card';
    const res = await api('/api/advise', { state, samples, depth }, signal);
    if (S.battle !== battle) { scrollAfter = null; return; } // the battle was reset meanwhile
    const recs = res.recommendations || [];
    if (!recs.length && !res.cancelled) {
      throw new Error(`이 상황을 계산하지 못했습니다${(res.log || []).length ? `: ${res.log.join(' / ')}` : '. 입력한 상황을 확인하세요.'}`);
    }
    S.advice = { res, key, turn: battle.turn, samples, depth, preset: pr.id, active: battle.me.active, foe: battle.foe.active };
    saveNow();
    if (res.cancelled) toast('계산이 중간에 멈춰 부분 결과만 표시합니다.', 'warn');
    else toast(`추천: ${recLabel(recs[0]).main} (승률 ${pct(recs[0].win_rate)})`, 'ok');
  });
}

async function runSwitchAdvice() {
  const b = S.battle;
  const fainted = b.me.active;
  const alive = b.me.brought.filter((n) => n !== fainted && !myMon(n).fainted);
  if (!alive.length) {
    toast('남은 포켓몬이 없습니다.', 'error');
    return;
  }
  if (!b.foe.active || foeMon(b.foe.active).fainted) {
    toast('상대 출전 포켓몬을 먼저 선택하세요.', 'error');
    return;
  }
  const state = buildState();
  await runAI('switch', async (signal) => {
    scrollAfter = 'faint-card';
    const res = await api('/api/advise_switch', Object.assign({ state }, SWITCH_PRESET), signal);
    if (S.battle !== b) { scrollAfter = null; return; } // the battle was reset meanwhile
    if (!(res.options || []).length) throw new Error(res.cancelled ? '계산이 중간에 멈췄습니다. 다시 눌러 주세요.' : '교체 후보를 계산하지 못했습니다.');
    S.switchAdvice = { res, fainted, turn: b.turn };
    saveNow();
    if (res.cancelled) toast('계산이 중간에 멈춰 일부 후보만 표시합니다.', 'warn');
  });
}

function sendIn(name) {
  setMyActive(name, true);
  S.switchAdvice = null;
  const fx = entryFieldEffects('me', name);
  saveNow();
  toast(`${spMain(name)}을(를) 내보냈습니다.${fx.length ? ` ${fieldNoticeText(fx)}` : ''}`, 'info', fx.length ? { label: '필드 확인', fn: showFieldCard } : null);
  rerender();
  if (S.autoAdvise && !adviseProblems().length) runAdvise();
}

// labels of AI actions ------------------------------------------------------

function recLabel(rec) {
  const mega = /Mega Evolve/.test(rec.label || '');
  const parts = (rec.meaning || []).filter(Boolean);
  if (!parts.length) return { main: rec.label_ko || rec.label, en: rec.label, type: '', kind: '' };
  const [kind, value] = parts[0];
  if (kind === 'switch') {
    return { main: `교체 → ${spMain(value)}`, en: rec.label, type: '', kind, value };
  }
  return { main: `${mega ? '메가진화 + ' : ''}${moveMain(value)}`, en: rec.label, type: moveTypeOf(value), kind, value };
}

function foeReplyLabel(label) {
  const m = /^(.+?): (.*)$/.exec(label || '');
  if (!m) return label;
  let act = m[2].replace(/ -> .*$/, '');
  let mega = false;
  if (act.startsWith('Mega Evolve + ')) { mega = true; act = act.slice(14); }
  const sw = /^switch to (.+)$/i.exec(act);
  const what = sw ? `교체 → ${spMain(sw[1])}` : moveMain(act);
  return `${spMain(m[1])}: ${mega ? '메가진화 + ' : ''}${what}`;
}

// ---------------------------------------------------------------------------
// rendering: shell

function go(tab) {
  // errors about the previous screen are no longer relevant
  document.querySelectorAll('#toasts .toast-error').forEach((t) => t.remove());
  S.tab = tab;
  save();
  render();
  window.scrollTo(0, 0);
}

function renderTabs() {
  const nav = $('#tabs');
  const badge = {
    team: S.team ? '✓' : '',
    preview: S.battle ? '✓' : '',
    battle: S.battle ? `T${S.battle.turn}` : '',
  };
  nav.replaceChildren(...TABS.map((t) => h('button', {
    type: 'button', class: 'tab', role: 'tab', id: `tab-${t.id}`, 'aria-selected': String(S.tab === t.id),
    tabindex: S.tab === t.id ? '0' : '-1', 'aria-controls': `panel-${t.id}`, onclick: () => go(t.id),
  }, h('span', { class: 'tab-n', 'aria-hidden': 'true' }, t.n), h('span', { class: 'tab-long' }, t.ko),
  h('span', { class: 'tab-short', 'aria-hidden': 'true' }, t.short),
  badge[t.id] ? h('span', { class: 'tab-badge' }, badge[t.id]) : null)));
}

/** Arrow keys / Home / End move between the step tabs (WAI-ARIA tabs pattern). */
function onTabKey(e) {
  const i = TABS.findIndex((t) => t.id === S.tab);
  const keys = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: TABS.length - 1 };
  if (!(e.key in keys) || i < 0) return;
  e.preventDefault();
  const next = TABS[(keys[e.key] + TABS.length) % TABS.length].id;
  go(next);
  const el = document.getElementById(`tab-${next}`);
  if (el) el.focus();
}

/**
 * Toasts sit above the sticky "최선의 행동 계산" bar when it is on screen. While it is hidden for the on-screen keyboard
 * (body.kb-open, phone) they keep its last height: a toast dropped into the bottom band, where the bar comes back,
 * would sit right where "기절" / "다음 턴" are tapped to end the typing.
 */
function syncBarHeight() {
  const bar = document.querySelector('#panel-battle:not([hidden]) .action-bar');
  if (bar && !bar.offsetHeight && document.body.classList.contains('kb-open')) return;
  document.documentElement.style.setProperty('--bar-h', `${bar ? bar.offsetHeight : 0}px`);
}

/*
 * On a phone the sticky bar is hidden while the on-screen keyboard is up (body.kb-open, see style.css), so it does
 * not cover the field being typed in. It must not come back in the middle of a tap: the tap that moves the focus
 * out of the field (e.g. on "기절" or "다음 턴" near the bottom of the screen) would otherwise end on the bar and
 * neither button would get the click. So the bar reappears only ~350 ms after the focus left the field and the
 * finger / mouse button is up, and it ignores taps (body.bar-settle) for another ~400 ms after it reappeared.
 */
const KB_FIELDS = 'input.cb-input, input[type="number"], input[type="text"], input[type="search"], textarea';
const KB = { timer: null, settle: null, down: false };
const isKbField = (el) => !!(el && el.matches && el.matches(KB_FIELDS) && !el.disabled && !el.readOnly);

function kbSet(open) {
  const cl = document.body.classList;
  if (open === cl.contains('kb-open')) return;
  cl.toggle('kb-open', open);
  clearTimeout(KB.settle);
  cl.toggle('bar-settle', !open);
  if (!open) KB.settle = setTimeout(() => { document.body.classList.remove('bar-settle'); }, 400);
  syncBarHeight();
}

function kbShowLater(ms) {
  clearTimeout(KB.timer);
  KB.timer = setTimeout(() => {
    if (isKbField(document.activeElement)) return; // the focus went on to another text field
    if (KB.down) { kbShowLater(150); return; } // wait until the finger / button is up
    kbSet(false);
  }, ms);
}

function watchKeyboard() {
  document.addEventListener('focusin', (e) => {
    if (!isKbField(e.target)) return;
    clearTimeout(KB.timer);
    kbSet(true);
  });
  document.addEventListener('focusout', () => {
    if (document.body.classList.contains('kb-open')) kbShowLater(350);
  });
  document.addEventListener('pointerdown', () => { KB.down = true; }, true);
  for (const ev of ['pointerup', 'pointercancel']) document.addEventListener(ev, () => { KB.down = false; }, true);
}

function render() {
  renderTabs();
  for (const t of TABS) {
    const p = document.getElementById(`panel-${t.id}`);
    p.hidden = S.tab !== t.id;
  }
  if (S.tab === 'team') renderTeam();
  else if (S.tab === 'preview') renderPreview();
  else if (S.tab === 'battle') renderBattle();
  else renderHelp();
  syncBarHeight();
  // the focused text field was replaced without a focusout (rerender() puts the focus back right after this)
  if (document.body.classList.contains('kb-open') && !isKbField(document.activeElement)) kbShowLater(350);
}

/**
 * Run `fn`, which replaces part of the page, without moving what the user is looking at: the focused control keeps
 * its place on the screen and gets the focus back. (Chrome's scroll anchoring prefers the focused element; once that
 * element has been replaced it would jump the page, e.g. by ~340 px after tapping "기절".)
 */
function keepPlace(fn) {
  const a = document.activeElement;
  const id = a && a !== document.body ? a.id : '';
  const x0 = window.scrollX;
  const y0 = window.scrollY;
  const top0 = id && !a.closest('.action-bar, .app-header, #toasts') ? a.getBoundingClientRect().top : null;
  fn();
  if (window.scrollX !== x0 || window.scrollY !== y0) window.scrollTo(x0, y0);
  const el = id ? document.getElementById(id) : null;
  if (el && top0 !== null && el.getClientRects().length) {
    const dy = el.getBoundingClientRect().top - top0;
    if (Math.abs(dy) >= 1) window.scrollBy(0, dy);
  }
  if (el && el !== a && !el.disabled) {
    Combobox.suppressOpen = true;
    try { el.focus({ preventScroll: true }); } finally { Combobox.suppressOpen = false; }
  }
}

/** Re-render the current tab, keeping keyboard focus on the same control (and the page where it was). */
function rerender() {
  keepPlace(render);
}

function detailsBlock(key, summary, body, defaultOpen) {
  const open = S.open[key] !== undefined ? S.open[key] : !!defaultOpen;
  const d = h('details', { class: 'block', id: `d-${key}`, open }, h('summary', null, summary), h('div', { class: 'block-b' }, body));
  d.addEventListener('toggle', () => { S.open[key] = d.open; save(); });
  return d;
}

// ---------------------------------------------------------------------------
// TAB 1: recommended team / my team

let REC = null; // {loading, data, error}

async function loadRecommended() {
  REC = { loading: true };
  try {
    REC = { data: await api('/api/recommended_teams') };
  } catch (e) {
    REC = { error: e.message };
  }
  if (S.tab === 'team') rerender();
}

function monCard(set, stats, opts) {
  const o = opts || {};
  const st = stats || set.stats || calcStats(set);
  const sp = D.species.get(set.species);
  const line = (label, value, cls) => h('div', { class: `mon-l ${cls || ''}` }, h('span', { class: 'lbl' }, label), h('span', { class: 'mon-v' }, value));
  return h('div', { class: `mon ${o.cls || ''}` },
    h('div', { class: 'mon-h' }, spDual(set.species, 'mon-name'), typeChips(sp ? sp.types : [])),
    line('도구', [set.item ? dual(itemKo(set.item), set.item) : h('span', { class: 'muted' }, '없음'),
      megaStoneOf(set) ? h('span', { class: 'tag tag-mega' }, '메가') : null]),
    line('특성', set.ability ? dual(abilKo(set.ability), set.ability) : '–'),
    line('성격', [set.nature ? dual(natureKo(set.nature), set.nature) : '–',
      set.nature ? h('span', { class: 'muted small' }, ` ${natureEffect(set.nature)}`) : null]),
    line('SP', spLine(set.evs), 'small'),
    st ? line('실능', statLine(st), 'small mono') : null,
    h('ul', { class: 'moves' }, (set.moves || []).filter(Boolean).map((m) => h('li', {
      class: `mv t-b-${(moveTypeOf(m) || 'none').toLowerCase()}`,
    }, moveDual(m)))));
}

function renderTeam() {
  const p = $('#panel-team');
  const kids = [];
  if (S.team) kids.push(currentTeamCard());
  kids.push(recommendedCard());
  kids.push(topSpeciesCard());
  kids.push(pasteCard());
  kids.push(builderCard());
  p.replaceChildren(...kids);
  if (!REC) loadRecommended();
}

function currentTeamCard() {
  const t = S.team;
  return card('현재 선택된 팀', {
    id: 'current-team', cls: 'card-accent', sub: t.label,
    actions: [
      h('button', { type: 'button', class: 'btn btn-primary', onclick: () => go('preview') }, '선출하러 가기 →'),
      h('button', { type: 'button', class: 'btn', onclick: () => copyText(t.text || '') }, '텍스트 복사'),
      h('button', { type: 'button', class: 'btn btn-ghost', onclick: () => { loadBuilderFrom(t.sets); } }, '빌더에서 편집'),
    ],
  }, h('div', { class: 'chips' }, t.sets.map((s) => h('span', { class: 'chip chip-static' }, spDual(s.species)))));
}

function recommendedCard() {
  let body;
  if (!REC || REC.loading) body = spinner('AI 추천 파티 불러오는 중…');
  else if (REC.error) {
    body = [errBox(REC.error), h('button', { type: 'button', class: 'btn', onclick: () => { REC = null; rerender(); } }, '다시 시도')];
  } else if (!REC.data.teams.length) {
    body = h('p', { class: 'muted' }, REC.data.note || '추천 팀이 없습니다. (python -m pokechamp evolve 로 팀 라이브러리를 만드세요)');
  } else {
    body = REC.data.teams.map((t) => {
      const isCur = sameTeam(t.sets);
      const label = `추천 #${t.rank}`;
      return h('article', { class: `team-card ${isCur ? 'is-current' : ''}`, 'aria-label': label },
        h('div', { class: 'team-h' },
          h('div', { class: 'team-title' }, h('span', { class: 'rank' }, `#${t.rank}`),
            h('span', { class: 'tag' }, t.source),
            t.fitness !== null && t.fitness !== undefined ? h('span', { class: 'muted small' }, `적합도 ${Number(t.fitness).toFixed(2)}`) : null,
            isCur ? h('span', { class: 'tag tag-ok' }, '사용 중') : null),
          h('div', { class: 'team-actions' },
            confirmBtn({ class: 'btn btn-primary', id: `use-team-${t.rank}` }, isCur && S.battle ? '배틀로 돌아가기' : '이 팀으로 배틀', RESET_CONFIRM,
              () => resetsBattle(t.sets), () => useTeam(t.sets.map((s, i) => Object.assign({}, s, { stats: t.stats && t.stats[i] })), t.text, label)),
            h('button', { type: 'button', class: 'btn', onclick: () => copyText(t.text) }, '텍스트 복사'))),
        h('div', { class: 'mons-grid' }, t.sets.map((s, i) => monCard(s, t.stats && t.stats[i]))),
        h('details', { class: 'export' }, h('summary', null, '텍스트 보기 (Showdown)'),
          h('textarea', { class: 'export-text', readonly: true, rows: 12, 'aria-label': `${label} Showdown 텍스트`, value: t.text })));
    });
  }
  return card('AI 추천 파티', {
    id: 'rec-card', sub: REC && REC.data && REC.data.generation ? `팀 빌딩 AI · ${REC.data.generation}세대 진화 결과` : '팀 빌딩 AI',
  }, h('p', { class: 'muted small' }, '자가 대전으로 진화시킨 팀 중 성적이 좋은 순서입니다. "이 팀으로 배틀"을 누르면 선출 단계로 넘어갑니다.'), body);
}

function topSpeciesCard() {
  const list = REC && REC.data ? REC.data.top_species || [] : [];
  if (!list.length) return null;
  return detailsBlock('top-species', h('span', null, '학습된 강한 포켓몬 ', h('span', { class: 'muted small' }, `(상위 ${list.length})`)),
    h('table', { class: 'tbl' },
      h('thead', null, h('tr', null, h('th', { scope: 'col' }, '#'), h('th', { scope: 'col' }, '포켓몬'),
        h('th', { scope: 'col', class: 'num' }, '승률'), h('th', { scope: 'col', class: 'num' }, '게임 수'))),
      h('tbody', null, list.map((x, i) => h('tr', null, h('td', null, i + 1),
        h('td', null, dual(x.ko || spKo(x.species), x.species), ' ', typeChips(typeTypes(x.species))),
        h('td', { class: 'num' }, pct(x.win_rate, 1)), h('td', { class: 'num' }, Math.round(x.games)))))), true);
}

// paste ----------------------------------------------------------------------

function parseResultView(res, onUse) {
  if (!res) return null;
  const ok = !res.problems.length;
  return h('div', { class: 'parse-res' },
    ok ? h('div', { class: 'msg msg-ok' }, `문제 없음 · ${res.sets.length}마리`) : h('div', { class: 'msg msg-warn', role: 'alert' },
      h('strong', null, `규칙 위반 ${res.problems.length}건`), h('ul', null, res.problems.map((x) => h('li', null, x)))),
    h('div', { class: 'mons-grid' }, res.sets.map((s) => monCard(s, s.stats))),
    res.sets.length ? h('div', { class: 'row' },
      ok ? confirmBtn({ class: 'btn btn-primary' }, '이 팀 사용', RESET_CONFIRM, () => resetsBattle(res.sets), onUse)
        : confirmBtn({ class: 'btn' }, '경고 무시하고 사용', S.battle ? `정말 사용? (${RESET_CONFIRM})` : '정말 사용? 다시 누르면 확정', () => true, onUse)) : null);
}

function pasteCard() {
  const ta = h('textarea', { id: 'paste-text', rows: 10, placeholder: 'Garchomp @ Choice Scarf\nAbility: Rough Skin\nSPs: 2 HP / 32 Atk / 32 Spe\nJolly Nature\n- Earthquake\n...', value: S.paste.text });
  ta.addEventListener('input', () => { S.paste.text = ta.value; save(); });
  const busy = RUN && RUN.kind === 'parse';
  const res = S.paste.result;
  return detailsBlock('paste', h('span', null, 'Showdown 텍스트로 팀 불러오기'), [
    fld('Showdown 내보내기 텍스트 (SPs: 줄 포함)', ta),
    h('div', { class: 'row' }, h('button', {
      type: 'button', class: 'btn', id: 'paste-parse', disabled: !!RUN,
      onclick: () => runAI('parse', async (signal) => {
        if (!S.paste.text.trim()) throw new Error('텍스트를 붙여 넣으세요.');
        S.paste.result = await api('/api/team/parse', { text: S.paste.text }, signal);
        save();
      }),
    }, busy ? '검증 중…' : '불러오기 · 검증')),
    errBox(ERR.parse),
    parseResultView(res, () => useTeam(res.sets, res.text, '붙여넣은 팀')),
  ], false);
}

// builder --------------------------------------------------------------------

function loadBuilderFrom(sets) {
  S.builder = [0, 1, 2, 3, 4, 5].map((i) => {
    const s = sets[i];
    if (!s) return emptySlot();
    const moves = (s.moves || []).slice(0, 4);
    while (moves.length < 4) moves.push('');
    return { species: s.species, item: s.item || '', ability: s.ability || '', nature: s.nature || '', moves,
      evs: Object.assign(emptySlot().evs, s.evs || {}) };
  });
  S.open.builder = true;
  S.builderResult = null;
  saveNow();
  rerender();
  const el = document.getElementById('d-builder');
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  toast('빌더로 불러왔습니다.');
}

function builderSets() {
  return S.builder.filter((b) => b.species).map((b) => plainSet(b));
}

function builderSlot(i) {
  const b = S.builder[i];
  const sp = D.species.get(b.species);
  const wrap = h('div', { class: 'slot', id: `bslot-${i}` });
  const total = spTotal(b.evs);
  const stats = sp ? calcStats(b) : null;
  const statsEl = h('div', { class: 'small mono muted', id: `bstats-${i}` }, stats ? `실능 ${statLine(stats)}` : '');
  const totalEl = h('span', { class: `sp-total ${total > 66 ? 'bad' : ''}`, id: `btotal-${i}` }, `합계 ${total} / 66`);
  const rerenderSlot = () => {
    save();
    const fresh = builderSlot(i);
    const old = document.getElementById(`bslot-${i}`);
    if (old) keepPlace(() => old.replaceWith(fresh));
  };
  const species = new Combobox({
    id: `b${i}-species`, options: optSpecies, value: b.species, clearable: true, label: `${i + 1}번 포켓몬`,
    placeholder: '포켓몬 검색 (예: ㅎㅋㄹ, garchomp)',
    onChange: (v) => {
      b.species = v;
      const s2 = D.species.get(v);
      if (s2) {
        if (!s2.abilities.includes(b.ability)) b.ability = s2.abilities[0] || '';
        b.moves = b.moves.map((m) => (s2.moves.includes(m) ? m : ''));
      }
      rerenderSlot();
    },
  });
  const item = new Combobox({
    id: `b${i}-item`, options: optItems, value: b.item, clearable: true, label: `${i + 1}번 도구`, placeholder: '도구 검색',
    onChange: (v) => { b.item = v; rerenderSlot(); },
  });
  const ability = new Combobox({
    id: `b${i}-ability`, options: () => optAbilities(b.species), value: b.ability, label: `${i + 1}번 특성`,
    placeholder: '특성', disabled: !sp, onChange: (v) => { b.ability = v; save(); },
  });
  const nature = new Combobox({
    id: `b${i}-nature`, options: optNatures, value: b.nature, label: `${i + 1}번 성격`, placeholder: '성격 검색',
    onChange: (v) => { b.nature = v; rerenderSlot(); },
  });
  const moves = b.moves.map((m, j) => new Combobox({
    id: `b${i}-move${j}`, options: () => optMoves(b.species), value: m, clearable: true, label: `${i + 1}번 기술 ${j + 1}`,
    placeholder: `기술 ${j + 1}`, disabled: !sp,
    onChange: (v) => {
      if (v && b.moves.some((x, k) => x === v && k !== j)) {
        toast('같은 기술을 두 번 넣을 수 없습니다.', 'error');
        moves[j].setValue(b.moves[j]);
        return;
      }
      b.moves[j] = v;
      save();
    },
  }));
  const spInputs = STATS.map((k) => {
    const input = h('input', {
      type: 'number', id: `b${i}-sp-${k}`, min: 0, max: 32, step: 1, inputmode: 'numeric', value: String(b.evs[k] || 0),
    });
    input.addEventListener('input', () => {
      const v = input.value === '' ? 0 : clamp(Math.round(Number(input.value) || 0), 0, 32);
      b.evs[k] = v;
      const t = spTotal(b.evs);
      totalEl.textContent = `합계 ${t} / 66`;
      totalEl.classList.toggle('bad', t > 66);
      const st = D.species.get(b.species) ? calcStats(b) : null;
      statsEl.textContent = st ? `실능 ${statLine(st)}` : '';
      save();
    });
    input.addEventListener('change', () => { input.value = String(b.evs[k]); });
    return h('div', { class: 'sp-in' }, h('label', { for: input.id }, STAT_KO[k]), input);
  });
  wrap.append(
    h('div', { class: 'slot-h' }, h('span', { class: 'slot-n' }, `${i + 1}`), fld('포켓몬', species, 'grow'),
      sp ? typeChips(sp.types) : null),
    h('div', { class: 'grid-2' }, fld('도구', item), fld('특성', ability)),
    fld('성격', nature),
    h('div', { class: 'grid-2 moves-in' }, moves.map((c, j) => fld(`기술 ${j + 1}`, c))),
    h('div', { class: 'sp-h' }, h('span', { class: 'lbl' }, 'SP (각 0~32)'), totalEl),
    h('div', { class: 'sp-grid' }, spInputs),
    statsEl,
  );
  return wrap;
}

function builderCard() {
  const busy = RUN && RUN.kind === 'builder';
  const res = S.builderResult;
  const validate = (use) => runAI('builder', async (signal) => {
    const sets = builderSets();
    if (!sets.length) throw new Error('포켓몬을 한 마리 이상 고르세요.');
    const r = await api('/api/team/parse', { sets }, signal);
    S.builderResult = r;
    save();
    if (use && !r.problems.length) useTeam(r.sets, r.text, '직접 만든 팀');
    else if (use) toast(`규칙 위반 ${r.problems.length}건 - 아래 목록을 확인하세요.`, 'error');
  });
  return detailsBlock('builder', h('span', null, '드롭다운으로 직접 팀 만들기'), [
    h('p', { class: 'muted small' }, 'Lv.50 · 각 능력치 SP 0~32, 합계 66 이하 · 같은 포켓몬/도구 중복 불가. 이름은 한글·영어·초성(ㅎㅋㄹ)으로 검색됩니다.'),
    h('div', { class: 'slots' }, S.builder.map((_, i) => builderSlot(i))),
    h('div', { class: 'row' },
      h('button', { type: 'button', class: 'btn', id: 'builder-validate', disabled: !!RUN, onclick: () => validate(false) }, busy ? '검증 중…' : '검증하기'),
      confirmBtn({ class: 'btn btn-primary', id: 'builder-use', disabled: !!RUN }, '검증 후 이 팀 사용', RESET_CONFIRM,
        () => resetsBattle(builderSets()), () => validate(true)),
      S.team ? h('button', { type: 'button', class: 'btn btn-ghost', onclick: () => loadBuilderFrom(S.team.sets) }, '현재 팀 불러오기') : null,
      confirmBtn({ class: 'btn btn-ghost' }, '모두 지우기', '다시 누르면 지웁니다', () => true, () => {
        S.builder = [0, 1, 2, 3, 4, 5].map(emptySlot);
        S.builderResult = null;
        saveNow();
        rerender();
      })),
    errBox(ERR.builder),
    res ? parseResultView(res, () => useTeam(res.sets, res.text, '직접 만든 팀')) : null,
  ], false);
}

// ---------------------------------------------------------------------------
// TAB 2: team preview

function renderPreview() {
  const p = $('#panel-preview');
  if (!S.team) {
    p.replaceChildren(card('선출', null, h('p', null, '먼저 1단계에서 사용할 팀을 정하세요.'),
      h('button', { type: 'button', class: 'btn btn-primary', onclick: () => go('team') }, '← 팀 고르기')));
    return;
  }
  const kids = [];
  if (S.battle) {
    kids.push(h('div', { class: 'msg msg-info' }, `진행 중인 배틀이 있습니다 (턴 ${S.battle.turn}). `,
      h('button', { type: 'button', class: 'btn btn-sm', onclick: () => go('battle') }, '배틀로 돌아가기')));
  }
  kids.push(card('내 팀', { sub: S.team.label, actions: h('button', { type: 'button', class: 'btn btn-ghost btn-sm', onclick: () => go('team') }, '변경') },
    h('div', { class: 'chips' }, S.team.sets.map((s) => h('span', { class: 'chip chip-static' }, spDual(s.species), typeChips(D.species.get(s.species) ? D.species.get(s.species).types : []))))));
  kids.push(foeEntryCard());
  kids.push(previewAICard());
  kids.push(manualPickCard());
  p.replaceChildren(...kids);
}

/** Species options for opponent preview box i: species already entered in another box are marked. */
function optFoeSpecies(i) {
  const taken = new Set(S.foeTeam.filter((x, k) => x && k !== i));
  if (!taken.size) return optSpecies();
  return optSpecies().map((o) => (taken.has(o.value)
    ? Object.assign({}, o, { hint: `이미 입력함 · ${o.hint}`, cls: 'cb-taken' }) : o));
}

function foeEntryCard() {
  const dups = foeDups();
  const boxes = S.foeTeam.map((v, i) => {
    const cb = new Combobox({
      id: `foe-pv-${i}`, options: () => optFoeSpecies(i), value: v, clearable: true, label: `상대 ${i + 1}번 포켓몬`,
      placeholder: `상대 ${i + 1}번 (검색)`,
      onChange: (val) => {
        S.foeTeam[i] = val;
        save();
        rerender();
        // move on to the next empty box for fast entry
        if (val) {
          const next = S.foeTeam.findIndex((x) => !x);
          const el = next >= 0 ? document.getElementById(`foe-pv-${next}`) : null;
          if (el && !(window.matchMedia && window.matchMedia('(pointer: coarse)').matches)) el.focus();
        }
      },
    });
    const sp = D.species.get(v);
    return h('div', { class: 'pv-slot' }, fld(`${i + 1}`, cb), sp ? typeChips(sp.types) : null);
  });
  const filled = S.foeTeam.filter(Boolean).length;
  return card('상대 엔트리 (팀 미리보기 6마리)', {
    id: 'foe-entry', sub: `${filled} / 6`,
    actions: confirmBtn({ class: 'btn btn-ghost btn-sm' }, '지우기', '다시 누르면 지웁니다', () => filled > 0, () => {
      S.foeTeam = ['', '', '', '', '', ''];
      S.preview = null;
      saveNow();
      rerender();
    }),
  }, h('div', { class: 'pv-grid' }, boxes),
  dups.length ? h('div', { class: 'msg msg-warn', role: 'alert' }, `중복된 포켓몬: ${dups.map(spMain).join(', ')} — 같은 포켓몬은 한 마리만 낼 수 있습니다. 하나를 고치세요.`) : null);
}

function previewAICard() {
  const busy = RUN && RUN.kind === 'preview';
  const ready = foeEntryReady();
  const pv = S.preview;
  const stale = pv && (pv.foe.join('|') !== S.foeTeam.join('|') || pv.teamKey !== teamKey(S.team.sets));
  const run = () => runAI('preview', async (signal) => {
    if (!S.foeTeam.every(Boolean)) throw new Error('상대 엔트리 6마리를 모두 입력하세요.');
    if (foeDups().length) throw new Error('상대 엔트리에 같은 포켓몬이 두 번 있습니다.');
    const foe = [...S.foeTeam];
    const tk = teamKey(S.team.sets);
    const res = await api('/api/preview', { my_team: S.team.sets.map(plainSet), foe, sims: 12 }, signal);
    if (!S.team || teamKey(S.team.sets) !== tk) return; // the team was changed meanwhile
    if (!(res.options || []).length) throw new Error(res.cancelled ? '계산이 중간에 멈췄습니다. 다시 눌러 주세요.' : '선출 조합을 계산하지 못했습니다.');
    S.preview = Object.assign(res, { foe, teamKey: tk });
    saveNow();
    if (res.cancelled) toast('계산이 중간에 멈춰 일부 조합만 표시합니다.', 'warn');
    else if (S.tab === 'preview') toast('선출 추천이 끝났습니다.', 'ok');
  });
  const best = pv && pv.options && pv.options.length ? pv.options[0] : null;
  return card('AI 선출 추천', { id: 'preview-ai' },
    h('p', { class: 'muted small' }, '내 6마리 중 3마리 조합(첫 번째 = 선봉)을 상대 엔트리와 시뮬레이션해 승률 순으로 보여줍니다. 10~20초 걸립니다.'),
    h('div', { class: 'row' }, h('button', {
      type: 'button', class: 'btn btn-primary btn-big', id: 'preview-run', disabled: !!RUN || !ready, onclick: run,
    }, busy ? '계산 중…' : 'AI 선출 추천'), !ready ? h('span', { class: 'muted small' }, foeDups().length ? '중복된 상대 포켓몬을 고치면 활성화됩니다.' : '상대 6마리를 모두 입력하면 활성화됩니다.') : null),
    busy ? spinner('선출 조합을 시뮬레이션하는 중…', true) : null,
    errBox(ERR.preview),
    pv && pv.options ? h('div', { class: 'pv-results' },
      stale ? h('div', { class: 'msg msg-warn' }, '엔트리가 바뀌었습니다. 다시 계산하세요.') : null,
      h('div', { class: 'muted small' }, `후보 ${pv.considered}개 중 상위 ${pv.options.length}개 시뮬레이션 · ${pv.elapsed}초`,
        pv.cancelled ? h('span', { class: 'tag tag-warn' }, ' 부분 결과 (중단됨)') : null),
      h('ol', { class: 'opt-list' }, pv.options.map((o, i) => h('li', { class: `opt ${o === best ? 'opt-best' : ''}` },
        h('div', { class: 'opt-main' },
          h('span', { class: 'rank' }, `#${i + 1}`),
          h('div', { class: 'opt-names' }, o.names.map((n, k) => h('span', { class: `pick ${k === 0 ? 'pick-lead' : ''}` },
            k === 0 ? h('span', { class: 'tag tag-lead' }, '선봉') : null, spDual(n))))),
        h('div', { class: 'opt-stats' },
          h('div', { class: 'winbar', 'aria-hidden': 'true' }, h('div', { class: 'winbar-fill', style: `width:${Math.round(o.win_rate * 100)}%` })),
          h('span', { class: 'big-num' }, pct(o.win_rate)), h('span', { class: 'muted small' }, ` 승률 (${o.games}판)`),
          o.policy !== null && o.policy !== undefined ? h('span', { class: 'muted small' }, ` · 정책망 ${pct(o.policy, 1)}`) : null),
        confirmBtn({ class: 'btn btn-primary btn-sm', id: `pv-start-${i}`, disabled: !ready }, '이 선출로 시작', '진행 중 배틀 초기화 - 다시 누르면 시작',
          () => !!(S.battle && S.battle.turn > 1), () => startBattle(o.names, o.lead)))))) : null);
}

function manualPickCard() {
  const m = S.manual;
  m.picked = m.picked.filter((n) => teamSet(n));
  if (m.lead && !m.picked.includes(m.lead)) m.lead = m.picked[0] || '';
  const ready = m.picked.length === 3 && foeEntryReady();
  return detailsBlock('manual-pick', h('span', null, '직접 선출하기 (3마리 + 선봉)'), [
    h('p', { class: 'muted small' }, '3마리를 고르고 선봉을 정하세요. 처음 고른 포켓몬이 기본 선봉입니다.'),
    h('div', { class: 'chips' }, S.team.sets.map((s, i) => chipToggle(`mp-${i}`, spDual(s.species), m.picked.includes(s.species), (on) => {
      if (on) {
        if (m.picked.length >= 3) { toast('3마리까지만 고를 수 있습니다.', 'error'); return; }
        m.picked.push(s.species);
        if (!m.lead) m.lead = s.species;
      } else {
        m.picked = m.picked.filter((x) => x !== s.species);
        if (m.lead === s.species) m.lead = m.picked[0] || '';
      }
      save();
      rerender();
    }))),
    m.picked.length ? h('div', { class: 'fld' }, h('span', { class: 'lbl' }, '선봉'),
      seg('mp-lead', '선봉 선택', m.picked.map((n) => [n, spMain(n)]), m.lead, (v) => { m.lead = v; save(); rerender(); })) : null,
    h('div', { class: 'row' }, confirmBtn({ class: 'btn btn-primary', id: 'mp-start', disabled: !ready || !!RUN }, '이 선출로 시작',
      '진행 중 배틀 초기화 - 다시 누르면 시작', () => !!(S.battle && S.battle.turn > 1), () => startBattle(m.picked, m.lead)),
    !ready ? h('span', { class: 'muted small' }, m.picked.length !== 3 ? `${m.picked.length}/3 선택` : foeDups().length ? '상대 엔트리의 중복을 고치세요.' : '상대 엔트리 6마리를 입력하세요.') : null),
  ], true);
}

// ---------------------------------------------------------------------------
// TAB 3: battle

function hpBar(frac) {
  const f = clamp(frac, 0, 1);
  const cls = f > 0.5 ? 'hp-g' : f > 0.2 ? 'hp-y' : 'hp-r';
  return h('div', { class: 'hpbar', role: 'img', 'aria-label': `HP ${Math.round(f * 100)}%` },
    h('div', { class: `hpbar-fill ${cls}`, style: `width:${(f * 100).toFixed(1)}%` }));
}

function statusKo(id) {
  const s = (D.raw.statuses || []).find((x) => x.id === id);
  return s ? s.ko : id;
}

function boostsText(b) {
  return Object.entries(nonZero(b)).map(([k, v]) => `${(D.raw.boosts.find((x) => x.id === k) || {}).ko || k}${v > 0 ? '+' : ''}${v}`).join(' ');
}

function renderBattle() {
  const p = $('#panel-battle');
  if (!S.team || !S.battle) {
    p.replaceChildren(card('배틀', null, h('p', null, '먼저 2단계(선출)에서 3마리를 고르고 "이 선출로 시작"을 누르세요.'),
      h('button', { type: 'button', class: 'btn btn-primary', onclick: () => go(S.team ? 'preview' : 'team') }, S.team ? '← 선출하러 가기' : '← 팀 고르기')));
    return;
  }
  const b = S.battle;
  const kids = [matchupCard()];
  if (!b.foe.active) kids.push(foeLeadCard());
  else if (foeMon(b.foe.active).fainted) kids.push(foeReplaceCard());
  if (myMon(b.me.active).fainted) kids.push(faintCard());
  kids.push(resultsCard());
  kids.push(quickCard());
  kids.push(h('div', { class: 'sides' }, mySideCard(), foeSideCard()));
  kids.push(fieldCard());
  kids.push(actionBar());
  p.replaceChildren(...kids);
}

function matchupCard() {
  const b = S.battle;
  const me = b.me.active;
  const mm = myMon(me);
  const mx = maxHp(me);
  const fa = b.foe.active;
  const fm = fa ? foeMon(fa) : null;
  const side = (who, name, frac, hpText, mon, extra) => h('div', { class: `vs-side vs-${who}` },
    h('div', { class: 'vs-who' }, who === 'me' ? '나' : '상대'),
    name ? spDual(name, 'vs-name') : h('div', { class: 'muted' }, '선봉 미선택'),
    name ? hpBar(frac) : null,
    name ? h('div', { class: 'vs-meta' }, h('span', { class: 'mono' }, hpText),
      mon.status ? h('span', { class: `tag st-${mon.status}` }, statusKo(mon.status)) : null,
      mon.fainted ? h('span', { class: 'tag tag-bad' }, '기절') : null,
      mon.mega ? h('span', { class: 'tag tag-mega' }, '메가') : null,
      boostsText(mon.boosts) ? h('span', { class: 'tag' }, boostsText(mon.boosts)) : null, extra) : null);
  return h('section', { class: 'card matchup', id: 'matchup', 'aria-label': '현재 대면' },
    h('div', { class: 'turnbox' },
      h('button', { type: 'button', class: 'btn btn-sm btn-round', id: 'turn-dec', 'aria-label': '턴 1 감소', onclick: () => { b.turn = Math.max(1, b.turn - 1); save(); rerender(); } }, '−'),
      h('div', { class: 'turn', 'aria-live': 'polite' }, h('span', { class: 'muted small' }, '턴'), h('strong', null, String(b.turn))),
      h('button', { type: 'button', class: 'btn btn-sm btn-round', id: 'turn-inc', 'aria-label': '턴 1 증가', onclick: () => { b.turn += 1; save(); rerender(); } }, '+')),
    h('div', { class: 'vs' },
      side('me', me, (mm.fainted ? 0 : mm.hp) / mx, `${mm.fainted ? 0 : mm.hp}/${mx}`, mm),
      h('div', { class: 'vs-x', 'aria-hidden': 'true' }, 'VS'),
      side('foe', fa, fm ? (fm.fainted ? 0 : fm.hp) / 100 : 0, fm ? `${fm.fainted ? 0 : fm.hp}%` : '', fm || {})));
}

function foeLeadCard() {
  const b = S.battle;
  return card('상대 선봉은?', { id: 'foe-lead', cls: 'card-accent' },
    h('p', { class: 'muted small' }, '배틀이 시작되면 상대가 처음 내보낸 포켓몬을 누르세요.'),
    h('div', { class: 'pick-grid' }, b.foe.team.map((n, i) => h('button', {
      type: 'button', class: 'btn pick-btn', id: `foe-lead-${i}`,
      onclick: () => {
        setFoeActive(n, b.turn === 1);
        // its ability may be known already (entered in 「상대 포켓몬」 beforehand): weather / terrain of the lead
        const fx = lateEntryEffects('foe', n);
        saveNow();
        rerender();
        fieldToast(fx);
      },
    }, spDual(n), typeChips(typeTypes(n))))));
}

function foeReplaceCard() {
  const b = S.battle;
  const left = b.foe.team.filter((n) => n !== b.foe.active && !(b.foe.mons[n] && b.foe.mons[n].fainted) &&
    (b.foe.seen.length < 3 || b.foe.seen.includes(n)));
  return card(`상대 ${spMain(b.foe.active)} 쓰러짐 — 다음 포켓몬은?`, { id: 'foe-replace', cls: 'card-accent' },
    left.length ? h('div', { class: 'pick-grid' }, left.map((n, i) => h('button', {
      type: 'button', class: 'btn pick-btn', id: `foe-next-${i}`,
      onclick: () => {
        setFoeActive(n, true);
        const fx = entryFieldEffects('foe', n);
        saveNow();
        rerender();
        fieldToast(fx);
      },
    }, spDual(n), typeChips(typeTypes(n))))) : h('p', null, '상대의 남은 포켓몬이 없습니다. 승리!'));
}

function faintCard() {
  const b = S.battle;
  const busy = RUN && RUN.kind === 'switch';
  const sa = S.switchAdvice && S.switchAdvice.fainted === b.me.active ? S.switchAdvice.res : null;
  const alive = b.me.brought.filter((n) => n !== b.me.active && !myMon(n).fainted);
  return card(`내 ${spMain(b.me.active)} 쓰러짐 — 누구를 내보낼까?`, { id: 'faint-card', cls: 'card-warn' },
    alive.length ? [
      h('div', { class: 'row' },
        h('button', { type: 'button', class: 'btn btn-primary', id: 'switch-advice', disabled: !!RUN, onclick: runSwitchAdvice }, busy ? '계산 중…' : '교체 추천'),
        h('span', { class: 'muted small' }, '또는 직접:'),
        alive.map((n) => h('button', { type: 'button', class: 'btn btn-sm', onclick: () => sendIn(n) }, spMain(n)))),
      busy ? spinner('남은 포켓몬마다 다음 턴을 계산하는 중 (보통 3~10초)…', true) : null,
      errBox(ERR.switch),
      sa ? h('ol', { class: 'opt-list' }, sa.options.map((o, i) => h('li', { class: `opt ${i === 0 ? 'opt-best' : ''}` },
        h('div', { class: 'opt-main' }, h('span', { class: 'rank' }, `#${i + 1}`), spDual(o.switch_to)),
        h('div', { class: 'opt-stats' },
          h('div', { class: 'winbar', 'aria-hidden': 'true' }, h('div', { class: 'winbar-fill', style: `width:${Math.round((o.win_rate || 0) * 100)}%` })),
          h('span', { class: 'big-num' }, pct(o.win_rate)), h('span', { class: 'muted small' }, ' 승률'),
          o.best_next_action ? h('span', { class: 'muted small' }, ` · 다음 행동: ${o.best_next_action}`) : null),
        h('button', { type: 'button', class: 'btn btn-primary btn-sm', id: `send-${i}`, onclick: () => sendIn(o.switch_to) }, '이 포켓몬 내보내기')))) : null,
    ] : h('p', null, '남은 포켓몬이 없습니다.'));
}

// results ---------------------------------------------------------------------

function resultsCard() {
  const a = S.advice;
  const busy = RUN && RUN.kind === 'advise';
  const kids = [];
  if (busy) {
    const pr = presetOf(S.preset);
    kids.push(spinner(`최선의 행동 계산 중 (${presetText(pr)} · 보통 ${pr.time}, 행동이 많거나 PC가 느리면 더 걸림)…`, true));
  }
  // a "cannot calculate yet" message disappears once the situation has been completed
  kids.push(errBox(ERR.advise && !(ERR.adviseLocal && !adviseProblems().length) ? ERR.advise : null));
  if (!a && !busy) {
    kids.push(h('p', { class: 'muted' }, '상황을 입력한 뒤 아래 ', h('strong', null, '"최선의 행동 계산"'), ' 버튼을 누르면 행동별 예상 승률이 표시됩니다.'));
  }
  if (a) {
    const res = a.res;
    const stale = a.key !== stateKey();
    if (stale) kids.push(h('div', { class: 'msg msg-warn' }, `턴 ${a.turn} 기준 결과입니다. 이후 상황이 바뀌었으니 다시 계산하세요.`));
    if (res.cancelled) kids.push(h('div', { class: 'msg msg-warn' }, h('span', { class: 'tag tag-warn' }, '부분 결과 (중단됨)'), ' 계산이 중간에 멈춰 일부 샘플만 반영됐습니다. 다시 계산하면 정확해집니다.'));
    const recs = res.recommendations || [];
    if (!recs.length) kids.push(h('p', { class: 'muted' }, '가능한 행동이 없습니다.'));
    else {
      kids.push(h('table', { class: 'tbl recs' },
        h('caption', { class: 'sr-only' }, '행동별 예상 승률'),
        h('thead', null, h('tr', null, h('th', { scope: 'col' }, '#'), h('th', { scope: 'col' }, '행동'),
          h('th', { scope: 'col', class: 'num' }, '승률'), h('th', { scope: 'col', class: 'num' }, '정책'))),
        h('tbody', null, recs.map((r, i) => {
          const L = recLabel(r);
          return h('tr', { class: i === 0 ? 'best' : '' },
            h('td', { class: 'rank-c' }, i === 0 ? h('span', { class: 'best-badge' }, '추천') : `${i + 1}`),
            h('td', null, h('div', { class: 'act' }, L.type ? typeChip(L.type) : null,
              h('span', { class: 'act-main' }, L.main)), h('div', { class: 'act-en' }, L.en)),
            h('td', { class: 'num' }, h('div', { class: 'win' }, h('strong', null, pct(r.win_rate, 1)),
              h('span', { class: 'muted small' }, ` ±${((r.stderr || 0) * 100).toFixed(1)}`)),
            h('div', { class: 'winbar', 'aria-hidden': 'true' }, h('div', { class: 'winbar-fill', style: `width:${Math.round(r.win_rate * 100)}%` }))),
            h('td', { class: 'num muted' }, r.policy !== null && r.policy !== undefined ? pct(r.policy, r.policy < 0.1 ? 1 : 0) : '–'));
        }))));
    }
    // the value network's one-look estimate of the position: secondary to the simulated win rates above
    if (res.value !== null && res.value !== undefined) {
      kids.push(h('div', { class: 'value-row', title: '신경망이 현재 국면만 보고 바로 낸 대략적인 값입니다. 행동별 승률은 실제로 시뮬레이션한 결과라 더 정확합니다.' },
        h('span', null, '현재 국면 승률 ', h('span', { class: 'value-note' }, '(신경망 즉석 추정 · 위 시뮬레이션 승률이 더 정확)')),
        h('div', { class: 'winbar winbar-thin', 'aria-hidden': 'true' }, h('div', { class: 'winbar-fill', style: `width:${Math.round(res.value * 100)}%` })),
        h('span', { class: 'value-num' }, pct(res.value, 1))));
    }
    const fr = res.foe_replies || [];
    if (fr.length) {
      kids.push(h('div', { class: 'sub-sec' }, h('h3', null, '상대의 예상 행동'),
        h('ul', { class: 'replies' }, fr.map((x) => h('li', null,
          h('span', { class: 'reply-l' }, foeReplyLabel(x.label)),
          h('div', { class: 'winbar', 'aria-hidden': 'true' }, h('div', { class: 'winbar-fill foe', style: `width:${Math.round(x.prob * 100)}%` })),
          h('span', { class: 'num' }, pct(x.prob)))))));
    }
    const bel = res.beliefs || {};
    const names = Object.keys(bel);
    kids.push(h('div', { class: 'sub-sec', id: 'beliefs-sec' }, h('h3', null, '상대 능력치 추정'),
      names.length ? h('div', { class: 'beliefs' }, names.map((n) => beliefView(n, bel[n])))
        : h('p', { class: 'muted small belief-none' }, '아직 관측 없음 — 같은 우선도 기술의 행동 순서를 기록하면 상대 스피드/구애스카프를 추정합니다')));
    const pr = a.preset ? presetOf(a.preset) : null;
    kids.push(h('div', { class: 'muted small res-meta' }, `턴 ${a.turn} · ${pr ? `${pr.ko} · ` : ''}샘플 ${res.samples || a.samples}개`,
      a.depth ? ` · ${a.depth}턴 앞까지` : '', ` · ${res.elapsed}초`,
      (res.log || []).length ? ` · 참고: ${res.log.join(' / ')}` : ''));
  }
  return card('AI 추천 행동', { id: 'results-card', cls: 'results' }, kids);
}

function beliefView(name, b) {
  if (!b) return null;
  const rows = [];
  if (b.speed_10_90) rows.push(['스피드', `${b.speed_10_90[0]}–${b.speed_10_90[1]}`, '(10~90% 범위)']);
  if (b.p_choice_scarf !== undefined && b.p_choice_scarf !== null) rows.push(['구애스카프', pct(b.p_choice_scarf), '확률']);
  if (b.max_hp_range) rows.push(['최대 HP', `${b.max_hp_range[0]}–${b.max_hp_range[1]}`, '']);
  if (b.mean_stats) rows.push(['평균 실능', statLine(b.mean_stats), '']);
  if (b.observations !== undefined) rows.push(['관측', `${b.observations}회`, '']);
  return h('div', { class: 'belief' }, h('div', { class: 'belief-h' }, spDual(name)),
    h('dl', null, rows.map(([k, v, s]) => [h('dt', null, k), h('dd', null, h('span', { class: 'mono' }, v), s ? h('span', { class: 'muted small' }, ` ${s}`) : null)])));
}

// quick per-turn entry ------------------------------------------------------------

const megasOf = (species) => ((D.species.get(species) || {}).megas || []);

/** The opponent's `m` (species `species`) has Mega Evolved; `forme` picks one of two Megas ('' = the only one / unknown). */
function setFoeMega(m, species, forme) {
  const megas = megasOf(species);
  const mg = forme ? megas.find((x) => x.forme === forme) : megas.length === 1 ? megas[0] : null;
  m.mega = true;
  S.battle.foe.megaUsed = true;
  // a Mega Evolved Pokemon holds its Mega Stone (it cannot lose it): the stone tells the AI which Mega it is
  if (mg && (forme || m.item === null || m.item === undefined)) m.item = mg.stone;
}

function unsetFoeMega(m, species) {
  m.mega = false;
  if (megasOf(species).some((x) => x.stone === m.item)) m.item = null;
  S.battle.foe.megaUsed = S.battle.foe.seen.some((x) => x !== species && foeMon(x).mega);
}

/** Which Mega forme the opponent's Pokemon is: '' = not Mega, '?' = Mega but the forme is not known. */
function foeMegaForme(m, species) {
  if (!m.mega) return '';
  const mg = megasOf(species).find((x) => x.stone === m.item);
  return mg ? mg.forme : '?';
}

/** "메가진화함" control for an opponent: a checkbox, or a forme select for species with two Megas. */
function foeMegaControl(id, m, species, onDone) {
  const megas = megasOf(species);
  if (megas.length > 1) {
    const opts = [['', '메가진화 안 함'], ['?', '메가진화함 (어느 쪽인지 모름)'],
      ...megas.map((x) => [x.forme, `${spMain(x.forme)} (${itemMain(x.stone)})`])];
    return fld('메가진화', sel(id, opts, foeMegaForme(m, species), (v) => {
      if (v) setFoeMega(m, species, v === '?' ? '' : v); else unsetFoeMega(m, species);
      onDone();
    }, `상대 ${spMain(species)} 메가진화`));
  }
  return chk(id, `메가진화함 (${megas.map((x) => spMain(x.forme)).join(' / ')})`, m.mega, (v) => {
    if (v) setFoeMega(m, species, ''); else unsetFoeMega(m, species);
    onDone();
  }, false, `상대 ${spMain(species)} 메가진화함`);
}

// abilities that change Speed (the AI compares plain Speed stats, so such turns are not recorded)
const WEATHER_SPEED = { 'Swift Swim': 'raindance', Chlorophyll: 'sunnyday', 'Sand Rush': 'sandstorm', 'Slush Rush': 'snowscape' };
function speedAbilityActive(ab, mon, itemLost, b) {
  if (!ab) return false;
  if (WEATHER_SPEED[ab]) return b.field.weather === WEATHER_SPEED[ab];
  if (ab === 'Surge Surfer') return b.field.terrain === 'electricterrain';
  if (ab === 'Quick Feet') return !!mon.status;
  if (ab === 'Unburden') return !!itemLost;
  return false;
}

/** Priority of `move` used by a Pokemon with ability `ab` (Prankster / Gale Wings change it). */
function effPriority(move, ab, hpFull) {
  const mv = D.moves(move) || {};
  let p = mv.priority || 0;
  if (ab === 'Prankster' && mv.category === 'Status') p += 1;
  if (ab === 'Gale Wings' && mv.type === 'Flying' && hpFull) p += 1;
  return p;
}

/**
 * "Who moved first" -> faster_than / slower_than of the opponent, only when the order says something about the
 * plain Speed stats the AI compares. Returns a note when nothing was recorded.
 */
function recordSpeedOrder(b, q, meA, foeA) {
  const mm = myMon(meA);
  const fm = foeMon(foeA);
  const mySet = teamSet(meA) || {};
  if (!q.myMove) return '내가 쓴 기술을 고르지 않아 스피드 비교는 기록하지 않았습니다.';
  // what was revealed this turn counts already (no switch happened when this is called)
  const foeAbility = q.foeAbility !== '?' ? q.foeAbility : fm.ability;
  const foeItem = q.foeItem !== '?' ? q.foeItem : fm.item;
  const foeAbs = foeAbility ? [foeAbility] : (D.species.get(foeA) || {}).abilities || [];
  const mp = effPriority(q.myMove, mySet.ability, Number(mm.hp) >= maxHp(meA));
  const fps = uniq((foeAbs.length ? foeAbs : ['']).map((a) => effPriority(q.foeMove, a, Number(fm.hp) >= 100)));
  if (fps.length > 1) return '상대 특성에 따라 우선도가 달라질 수 있어 스피드 비교는 기록하지 않았습니다.';
  if (fps[0] !== mp) return '우선도가 달라 스피드 비교는 기록하지 않았습니다.';
  const confound = [];
  if ((Number(b.me.side.tailwind) || 0) > 0 || (Number(b.foe.side.tailwind) || 0) > 0) confound.push('순풍');
  if (mm.status === 'par' || fm.status === 'par') confound.push('마비');
  if (Number((mm.boosts || {}).spe) || Number((fm.boosts || {}).spe)) confound.push('스피드 랭크 변화');
  // the AI uses the base forme's Speed: a Mega (or one Mega Evolving this turn) has a different one
  if (mm.mega || fm.mega || q.myMega || q.foeMega) confound.push('메가진화');
  if (mySet.item === 'Choice Scarf' && mm.itemLost) confound.push('구애스카프 잃음');
  if (speedAbilityActive(mySet.ability, mm, mm.itemLost, b)) confound.push(`내 특성 ${abilMain(mySet.ability)}`);
  const fa = foeAbs.find((a) => speedAbilityActive(a, fm, foeItem === '', b));
  if (fa) confound.push(`상대 특성 ${abilMain(fa)}${foeAbility ? '' : ' 가능성'}`);
  if (confound.length) return `${confound.join('·')} 때문에 스피드 비교는 기록하지 않았습니다.`;
  fm.faster = fm.faster || [];
  fm.slower = fm.slower || [];
  // the AI compares raw Speed stats: under Trick Room the slower Pokemon moves first
  const foeFaster = (q.order === 'foe') !== (Number(b.field.trickroom) > 0);
  if (foeFaster) {
    if (!fm.faster.includes(meA)) fm.faster.push(meA);
    fm.slower = fm.slower.filter((x) => x !== meA);
  } else {
    if (!fm.slower.includes(meA)) fm.slower.push(meA);
    fm.faster = fm.faster.filter((x) => x !== meA);
  }
  return '';
}

// field effects of what was recorded --------------------------------------------------------
// Moves used by either side and weather / terrain abilities of Pokemon entering the field change the field by
// themselves; a notice lists what was applied so it can be corrected in 「필드」. Counters mean "turns left at the
// start of the next decision": an effect set during a turn has already lost that turn (5-turn Trick Room -> 4).

const ROOM_MOVES = { 'Trick Room': 'trickroom', Gravity: 'gravity', 'Magic Room': 'magicroom', 'Wonder Room': 'wonderroom' };
const ROOM_TOGGLE = new Set(['trickroom', 'magicroom', 'wonderroom']); // used again while up: it ends
const WEATHER_MOVES = { 'Sunny Day': 'sunnyday', 'Rain Dance': 'raindance', Sandstorm: 'sandstorm', Snowscape: 'snowscape',
  'Chilly Reception': 'snowscape' };
const WEATHER_ABILITIES = { Drought: 'sunnyday', Drizzle: 'raindance', 'Sand Stream': 'sandstorm', 'Snow Warning': 'snowscape' };
const WEATHER_ROCKS = { sunnyday: 'Heat Rock', raindance: 'Damp Rock', sandstorm: 'Smooth Rock', snowscape: 'Icy Rock' };
const TERRAIN_MOVES = { 'Electric Terrain': 'electricterrain', 'Grassy Terrain': 'grassyterrain', 'Misty Terrain': 'mistyterrain',
  'Psychic Terrain': 'psychicterrain' };
const TERRAIN_ABILITIES = { 'Electric Surge': 'electricterrain', 'Grassy Surge': 'grassyterrain', 'Misty Surge': 'mistyterrain',
  'Psychic Surge': 'psychicterrain' };
// abilities that change the field when their holder is hit: applied when they are revealed
const ON_HIT_ABILITIES = { 'Sand Spit': ['weather', 'sandstorm'], 'Seed Sower': ['terrain', 'grassyterrain'] };
const SCREEN_MOVES = { Reflect: 'reflect', 'Light Screen': 'lightscreen', 'Aurora Veil': 'auroraveil' };
const SIDE_MOVES = { Tailwind: ['tailwind', 4], Safeguard: ['safeguard', 5] }; // on the user's side
const HAZARD_MOVES = { 'Stealth Rock': 'stealthrock', Spikes: 'spikes', 'Toxic Spikes': 'toxicspikes', 'Sticky Web': 'stickyweb' };
const HAZARD_HIT_MOVES = { 'Stone Axe': 'stealthrock', 'Ceaseless Edge': 'spikes' }; // damaging moves: on a hit
const HAZARDS = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb'];
const PROTECT_MOVES = new Set(['Protect', 'Detect', "King's Shield", 'Spiky Shield', 'Baneful Bunker', 'Silk Trap',
  'Burning Bulwark', 'Obstruct']);
// the user switches out after the move: the Pokemon coming in arrives after it
const PIVOT_MOVES = new Set(['U-turn', 'Volt Switch', 'Flip Turn', 'Parting Shot', 'Teleport', 'Baton Pass', 'Shed Tail',
  'Chilly Reception']);
const entryFx = (ab) => !!(ab && (WEATHER_ABILITIES[ab] || TERRAIN_ABILITIES[ab]));
/** The weather / terrain that ability `ab` sets is up on field `f`. */
function abilityOnField(f, ab) {
  const x = WEATHER_ABILITIES[ab] ? ['weather', WEATHER_ABILITIES[ab]]
    : TERRAIN_ABILITIES[ab] ? ['terrain', TERRAIN_ABILITIES[ab]] : ON_HIT_ABILITIES[ab];
  return !!x && f[x[0]] === x[1];
}

const fieldKo = (kind, id) => (((D.raw[kind] || []).find((x) => x.id === id) || {}).ko || id);
const sideKo = (side) => (side === 'me' ? '내 필드' : '상대 필드');
const megaFormeOf = (species, item) => megasOf(species).find((x) => x.stone === item) || null;

/** Types / ability of a Pokemon on the field (its Mega forme's when it has Mega Evolved and the forme is known). */
function fieldMon(side, name) {
  if (side === 'me') {
    const set = teamSet(name) || {};
    const mg = myMon(name).mega ? megaFormeOf(name, set.item) : null;
    return { types: mg ? mg.types : typeTypes(name), ability: (mg && mg.ability) || set.ability || '' };
  }
  const m = foeMon(name);
  const mg = m.mega ? megaFormeOf(name, m.item) : null;
  return { types: mg ? mg.types : typeTypes(name), ability: (mg && mg.ability) || m.ability || '' };
}

/** Ability / item of a Pokemon entering the field (mine: its set; the opponent's: what is known or revealed now). */
function entryInfo(side, name, q) {
  if (side === 'me') {
    const set = teamSet(name) || {};
    return { ability: set.ability || '', item: itemGone(myMon(name), set) ? '' : set.item || '' };
  }
  const m = foeMon(name);
  return { ability: q && q.foeAbility !== '?' ? q.foeAbility : m.ability || '',
    item: q && q.foeItem !== '?' ? q.foeItem : m.item };
}

/**
 * The field events of the turn recorded in `q`, in the order they happened (built from the state at the start of
 * the turn): switches (after the move for U-turn & co.), abilities of Pokemon entering, Mega Evolution, moves.
 */
function turnFieldEvents(b, q, mySet) {
  const meA = b.me.active;
  const foeA = b.foe.active;
  const mm = myMon(meA);
  const fm = foeMon(foeA);
  const foeAbility = !q.foeSwitch && q.foeAbility !== '?' ? q.foeAbility : fm.ability;
  const mv = { me: q.myMove, foe: q.foeMove };
  const sw = { me: q.mySwitch, foe: q.foeSwitch };
  // who moved first: as recorded, else by priority (ties: mine first)
  let first = q.order;
  if (!first) {
    const mp = mv.me ? effPriority(mv.me, (mySet || {}).ability, Number(mm.hp) >= maxHp(meA)) : 0;
    const fp = mv.foe ? effPriority(mv.foe, foeAbility || '', Number(fm.hp) >= 100) : 0;
    first = fp > mp ? 'foe' : 'me';
  }
  const at = { me: first === 'me' ? 2 : 3, foe: first === 'me' ? 3 : 2 };
  const ev = [];
  for (const side of ['me', 'foe']) {
    if (!sw[side]) continue;
    const t = mv[side] && PIVOT_MOVES.has(mv[side]) ? at[side] + 0.5 : 0;
    ev.push({ t, side, kind: 'switch', to: sw[side] });
    const inc = entryInfo(side, sw[side], side === 'foe' ? q : null);
    if (entryFx(inc.ability)) ev.push({ t: t + 0.1, side, kind: 'ability', name: inc.ability, item: inc.item });
  }
  // the opponent's ability revealed this turn without a switch (it came in earlier, or it was hit: Sand Spit). One
  // already entered in 「상대 포켓몬」 still counts while its weather / terrain is not on the field (it was entered
  // when it could not take effect any more), but not once it has taken effect since this Pokemon came in.
  const rv = q.foeAbility;
  if (!q.foeSwitch && rv !== '?' && (entryFx(rv) || ON_HIT_ABILITIES[rv]) && fm.abilityFx !== rv
    && (rv !== fm.ability || !abilityOnField(b.field, rv))) {
    ev.push({ t: 0.2, side: 'foe', kind: 'ability', name: rv, item: q.foeItem !== '?' ? q.foeItem : fm.item });
  }
  // Mega Evolution happens before the moves, and the Mega's ability takes effect (e.g. Mega Charizard Y: Drought)
  if (q.myMega && mySet) {
    const mg = megaFormeOf(meA, mySet.item);
    if (mg && entryFx(mg.ability)) ev.push({ t: 1, side: 'me', kind: 'ability', name: mg.ability, item: mySet.item });
  }
  if (q.foeMega) {
    const megas = megasOf(foeA);
    const mg = typeof q.foeMega === 'string' ? megas.find((x) => x.forme === q.foeMega) : megas.length === 1 ? megas[0] : null;
    if (mg && entryFx(mg.ability)) ev.push({ t: 1.1, side: 'foe', kind: 'ability', name: mg.ability, item: mg.stone });
  }
  const items = { me: mySet && !itemGone(mm, mySet) ? mySet.item || '' : '', foe: !q.foeSwitch && q.foeItem !== '?' ? q.foeItem : fm.item };
  for (const side of ['me', 'foe']) {
    if (mv[side]) ev.push({ t: at[side], side, kind: 'move', name: mv[side], item: items[side] });
  }
  return ev.sort((x, y) => x.t - y.t);
}

/**
 * Apply field events to the battle. opts: start = {me, foe} actives when the events begin, protect = {me, foe}
 * (that side used Protect & co. this turn), endOfTurn = the events happened during a turn that is now over (what they
 * set loses one turn). Returns {notes: [Korean text], fresh: Set of the counters set (for tickField)}.
 */
function applyFieldEvents(b, events, opts) {
  const o = opts || {};
  const f = b.field;
  const notes = [];
  const fresh = new Set();
  const cur = Object.assign({ me: b.me.active, foe: b.foe.active }, o.start || {});
  const protect = o.protect || {};
  const left = (n) => n - (o.endOfTurn ? 1 : 0);
  const other = (side) => (side === 'me' ? 'foe' : 'me');
  const layersOf = (id) => ((D.raw.side_conditions || []).find((c) => c.id === id) || {}).layers || 1;
  const ext = (item, want, n) => (item && item === want ? [8, ` (${itemMain(want)})`] : [n, '']);
  const setWeather = (w, item) => {
    if (f.weather === w) { notes.push(`${fieldKo('weathers', w)} 이미 있음`); return; }
    const [n, why] = ext(item, WEATHER_ROCKS[w], 5);
    Object.assign(f, { weather: w, weather_turns: n });
    fresh.add('weather');
    notes.push(`${fieldKo('weathers', w)} ${left(n)}턴${why}`);
  };
  const setTerrain = (t, item) => {
    if (f.terrain === t) { notes.push(`${fieldKo('terrains', t)} 이미 있음`); return; }
    const [n, why] = ext(item, 'Terrain Extender', 5);
    Object.assign(f, { terrain: t, terrain_turns: n });
    fresh.add('terrain');
    notes.push(`${fieldKo('terrains', t)} ${left(n)}턴${why}`);
  };
  const clearTerrain = (why) => {
    if (!f.terrain) return;
    notes.push(`${fieldKo('terrains', f.terrain)} 사라짐 (${why})`);
    Object.assign(f, { terrain: '', terrain_turns: 0 });
  };
  const clearConds = (side, ids, why) => {
    const so = b[side].side;
    const gone = ids.filter((c) => Number(so[c]) > 0);
    for (const c of gone) { delete so[c]; fresh.delete(`${side}:${c}`); }
    if (gone.length) notes.push(`${sideKo(side)} ${gone.map((c) => fieldKo('side_conditions', c)).join('·')} 제거 (${why})`);
  };
  const addHazard = (side, c) => {
    const so = b[side].side;
    const max = layersOf(c);
    const v = Number(so[c]) || 0;
    if (v >= max) { notes.push(`${sideKo(side)} ${fieldKo('side_conditions', c)} 이미 ${max > 1 ? '최대' : '있음'}`); return; }
    so[c] = v + 1;
    notes.push(`${sideKo(side)} ${fieldKo('side_conditions', c)}${max > 1 ? ` ${v + 1}층` : ''}`);
  };
  const addTimed = (side, c, n0, item) => {
    const so = b[side].side;
    if (Number(so[c]) > 0) { notes.push(`${sideKo(side)} ${fieldKo('side_conditions', c)} 이미 있음`); return; }
    const [n, why] = ext(item, 'Light Clay', n0);
    so[c] = n;
    fresh.add(`${side}:${c}`);
    notes.push(`${sideKo(side)} ${fieldKo('side_conditions', c)} ${left(n)}턴${why}`);
  };
  for (const ev of events) {
    const side = ev.side;
    if (ev.kind === 'switch') { cur[side] = ev.to; continue; }
    if (ev.kind === 'ability') {
      const a = ev.name;
      const hit = ON_HIT_ABILITIES[a];
      // remembered for this stay on the field: revealing it again later must not apply it a second time
      if (entryFx(a)) {
        const mon = side === 'me' ? myMon(cur.me) : foeMon(cur.foe);
        mon.abilityFx = a;
        if (mon.fx && mon.fx.ability !== a) delete mon.fx;
      }
      if (WEATHER_ABILITIES[a]) setWeather(WEATHER_ABILITIES[a], ev.item);
      else if (TERRAIN_ABILITIES[a]) setTerrain(TERRAIN_ABILITIES[a], ev.item);
      else if (hit && hit[0] === 'weather') setWeather(hit[1], ev.item);
      else if (hit) setTerrain(hit[1], ev.item);
      continue;
    }
    const m = ev.name;
    const tside = other(side);
    const target = fieldMon(tside, cur[tside]);
    const moveKo2 = moveMain(m);
    // moves that hit the opposing Pokemon do nothing when it protected itself
    const blocked = !!protect[tside];
    if (ROOM_MOVES[m]) {
      const k = ROOM_MOVES[m];
      const ko = fieldKo('pseudo', k);
      if (Number(f[k]) > 0) {
        if (ROOM_TOGGLE.has(k)) { f[k] = 0; fresh.delete(k); notes.push(`${ko} 해제 (${ko} 중에 다시 사용)`); } else notes.push(`${ko} 이미 있음`);
      } else { f[k] = 5; fresh.add(k); notes.push(`${ko} ${left(5)}턴`); }
    } else if (WEATHER_MOVES[m]) setWeather(WEATHER_MOVES[m], ev.item);
    else if (TERRAIN_MOVES[m]) setTerrain(TERRAIN_MOVES[m], ev.item);
    else if (SCREEN_MOVES[m]) {
      if (m === 'Aurora Veil' && f.weather !== 'snowscape') notes.push(`${moveKo2} 실패로 처리 (설경이 아님)`);
      else addTimed(side, SCREEN_MOVES[m], 5, ev.item);
    } else if (SIDE_MOVES[m]) addTimed(side, SIDE_MOVES[m][0], SIDE_MOVES[m][1], '');
    else if (HAZARD_MOVES[m]) addHazard(tside, HAZARD_MOVES[m]);
    else if (HAZARD_HIT_MOVES[m]) { if (!blocked) addHazard(tside, HAZARD_HIT_MOVES[m]); } else if (m === 'Defog') {
      if (blocked || target.ability === 'Good as Gold') notes.push(`${moveKo2} 실패로 처리 (${blocked ? '방어' : abilMain('Good as Gold')})`);
      else {
        clearConds(tside, ['reflect', 'lightscreen', 'auroraveil', 'safeguard', ...HAZARDS], moveKo2);
        clearConds(side, HAZARDS, moveKo2);
        clearTerrain(moveKo2);
      }
    } else if (m === 'Rapid Spin' || m === 'Mortal Spin') {
      const immune = m === 'Rapid Spin' ? target.types.includes('Ghost') : target.types.includes('Steel');
      if (!blocked && !immune) clearConds(side, HAZARDS, moveKo2);
    } else if (m === 'Tidy Up') {
      clearConds(side, HAZARDS, moveKo2);
      clearConds(tside, HAZARDS, moveKo2);
    } else if (m === 'Court Change') {
      const mine = b.me.side;
      b.me.side = b.foe.side;
      b.foe.side = mine;
      for (const k of [...fresh]) {
        if (/^(me|foe):/.test(k)) { fresh.delete(k); fresh.add(k.replace(/^(me|foe):/, (x, s1) => `${other(s1)}:`)); }
      }
      notes.push(`양쪽 필드 상태 교체 (${moveKo2})`);
    } else if ((m === 'Ice Spinner' || m === 'Steel Roller') && !blocked) clearTerrain(moveKo2);
  }
  return { notes, fresh };
}

/** [kind, id] of the weather / terrain an entry ability sets (Drizzle -> ['weather', 'raindance']). */
const fxOf = (ab) => (WEATHER_ABILITIES[ab] ? ['weather', WEATHER_ABILITIES[ab]] : TERRAIN_ABILITIES[ab] ? ['terrain', TERRAIN_ABILITIES[ab]] : null);
const fxKo = (kind, id) => fieldKo(`${kind}s`, id);

/**
 * What an entry ability set (m.fx, see entryFieldEffects) is still on the field, counting down from then: it was not
 * replaced, cleared or corrected by hand since.
 */
function fxOnField(f, fx, turn) {
  return !!fx && fx.applied && f[fx.kind] === fx.id && Number(f[`${fx.kind}_turns`]) === fx.n - (turn - fx.turn);
}

/**
 * Actual Speed range [lo, hi] of a Pokemon coming in (Lv.50): mine from its set; the opponent's from its base stat
 * (any nature and SP, and a Choice Scarf unless its item is known). Paralysis halves it.
 */
function entrySpeedRange(side, name) {
  if (side === 'me') {
    const set = teamSet(name);
    const spe = set && ((set.stats && set.stats.spe) || (calcStats(set) || {}).spe);
    if (!spe) return null;
    const m = myMon(name);
    let v = set.item === 'Choice Scarf' && !itemGone(m, set) ? Math.floor(spe * 1.5) : spe;
    if (m.status === 'par') v = Math.floor(v / 2);
    return [v, v];
  }
  const sp = D.species.get(name);
  if (!sp || !sp.baseStats) return null;
  const m = foeMon(name);
  const base = sp.baseStats.spe;
  let lo = Math.floor(((base + 20) * 90) / 100);
  let hi = Math.floor(((base + 32 + 20) * 110) / 100);
  const unknown = m.item === null || m.item === undefined;
  if (m.item === 'Choice Scarf') lo = Math.floor(lo * 1.5);
  if (unknown || m.item === 'Choice Scarf') hi = Math.floor(hi * 1.5);
  if (m.status === 'par') { lo = Math.floor(lo / 2); hi = Math.floor(hi / 2); }
  return [lo, hi];
}

/**
 * Which of my `myName` and the opponent's `foeName`, coming in at the same moment, has its ability take effect
 * first: 'me', 'foe', or '' when that cannot be told (speed tie, ranges overlap). Recorded speed comparisons count;
 * Trick Room reverses the order.
 */
function firstOnEntry(myName, foeName) {
  const fm = foeMon(foeName);
  let first = '';
  if ((fm.faster || []).includes(myName)) first = 'foe';
  else if ((fm.slower || []).includes(myName)) first = 'me';
  else {
    const mine = entrySpeedRange('me', myName);
    const theirs = entrySpeedRange('foe', foeName);
    if (mine && theirs) first = theirs[0] > mine[1] ? 'foe' : theirs[1] < mine[0] ? 'me' : '';
  }
  if (first && Number(S.battle.field.trickroom) > 0) first = first === 'me' ? 'foe' : 'me';
  return first;
}

/**
 * A Pokemon entered between turns (lead, replacement after a faint): its weather / terrain ability (`q`: the
 * per-turn reveals count, see entryInfo). `endOfTurn`: it came in during the turn just recorded, so what it set has
 * already lost that turn. What it set is remembered in m.fx (the field it replaced, for undoEntryFx; its turns, for
 * retimeEntryFx).
 */
function entryFieldEffects(side, name, q, endOfTurn) {
  const b = S.battle;
  const f = b.field;
  const inc = entryInfo(side, name, q || null);
  if (!entryFx(inc.ability)) return [];
  const m = side === 'me' ? myMon(name) : foeMon(name);
  const [kind, id] = fxOf(inc.ability);
  const label = `${side === 'me' ? '내' : '상대'} ${spMain(name)} 등장 (${abilMain(inc.ability)})`;
  // both Pokemon came in at the same moment (the leads, or replacements after both fainted) and the other one's
  // weather / terrain is up: the abilities take effect fastest first, so the slower one's is what stays
  const os = side === 'me' ? 'foe' : 'me';
  const on = b[os].active;
  const om = on ? (os === 'me' ? myMon(on) : foeMon(on)) : null;
  let unsure = '';
  if (!endOfTurn && om && om.fresh && !om.midTurn && !om.fainted && !m.midTurn && om.fx && om.fx.kind === kind
    && om.fx.turn === b.turn && fxOnField(f, om.fx, b.turn)) {
    const first = side === 'me' ? firstOnEntry(name, on) : firstOnEntry(on, name);
    if (first === side) {
      m.abilityFx = inc.ability;
      m.fx = { ability: inc.ability, kind, id, turn: b.turn, applied: false };
      const notes = [`${spMain(name)}이(가) 더 빨라 먼저 발동 → ${fxKo(kind, f[kind])} 유지`];
      b.fieldLog = { label, notes };
      return notes;
    }
    if (!first) unsure = `스피드를 알 수 없어 나중에 입력한 쪽으로 반영 (실제로는 느린 쪽의 ${kind === 'weather' ? '날씨' : '필드'}가 남음)`;
  }
  const prev = { [kind]: f[kind], [`${kind}_turns`]: f[`${kind}_turns`] };
  const { notes, fresh } = applyFieldEvents(b, [{ side, kind: 'ability', name: inc.ability, item: inc.item }],
    { start: { [side]: name }, endOfTurn: !!endOfTurn });
  if (endOfTurn) tickField(b, fresh);
  const applied = fresh.has(kind);
  m.fx = { ability: inc.ability, kind, id, turn: b.turn, applied, prev, end: !!endOfTurn, n: applied ? Number(f[`${kind}_turns`]) : 0 };
  if (applied && unsure) notes.push(unsure);
  if (notes.length) b.fieldLog = { label, notes };
  return notes;
}

/**
 * The weather / terrain ability that took effect on entry was entered by mistake (corrected on the same turn, or the
 * Pokemon on the field was corrected): what it replaced comes back, unless the field was changed since.
 */
function undoEntryFx(side, name) {
  const b = S.battle;
  const m = side === 'me' ? myMon(name) : foeMon(name);
  const fx = m.fx;
  if (!fx || fx.turn !== b.turn) return [];
  delete m.fx;
  if (m.abilityFx === fx.ability) delete m.abilityFx;
  if (!fxOnField(b.field, fx, b.turn)) return [];
  Object.assign(b.field, fx.prev);
  const was = fx.prev[fx.kind];
  const notes = [`${fxKo(fx.kind, fx.id)} 취소${was ? ` → ${fxKo(fx.kind, was)} ${fx.prev[`${fx.kind}_turns`]}턴 복원` : ''}`];
  b.fieldLog = { label: `${side === 'me' ? '내' : '상대'} ${spMain(name)} ${abilMain(fx.ability)} 취소`, notes };
  return notes;
}

/**
 * The item of the Pokemon on the field became known after its weather / terrain ability took effect on entry:
 * Damp Rock & co. make it last 8 turns (and a corrected item takes that back). Returns the notes.
 */
function retimeEntryFx(side, name, q) {
  const b = S.battle;
  const f = b.field;
  const m = side === 'me' ? myMon(name) : foeMon(name);
  const fx = m.fx;
  if (b[side].active !== name || !fx || !fxOnField(f, fx, b.turn)) return [];
  if (q === undefined && side === 'foe' && !b.quick.foeSwitch) q = b.quick;
  const item = entryInfo(side, name, q || null).item;
  const want = fx.kind === 'weather' ? WEATHER_ROCKS[fx.id] : 'Terrain Extender';
  const n = (item && item === want ? 8 : 5) - (fx.end ? 1 : 0);
  if (n === fx.n) return [];
  const left = n - (b.turn - fx.turn);
  if (left <= 0) return [];
  f[`${fx.kind}_turns`] = left;
  fx.n = n;
  const notes = [`${fxKo(fx.kind, fx.id)} ${left}턴${item === want ? ` (${itemMain(want)})` : ''}`];
  b.fieldLog = { label: `${side === 'me' ? '내' : '상대'} ${spMain(name)} 도구 ${item ? itemMain(item) : '모름'}`, notes };
  return notes;
}

/**
 * The ability of the Pokemon on the field became known after it came in (「상대 포켓몬」 특성, 특성 공개 in 이번 턴 기록,
 * or the active Pokemon corrected by hand) while it is still fresh: the turn-1 lead, or it came in this turn. Its
 * weather / terrain takes effect now, unless that ability already did since it came in; one entered by mistake
 * before is taken back. For the opponent's Pokemon the reveals of this turn count (q defaults to 이번 턴 기록).
 * Returns the notes.
 */
function lateEntryEffects(side, name, q) {
  const b = S.battle;
  const m = side === 'me' ? myMon(name) : foeMon(name);
  if (b[side].active !== name || !m.fresh || m.fainted) return [];
  if (q === undefined && side === 'foe' && !b.quick.foeSwitch) q = b.quick;
  const ab = entryInfo(side, name, q || null).ability;
  const notes = m.fx && m.fx.ability !== ab ? undoEntryFx(side, name) : [];
  if (entryFx(ab) && m.abilityFx !== ab) {
    const log = b.fieldLog;
    const more = entryFieldEffects(side, name, q, !!m.midTurn);
    if (notes.length && more.length) b.fieldLog = { label: b.fieldLog.label, notes: [...log.notes, ...more] };
    notes.push(...more);
  } else if (m.fx && m.fx.ability === ab) notes.push(...retimeEntryFx(side, name, q));
  return notes;
}

const fieldNoticeText = (notes) => `필드 자동 반영: ${notes.join(', ')}`;
const fieldToast = (notes) => {
  if (notes.length) toast(fieldNoticeText(notes), 'info', { label: '필드 확인', fn: showFieldCard });
};

function quickCard() {
  const b = S.battle;
  const q = b.quick;
  const meA = b.me.active;
  const foeA = b.foe.active;
  if (!foeA) {
    return card('이번 턴 기록', { id: 'quick-card' }, h('p', { class: 'muted' }, '상대 선봉을 먼저 선택하세요.'));
  }
  const fm = foeMon(foeA);
  const mm = myMon(meA);
  if (mm.fainted || fm.fainted) {
    // the turn is over: first say who comes in next (cards above), then record the next turn
    return card('이번 턴 기록', { id: 'quick-card', sub: `턴 ${b.turn}` },
      h('div', { class: 'msg msg-info', role: 'status' }, mm.fainted
        ? `내 ${spMain(meA)}이(가) 쓰러졌습니다. 먼저 위에서 다음에 내보낼 포켓몬을 고르세요.`
        : `상대 ${spMain(foeA)}이(가) 쓰러졌습니다. 먼저 위에서 상대가 다음에 내보낸 포켓몬을 고르세요.`));
  }
  const mySet = teamSet(meA);
  const upd = (k, rr) => (v) => { q[k] = v; save(); if (rr) rerender(); };
  const myTarget = q.mySwitch || meA;
  const foeTarget = q.foeSwitch || foeA;
  const foeSwitching = foeTarget !== foeA;
  const kids = [];
  // opponent's move: revealed moves first
  const revealed = (fm.moves || []).filter(Boolean);
  const foeMoveOpts = [...revealed.map((m) => moveOpt(m, '공개됨')),
    ...optMoves(foeA).filter((o) => !revealed.includes(o.value))];
  const foeMove = new Combobox({
    id: 'q-foe-move', options: foeMoveOpts, value: q.foeMove, clearable: true, label: `상대 ${spMain(foeA)}이(가) 쓴 기술`,
    placeholder: '기술 검색', onChange: upd('foeMove', true),
  });
  kids.push(h('div', { class: 'q-grid' },
    fld(`상대 ${spMain(foeA)}이(가) 쓴 기술`, foeMove),
    fld(`내 ${spMain(meA)}이(가) 쓴 기술`, sel('q-my-move', [['', '— 선택 안 함 —'], ...(mySet ? mySet.moves : []).map((m) => [m, moveMain(m)])], q.myMove, upd('myMove', true))),
    h('div', { class: 'fld' }, h('span', { class: 'lbl', id: 'q-order-l' }, '먼저 행동한 쪽'),
      seg('q-order', '먼저 행동한 쪽', [['me', '내가 먼저'], ['foe', '상대가 먼저'], ['', '모름']], q.order, upd('order', true))),
  ));
  if (q.order && (!q.foeMove || !q.myMove)) {
    kids.push(h('div', { class: 'muted small' }, '※ 스피드 비교는 양쪽이 쓴 기술을 모두 골라야 기록됩니다 (같은 우선도일 때만, 트릭룸이면 자동으로 반대로 기록).'));
  }
  const fieldMoves = [q.foeMove, q.myMove].filter((m) => m && (ROOM_MOVES[m] || WEATHER_MOVES[m] || TERRAIN_MOVES[m] || SCREEN_MOVES[m]
    || SIDE_MOVES[m] || HAZARD_MOVES[m] || HAZARD_HIT_MOVES[m] || ['Defog', 'Rapid Spin', 'Mortal Spin', 'Tidy Up', 'Court Change'].includes(m)));
  if (fieldMoves.length) {
    kids.push(h('div', { class: 'muted small q-note', id: 'q-field-hint' }, `※ ${fieldMoves.map(moveMain).join('·')}의 효과는 "다음 턴"을 누르면 「필드」에 자동 반영됩니다.`));
  }
  // switches come before the reveals: an item / ability revealed on a switch belongs to the Pokemon coming in
  const foeSwitchOpts = b.foe.team.filter((n) => n !== foeA && !(b.foe.mons[n] && b.foe.mons[n].fainted) &&
    (b.foe.seen.length < 3 || b.foe.seen.includes(n)));
  const mySwitchOpts = b.me.brought.filter((n) => n !== meA && !myMon(n).fainted);
  kids.push(h('div', { class: 'q-grid' },
    fld('상대 교체 (기술 대신 교체)', sel('q-foe-switch', [['', '교체 안 함'], ...foeSwitchOpts.map((n) => [n, spText(n)])], q.foeSwitch, (v) => {
      q.foeSwitch = v;
      const t = v || foeA;
      if (q.foeAbility !== '?' && !optAbilities(t).some((o) => o.value === q.foeAbility)) q.foeAbility = '?';
      save();
      rerender();
    })),
    fld('내 교체', sel('q-my-switch', [['', '교체 안 함'], ...mySwitchOpts.map((n) => [n, spText(n)])], q.mySwitch, upd('mySwitch', true))),
  ));
  const foeMegas = megasOf(foeA);
  const foeMegaOk = foeMegas.length && !b.foe.megaUsed;
  const myMegaOk = megaStoneOf(mySet) && !b.me.megaUsed;
  let foeMegaCtl = null;
  if (foeMegaOk && foeMegas.length > 1) {
    foeMegaCtl = sel('q-foe-mega', [['', '상대 메가진화 안 함'], ...foeMegas.map((x) => [x.forme, `상대가 ${spMain(x.forme)}로 메가진화`])],
      typeof q.foeMega === 'string' ? q.foeMega : '', (v) => { q.foeMega = v || false; save(); }, `이번 턴 상대 ${spMain(foeA)} 메가진화`);
  } else if (foeMegaOk) {
    foeMegaCtl = chk('q-foe-mega', '상대가 메가진화', !!q.foeMega, upd('foeMega'), false, `상대 ${spMain(foeA)}이(가) 메가진화`);
  }
  kids.push(h('div', { class: 'q-grid' },
    fld(`상대 ${spMain(foeTarget)} 도구 공개`, new Combobox({
      id: 'q-foe-item', options: optItemsReveal, value: q.foeItem, label: `상대 ${spMain(foeTarget)} 도구 공개`,
      onChange: (v) => {
        q.foeItem = v;
        // e.g. Damp Rock of an opponent whose Drizzle took effect when it came in: the rain lasts 8 turns
        const fx = q.foeSwitch ? [] : retimeEntryFx('foe', foeA, q);
        save();
        if (fx.length) { rerender(); fieldToast(fx); }
      },
    })),
    fld(`상대 ${spMain(foeTarget)} 특성 공개`, new Combobox({
      id: 'q-foe-ability', options: () => optAbilitiesReveal(foeTarget), value: q.foeAbility, label: `상대 ${spMain(foeTarget)} 특성 공개`,
      onChange: (v) => {
        q.foeAbility = v;
        // the ability of an opponent that has just come in (turn-1 lead, replacement): its weather / terrain is up
        // already at this decision (with a switch selected the reveal belongs to the incoming one: 다음 턴 applies it)
        // (a different one, or 변경 없음, takes back what a mistaken one set)
        const fx = q.foeSwitch ? [] : lateEntryEffects('foe', foeA, q);
        save();
        if (fx.length) { rerender(); fieldToast(fx); }
      },
    })),
    h('div', { class: 'fld' }, h('span', { class: 'lbl' }, '메가진화'),
      h('div', { class: 'row' },
        myMegaOk ? chk('q-my-mega', '내가 메가진화', q.myMega, upd('myMega'), false, `내 ${spMain(meA)}이(가) 메가진화`) : null,
        foeMegaCtl,
        !myMegaOk && !foeMegaCtl ? h('span', { class: 'muted small' }, '해당 없음') : null)),
  ));
  if (foeSwitching) {
    kids.push(h('div', { class: 'muted small q-note' }, `도구·특성 공개는 교체로 들어온 상대 ${spMain(foeTarget)}에게 기록됩니다. 나간 ${spMain(foeA)}의 정보는 아래 「상대 포켓몬」에서 고치세요.`));
  }
  // HP after the turn
  const mx = maxHp(myTarget);
  const curMy = myMon(myTarget).hp;
  const curFoe = foeMon(foeTarget).hp;
  const myHp = h('input', {
    type: 'number', id: 'q-my-hp', min: 0, max: mx, step: 1, inputmode: 'numeric', placeholder: `${curMy}`,
    value: q.myHp === null ? '' : String(q.myHp),
  });
  myHp.addEventListener('input', () => {
    q.myHp = myHp.value === '' ? null : clamp(Math.round(Number(myHp.value) || 0), 0, mx);
    save();
  });
  const foeHpNum = h('input', {
    type: 'number', id: 'q-foe-hp', min: 0, max: 100, step: 1, inputmode: 'numeric', placeholder: `${curFoe}`,
    value: q.foeHp === null ? '' : String(q.foeHp),
  });
  const foeHpRange = h('input', {
    type: 'range', id: 'q-foe-hp-r', min: 0, max: 100, step: 1, value: String(q.foeHp === null ? curFoe : q.foeHp),
    'aria-label': `상대 ${spMain(foeTarget)} 남은 HP % 슬라이더`,
  });
  foeHpNum.addEventListener('input', () => {
    q.foeHp = foeHpNum.value === '' ? null : clamp(Math.round(Number(foeHpNum.value) || 0), 0, 100);
    foeHpRange.value = String(q.foeHp === null ? curFoe : q.foeHp);
    save();
  });
  foeHpRange.addEventListener('input', () => {
    q.foeHp = Number(foeHpRange.value);
    foeHpNum.value = String(q.foeHp);
    save();
  });
  const statusOpts = [['?', '변경 없음'], ...D.raw.statuses.map((s) => [s.id, s.ko])];
  kids.push(h('div', { class: 'q-grid' },
    h('div', { class: 'fld' }, h('label', { for: 'q-my-hp' }, `턴 종료 후 내 ${spMain(myTarget)} HP`),
      h('div', { class: 'hp-in' }, myHp, h('span', { class: 'muted' }, `/ ${mx}`),
        h('button', { type: 'button', class: 'btn btn-sm btn-ghost', id: 'q-my-faint', 'aria-label': `내 ${spMain(myTarget)} 기절로 기록`, onclick: () => { q.myHp = 0; save(); rerender(); } }, '기절'))),
    h('div', { class: 'fld' }, h('label', { for: 'q-foe-hp' }, `턴 종료 후 상대 ${spMain(foeTarget)} HP %`),
      h('div', { class: 'hp-in' }, foeHpRange, foeHpNum, h('span', { class: 'muted' }, '%'))),
    fld(`턴 종료 후 내 ${spMain(myTarget)} 상태이상`, sel('q-my-status', statusOpts, q.myStatus, upd('myStatus'))),
    fld(`턴 종료 후 상대 ${spMain(foeTarget)} 상태이상`, sel('q-foe-status', statusOpts, q.foeStatus, upd('foeStatus'))),
  ));
  kids.push(h('div', { class: 'row q-foot' },
    chk('q-tick', '날씨·필드·벽 남은 턴 1 감소', q.tick, upd('tick')),
    chk('q-auto', '기록 후 바로 AI 계산', S.autoAdvise, (v) => { S.autoAdvise = v; save(); }),
  ));
  kids.push(h('div', { class: 'row' },
    h('button', { type: 'button', class: 'btn btn-primary btn-big', id: 'next-turn', disabled: !!RUN, onclick: nextTurn }, `턴 ${b.turn} 기록 → 다음 턴`),
    h('button', { type: 'button', class: 'btn btn-ghost', id: 'quick-clear', onclick: () => { b.quick = newQuick(); save(); rerender(); } }, '입력 지우기')));
  return card('이번 턴 기록', {
    id: 'quick-card', sub: `턴 ${b.turn}에 일어난 일을 고르고 "다음 턴"을 누르세요`,
  }, kids);
}

function nextTurn() {
  const b = S.battle;
  const meA = b.me.active;
  const foeA = b.foe.active;
  if (!foeA) { toast('상대 선봉을 먼저 선택하세요.', 'error'); return; }
  const mm = myMon(meA);
  const fm = foeMon(foeA);
  if (mm.fainted || fm.fainted) { toast('쓰러진 포켓몬 대신 나온 포켓몬을 먼저 고르세요.', 'error'); return; }
  // a copy: the switches below clear b.quick
  const q = Object.assign(newQuick(), b.quick);
  const mySet = teamSet(meA);
  const notes = [];
  // drop entries that cannot belong to the Pokemon on the field (e.g. left over from an earlier matchup)
  if (q.foeMove && !((D.species.get(foeA) || {}).moves || []).includes(q.foeMove)) q.foeMove = '';
  if (q.myMove && !(mySet && mySet.moves.includes(q.myMove))) q.myMove = '';
  if (q.foeSwitch && (q.foeSwitch === foeA || !b.foe.team.includes(q.foeSwitch) || (b.foe.mons[q.foeSwitch] || {}).fainted
    || (b.foe.seen.length >= 3 && !b.foe.seen.includes(q.foeSwitch)))) q.foeSwitch = '';
  if (q.mySwitch && (q.mySwitch === meA || !b.me.brought.includes(q.mySwitch) || myMon(q.mySwitch).fainted)) q.mySwitch = '';
  const foeT = q.foeSwitch || foeA;
  if (q.foeAbility !== '?' && !((D.species.get(foeT) || {}).abilities || []).includes(q.foeAbility)) q.foeAbility = '?';
  if (q.myMega && !(megaStoneOf(mySet) && !b.me.megaUsed)) q.myMega = false;
  const foeMegas = megasOf(foeA);
  if (q.foeMega && (!foeMegas.length || b.foe.megaUsed || (typeof q.foeMega === 'string' && !foeMegas.some((x) => x.forme === q.foeMega)))) q.foeMega = false;
  // field effects of the moves / abilities (from the state at the start of the turn; applied in step 6)
  const fieldEv = turnFieldEvents(b, q, mySet);
  const protect = { me: PROTECT_MOVES.has(q.myMove), foe: PROTECT_MOVES.has(q.foeMove) };
  // 1) what the opponent's Pokemon that started the turn did
  if (q.foeMove && !fm.moves.includes(q.foeMove)) {
    const known = fm.moves.filter(Boolean);
    if (known.length >= 4) notes.push('상대 기술이 이미 4개라 새 기술을 추가하지 않았습니다.');
    else fm.moves = [...known, q.foeMove];
  }
  // 2) who moved first (uses the state at the start of the turn)
  if (q.order && q.foeMove && !q.mySwitch && !q.foeSwitch) {
    const note = recordSpeedOrder(b, q, meA, foeA);
    if (note) notes.push(note);
  }
  if (q.foeMega) setFoeMega(fm, foeA, typeof q.foeMega === 'string' ? q.foeMega : '');
  // 3) what I did
  if (q.myMove && mySet && isChoice(mySet.item) && !mm.itemLost) mm.locked = q.myMove;
  if (q.myMega) { mm.mega = true; b.me.megaUsed = true; }
  // 4) switches (happen before moves; reveals / HP / status below apply to the Pokemon now on the field)
  if (q.mySwitch) setMyActive(q.mySwitch, true);
  else mm.fresh = false;
  if (q.foeSwitch) setFoeActive(q.foeSwitch, true);
  else fm.fresh = false;
  const me2 = myMon(b.me.active);
  const foe2 = foeMon(b.foe.active);
  // one that switched in did so during the turn, not between turns (see lateEntryEffects)
  me2.midTurn = !!q.mySwitch;
  foe2.midTurn = !!q.foeSwitch;
  if (q.foeItem !== '?') foe2.item = q.foeItem;
  if (q.foeAbility !== '?') foe2.ability = q.foeAbility;
  // 5) HP and status after the turn
  for (const [mon, st, switched] of [[me2, q.myStatus, !!q.mySwitch], [foe2, q.foeStatus, !!q.foeSwitch]]) {
    const before = mon.status;
    if (st !== '?') mon.status = st;
    if (mon.status === 'tox') mon.toxic = before === 'tox' ? (Number(mon.toxic) || 0) + 1 : 1;
    if (mon.status === 'slp') mon.sleep = before === 'slp' && !switched ? (Number(mon.sleep) || 0) + 1 : before === 'slp' ? Number(mon.sleep) || 0 : 0;
  }
  if (q.myHp !== null && q.myHp !== undefined) {
    me2.hp = clamp(q.myHp, 0, maxHp(b.me.active));
    if (me2.hp === 0) { me2.fainted = true; leaveField(me2); }
  }
  if (q.foeHp !== null && q.foeHp !== undefined) {
    foe2.hp = clamp(q.foeHp, 0, 100);
    if (foe2.hp === 0) { foe2.fainted = true; leaveField(foe2); }
  }
  // 6) field effects, then the timers: what was set this turn has lost this turn either way
  const fx = applyFieldEvents(b, fieldEv, { start: { me: meA, foe: foeA }, protect, endOfTurn: true });
  if (q.tick) tickField(b);
  else if (fx.fresh.size) tickField(b, fx.fresh);
  const done = b.turn;
  b.fieldLog = fx.notes.length ? { label: `턴 ${done} 기록`, notes: fx.notes } : null;
  b.turn += 1;
  b.quick = Object.assign(newQuick(), { tick: q.tick });
  S.switchAdvice = null;
  saveNow();
  rerender();
  toast(`턴 ${done} 기록 완료 → 턴 ${b.turn}${fx.notes.length ? ` · ${fieldNoticeText(fx.notes)}` : ''}${notes.length ? ` (${notes.join(' ')})` : ''}`,
    notes.length ? 'warn' : 'ok', fx.notes.length ? { label: '필드 확인', fn: showFieldCard } : null);
  // the calculation starts first: its first re-render (spinner, buttons off) happens right away and keeps the focused
  // "다음 턴" in place with an instant scroll, which would stop the smooth scroll below if it came after it
  if (S.autoAdvise && !adviseProblems().length) runAdvise();
  const card1 = document.getElementById(me2.fainted ? 'faint-card' : foe2.fainted ? 'foe-replace' : 'matchup');
  if (card1) card1.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/** One turn passes: weather / terrain / rooms / timed side conditions lose a turn (`only`: just these counters). */
function tickField(b, only) {
  const f = b.field;
  const on = (k) => !only || only.has(k);
  if (on('weather') && f.weather && f.weather_turns > 0) { f.weather_turns -= 1; if (!f.weather_turns) f.weather = ''; }
  if (on('terrain') && f.terrain && f.terrain_turns > 0) { f.terrain_turns -= 1; if (!f.terrain_turns) f.terrain = ''; }
  for (const k of PSEUDO) if (on(k) && f[k] > 0) f[k] -= 1;
  const timed = D.raw.side_conditions.filter((c) => c.timed).map((c) => c.id);
  for (const key of ['me', 'foe']) {
    const side = b[key].side;
    for (const c of timed) if (on(`${key}:${c}`) && side[c] > 0) side[c] -= 1;
  }
}

function showFieldCard() {
  const el = document.getElementById('field-card');
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// detailed sides ----------------------------------------------------------------

function marked(el) {
  el.classList.add('mark');
  return el;
}

function boostGrid(prefix, mon, onDone, who) {
  return h('div', { class: 'boosts' }, D.raw.boosts.map((bo) => fld(bo.ko, marked(sel(`${prefix}-boost-${bo.id}`,
    BOOST_RANGE.map((v) => [v, v > 0 ? `+${v}` : `${v}`]), (mon.boosts || {})[bo.id] || 0, (v) => {
      mon.boosts = mon.boosts || {};
      if (Number(v)) mon.boosts[bo.id] = Number(v); else delete mon.boosts[bo.id];
      save();
      onDone();
    }, who ? `${who} ${bo.ko} 랭크` : null)), Number((mon.boosts || {})[bo.id]) ? 'changed' : '')));
}

function volatileChips(prefix, mon) {
  return h('div', { class: 'chips' }, D.raw.volatiles.map((v) => chipToggle(`${prefix}-vol-${v.id}`, v.ko,
    (mon.volatiles || []).includes(v.id), (on) => {
      mon.volatiles = (mon.volatiles || []).filter((x) => x !== v.id);
      if (on) mon.volatiles.push(v.id);
      save();
      rerender();
    }, v.id)));
}

function subHpSel(prefix, mon) {
  if (!(mon.volatiles || []).includes('substitute')) return null;
  return fld('대타 남은 HP', sel(`${prefix}-subhp`, [[25, '온전 (최대 HP의 25%)'], [15, '조금 깎임'], [8, '많이 깎임']],
    mon.subHp || 25, (v) => { mon.subHp = Number(v); save(); }));
}

function statusCounters(prefix, mon) {
  return [
    mon.status === 'slp' ? fld('잠든 턴 수 (이미 지난)', sel(`${prefix}-sleep`, [0, 1, 2, 3].map((v) => [v, `${v}턴`]), mon.sleep || 0, (v) => { mon.sleep = Number(v); save(); })) : null,
    mon.status === 'tox' ? fld('맹독 경과 턴', sel(`${prefix}-toxic`, Array.from({ length: 16 }, (_, v) => [v, `${v}턴`]), mon.toxic || 0, (v) => { mon.toxic = Number(v); save(); })) : null,
  ];
}

function statusSel(id, mon, label, ariaLabel) {
  return fld(label || '상태이상', sel(id, D.raw.statuses.map((s) => [s.id, s.ko]), mon.status || '', (v) => {
    mon.status = v;
    if (v !== 'slp') mon.sleep = 0;
    if (v !== 'tox') mon.toxic = 0;
    save();
    rerender();
  }, ariaLabel));
}

function mySideCard() {
  const b = S.battle;
  const rows = b.me.brought.map((n, idx) => {
    const m = myMon(n);
    const set = teamSet(n) || {};
    const mx = maxHp(n);
    const isActive = n === b.me.active;
    const p = `me${idx}`;
    const who = `내 ${spMain(n)}`;
    const num = h('input', { type: 'number', id: `${p}-hp`, min: 0, max: mx, step: 1, inputmode: 'numeric', value: String(m.fainted ? 0 : m.hp), 'aria-label': `${who} HP (최대 ${mx})` });
    const range = h('input', { type: 'range', id: `${p}-hpr`, min: 0, max: 100, step: 1, value: String(Math.round(((m.fainted ? 0 : m.hp) / mx) * 100)), 'aria-label': `${spMain(n)} HP % 슬라이더` });
    const pctEl = h('span', { class: 'muted small mono', id: `${p}-hpp` }, `${Math.round(((m.fainted ? 0 : m.hp) / mx) * 100)}%`);
    const commit = () => {
      if (m.hp <= 0 && !m.fainted) { m.fainted = true; leaveField(m); rerender(); } else if (m.hp > 0 && m.fainted) { m.fainted = false; rerender(); }
    };
    num.addEventListener('input', () => {
      if (num.value === '') return;
      m.hp = clamp(Math.round(Number(num.value) || 0), 0, mx);
      range.value = String(Math.round((m.hp / mx) * 100));
      pctEl.textContent = `${range.value}%`;
      save();
    });
    num.addEventListener('change', () => { num.value = String(m.hp); commit(); refreshMatchup(); });
    range.addEventListener('input', () => {
      m.hp = Math.round((Number(range.value) / 100) * mx);
      num.value = String(m.hp);
      pctEl.textContent = `${range.value}%`;
      save();
    });
    range.addEventListener('change', () => { commit(); refreshMatchup(); });
    const radio = h('input', { type: 'radio', name: 'my-active', id: `${p}-active`, checked: isActive, disabled: m.fainted && !isActive, 'aria-label': `${who} 출전` });
    radio.addEventListener('change', () => {
      if (!radio.checked) return;
      // e.g. the lead corrected on turn 1: the weather / terrain of the wrong one is taken back, and that of this
      // one (from my set) takes effect
      const fx = b.me.active ? undoEntryFx('me', b.me.active) : [];
      setMyActive(n);
      fx.push(...lateEntryEffects('me', n));
      saveNow();
      rerender();
      fieldToast(fx);
    });
    const stone = megaStoneOf(set);
    const otherMega = b.me.brought.some((x) => x !== n && myMon(x).mega);
    const activeExtras = isActive && !m.fainted ? [
      h('div', { class: 'sub-sec' }, h('h4', null, '능력 변화'), boostGrid(p, m, refreshMatchup, who)),
      h('div', { class: 'row wrap' },
        // only one Mega Evolution per battle: hidden once our side has used it (unless it is this one)
        stone && (!b.me.megaUsed || m.mega) ? chk(`${p}-mega`, `메가진화함 (${itemMain(stone.name)})`, m.mega, (v) => { m.mega = v; b.me.megaUsed = v || otherMega; save(); rerender(); }, false, `${who} 메가진화함`) : null,
        // a Mega Stone can be neither knocked off nor consumed
        set.item && !stone ? chk(`${p}-itemlost`, `도구 사용/잃음 (${itemMain(set.item)})`, m.itemLost, (v) => { m.itemLost = v; if (v) m.locked = ''; save(); rerender(); }, false, `${who} 도구 사용/잃음`) : null,
        chk(`${p}-fresh`, '이번 턴에 막 나옴 (속이기 가능)', m.fresh, (v) => { m.fresh = v; save(); }, false, `${who} 이번 턴에 막 나옴`)),
      isChoice(set.item) && !m.itemLost ? fld('구애 고정 기술', sel(`${p}-locked`, [['', '고정 없음'], ...set.moves.map((mv) => [mv, moveMain(mv)])], m.locked || '', (v) => { m.locked = v; save(); }, `${who} 구애 고정 기술`)) : null,
      h('div', { class: 'sub-sec' }, h('h4', null, '상태 변화'), volatileChips(p, m), subHpSel(p, m)),
    ] : [];
    return h('div', { class: `mon-edit ${isActive ? 'is-active' : ''} ${m.fainted ? 'is-fainted' : ''}` },
      h('div', { class: 'me-h' },
        h('label', { class: 'radio', for: `${p}-active` }, radio, h('span', null, isActive ? '출전 중' : '출전')),
        spDual(n), typeChips((D.species.get(n) || {}).types),
        m.mega ? h('span', { class: 'tag tag-mega' }, '메가') : null),
      h('div', { class: 'me-row' },
        h('div', { class: 'fld' }, h('label', { for: `${p}-hp` }, 'HP'), h('div', { class: 'hp-in' }, num, h('span', { class: 'muted' }, `/ ${mx}`), range, pctEl)),
        statusSel(`${p}-status`, m, null, `${who} 상태이상`),
        h('div', { class: 'fld fld-chk' }, chk(`${p}-fainted`, '기절', m.fainted, (v) => {
          m.fainted = v;
          if (v) { m.hp = 0; leaveField(m); } else if (m.hp <= 0) m.hp = 1;
          save();
          rerender();
        }, false, `${who} 기절`))),
      statusCounters(p, m),
      activeExtras);
  });
  return card('내 포켓몬', { id: 'my-side', sub: `메가진화 ${b.me.megaUsed ? '사용함' : '가능'}` }, rows);
}

function refreshMatchup() {
  const old = document.getElementById('matchup');
  if (old) old.replaceWith(matchupCard());
}

function foeSideCard() {
  const b = S.battle;
  const roster = h('div', { class: 'roster' }, b.foe.team.map((n, i) => {
    const seen = b.foe.seen.includes(n);
    const m = b.foe.mons[n];
    const state = n === b.foe.active ? '출전 중' : m && m.fainted ? '기절' : seen ? '등장' : '미등장';
    return h('button', {
      type: 'button', class: `chip roster-chip ${n === b.foe.active ? 'is-active' : ''} ${m && m.fainted ? 'is-fainted' : ''}`,
      id: `foe-roster-${i}`, 'aria-pressed': String(seen),
      title: seen ? '눌러서 등장 표시 해제' : '눌러서 등장으로 표시',
      onclick: () => {
        if (seen) {
          if (n === b.foe.active) { toast('출전 중인 포켓몬은 해제할 수 없습니다.', 'error'); return; }
          b.foe.seen = b.foe.seen.filter((x) => x !== n);
        } else {
          if (b.foe.seen.length >= 3) { toast('상대는 3마리만 선출합니다.', 'error'); return; }
          b.foe.seen.push(n);
          foeMon(n);
        }
        save();
        rerender();
      },
    }, spDual(n), h('span', { class: 'roster-state' }, state));
  }));
  const order = [...b.foe.seen].sort((x, y) => (x === b.foe.active ? -1 : y === b.foe.active ? 1 : 0));
  const details = order.map((n) => foeMonEditor(n));
  return card('상대 포켓몬', { id: 'foe-side', sub: `메가진화 ${b.foe.megaUsed ? '사용함' : '아직'}` },
    h('p', { class: 'muted small' }, '엔트리를 눌러 등장한 포켓몬을 표시하세요 (최대 3마리).'), roster,
    details.length ? details : h('p', { class: 'muted' }, '아직 등장한 상대 포켓몬이 없습니다.'));
}

function foeMonEditor(n) {
  const b = S.battle;
  const m = foeMon(n);
  const idx = b.foe.team.indexOf(n);
  const p = `foe${idx}`;
  const sp = D.species.get(n) || { types: [], megas: [], baseStats: null, abilities: [] };
  const isActive = n === b.foe.active;
  const who = `상대 ${spMain(n)}`;
  const num = h('input', { type: 'number', id: `${p}-hp`, min: 0, max: 100, step: 1, inputmode: 'numeric', value: String(m.fainted ? 0 : m.hp), 'aria-label': `${who} HP %` });
  const range = h('input', { type: 'range', id: `${p}-hpr`, min: 0, max: 100, step: 1, value: String(m.fainted ? 0 : m.hp), 'aria-label': `상대 ${spMain(n)} HP % 슬라이더` });
  const commit = () => {
    if (m.hp <= 0 && !m.fainted) { m.fainted = true; leaveField(m); rerender(); } else if (m.hp > 0 && m.fainted) { m.fainted = false; rerender(); } else refreshMatchup();
  };
  num.addEventListener('input', () => {
    if (num.value === '') return;
    m.hp = clamp(Math.round(Number(num.value) || 0), 0, 100);
    range.value = String(m.hp);
    save();
  });
  num.addEventListener('change', () => { num.value = String(m.hp); commit(); });
  range.addEventListener('input', () => { m.hp = Number(range.value); num.value = String(m.hp); save(); });
  range.addEventListener('change', commit);
  const moves = [0, 1, 2, 3].map((j) => {
    const cb = new Combobox({
      id: `${p}-move${j}`, options: () => optMoves(n), value: m.moves[j] || '', clearable: true,
      label: `상대 ${spMain(n)} 기술 ${j + 1}`, placeholder: `기술 ${j + 1}`,
      onChange: (v) => {
        if (v && m.moves.some((x, k) => x === v && k !== j)) {
          toast('이미 입력한 기술입니다.', 'error');
          cb.setValue(m.moves[j] || '');
          return;
        }
        const arr = [0, 1, 2, 3].map((k) => m.moves[k] || '');
        arr[j] = v;
        m.moves = arr;
        save();
      },
    });
    return fld(`기술 ${j + 1}`, cb);
  });
  const item = new Combobox({
    id: `${p}-item`, options: optItemsFoe, value: m.item === null || m.item === undefined ? '?' : m.item,
    label: `상대 ${spMain(n)} 도구`,
    onChange: (v) => {
      m.item = v === '?' ? null : v;
      // e.g. Heat Rock of the lead whose Drought took effect: the sun lasts 8 turns
      const fx = retimeEntryFx('foe', n);
      save();
      if (fx.length) { rerender(); fieldToast(fx); }
    },
  });
  const ability = new Combobox({
    id: `${p}-ability`, options: () => optAbilitiesFoe(n), value: m.ability || '?', label: `상대 ${spMain(n)} 특성`,
    onChange: (v) => {
      m.ability = v === '?' ? null : v;
      // e.g. Sand Stream of the turn-1 lead: the sandstorm is up from now on
      const fx = lateEntryEffects('foe', n);
      save();
      if (fx.length) { rerender(); fieldToast(fx); }
    },
  });
  const myNames = b.me.brought;
  const belief = S.advice && S.advice.res && S.advice.res.beliefs ? S.advice.res.beliefs[n] : null;
  const summary = h('span', { class: 'foe-sum' }, spDual(n), typeChips(sp.types),
    h('span', { class: 'mono small' }, `${m.fainted ? 0 : m.hp}%`),
    isActive ? h('span', { class: 'tag tag-ok' }, '출전 중') : null,
    m.fainted ? h('span', { class: 'tag tag-bad' }, '기절') : null,
    m.status && !m.fainted ? h('span', { class: `tag st-${m.status}` }, statusKo(m.status)) : null,
    m.mega ? h('span', { class: 'tag tag-mega' }, '메가') : null);
  const body = [
    h('div', { class: 'me-row' },
      isActive ? null : h('button', {
        type: 'button', class: 'btn btn-sm', id: `${p}-setactive`, disabled: m.fainted,
        onclick: () => {
          const fx = b.foe.active ? undoEntryFx('foe', b.foe.active) : [];
          setFoeActive(n, undefined, true);
          fx.push(...lateEntryEffects('foe', n));
          saveNow();
          rerender();
          fieldToast(fx);
        },
      }, '출전 중으로 설정'),
      h('div', { class: 'fld' }, h('label', { for: `${p}-hp` }, 'HP %'), h('div', { class: 'hp-in' }, range, num, h('span', { class: 'muted' }, '%'))),
      statusSel(`${p}-status`, m, null, `${who} 상태이상`),
      h('div', { class: 'fld fld-chk' }, chk(`${p}-fainted`, '기절', m.fainted, (v) => {
        m.fainted = v;
        if (v) { m.hp = 0; leaveField(m); } else if (m.hp <= 0) m.hp = 1;
        save();
        rerender();
      }, false, `${who} 기절`))),
    statusCounters(p, m),
    h('div', { class: 'sub-sec' }, h('h4', null, '공개된 기술'), h('div', { class: 'grid-2' }, moves)),
    h('div', { class: 'grid-2' }, fld('도구', item), fld('특성', ability)),
    // only one Mega Evolution per battle: hidden once the opponent has used it (unless it is this one)
    sp.megas && sp.megas.length && (!b.foe.megaUsed || m.mega)
      ? h('div', { class: 'row' }, foeMegaControl(`${p}-mega`, m, n, () => { save(); rerender(); })) : null,
    isActive && !m.fainted ? [
      h('div', { class: 'sub-sec' }, h('h4', null, '능력 변화'), boostGrid(p, m, refreshMatchup, who)),
      h('div', { class: 'sub-sec' }, h('h4', null, '상태 변화'), volatileChips(p, m), subHpSel(p, m)),
      chk(`${p}-fresh`, '이번 턴에 막 나옴', m.fresh, (v) => { m.fresh = v; save(); }, false, `${who} 이번 턴에 막 나옴`),
    ] : null,
    h('div', { class: 'sub-sec' }, h('h4', null, '스피드 비교 (실제 스피드 기준 · 트릭룸이면 반대로)'),
      h('p', { class: 'muted small' }, '같은 우선도 기술끼리, 양쪽 모두 메가진화·순풍·마비·스피드 랭크 변화·스피드 특성이 없을 때만 표시하세요. "이번 턴 기록"을 쓰면 자동으로 처리됩니다.'),
      h('div', { class: 'speed-row' }, h('span', { class: 'lbl' }, '상대가 더 빠름 (먼저 행동):'),
        h('div', { class: 'chips' }, myNames.map((x, k) => chipToggle(`${p}-faster-${k}`, `내 ${spMain(x)}보다 빠름`, (m.faster || []).includes(x), (on) => {
          m.faster = (m.faster || []).filter((y) => y !== x);
          m.slower = (m.slower || []).filter((y) => y !== x || !on);
          if (on) m.faster.push(x);
          save();
          rerender();
        })))),
      h('div', { class: 'speed-row' }, h('span', { class: 'lbl' }, '상대가 더 느림 (늦게 행동):'),
        h('div', { class: 'chips' }, myNames.map((x, k) => chipToggle(`${p}-slower-${k}`, `내 ${spMain(x)}보다 느림`, (m.slower || []).includes(x), (on) => {
          m.slower = (m.slower || []).filter((y) => y !== x);
          m.faster = (m.faster || []).filter((y) => y !== x || !on);
          if (on) m.slower.push(x);
          save();
          rerender();
        }))))),
    h('div', { class: 'info small muted' },
      sp.baseStats ? h('div', { class: 'mono' }, `종족값 ${statLine(sp.baseStats)}`) : null,
      sp.abilities && sp.abilities.length ? h('div', null, `가능한 특성: ${sp.abilities.map(abilMain).join(', ')}`) : null,
      belief ? h('div', null, `AI 추정: 스피드 ${belief.speed_10_90 ? `${belief.speed_10_90[0]}–${belief.speed_10_90[1]}` : '?'}`,
        belief.p_choice_scarf !== undefined ? ` · 구애스카프 ${pct(belief.p_choice_scarf)}` : '',
        belief.mean_stats ? ` · 평균 ${statLine(belief.mean_stats)}` : '') : null),
  ];
  const key = `foe-${n}`;
  const d = h('details', { class: `mon-edit foe-edit ${isActive ? 'is-active' : ''} ${m.fainted ? 'is-fainted' : ''}`, id: `d-${key}`, open: isActive || !!S.open[key] },
    h('summary', null, summary), h('div', { class: 'block-b' }, body));
  d.addEventListener('toggle', () => { if (!isActive) { S.open[key] = d.open; save(); } });
  return d;
}

// field --------------------------------------------------------------------------

function turnsSel(id, value, max, onChange, zeroLabel, ariaLabel) {
  const opts = [[0, zeroLabel || '없음']];
  for (let i = 1; i <= max; i++) opts.push([i, `${i}턴`]);
  return marked(sel(id, opts, Number(value) || 0, (v) => onChange(Number(v)), ariaLabel));
}

function sideCondEditor(prefix, sideObj, title, megaUsed, onMega) {
  const rows = D.raw.side_conditions.map((c) => {
    const id = `${prefix}-${c.id}`;
    const v = Number(sideObj[c.id]) || 0;
    const name = `${title} ${c.ko}`; // both sides have the same controls: say which side
    const set = (x) => { if (x) sideObj[c.id] = x; else delete sideObj[c.id]; save(); };
    if (c.timed) return fld(c.ko, turnsSel(id, v, 8, set, '없음', name), v ? 'changed' : '');
    if (c.layers > 1) return fld(c.ko, marked(sel(id, Array.from({ length: c.layers + 1 }, (_, i) => [i, i ? `${i}층` : '없음']), v, (x) => set(Number(x)), name)), v ? 'changed' : '');
    return h('div', { class: 'fld fld-chk' }, chk(id, c.ko, !!v, (on) => set(on ? 1 : 0), false, name));
  });
  return h('div', { class: 'side-conds' }, h('h4', null, title), h('div', { class: 'cond-grid' }, rows,
    h('div', { class: 'fld fld-chk' }, chk(`${prefix}-megaused`, '메가진화 사용함', megaUsed, onMega, false, `${title} 메가진화 사용함`))));
}

function fieldCard() {
  const b = S.battle;
  const f = b.field;
  const pseudoKo = Object.fromEntries(D.raw.pseudo.map((x) => [x.id, x.ko]));
  const log = b.fieldLog;
  return card('필드', { id: 'field-card', sub: '남은 턴 기준 (0 = 없음)' },
    log && (log.notes || []).length ? h('div', { class: 'msg msg-info field-note', id: 'field-note', role: 'status' },
      h('span', null, h('strong', null, `${log.label}에서 자동 반영: `), log.notes.join(' · '), ' — 다르면 아래에서 고치세요.')) : null,
    h('div', { class: 'cond-grid' },
      fld('날씨', sel('f-weather', D.raw.weathers.map((w) => [w.id, w.ko]), f.weather, (v) => { f.weather = v; if (!v) f.weather_turns = 0; save(); rerender(); })),
      f.weather ? fld('날씨 남은 턴', turnsSel('f-weather-turns', f.weather_turns, 8, (v) => { f.weather_turns = v; save(); }, '모름')) : null,
      fld('필드', sel('f-terrain', D.raw.terrains.map((w) => [w.id, w.ko]), f.terrain, (v) => { f.terrain = v; if (!v) f.terrain_turns = 0; save(); rerender(); })),
      f.terrain ? fld('필드 남은 턴', turnsSel('f-terrain-turns', f.terrain_turns, 8, (v) => { f.terrain_turns = v; save(); }, '모름')) : null,
      PSEUDO.map((k) => fld(pseudoKo[k] || k, turnsSel(`f-${k}`, f[k], 8, (v) => { f[k] = v; save(); }), f[k] ? 'changed' : ''))),
    h('div', { class: 'grid-2 sides-cond' },
      sideCondEditor('sc-me', b.me.side, '내 필드', b.me.megaUsed, (v) => { b.me.megaUsed = v; save(); rerender(); }),
      sideCondEditor('sc-foe', b.foe.side, '상대 필드', b.foe.megaUsed, (v) => { b.foe.megaUsed = v; save(); rerender(); })));
}

function actionBar() {
  const busy = RUN && RUN.kind === 'advise';
  return h('div', { class: 'action-bar', role: 'region', 'aria-label': '계산' },
    h('div', { class: 'action-in' },
      seg('samples', '계산 정밀도', PRESETS.map((p) => [p.id, h('span', null, p.ko, h('span', { class: 'seg-n' }, ` ${p.time}`)),
        `${presetText(p)} 시뮬레이션 · 보통 ${p.time}`]), S.preset, (v) => { S.preset = v; save(); rerender(); }),
      busy ? h('button', { type: 'button', class: 'btn btn-ghost', onclick: cancelRun }, '취소') : null,
      h('button', {
        type: 'button', class: 'btn btn-primary btn-big grow', id: 'advise-run', disabled: !!RUN,
        onclick: runAdvise,
      }, busy ? h('span', null, h('span', { class: 'spin spin-inv', 'aria-hidden': 'true' }), ' 계산 중… ', h('span', { 'data-elapsed': '1' }, '0.0초')) : '최선의 행동 계산')));
}

// ---------------------------------------------------------------------------
// TAB 4: help

let STATUS = null;

function renderHelp() {
  const p = $('#panel-help');
  const st = STATUS;
  const steps = [
    ['팀 고르기', '「추천 파티 / 내 팀」에서 AI가 추천한 팀의 "이 팀으로 배틀"을 누릅니다. 내 팀이 따로 있으면 Showdown 텍스트를 붙여 넣거나 드롭다운 빌더로 만드세요.'],
    ['선출', '배틀 매칭 후 팀 미리보기에 보이는 상대 6마리를 검색창에서 고릅니다 (한글·영어·초성 검색, 예: ㅎㅋㄹ → 한카리아스). "AI 선출 추천"을 누르면 3마리 조합과 선봉을 승률 순으로 알려줍니다. "이 선출로 시작"을 누르세요.'],
    ['배틀 시작', '상대 선봉을 누르면 대면이 설정됩니다. 하단의 "최선의 행동 계산"을 누르면 기술/교체별 예상 승률이 나옵니다. 1위(추천)를 참고해 행동하세요.'],
    ['매 턴 기록', '「이번 턴 기록」에서 상대가 쓴 기술, 먼저 행동한 쪽, 공개된 도구/특성, 턴 종료 후 남은 HP, 교체를 고른 뒤 "다음 턴"을 누르면 자동으로 반영되고 (옵션) 바로 다시 계산합니다.'],
    ['기절했을 때', '내 포켓몬 HP를 0으로 기록하면 "교체 추천" 버튼이 나타나 다음에 내보낼 포켓몬을 알려줍니다. 상대가 쓰러지면 상대가 다음에 내보낸 포켓몬을 누르세요.'],
    ['필드 자동 반영', '양쪽이 쓴 트릭룸·중력·매직룸·원더룸, 날씨 기술(쾌청·비바라기 등)과 날씨 특성(가뭄·잔비·모래날림·눈퍼뜨리기), 필드 기술과 ○○메이커 특성, 리플렉터·빛의장막·오로라베일·순풍·신비의부적(쓴 쪽 필드), 스텔스록·압정뿌리기·독압정·끈적끈적네트(상대 필드), 안개제거·고속스핀·정리정돈은 「필드」에 자동으로 반영됩니다. 막 나온 포켓몬(1턴의 선봉, 이번 턴에 나온 포켓몬)의 날씨·필드 특성은 「상대 포켓몬」의 특성이나 「이번 턴 기록」의 특성 공개에 입력하는 즉시 반영됩니다. 양쪽이 동시에 나왔으면(선봉끼리 등) 더 느린 쪽의 날씨·필드가 남고, 특성을 잘못 입력해 고치면 원래 날씨·필드로 돌아가며, 바위·그라운드코트를 나중에 입력해도 남은 턴이 늘어납니다. 남은 턴은 "다음 결정 때 남은 턴"이라 5턴 효과는 다음 턴에 4로 보입니다 (도구가 알려진 경우 바위·빛의점토·그라운드코트는 8턴). 트릭룸 중에 트릭룸을 다시 쓰면 해제됩니다. 반영한 내용은 알림과 「필드」 카드 위에 표시되니 다르면 고치세요.'],
    ['세부 수정', '「내 포켓몬」「상대 포켓몬」「필드」 카드에서 능력 변화, 상태이상, 메가진화, 구애 고정, 날씨/필드/벽 남은 턴 등을 언제든지 직접 고칠 수 있습니다. 상대의 도구·특성은 모르면 "모름"으로 두세요 — AI가 가능성을 추정합니다.'],
    ['스피드 관찰', '같은 우선도 기술끼리 상대가 먼저/늦게 행동했다는 정보는 상대 스피드(구애스카프 여부 등) 추정에 쓰입니다. 양쪽이 쓴 기술을 모두 골라야 기록되고, 메가진화·순풍·마비·스피드 랭크 변화·스피드 특성(쓱쓱 등)·구애스카프 상실이 끼면 AI가 잘못 추정하지 않도록 자동으로 기록하지 않습니다.'],
    ['저장', '입력한 내용은 이 브라우저에 자동 저장되어 새로고침해도 유지됩니다. 새 배틀을 시작할 때는 오른쪽 위 "새 배틀"을 누르세요 (팀은 유지). 실수로 초기화했다면 바로 뜨는 알림의 "되돌리기"를 누르세요.'],
    ['휴대폰에서 쓰기', 'PC에서 "run_ui.bat --host 0.0.0.0" (macOS/Linux: "./run_ui.sh --host 0.0.0.0", 직접 실행: "python -m pokechamp ui --host 0.0.0.0") 으로 실행하면 검은 창에 휴대폰용 주소 (http://<PC의 IP 주소>:8765) 가 표시됩니다. 같은 Wi-Fi에 연결된 휴대폰 브라우저에서 그 주소를 여세요. Windows 방화벽 창이 뜨면 "허용"을 누르세요. 같은 네트워크의 누구나 접속할 수 있으니 집 등 믿을 수 있는 네트워크에서만 쓰세요.'],
  ];
  const local = /^(127\.|localhost|\[::1\])/.test(window.location.hostname);
  p.replaceChildren(
    card('사용 방법', null, h('ol', { class: 'help-steps' }, steps.map(([t, d]) => h('li', null, h('strong', null, t), h('p', null, d))))),
    card('계산 정밀도', { id: 'help-presets' }, h('ul', null,
      PRESETS.map((pr) => h('li', null, h('strong', null, pr.ko), `: 샘플 ${pr.samples}개 · ${pr.depth}턴 앞까지 시뮬레이션 · 약 ${pr.time}`,
        { fast: ' · 대략적인 순위', normal: ' · 권장', precise: ' · 승률 차이가 작을 때' }[pr.id]))),
    h('p', { class: 'muted small' }, '샘플 = 상대의 모르는 정보(도구·특성·기술·능력치)를 무작위로 채워 넣은 가상 배틀의 수입니다. 많을수록 승률이 안정됩니다. ',
      '깊이(몇 턴 앞까지) = 각 행동 뒤 몇 턴을 더 시뮬레이션한 다음 신경망으로 판정하는지입니다. 깊을수록 몇 턴 뒤의 전개(교체, 랭크업 등)까지 보지만 오래 걸립니다.'),
    h('p', { class: 'muted small' }, '걸리는 시간은 선택할 수 있는 행동 수와 PC 성능에 따라 달라집니다 (구애 고정이면 빠르고, 기술 4개 + 교체 2개 + 메가진화처럼 행동이 많으면 느림). 위 시간은 행동 6개 기준입니다.'),
    h('p', { class: 'muted small' }, '승률 옆의 ± 값은 표본 오차입니다. 차이가 오차보다 작으면 두 행동은 비슷하다고 보면 됩니다. "정책"은 신경망이 본능적으로 고른 확률입니다.'),
    h('p', { class: 'muted small', id: 'help-value' }, '표 아래의 "현재 국면 승률 (신경망 즉석 추정)"은 신경망이 지금 국면만 보고 한 번에 낸 대략적인 값입니다. 시뮬레이션을 하지 않아 위의 행동별 승률과 다를 수 있으며, 행동을 고를 때는 행동별 시뮬레이션 승률을 보세요.')),
    card('상태', { id: 'status-card' }, st ? (st.error ? errBox(st.error) : h('dl', { class: 'kv' },
      h('dt', null, '포맷'), h('dd', null, st.format),
      h('dt', null, '배틀 신경망'), h('dd', null, st.model || (st.model_error ? `불러오지 못함 (휴리스틱 사용): ${st.model_error}` : '없음 (휴리스틱 사용)')),
      h('dt', null, '팀 라이브러리'), h('dd', null, st.library || '없음'),
      h('dt', null, '한글 이름'), h('dd', null, st.korean_names ? '사용 가능' : '없음 (영어 이름 표시)'),
      h('dt', null, '데이터'), h('dd', null, `포켓몬 ${D.raw.species.length} · 기술 ${Object.keys(D.raw.moves).length} · 도구 ${D.raw.items.length}`),
      h('dt', null, '접속 주소'), h('dd', null, window.location.host, local ? h('span', { class: 'muted small' }, ' (이 PC에서만 접속 가능 · 휴대폰은 위 "휴대폰에서 쓰기" 참고)') : null))) : spinner('불러오는 중…')),
    card('데이터 초기화', null, h('p', { class: 'muted small' }, '저장된 팀·배틀 기록을 모두 지웁니다.'),
      confirmBtn({ class: 'btn btn-danger' }, '모든 저장 데이터 지우기', '정말 모두 지울까요? 다시 누르면 삭제', () => true, () => {
        cancelRun(true);
        UNDO = null;
        try { localStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ }
        S = defaultState();
        S.tab = 'help';
        saveNow();
        render();
        toast('모든 데이터를 지웠습니다.');
      })),
  );
  if (!STATUS) {
    STATUS = { loading: true };
    api('/api/status').then((x) => { STATUS = x; }, (e) => { STATUS = { error: e.message }; }).then(() => { if (S.tab === 'help') rerender(); });
  } else if (STATUS.loading) {
    // still loading
  }
}

// ---------------------------------------------------------------------------
// new battle / boot

function newBattle() {
  const snap = snapshotForUndo();
  cancelRun(true);
  S.foeTeam = ['', '', '', '', '', ''];
  S.preview = null;
  S.battle = null;
  S.advice = null;
  S.switchAdvice = null;
  S.manual = { picked: [], lead: '' };
  saveNow();
  offerUndo(snap, '새 배틀을 준비했습니다. 상대 엔트리를 입력하세요.');
  go(S.team ? 'preview' : 'team');
}

function sanitizeState() {
  // drop anything that refers to data that no longer exists
  if (S.team && (!Array.isArray(S.team.sets) || !S.team.sets.length)) S.team = null;
  if (!Array.isArray(S.foeTeam) || S.foeTeam.length !== 6) S.foeTeam = ['', '', '', '', '', ''];
  S.foeTeam = S.foeTeam.map((x) => (x && D.species.has(x) ? x : ''));
  if (!Array.isArray(S.builder) || S.builder.length !== 6) S.builder = [0, 1, 2, 3, 4, 5].map(emptySlot);
  if (S.battle && (!S.team || !S.battle.me || !S.battle.foe)) S.battle = null;
  if (S.battle && !S.battle.quick) S.battle.quick = newQuick();
  if (!PRESETS.some((p) => p.id === S.preset)) S.preset = 'normal';
  S.open = S.open || {};
}

async function boot() {
  const main = $('#main');
  const loading = $('#boot');
  S = Object.assign(defaultState(), loadState() || {});
  renderTabs();
  $('#tabs').addEventListener('keydown', onTabKey);
  window.addEventListener('resize', syncBarHeight);
  watchKeyboard();
  $('#new-battle').replaceWith(confirmBtn({ class: 'btn btn-sm btn-ghost hdr-btn', id: 'new-battle' }, '새 배틀', '초기화? 다시 누르기',
    () => !!(S.battle || S.foeTeam.some(Boolean)), newBattle));
  try {
    await loadData();
  } catch (e) {
    loading.replaceChildren(errBox(`게임 데이터를 불러오지 못했습니다: ${e.message}`),
      h('button', { type: 'button', class: 'btn', onclick: () => window.location.reload() }, '다시 시도'));
    return;
  }
  sanitizeState();
  loading.remove();
  main.hidden = false;
  render();
  setInterval(tickTimers, 100);
}

window.addEventListener('pagehide', saveNow);
// highlight boost / turn selects that are not at their neutral value
document.addEventListener('change', (e) => {
  const t = e.target;
  if (t && t.matches && t.matches('select.mark')) {
    const f = t.closest('.fld');
    if (f) f.classList.toggle('changed', t.value !== '0' && t.value !== '');
  }
});
window.addEventListener('error', (e) => { toast(`스크립트 오류: ${e.message}`, 'error'); });
window.addEventListener('unhandledrejection', (e) => { toast(`오류: ${(e.reason && e.reason.message) || e.reason}`, 'error'); });
document.addEventListener('DOMContentLoaded', boot);
