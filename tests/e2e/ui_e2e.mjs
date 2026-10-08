#!/usr/bin/env node
/* End-to-end test of the battle assistant UI (pokechamp/ui/static) with Playwright + headless Chromium.
 *
 *   node tests/e2e/ui_e2e.mjs
 *
 * Starts its own server (python -m pokechamp ui --port 8781), walks the whole flow as a user would at
 * 1280x900 (desktop, keyboard selection in the dropdowns) and 390x844 (phone emulation, tapping options):
 *   recommended team -> use it; paste a Showdown export; build a team with the dropdown builder;
 *   team preview (6 opponent species via dropdowns) -> AI 선출 추천 -> 이 선출로 시작;
 *   battle: my HP / status / boosts / substitute, opponent lead + HP + moves + item + ability, speed
 *   observation, the per-turn panel + 다음 턴, field, 최선의 행동 계산 (request body checked against
 *   pokechamp/ui/API.md, samples / depth of each 빠름 / 보통 / 정밀 preset), opponent switch, faint -> 교체 추천,
 *   reload persistence (+ an old saved preset), API error display, help, 새 배틀 reset; a second battle with
 *   Mega Evolution, 취소, speed notes, field effects applied from the recorded moves (Trick Room, weather, screen,
 *   hazard: counts on the next turn) and, on the phone, taps on "기절" / "다음 턴" while a number field still has
 *   the keyboard focus and a message is on screen; a third battle with weather abilities of the leads (my Drought
 *   from my set, the opponent's Sand Stream entered in 「상대 포켓몬」 on turn 1, not applied twice when revealed
 *   again) and the page scrolling up to the matchup while auto-advise calculates after "다음 턴".
 * Console errors, failed requests, external requests and layout problems (horizontal overflow, unlabeled
 * controls, clipped text) are recorded. Screenshots + results.json go to $E2E_OUT.
 * Exit code 1 when any check failed.
 *
 * Env: E2E_PORT (8781), E2E_OUT, E2E_PREFIX (screenshot file prefix, "e2e"), E2E_VIEWPORTS ("desktop,phone"),
 * PYTHON ("python"), PLAYWRIGHT_PATH.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
function loadPlaywright() {
  const tries = [process.env.PLAYWRIGHT_PATH, 'playwright', '/opt/node-tools/node_modules/playwright'].filter(Boolean);
  for (const t of tries) {
    try { return require(t); } catch (e) { /* try the next one */ }
  }
  throw new Error('playwright not found (set PLAYWRIGHT_PATH or NODE_PATH)');
}
const { chromium } = loadPlaywright();

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const PORT = Number(process.env.E2E_PORT || 8781);
const BASE = `http://127.0.0.1:${PORT}`;
const SCRATCH = '/tmp/claude-0/-home-user-pokemon/dde426f4-2280-5829-92fd-45e1777b7c8b/scratchpad';
const OUT = process.env.E2E_OUT || (fs.existsSync(SCRATCH) ? path.join(SCRATCH, 'ui_e2e') : path.join(os.tmpdir(), 'pokechamp_ui_e2e'));
const PY = process.env.PYTHON || 'python';
const PREFIX = (process.env.E2E_PREFIX || 'e2e').replace(/[^\w-]/g, '_');
const STORE_KEY = 'pokechamp-ui-v1';
const T_AI = 420000; // generous: preview / advise / advise_switch run on CPU
const VIEWPORTS = {
  desktop: { name: 'desktop', phone: false, ctx: { viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1, locale: 'ko-KR' } },
  phone: {
    name: 'phone', phone: true,
    ctx: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, locale: 'ko-KR' },
  },
};
const RUN_VPS = (process.env.E2E_VIEWPORTS || 'desktop,phone').split(',').map((s) => s.trim()).filter(Boolean);
const FOE = ['Dragonite', 'Gholdengo', 'Hippowdon', 'Primarina', 'Tyranitar', 'Corviknight'];
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const STATUSES = ['', 'brn', 'par', 'slp', 'frz', 'psn', 'tox'];

fs.mkdirSync(OUT, { recursive: true });
for (const f of fs.readdirSync(OUT)) if (f.startsWith(`${PREFIX}_`) && f.endsWith('.png')) fs.rmSync(path.join(OUT, f));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const TEAM_TXT = fs.readFileSync(path.join(ROOT, 'examples', 'team_singles.txt'), 'utf8');
// the third battle's team: Torkoal (Drought, Heat Rock) instead of Talonflame
const TEAM3_TXT = TEAM_TXT.replace(/Talonflame @ Focus Sash[\s\S]*?- Protect\r?\n/,
  'Torkoal @ Heat Rock\nAbility: Drought\nSPs: 32 HP / 32 SpA / 2 SpD\nQuiet Nature\n- Eruption\n- Earth Power\n- Stealth Rock\n- Protect\n');
// the fourth battle's team: a fast Pelipper (Drizzle, Damp Rock) instead of Talonflame
const TEAM4_TXT = TEAM_TXT.replace(/Talonflame @ Focus Sash[\s\S]*?- Protect\r?\n/,
  'Pelipper @ Damp Rock\nAbility: Drizzle\nSPs: 32 HP / 2 Def / 32 Spe\nTimid Nature\n- Hurricane\n- Hydro Pump\n- U-turn\n- Roost\n');

// ---------------------------------------------------------------------------
// results

const R = {
  started: new Date().toISOString(), base: BASE, passed: [], failed: [], warnings: [], screenshots: [],
  console: {}, requestFailures: {}, badResponses: {}, external: {}, forcedClicks: {}, layout: {}, bodies: {}, timings: {},
};
const evid = (e) => (e === undefined || e === null ? '' : (typeof e === 'string' ? e : JSON.stringify(e)));
function pass(vp, msg) { R.passed.push(`[${vp}] ${msg}`); console.log(`  ok   [${vp}] ${msg}`); }
function fail(vp, msg, evidence) {
  R.failed.push({ vp, msg, evidence: evid(evidence).slice(0, 2500) });
  console.log(`  FAIL [${vp}] ${msg}${evidence !== undefined ? `\n         evidence: ${evid(evidence).slice(0, 600)}` : ''}`);
}
function warn(vp, msg, evidence) {
  R.warnings.push({ vp, msg, evidence: evid(evidence).slice(0, 1500) });
  console.log(`  warn [${vp}] ${msg}${evidence !== undefined ? ` -- ${evid(evidence).slice(0, 300)}` : ''}`);
}
function check(vp, cond, msg, evidence) {
  if (cond) pass(vp, msg); else fail(vp, msg, evidence);
  return !!cond;
}

// ---------------------------------------------------------------------------
// server

let server = null;
async function isUp() {
  try { const r = await fetch(`${BASE}/api/status`); return r.ok; } catch (e) { return false; }
}
async function startServer() {
  if (await isUp()) throw new Error(`something already listens on port ${PORT}; stop it or set E2E_PORT`);
  const logPath = path.join(OUT, `server_${PORT}.log`);
  const fd = fs.openSync(logPath, 'w');
  server = spawn(PY, ['-m', 'pokechamp', 'ui', '--port', String(PORT)], {
    cwd: ROOT, env: { ...process.env, PYTHONPATH: ROOT, POKECHAMP_UI_LOG: '1' }, stdio: ['ignore', fd, fd],
  });
  const t0 = Date.now();
  while (Date.now() - t0 < 90000) {
    if (server.exitCode !== null) throw new Error(`server exited with code ${server.exitCode}; see ${logPath}`);
    if (await isUp()) {
      console.log(`server up on ${BASE} (pid ${server.pid}) after ${((Date.now() - t0) / 1000).toFixed(1)} s; log ${logPath}`);
      return;
    }
    await sleep(300);
  }
  throw new Error('server did not answer /api/status within 90 s');
}
async function stopServer() {
  if (!server || server.exitCode !== null) return;
  server.kill('SIGTERM');
  for (let i = 0; i < 60 && server.exitCode === null; i++) await sleep(100);
  if (server.exitCode === null) server.kill('SIGKILL');
  console.log('server stopped');
}
process.on('exit', () => { try { if (server && server.exitCode === null) server.kill('SIGKILL'); } catch (e) { /* ignore */ } });
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => { stopServer().finally(() => process.exit(130)); });

// ---------------------------------------------------------------------------
// game data helpers (the same /api/data the page uses)

let DATA = null;
let REC = null;
let SP = null;
let ITEMS = null;
const CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ';
const initials = (s) => [...s].map((ch) => {
  const c = ch.charCodeAt(0) - 0xac00;
  return c >= 0 && c <= 11171 ? CHO[Math.floor(c / 588)] : ch;
}).join('');
const koSp = (n) => (SP.get(n) && SP.get(n).ko) || '';
const koMove = (n) => (DATA.moves[n] && DATA.moves[n].ko) || '';
const koItem = (n) => (ITEMS.get(n) && ITEMS.get(n).ko) || '';
const koAb = (n) => DATA.abilities[n] || '';
const koNat = (n) => (DATA.natures.find((x) => x.name === n) || {}).ko || '';
const mainSp = (n) => koSp(n) || n;
const shown = (ko, en) => (ko ? `${ko} (${en})` : en);
const hasHangul = (s) => /[가-힣]/.test(s || '');
const pctTxt = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const STAT_KO = { hp: 'HP', atk: '공격', def: '방어', spa: '특공', spd: '특방', spe: '스피드' };
const statLine = (st) => ['H', 'A', 'B', 'C', 'D', 'S'].map((a, i) => `${a}${st[STATS[i]]}`).join(' ');

async function getJSON(p, body) {
  const r = await fetch(`${BASE}${p}`, body === undefined ? undefined
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return { status: r.status, json: await r.json() };
}

// ---------------------------------------------------------------------------
// page helpers

async function openPage(browser, vp) {
  const ctx = await browser.newContext(vp.ctx);
  try { await ctx.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: BASE }); } catch (e) { /* ignore */ }
  const page = await ctx.newPage();
  page.setDefaultTimeout(15000);
  const P = {
    vp: vp.name, phone: vp.phone, ctx, page, n: 0, console: [], reqFail: [], bad: [], external: [], forced: [],
    expectBad: 0, expectConsole: 0, expectAbort: 0, layout: [],
  };
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    if (P.expectConsole > 0 && /status of 400/.test(m.text())) { P.expectConsole -= 1; return; }
    P.console.push(`console.error: ${m.text()}`);
  });
  page.on('pageerror', (e) => P.console.push(`pageerror: ${e.message}`));
  page.on('requestfailed', (r) => {
    const err = (r.failure() && r.failure().errorText) || '';
    if (P.expectAbort > 0 && /ERR_ABORTED/.test(err) && /\/api\/(advise|preview|advise_switch)$/.test(r.url())) { P.expectAbort -= 1; return; }
    P.reqFail.push(`${r.method()} ${r.url()} -> ${err}`);
  });
  page.on('request', (r) => {
    const u = r.url();
    if (!u.startsWith(BASE) && !/^(data|blob|about):/.test(u)) P.external.push(u);
  });
  page.on('response', async (r) => {
    if (r.status() < 400) return;
    if (P.expectBad > 0) { P.expectBad -= 1; return; }
    let body = '';
    try { body = (await r.text()).slice(0, 300); } catch (e) { /* ignore */ }
    P.bad.push(`${r.status()} ${r.request().method()} ${r.url()} ${body}`);
  });
  return P;
}

async function shot(P, name, opts = {}) {
  const file = path.join(OUT, `${PREFIX}_${P.vp}_${String(++P.n).padStart(2, '0')}_${name}.png`);
  try {
    if (opts.sel) {
      const loc = P.page.locator(opts.sel).first();
      await loc.scrollIntoViewIfNeeded();
      await loc.screenshot({ path: file });
    } else {
      await P.page.screenshot({ path: file, fullPage: !!opts.full });
    }
    R.screenshots.push(file);
  } catch (e) {
    warn(P.vp, `screenshot ${name} failed`, e.message.split('\n')[0]);
  }
  return file;
}

async function appState(P) {
  await P.page.waitForTimeout(350); // the page debounces saves by 150 ms
  return P.page.evaluate((k) => {
    try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; }
  }, STORE_KEY);
}

/** Click like a user; if something covers the element, record it and force the click so the flow goes on. */
async function click(P, target, desc) {
  const loc = typeof target === 'string' ? P.page.locator(target).first() : target;
  try {
    await loc.click({ timeout: 8000 });
  } catch (e) {
    const lines = e.message.split('\n');
    const line = (lines.find((l) => /intercepts pointer|not visible|outside of the viewport|not stable|not enabled|detached/.test(l)) || lines[0]).trim();
    P.forced.push(`${desc || target}: ${line}`);
    await loc.click({ force: true, timeout: 5000 });
  }
}

async function blur(P) {
  await P.page.evaluate(() => { if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); });
}

async function setNumber(P, sel, value) {
  const loc = P.page.locator(sel);
  await loc.scrollIntoViewIfNeeded();
  await loc.fill(String(value));
  await loc.evaluate((e) => e.blur()); // fires "change" like leaving the field
  await P.page.waitForTimeout(120);
}

async function select(P, sel, value) {
  const loc = P.page.locator(sel);
  await loc.scrollIntoViewIfNeeded();
  await loc.selectOption(String(value));
  await P.page.waitForTimeout(120);
}

async function toastSeen(P, re, what) {
  const t = P.page.locator('#toasts .toast', { hasText: re });
  try {
    await t.first().waitFor({ state: 'attached', timeout: 6000 });
    pass(P.vp, `toast: ${what}`);
    return true;
  } catch (e) {
    fail(P.vp, `toast: ${what}`, `toasts now: "${(await P.page.locator('#toasts').innerText().catch(() => '')).replace(/\s+/g, ' ')}"`);
    return false;
  }
}

async function cbItems(P, id) {
  return P.page.locator(`[id="${id}-list"] li.cb-opt`).evaluateAll((els) => els.map((e) => ({
    main: (e.querySelector('.cb-main') || {}).textContent || '', sub: (e.querySelector('.cb-sub') || {}).textContent || '',
    hint: (e.querySelector('.cb-hint') || {}).textContent || '',
  })));
}

/**
 * Pick an option in a searchable dropdown by typing `query`.
 * want = {value: canonical English name} or {label: visible main label}. Desktop: arrows + Enter; phone: tap.
 */
async function cbPick(P, id, query, want, opts = {}) {
  const { page } = P;
  const input = page.locator(`[id="${id}"]`);
  const list = page.locator(`[id="${id}-list"]`);
  const what = opts.what || `#${id}`;
  const wanted = want.value !== undefined ? want.value : want.label;
  try {
    await input.scrollIntoViewIfNeeded();
    await click(P, input, `combobox ${what}`);
    await input.fill(query);
    await list.waitFor({ state: 'visible', timeout: 5000 });
  } catch (e) {
    fail(P.vp, `${what}: could not open the dropdown`, e.message.split('\n')[0]);
    return false;
  }
  const items = await cbItems(P, id);
  const isWant = (t) => (want.label !== undefined ? t.main === want.label : (t.sub === want.value || (!t.sub && t.main === want.value)));
  const idx = items.findIndex(isWant);
  const top = items.slice(0, 5).map((t) => `${t.main}${t.sub ? ` (${t.sub})` : ''}`).join(', ') || '(no options)';
  if (idx < 0) {
    fail(P.vp, `${what}: typing "${query}" lists ${wanted}`, `options: ${top}`);
    await input.press('Escape');
    return false;
  }
  const rankMsg = `${what}: typing "${query}" puts ${wanted} first`;
  if (idx === 0) pass(P.vp, rankMsg);
  else if (opts.rankWarnOnly) warn(P.vp, rankMsg, `it is #${idx + 1}; top: ${top}`);
  else fail(P.vp, rankMsg, `it is #${idx + 1}; top: ${top}`);
  if (P.phone) {
    const bb = await list.boundingBox();
    if (bb && bb.x + bb.width > 391) fail(P.vp, `${what}: open dropdown list fits the 390px screen`, `list x=${bb.x.toFixed(0)} width=${bb.width.toFixed(0)}`);
    await click(P, list.locator('li.cb-opt').nth(idx), `option ${wanted} in ${what}`);
  } else {
    for (let i = 0; i < idx; i++) await input.press('ArrowDown');
    await input.press('Enter');
  }
  await page.waitForTimeout(150);
  return true;
}

/** Layout / accessibility audit of the visible page (horizontal overflow, clipped text, labels, tiny text). */
async function audit(P, stepName) {
  const res = await P.page.evaluate((isPhone) => {
    const vw = document.documentElement.clientWidth;
    const desc = (el) => {
      let s = el.tagName.toLowerCase();
      if (el.id) s += `#${el.id}`;
      else if (typeof el.className === 'string' && el.className.trim()) s += `.${el.className.trim().split(/\s+/).slice(0, 2).join('.')}`;
      const t = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().replace(/\s+/g, ' ').slice(0, 40);
      return t ? `${s} "${t}"` : s;
    };
    const visible = (el) => {
      if (el.closest('[hidden], .sr-only, .offscreen, .skip, #toasts')) return false;
      for (let d = el.closest('details:not([open])'); d; d = d.parentElement && d.parentElement.closest('details:not([open])')) {
        const sum = d.querySelector(':scope > summary');
        if (!(sum && sum.contains(el))) return false;
      }
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) return false;
      const cs = getComputedStyle(el);
      return cs.visibility !== 'hidden' && cs.display !== 'none' && cs.opacity !== '0';
    };
    const clipped = (el) => {
      for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
        const ox = getComputedStyle(p).overflowX;
        if (ox !== 'visible') {
          const pr = p.getBoundingClientRect();
          if (pr.right <= vw + 1 && pr.left >= -1) return true;
        }
      }
      return false;
    };
    const out = (el) => { const r = el.getBoundingClientRect(); return r.right > vw + 1 || r.left < -1; };
    const all = [...document.querySelectorAll('body *')].filter((el) => !['SCRIPT', 'STYLE', 'OPTION'].includes(el.tagName));
    const overflow = [];
    for (const el of all) {
      if (!visible(el) || !out(el) || clipped(el)) continue;
      const p = el.parentElement;
      if (p && p !== document.body && visible(p) && out(p) && !clipped(p)) continue; // report the outermost only
      const r = el.getBoundingClientRect();
      overflow.push(`${desc(el)} [x ${Math.round(r.left)}..${Math.round(r.right)}]`);
    }
    const truncated = [];
    for (const el of document.querySelectorAll('input.cb-input, input[type=text], .btn, .tab, .seg-btn, .chip, .tag')) {
      if (!visible(el)) continue;
      // a combobox that is not focused shows its value in a two-line overlay (its input text is transparent)
      const cb = el.classList.contains('cb-input') && document.activeElement !== el ? el.closest('.cb-shown') : null;
      const parts = cb ? [...cb.querySelectorAll('.cb-show-main, .cb-show-sub')].filter((x) => x.textContent) : [el];
      for (const x of parts) {
        if (x.scrollWidth > x.clientWidth + 2) truncated.push(`${desc(el)}${x === el ? '' : ` ${x.className}`} (${x.scrollWidth}px text in ${x.clientWidth}px)`);
      }
    }
    // short labels (ranks, tags, badges, buttons) that break over two lines
    const wrapped = [];
    for (const el of document.querySelectorAll('.rank, .tag, .type-chip, .best-badge, .tab-badge, .big-num, .seg-btn, .tab, .btn, th, .roster-state')) {
      if (!visible(el)) continue;
      const t = (el.textContent || '').trim();
      if (!t || t.length > 12) continue;
      const fsz = parseFloat(getComputedStyle(el).fontSize) || 14;
      const tops = [];
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        if (!n.textContent.trim()) continue;
        const rg = document.createRange();
        rg.selectNodeContents(n);
        for (const r of rg.getClientRects()) if (r.width > 0.5) tops.push(r.top);
      }
      if (tops.length > 1 && Math.max(...tops) - Math.min(...tops) > fsz * 0.6) wrapped.push(`${desc(el)} breaks over ${new Set(tops.map(Math.round)).size} lines`);
    }
    const tiny = [];
    for (const el of all) {
      if (!visible(el)) continue;
      if (![...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) continue;
      const fs = parseFloat(getComputedStyle(el).fontSize);
      if (fs < 11) tiny.push(`${desc(el)} ${fs}px`);
    }
    const unlabeled = [];
    for (const el of document.querySelectorAll('input, select, textarea')) {
      if (el.type === 'hidden' || !visible(el)) continue;
      const lab = el.getAttribute('aria-label') || el.getAttribute('aria-labelledby') || el.title
        || (el.labels && [...el.labels].map((l) => (l.innerText || l.textContent || '').trim()).join(' '));
      if (!lab) unlabeled.push(desc(el));
    }
    const unnamed = [...document.querySelectorAll('button')].filter((b) => visible(b)
      && !((b.innerText || b.textContent || '').trim() || b.getAttribute('aria-label') || b.title)).map(desc);
    const small = [];
    if (isPhone) {
      for (const el of document.querySelectorAll('button:not(.cb-toggle):not(.cb-clear):not(.toast-x), select, input[type=text], input[type=number], summary')) {
        if (!visible(el)) continue;
        const r = el.getBoundingClientRect();
        if (r.height < 32 || r.width < 28) small.push(`${desc(el)} ${Math.round(r.width)}x${Math.round(r.height)}`);
      }
    }
    return {
      docOverflow: document.documentElement.scrollWidth - vw, overflow: overflow.slice(0, 20),
      wrapped: [...new Set(wrapped)].slice(0, 20), truncated: [...new Set(truncated)].slice(0, 40), tiny: [...new Set(tiny)].slice(0, 20), unlabeled: unlabeled.slice(0, 20),
      unnamed: unnamed.slice(0, 20), small: small.slice(0, 30),
    };
  }, P.phone);
  res.step = stepName;
  P.layout.push(res);
  return res;
}

const deepEq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sortEq = (a, b) => Array.isArray(a) && Array.isArray(b) && deepEq([...a].sort(), [...b].sort());

/** spec: {key: exact value | ABSENT | {$sorted: [...]}} */
const ABSENT = Symbol('absent');
function expectPartial(errs, obj, spec, where) {
  if (!obj || typeof obj !== 'object') { errs.push(`${where} missing`); return; }
  for (const [k, v] of Object.entries(spec)) {
    if (v === ABSENT) { if (k in obj && obj[k] !== null && obj[k] !== undefined) errs.push(`${where}.${k} should be omitted (unknown) but is ${JSON.stringify(obj[k])}`); continue; }
    if (v && v.$sorted) { if (!sortEq(obj[k], v.$sorted)) errs.push(`${where}.${k}=${JSON.stringify(obj[k])} expected ${JSON.stringify(v.$sorted)}`); continue; }
    if (!deepEq(obj[k], v)) errs.push(`${where}.${k}=${JSON.stringify(obj[k])} expected ${JSON.stringify(v)}`);
  }
}

function findNulls(o, p, out) {
  if (o === null) out.push(p);
  else if (Array.isArray(o)) o.forEach((x, i) => findNulls(x, `${p}[${i}]`, out));
  else if (typeof o === 'object') for (const [k, v] of Object.entries(o)) findNulls(v, `${p}.${k}`, out);
  return out;
}

/** Check a <state> (API.md) built by the page. exp has the values the test entered. */
function validateState(s, exp) {
  const errs = [];
  if (!s || typeof s !== 'object') return ['state is missing'];
  if (s.format !== DATA.format) errs.push(`format=${JSON.stringify(s.format)} expected ${DATA.format}`);
  if (!Number.isInteger(s.turn) || s.turn !== exp.turn) errs.push(`turn=${JSON.stringify(s.turn)} expected integer ${exp.turn}`);
  const me = s.me || {};
  const foe = s.foe || {};
  if (!Array.isArray(me.team) || me.team.length !== 6) errs.push(`me.team should be 6 sets: ${JSON.stringify(me.team).slice(0, 100)}`);
  else {
    me.team.forEach((t, i) => {
      for (const k of ['species', 'item', 'ability', 'moves', 'nature', 'evs']) if (!(k in t)) errs.push(`me.team[${i}] lacks "${k}"`);
      if (!SP.has(t.species)) errs.push(`me.team[${i}].species "${t.species}" is not a canonical name`);
      if (t.item && !ITEMS.has(t.item)) errs.push(`me.team[${i}].item "${t.item}" is not canonical`);
      if (t.ability && !(t.ability in DATA.abilities)) errs.push(`me.team[${i}].ability "${t.ability}" is not canonical`);
      if (!Array.isArray(t.moves) || !t.moves.length || t.moves.length > 4 || t.moves.some((m) => !DATA.moves[m])) errs.push(`me.team[${i}].moves ${JSON.stringify(t.moves)}`);
      if (!t.evs || STATS.some((k) => typeof t.evs[k] !== 'number')) errs.push(`me.team[${i}].evs ${JSON.stringify(t.evs)}`);
      const want = exp.team[i];
      if (want && (want.species !== t.species || want.item !== t.item || want.nature !== t.nature || !deepEq(want.evs, t.evs)
        || !deepEq(want.moves.filter(Boolean), t.moves))) errs.push(`me.team[${i}] differs from the chosen team: ${JSON.stringify(t)}`);
    });
  }
  if (!deepEq(me.brought, exp.brought)) errs.push(`me.brought=${JSON.stringify(me.brought)} expected ${JSON.stringify(exp.brought)}`);
  if (!deepEq(me.active, [exp.myActive])) errs.push(`me.active=${JSON.stringify(me.active)} expected ${JSON.stringify([exp.myActive])}`);
  const mp = me.pokemon || {};
  if (!sortEq(Object.keys(mp), exp.brought)) errs.push(`me.pokemon keys ${JSON.stringify(Object.keys(mp))} should be the brought Pokemon`);
  for (const [n, m] of Object.entries(mp)) {
    if (typeof m.hp !== 'string' || !/^\d+\/\d+$/.test(m.hp)) errs.push(`me.pokemon.${n}.hp=${JSON.stringify(m.hp)} should be "cur/max"`);
    else {
      const [c, x] = m.hp.split('/').map(Number);
      if (x !== exp.maxHp[n]) errs.push(`me.pokemon.${n}.hp max ${x} != HP stat ${exp.maxHp[n]}`);
      if (c > x) errs.push(`me.pokemon.${n}.hp cur > max (${m.hp})`);
    }
    if ('status' in m && !STATUSES.includes(m.status)) errs.push(`me.pokemon.${n}.status=${JSON.stringify(m.status)}`);
    if (n !== exp.myActive) for (const k of ['boosts', 'volatiles', 'locked_move', 'substitute_hp']) if (k in m) errs.push(`me.pokemon.${n} is not active but has "${k}"`);
    if (m.boosts) for (const [k, v] of Object.entries(m.boosts)) if (!Number.isInteger(v) || v === 0 || v < -6 || v > 6) errs.push(`me.pokemon.${n}.boosts.${k}=${v}`);
  }
  for (const [n, spec] of Object.entries(exp.mePokemon || {})) expectPartial(errs, mp[n], spec, `me.pokemon.${n}`);
  if (!deepEq(me.side, exp.meSide)) errs.push(`me.side=${JSON.stringify(me.side)} expected ${JSON.stringify(exp.meSide)}`);
  if (typeof me.mega_used !== 'boolean') errs.push(`me.mega_used=${JSON.stringify(me.mega_used)} should be boolean`);
  if (exp.meMegaUsed !== undefined && me.mega_used !== exp.meMegaUsed) errs.push(`me.mega_used=${me.mega_used} expected ${exp.meMegaUsed}`);
  const foeTeam = exp.foeTeam || FOE;
  if (!deepEq(foe.team, foeTeam)) errs.push(`foe.team=${JSON.stringify(foe.team)} expected ${JSON.stringify(foeTeam)}`);
  if (!sortEq(foe.brought, exp.foeSeen)) errs.push(`foe.brought=${JSON.stringify(foe.brought)} expected ${JSON.stringify(exp.foeSeen)}`);
  if (!deepEq(foe.active, [exp.foeActive])) errs.push(`foe.active=${JSON.stringify(foe.active)} expected ${JSON.stringify([exp.foeActive])}`);
  const fp = foe.pokemon || {};
  if (!sortEq(Object.keys(fp), exp.foeSeen)) errs.push(`foe.pokemon keys ${JSON.stringify(Object.keys(fp))} should be the seen Pokemon ${JSON.stringify(exp.foeSeen)}`);
  for (const [n, m] of Object.entries(fp)) {
    if (typeof m.hp !== 'string' || !/^\d{1,3}%$/.test(m.hp)) errs.push(`foe.pokemon.${n}.hp=${JSON.stringify(m.hp)} should be "NN%"`);
    if (m.item !== undefined && m.item !== '' && !ITEMS.has(m.item)) errs.push(`foe.pokemon.${n}.item "${m.item}" is not canonical`);
    if (m.item === '?') errs.push(`foe.pokemon.${n}.item is the UI placeholder "?"`);
    if (m.ability !== undefined && !(m.ability in DATA.abilities)) errs.push(`foe.pokemon.${n}.ability "${m.ability}" is not canonical`);
    if (m.moves && m.moves.some((x) => !DATA.moves[x])) errs.push(`foe.pokemon.${n}.moves ${JSON.stringify(m.moves)}`);
    for (const k of ['faster_than', 'slower_than']) if (m[k] && m[k].some((x) => !exp.brought.includes(x))) errs.push(`foe.pokemon.${n}.${k} names a Pokemon we did not bring`);
    if (n !== exp.foeActive) for (const k of ['boosts', 'volatiles', 'substitute_hp']) if (k in m) errs.push(`foe.pokemon.${n} is not active but has "${k}"`);
  }
  for (const [n, spec] of Object.entries(exp.foePokemon || {})) expectPartial(errs, fp[n], spec, `foe.pokemon.${n}`);
  if (!deepEq(foe.side, exp.foeSide)) errs.push(`foe.side=${JSON.stringify(foe.side)} expected ${JSON.stringify(exp.foeSide)}`);
  if (typeof foe.mega_used !== 'boolean') errs.push(`foe.mega_used=${JSON.stringify(foe.mega_used)} should be boolean`);
  if (exp.foeMegaUsed !== undefined && foe.mega_used !== exp.foeMegaUsed) errs.push(`foe.mega_used=${foe.mega_used} expected ${exp.foeMegaUsed}`);
  const f = s.field || {};
  for (const k of ['weather_turns', 'terrain_turns', 'trickroom', 'gravity', 'magicroom', 'wonderroom']) if (typeof f[k] !== 'number') errs.push(`field.${k}=${JSON.stringify(f[k])} should be a number`);
  if (exp.field) expectPartial(errs, f, exp.field, 'field');
  const nulls = findNulls(s, 'state', []);
  if (nulls.length) errs.push(`null values: ${nulls.join(', ')}`);
  return errs;
}

// ---------------------------------------------------------------------------
// the flow

async function runViewport(browser, vp) {
  console.log(`\n=== ${vp.name} ${vp.ctx.viewport.width}x${vp.ctx.viewport.height} ===`);
  const P = await openPage(browser, vp);
  const { page } = P;
  const V = vp.name;
  const T = {};
  const ctx = {}; // values carried between steps
  const step = async (name, fn) => {
    const t0 = Date.now();
    try {
      await fn();
    } catch (e) {
      fail(V, `step "${name}" threw`, e.message.split('\n').slice(0, 3).join(' | '));
      await shot(P, `FAIL_${name.replace(/[^a-z0-9]+/gi, '_')}`);
    }
    T[name] = ((Date.now() - t0) / 1000).toFixed(1);
  };

  // 1) recommended teams --------------------------------------------------------------------------------
  await step('recommended', async () => {
    await page.goto(`${BASE}/`);
    await page.locator('#main').waitFor({ state: 'visible', timeout: 30000 });
    const coarse = await page.evaluate(() => window.matchMedia('(pointer: coarse)').matches);
    if (P.phone) check(V, coarse, 'phone emulation reports a coarse (touch) pointer');
    check(V, (await page.locator('#tab-team').getAttribute('aria-selected')) === 'true', 'first screen is step 1 "추천 파티 / 내 팀" and its tab is highlighted');
    const tabs = await page.locator('#tabs .tab').evaluateAll((els) => els.map((e) => e.innerText.replace(/\s+/g, ' ').trim()));
    check(V, tabs.length === 4, 'four step tabs are shown', tabs.join(' | '));
    const tabFit = await page.locator('#tabs').evaluate((e) => [e.scrollWidth, e.clientWidth]);
    check(V, tabFit[0] <= tabFit[1] + 1, 'all step tabs fit the width without scrolling', tabFit);
    await page.locator('#rec-card article.team-card').first().waitFor({ timeout: 60000 });
    const firstId = await page.evaluate(() => (document.querySelector('#panel-team > *') || {}).id);
    check(V, firstId === 'rec-card', 'the AI recommended teams are the first card on the page', `first card: ${firstId}`);
    const bb = await page.locator('#rec-card').boundingBox();
    check(V, bb && bb.y < 300, 'the recommended teams start within the first screen', `#rec-card y=${bb && bb.y}`);
    const cards = page.locator('#rec-card article.team-card');
    check(V, (await cards.count()) === REC.teams.length, `all ${REC.teams.length} recommended teams are listed`, `${await cards.count()} cards`);
    const t = REC.teams[0];
    const c1 = cards.first();
    const head = (await c1.locator('.team-h').innerText()).replace(/\s+/g, ' ');
    check(V, head.includes(t.source), 'team card shows its source', head);
    if (t.fitness !== null && t.fitness !== undefined) check(V, head.includes(Number(t.fitness).toFixed(2)), 'team card shows its fitness', head);
    const mons = c1.locator('.mon');
    check(V, (await mons.count()) === 6, 'team card shows 6 Pokemon', `${await mons.count()}`);
    const probs = [];
    for (let i = 0; i < Math.min(6, await mons.count()); i++) {
      const s = t.sets[i];
      const m = mons.nth(i);
      const txt = (await m.innerText()).replace(/\s+/g, ' ');
      const need = [mainSp(s.species), s.species, koItem(s.item) || s.item, koAb(s.ability) || s.ability, koNat(s.nature) || s.nature,
        ...s.moves.map((mv) => koMove(mv) || mv),
        ...STATS.filter((k) => s.evs[k]).map((k) => `${STAT_KO[k]} ${s.evs[k]}`)];
      const missing = need.filter((x) => x && !txt.includes(x));
      const types = await m.locator('.mon-h .type-chip').count();
      if (types !== SP.get(s.species).types.length) missing.push(`type chips (${types})`);
      const nMoves = await m.locator('ul.moves li').count();
      if (nMoves !== s.moves.length) missing.push(`moves (${nMoves})`);
      if (!hasHangul((await m.locator('.mon-h .d-main').innerText()))) missing.push('Korean name');
      if (missing.length) probs.push(`${s.species}: missing ${missing.join(', ')} | "${txt.slice(0, 200)}"`);
    }
    check(V, !probs.length, 'each recommended Pokemon shows Korean+English name, types, item, ability, nature, SP spread, 4 moves', probs.join('\n'));
    check(V, await c1.getByRole('button', { name: '이 팀으로 배틀' }).isVisible(), 'team card has "이 팀으로 배틀"');
    check(V, await c1.getByRole('button', { name: '텍스트 복사' }).isVisible(), 'team card has "텍스트 복사"');
    // learned top species
    const rows = page.locator('#d-top-species tbody tr');
    const nTop = REC.top_species.length;
    if (nTop) {
      check(V, (await rows.count()) === nTop, `top species table lists ${nTop} learned species`, `${await rows.count()} rows`);
      const r0 = (await rows.first().innerText()).replace(/\s+/g, ' ');
      const ts = REC.top_species[0];
      check(V, r0.includes(mainSp(ts.species)) && r0.includes(pctTxt(ts.win_rate, 1)), 'top species row shows Korean name + win rate', r0);
    }
    await shot(P, 'team_recommended');
    await audit(P, 'team tab (recommended)');
    // copy export
    await click(P, c1.getByRole('button', { name: '텍스트 복사' }), '텍스트 복사');
    await toastSeen(P, /클립보드에 복사/, '"텍스트 복사" confirms the copy');
    const clip = await page.evaluate(() => navigator.clipboard.readText()).catch((e) => `ERR ${e.message}`);
    check(V, clip === t.text, '"텍스트 복사" puts the Showdown export of the team on the clipboard', clip.slice(0, 120));
    // use it
    await click(P, '#use-team-1', '이 팀으로 배틀');
    await toastSeen(P, /팀으로 설정/, 'team chosen');
    check(V, (await page.locator('#tab-preview').getAttribute('aria-selected')) === 'true', '"이 팀으로 배틀" goes to step 2 (선출)');
    const st = await appState(P);
    check(V, st && st.team && deepEq(st.team.sets.map((s) => s.species), t.species), 'the recommended team is stored as my team', st && st.team && st.team.sets.map((s) => s.species));
    await click(P, '#tab-team');
    const first = await page.evaluate(() => (document.querySelector('#panel-team > *') || {}).id);
    check(V, first === 'current-team', 'team tab then shows the current team first', first);
    check(V, await page.locator('#rec-card .team-card.is-current .tag-ok').first().isVisible().catch(() => false), 'the used recommended team is marked "사용 중"');
  });

  // 2) paste a Showdown export ---------------------------------------------------------------------------
  await step('paste', async () => {
    if ((await page.locator('#tab-team').getAttribute('aria-selected')) !== 'true') await click(P, '#tab-team');
    if (!(await page.locator('#d-paste').evaluate((e) => e.open))) await click(P, '#d-paste > summary', 'paste section');
    await page.fill('#paste-text', '');
    await click(P, '#paste-parse');
    const err = page.locator('#d-paste .msg-err');
    await err.waitFor({ timeout: 5000 }).catch(() => {});
    check(V, (await err.count()) && (await err.innerText()).includes('텍스트를 붙여 넣으세요'), 'empty paste shows an inline error');
    // illegal: SP total over 66
    const bad = TEAM_TXT.replace('SPs: 2 HP / 32 Atk / 32 Spe', 'SPs: 20 HP / 32 Atk / 32 Spe');
    await page.fill('#paste-text', bad);
    const r1 = page.waitForResponse((r) => r.url().endsWith('/api/team/parse'));
    await click(P, '#paste-parse');
    await r1;
    const warnBox = page.locator('#d-paste .parse-res .msg-warn');
    await warnBox.waitFor({ timeout: 10000 });
    const wtxt = (await warnBox.innerText()).replace(/\s+/g, ' ');
    check(V, /규칙 위반/.test(wtxt) && (await warnBox.locator('li').count()) >= 1, 'an illegal export (SP total 84) lists the legality problems', wtxt);
    await shot(P, 'paste_problems', { sel: '#d-paste' });
    // legal
    await page.fill('#paste-text', TEAM_TXT);
    const r2 = page.waitForResponse((r) => r.url().endsWith('/api/team/parse'));
    await click(P, '#paste-parse');
    await r2;
    const ok = page.locator('#d-paste .parse-res .msg-ok');
    await ok.waitFor({ timeout: 10000 });
    check(V, (await ok.innerText()).includes('문제 없음') && (await ok.innerText()).includes('6'), 'the example team parses with "문제 없음 · 6마리"', await ok.innerText());
    const names = await page.locator('#d-paste .parse-res .mon .mon-h .d-main').allInnerTexts();
    const want = ['Garchomp', 'Rotom-Wash', 'Kingambit', 'Gengar', 'Talonflame', 'Azumarill'].map(mainSp);
    check(V, deepEq(names, want), 'parsed sets are shown with Korean names', names.join(', '));
    await audit(P, 'team tab (paste result)');
    await click(P, page.locator('#d-paste .parse-res').getByRole('button', { name: '이 팀 사용' }), '이 팀 사용 (paste)');
    await toastSeen(P, /붙여넣은 팀/, 'pasted team chosen');
    const st = await appState(P);
    check(V, st.team && st.team.label === '붙여넣은 팀' && st.team.sets[0].species === 'Garchomp' && st.tab === 'preview',
      'pasted team becomes my team and the page moves to 선출', st.team && { label: st.team.label, tab: st.tab });
  });

  // 3) dropdown team builder -----------------------------------------------------------------------------
  await step('builder', async () => {
    await click(P, '#tab-team');
    if (!(await page.locator('#d-builder').evaluate((e) => e.open))) await click(P, '#d-builder > summary', 'builder section');
    await page.locator('#b0-species').waitFor();
    // species by Korean initial consonants
    await cbPick(P, 'b0-species', initials(koSp('Garchomp')), { value: 'Garchomp' }, { what: 'builder species (초성)' });
    let st = await appState(P);
    check(V, st.builder[0].species === 'Garchomp', 'builder slot 1 species = Garchomp', st.builder[0]);
    check(V, (await page.locator('#b0-species').inputValue()) === shown(koSp('Garchomp'), 'Garchomp'), 'the dropdown shows "한글 (English)" after choosing', await page.locator('#b0-species').inputValue());
    check(V, SP.get('Garchomp').abilities.includes(st.builder[0].ability), 'picking a species fills a legal ability', st.builder[0].ability);
    // item by English, ability / nature / moves by Korean
    await cbPick(P, 'b0-item', 'choice scarf', { value: 'Choice Scarf' }, { what: 'builder item (English)' });
    const abItems = await (async () => { await click(P, '#b0-ability'); await page.waitForTimeout(100); const x = await cbItems(P, 'b0-ability'); await page.locator('#b0-ability').press('Escape'); return x; })();
    check(V, abItems.length === SP.get('Garchomp').abilities.length, 'ability dropdown only offers the species\' legal abilities', abItems.map((x) => x.sub || x.main));
    await cbPick(P, 'b0-ability', koAb('Rough Skin'), { value: 'Rough Skin' }, { what: 'builder ability (Korean)' });
    await cbPick(P, 'b0-nature', koNat('Jolly'), { value: 'Jolly' }, { what: 'builder nature (Korean)' });
    await cbPick(P, 'b0-move0', koMove('Earthquake'), { value: 'Earthquake' }, { what: 'builder move (Korean)' });
    await cbPick(P, 'b0-move1', 'dragon claw', { value: 'Dragon Claw' }, { what: 'builder move (English)' });
    await cbPick(P, 'b0-move2', initials(koMove('Stone Edge')), { value: 'Stone Edge' }, { what: 'builder move (초성)', rankWarnOnly: true });
    // legal moves only
    await click(P, '#b0-move3');
    await page.locator('#b0-move3').fill('shadow ball');
    const none = await page.locator('#b0-move3-list .cb-none').innerText().catch(() => '');
    check(V, (await page.locator('#b0-move3-list li.cb-opt').count()) === 0 && none.includes('일치하는 항목이 없습니다'),
      'move dropdown is limited to the species\' learnset (Garchomp cannot learn Shadow Ball)', none);
    // keyboard: arrows move the highlight, Escape closes without choosing
    if (!P.phone) {
      await page.locator('#b0-move3').fill('dragon');
      const nDragon = await page.locator('#b0-move3-list li.cb-opt').count();
      const a0 = await page.locator('#b0-move3').getAttribute('aria-activedescendant');
      await page.locator('#b0-move3').press('ArrowDown');
      const a1 = await page.locator('#b0-move3').getAttribute('aria-activedescendant');
      await page.locator('#b0-move3').press('ArrowUp');
      const a2 = await page.locator('#b0-move3').getAttribute('aria-activedescendant');
      check(V, nDragon >= 2 && a0 && a1 && a0 !== a1 && a2 === a0, 'ArrowDown / ArrowUp move the highlighted option', `${nDragon} options; ${a0} -> ${a1} -> ${a2}`);
      await page.locator('#b0-move3').press('Escape');
      check(V, (await page.locator('#b0-move3').getAttribute('aria-expanded')) === 'false' && !(await page.locator('#b0-move3-list').isVisible()),
        'Escape closes the dropdown');
      st = await appState(P);
      check(V, st.builder[0].moves[3] === '', 'Escape does not choose anything', st.builder[0].moves);
    } else {
      await page.locator('#b0-move3').press('Escape').catch(() => {});
      await blur(P);
    }
    // duplicate move is refused
    await cbPick(P, 'b0-move3', koMove('Earthquake'), { value: 'Earthquake' }, { what: 'builder duplicate move' });
    await toastSeen(P, /같은 기술을 두 번/, 'duplicate move refused');
    st = await appState(P);
    check(V, st.builder[0].moves[3] === '', 'duplicate move is not stored', st.builder[0].moves);
    await cbPick(P, 'b0-move3', 'poison jab', { value: 'Poison Jab' }, { what: 'builder move (English 2)' });
    // SPs with a live total
    await setNumber(P, '#b0-sp-hp', 2);
    await setNumber(P, '#b0-sp-atk', 32);
    await setNumber(P, '#b0-sp-spe', 32);
    let tot = await page.locator('#btotal-0').innerText();
    check(V, tot.includes('66 / 66') && !(await page.locator('#btotal-0').getAttribute('class')).includes('bad'), 'SP total shows 66 / 66', tot);
    const stats = await page.locator('#bstats-0').innerText();
    check(V, stats.includes('S169') && stats.includes('A182'), 'builder shows the computed stats (Jolly 32 Spe Garchomp: S169, A182)', stats);
    await setNumber(P, '#b0-sp-spd', 10);
    tot = await page.locator('#btotal-0').innerText();
    check(V, tot.includes('76 / 66') && (await page.locator('#btotal-0').getAttribute('class')).includes('bad'), 'SP total over 66 turns red', tot);
    await setNumber(P, '#b0-sp-atk', 40);
    check(V, (await page.locator('#b0-sp-atk').inputValue()) === '32', 'an SP input clamps 40 to 32', await page.locator('#b0-sp-atk').inputValue());
    // validate a 1-Pokemon team with SP 76 -> problems listed
    const r1 = page.waitForResponse((r) => r.url().endsWith('/api/team/parse'));
    await click(P, '#builder-validate');
    await r1;
    const wb = page.locator('#d-builder .parse-res .msg-warn');
    await wb.waitFor({ timeout: 10000 });
    const wtxt = (await wb.innerText()).replace(/\s+/g, ' ');
    check(V, (await wb.locator('li').count()) >= 1, 'validating an incomplete builder team lists the problems', wtxt);
    await shot(P, 'builder_slot1', { sel: '#bslot-0' });
    await setNumber(P, '#b0-sp-spd', 0);
    // load my current team (the pasted one) into the builder and change slot 6 to Kleavor
    await click(P, page.locator('#d-builder').getByRole('button', { name: '현재 팀 불러오기' }), '현재 팀 불러오기');
    await page.waitForTimeout(400);
    st = await appState(P);
    check(V, st.builder.map((b) => b.species).join() === 'Garchomp,Rotom-Wash,Kingambit,Gengar,Talonflame,Azumarill', '"현재 팀 불러오기" fills the 6 builder slots', st.builder.map((b) => b.species));
    await cbPick(P, 'b5-species', initials(koSp('Kleavor')), { value: 'Kleavor' }, { what: 'builder species 6 (초성)', rankWarnOnly: true });
    st = await appState(P);
    check(V, st.builder[5].species === 'Kleavor' && st.builder[5].moves.every((m) => !m || SP.get('Kleavor').moves.includes(m)),
      'changing the species drops moves it cannot learn', st.builder[5]);
    const kmoves = ['Stone Axe', 'X-Scissor', 'Close Combat', 'Protect'];
    const qs = [koMove('Stone Axe'), 'x-scissor', koMove('Close Combat'), 'protect'];
    for (let j = 0; j < 4; j++) await cbPick(P, `b5-move${j}`, qs[j], { value: kmoves[j] }, { what: `builder Kleavor move ${j + 1}`, rankWarnOnly: j === 3 });
    await cbPick(P, 'b5-item', koItem('Life Orb'), { value: 'Life Orb' }, { what: 'builder item (Korean)' });
    // field moves for the second battle: Rotom-Wash Will-O-Wisp -> Reflect, Gengar Protect -> Trick Room
    await cbPick(P, 'b1-move2', 'reflect', { value: 'Reflect' }, { what: 'builder Rotom-Wash move 3', rankWarnOnly: true });
    await cbPick(P, 'b3-move3', koMove('Trick Room'), { value: 'Trick Room' }, { what: 'builder Gengar move 4 (Korean)', rankWarnOnly: true });
    await setNumber(P, '#b5-sp-hp', 2);
    await setNumber(P, '#b5-sp-atk', 32);
    await setNumber(P, '#b5-sp-spe', 32);
    await setNumber(P, '#b5-sp-def', 0);
    await cbPick(P, 'b5-nature', 'jolly', { value: 'Jolly' }, { what: 'builder nature (English)' });
    const uiStats = [];
    for (let i = 0; i < 6; i++) uiStats.push((await page.locator(`#bstats-${i}`).innerText()).replace('실능 ', '').trim());
    await shot(P, 'builder_full', { full: P.phone ? false : true });
    await audit(P, 'team tab (builder)');
    const r2 = page.waitForResponse((r) => r.url().endsWith('/api/team/parse'));
    await click(P, '#builder-use', '검증 후 이 팀 사용');
    const resp = await r2;
    const pj = await resp.json();
    if (!check(V, pj.problems && !pj.problems.length, 'the builder team is legal', pj.problems)) {
      await click(P, page.locator('#d-builder .parse-res').getByRole('button', { name: '경고 무시하고 사용' }));
      await click(P, page.locator('#d-builder .parse-res').getByRole('button', { name: /다시 누르면/ }));
    }
    await page.locator('#tab-preview[aria-selected="true"]').waitFor({ timeout: 8000 }).catch(() => {});
    st = await appState(P);
    check(V, st.team && st.team.label === '직접 만든 팀' && st.team.sets[5].species === 'Kleavor' && st.team.sets[0].item === 'Choice Scarf'
      && st.team.sets[1].moves.includes('Reflect') && st.team.sets[3].moves.includes('Trick Room'),
      'the builder team becomes my team', st.team && st.team.sets.map((s) => `${s.species}@${s.item} ${s.moves.join('/')}`));
    const server = st.team.sets.map((s) => statLine(s.stats));
    check(V, deepEq(uiStats, server), 'stats computed in the page match the server\'s stats', { ui: uiStats, server });
    ctx.team = st.team.sets;
  });

  // 4) team preview ------------------------------------------------------------------------------------------
  await step('preview', async () => {
    if ((await page.locator('#tab-preview').getAttribute('aria-selected')) !== 'true') await click(P, '#tab-preview');
    await page.locator('#foe-pv-0').waitFor();
    check(V, await page.locator('#preview-run').isDisabled(), '"AI 선출 추천" is disabled until 6 opponents are entered');
    const queries = [koSp('Dragonite'), 'gholdengo', initials(koSp('Hippowdon')), 'Primarina', koSp('Tyranitar'), 'corvi'];
    for (let i = 0; i < 6; i++) {
      await cbPick(P, `foe-pv-${i}`, queries[i], { value: FOE[i] }, { what: `opponent preview ${i + 1}`, rankWarnOnly: i === 2 });
    }
    await blur(P);
    let st = await appState(P);
    check(V, deepEq(st.foeTeam, FOE), 'the 6 opponent species are stored', st.foeTeam);
    const sub = await page.locator('#foe-entry .card-sub').innerText();
    check(V, sub.includes('6 / 6'), 'opponent entry counter shows 6 / 6', sub);
    // manual pick
    if (!(await page.locator('#d-manual-pick').evaluate((e) => e.open))) await click(P, '#d-manual-pick > summary');
    await click(P, '#mp-0'); await click(P, '#mp-1'); await click(P, '#mp-3');
    await click(P, '#mp-4');
    await toastSeen(P, /3마리까지만/, 'a 4th manual pick is refused');
    await click(P, '#mp-lead-1');
    st = await appState(P);
    check(V, deepEq(st.manual.picked, ['Garchomp', 'Rotom-Wash', 'Gengar']) && st.manual.lead === 'Rotom-Wash', 'manual pick of 3 + lead works', st.manual);
    check(V, !(await page.locator('#mp-start').isDisabled()), 'manual "이 선출로 시작" is enabled with 3 picks');
    await shot(P, 'preview_entry');
    // AI preview
    const run = page.locator('#preview-run');
    check(V, !(await run.isDisabled()), '"AI 선출 추천" is enabled with 6 opponents');
    const respP = page.waitForResponse((r) => r.url().endsWith('/api/preview'), { timeout: T_AI });
    const t0 = Date.now();
    await click(P, run, 'AI 선출 추천');
    await page.waitForTimeout(300);
    check(V, await run.isDisabled(), '"AI 선출 추천" is disabled while it runs');
    const sp = await page.locator('#preview-ai .running').innerText().catch(() => '');
    check(V, sp.includes('시뮬레이션') && /초/.test(sp), 'a progress line with elapsed time is shown while the preview runs', sp);
    await page.waitForTimeout(1500);
    await shot(P, 'preview_running', { sel: '#preview-ai' });
    const resp = await respP;
    R.timings[`${V} preview`] = ((Date.now() - t0) / 1000).toFixed(1);
    const pj = await resp.json();
    check(V, resp.status() === 200 && !pj.error, '/api/preview succeeds', pj.error);
    const req = resp.request().postDataJSON();
    check(V, deepEq(req.foe, FOE) && req.my_team.length === 6 && req.my_team.every((s) => SP.has(s.species)),
      '/api/preview request uses the 6 opponents and my 6 sets (English names)', JSON.stringify(req).slice(0, 300));
    await page.locator('#preview-ai .pv-results li.opt').first().waitFor({ timeout: 10000 });
    const opts = page.locator('#preview-ai .pv-results li.opt');
    check(V, (await opts.count()) === pj.options.length, 'all preview options are listed', `${await opts.count()} vs ${pj.options.length}`);
    const o0 = pj.options[0];
    const t1 = (await opts.first().innerText()).replace(/\s+/g, ' ');
    check(V, t1.includes('선봉') && o0.names.every((n) => t1.includes(mainSp(n))) && t1.includes(pctTxt(o0.win_rate)),
      'best option shows lead tag, the 3 Korean names and the win %', t1);
    if (o0.policy !== null && o0.policy !== undefined) check(V, t1.includes('정책망'), 'best option shows the policy %', t1);
    const leadTxt = await opts.first().locator('.pick-lead .d-main').innerText().catch(() => '');
    check(V, leadTxt === mainSp(o0.lead), 'the lead is the first, marked name', leadTxt);
    const rates = pj.options.map((o) => o.win_rate);
    check(V, rates.every((r, i) => i === 0 || r <= rates[i - 1]), 'preview options are sorted best first', rates);
    check(V, (await opts.first().getAttribute('class')).includes('opt-best'), 'best option is highlighted');
    await shot(P, 'preview_results', { sel: '#preview-ai' });
    await audit(P, 'preview tab (results)');
    await click(P, '#pv-start-0', '이 선출로 시작');
    await toastSeen(P, /배틀을 시작/, 'battle started');
    st = await appState(P);
    const b = st.battle;
    const expBrought = [o0.lead, ...o0.names.filter((n) => n !== o0.lead)];
    check(V, b && deepEq(b.me.brought, expBrought) && b.me.active === o0.lead && b.turn === 1 && deepEq(b.foe.team, FOE) && st.tab === 'battle',
      '"이 선출로 시작" sets brought (lead first), my active = lead, turn 1, foe team, and opens 배틀',
      b && { brought: b.me.brought, active: b.me.active, turn: b.turn, tab: st.tab });
    ctx.brought = b ? b.me.brought : expBrought;
  });

  // 5) battle -------------------------------------------------------------------------------------------------
  const setOf = (n) => (ctx.team || []).find((s) => s.species === n);
  const maxHp = (n) => (setOf(n) ? setOf(n).stats.hp : 100);
  let adviseCount = 0;
  page.on('request', (r) => { if (r.url().endsWith('/api/advise')) adviseCount += 1; });
  const dismissToasts = async () => {
    for (let i = 0; i < 4 && (await page.locator('#toasts .toast-x').count()); i++) await page.locator('#toasts .toast-x').first().click();
  };
  /** Hold the next /api/advise request until the returned function is called (the page stays "calculating"). */
  const holdAdvise = async () => {
    let release;
    const gate = new Promise((r) => { release = r; });
    await page.route('**/api/advise', async (route) => { await gate; await route.continue().catch(() => {}); }, { times: 1 });
    return release;
  };
  /** Right after "다음 턴" with auto-advise: the page scrolls up to the matchup while the AI calculates. */
  const runInView = async (what) => {
    await page.waitForTimeout(1200); // a smooth scroll takes a few hundred ms
    const r = await page.evaluate(() => {
      const m = document.getElementById('matchup').getBoundingClientRect();
      const sp = document.querySelector('#results-card .running');
      const s = sp ? sp.getBoundingClientRect() : null;
      const bar = document.querySelector('.action-bar');
      const barTop = bar && bar.offsetHeight ? bar.getBoundingClientRect().top : window.innerHeight;
      return { matchupTop: Math.round(m.top), progress: s && [Math.round(s.top), Math.round(s.bottom)], barTop: Math.round(barTop),
        scrollY: Math.round(window.scrollY), running: !!document.querySelector('#advise-run[disabled]') };
    });
    check(V, r.running && r.progress && r.matchupTop >= 0 && r.matchupTop <= 200 && r.progress[0] >= 0 && r.progress[1] <= r.barTop,
      `${what}: while the AI calculates, the page has scrolled up to the matchup and the progress line is on screen`, r);
  };

  await step('battle-setup', async () => {
    if (!ctx.brought) throw new Error('no battle started');
    if ((await page.locator('#tab-battle').getAttribute('aria-selected')) !== 'true') await click(P, '#tab-battle');
    const brought = ctx.brought;
    const lead = brought[0];
    check(V, await page.locator('#foe-lead').isVisible(), '"상대 선봉은?" card appears at the start of the battle');
    await shot(P, 'battle_start');
    await click(P, '#foe-lead-0', 'opponent lead Dragonite');
    let st = await appState(P);
    check(V, st.battle.foe.active === 'Dragonite' && deepEq(st.battle.foe.seen, ['Dragonite']), 'tapping the opponent lead makes it active and seen', st.battle.foe);
    const mt = (await page.locator('#matchup').innerText()).replace(/\s+/g, ' ');
    check(V, mt.includes(mainSp('Dragonite')) && mt.includes(mainSp(lead)) && mt.includes(`${maxHp(lead)}/${maxHp(lead)}`), 'the matchup card shows both actives in Korean and my HP x/max', mt);
    // turn +/-
    const turnTxt = async () => (await page.locator('#matchup .turn strong').innerText()).trim();
    await click(P, '#turn-inc'); await click(P, '#turn-inc');
    check(V, (await turnTxt()) === '3', 'turn + button increments', await turnTxt());
    await click(P, '#turn-dec'); await click(P, '#turn-dec'); await click(P, '#turn-dec');
    check(V, (await turnTxt()) === '1', 'turn − button stops at 1', await turnTxt());
    await click(P, '#turn-inc');
    check(V, (await turnTxt()) === '2', 'turn is 2', await turnTxt());
    // a per-turn entry made for one Pokemon is dropped when the active Pokemon changes
    await page.locator('#q-my-move').scrollIntoViewIfNeeded();
    const leadMove = setOf(lead).moves[0];
    await select(P, '#q-my-move', leadMove);
    // conditional controls for each of my Pokemon when active
    for (let k = 0; k < brought.length; k++) {
      const n = brought[k];
      const set = setOf(n);
      await click(P, `label[for="me${k}-active"]`, `make ${n} active`);
      await page.waitForTimeout(150);
      const has = async (sel) => (await page.locator(sel).count()) > 0;
      const it = ITEMS.get(set.item);
      const stone = !!(it && it.megaStone && it.megaFor.includes(n));
      const choice = /^Choice /.test(set.item);
      check(V, (await has(`#me${k}-locked`)) === choice, `${n} (${set.item || 'no item'}) active: choice-lock select only with a Choice item`);
      check(V, (await has(`#me${k}-mega`)) === stone, `${n} active: "메가진화함" only when holding its mega stone`);
      check(V, (await has(`#me${k}-itemlost`)) === (!!set.item && !stone),
        `${n} (${set.item || 'no item'}) active: "도구 사용/잃음" only for an item that is not a Mega Stone`);
      check(V, (await page.locator(`[id^="me${k}-boost-"]`).count()) === 7, `${n} active: 7 boost selects`);
      check(V, (await page.locator('[id^="me"][id$="-boost-atk"]').count()) === 1, 'only my active Pokemon has boost selects');
      if (stone) ctx.megaChecked = true;
    }
    await click(P, 'label[for="me0-active"]', 'lead back to active');
    st = await appState(P);
    check(V, st.battle.me.active === lead, 'radio "출전" switches my active Pokemon', st.battle.me.active);
    check(V, st.battle.quick.myMove === '' && (await page.locator('#q-my-move').inputValue()) === '',
      'changing my active Pokemon clears "내가 쓴 기술" of the per-turn panel', { stored: st.battle.quick.myMove });
    // my HP / status / boosts / substitute
    const mx0 = maxHp(lead);
    await setNumber(P, '#me0-hp', mx0 - 37);
    const r0 = await page.locator('#me0-hpr').inputValue();
    check(V, r0 === String(Math.round(((mx0 - 37) / mx0) * 100)), 'typing my HP moves the % slider', `${r0} for ${mx0 - 37}/${mx0}`);
    const hpLabel = (await page.locator('#me0-hp').locator('xpath=..').innerText()).replace(/\s+/g, ' ');
    check(V, hpLabel.includes(`/ ${mx0}`), 'my HP input shows "/maxHP"', hpLabel);
    await page.locator('#me1-hpr').fill('50');
    await page.waitForTimeout(150);
    const mx1 = maxHp(brought[1]);
    check(V, (await page.locator('#me1-hp').inputValue()) === String(Math.round(0.5 * mx1)), 'the % slider sets the exact HP', await page.locator('#me1-hp').inputValue());
    await select(P, '#me1-status', 'brn');
    await select(P, '#me2-status', 'slp');
    check(V, (await page.locator('#me2-sleep').count()) === 1, 'sleep turn counter appears for a sleeping Pokemon');
    await select(P, '#me2-sleep', '1');
    await select(P, '#me2-status', 'tox');
    check(V, (await page.locator('#me2-toxic').count()) === 1 && (await page.locator('#me2-sleep').count()) === 0, 'toxic counter replaces the sleep counter for 맹독');
    await select(P, '#me2-status', 'slp');
    await select(P, '#me2-sleep', '1');
    await select(P, '#me0-boost-atk', '1');
    await select(P, '#me0-boost-def', '-1');
    check(V, (await page.locator('#me0-boost-atk').locator('xpath=..').getAttribute('class')).includes('changed'), 'a changed boost is highlighted');
    const chip = page.locator('#me0-vol-substitute');
    await chip.evaluate((e) => e.scrollIntoView({ block: 'center' })); // not under the sticky bar: the click must not need a scroll
    const chipY0 = await chip.evaluate((e) => e.getBoundingClientRect().top);
    await click(P, '#me0-vol-substitute', 'substitute chip');
    await page.waitForTimeout(150);
    const chipY1 = await chip.evaluate((e) => e.getBoundingClientRect().top);
    check(V, Math.abs(chipY1 - chipY0) < 3, 'toggling a chip re-renders the page without moving it (the chip stays under the pointer)', { before: chipY0, after: chipY1 });
    check(V, (await page.locator('#me0-vol-substitute').getAttribute('aria-pressed')) === 'true' && (await page.locator('#me0-subhp').count()) === 1,
      'volatile chip toggles and the substitute HP select appears');
    const lset = setOf(lead);
    const prio0 = lset.moves.filter((m) => DATA.moves[m] && (DATA.moves[m].priority || 0) === 0);
    ctx.myMove = prio0.find((m) => DATA.moves[m].category !== 'Status') || prio0[0] || lset.moves[0];
    if (/^Choice /.test(lset.item)) {
      await select(P, '#me0-locked', ctx.myMove);
      ctx.locked = ctx.myMove;
    }
    const vs = (await page.locator('#matchup').innerText()).replace(/\s+/g, ' ');
    check(V, vs.includes(`${mx0 - 37}/${mx0}`) && vs.includes('공격+1') && vs.includes('방어-1'), 'matchup card reflects my HP and boosts', vs);
    await shot(P, 'battle_my_side', { sel: '#my-side' });

    // opponent: Dragonite active
    await setNumber(P, '#foe0-hp', 70);
    check(V, (await page.locator('#foe0-hpr').inputValue()) === '70', 'opponent HP number syncs the slider');
    await page.locator('#foe0-hpr').fill('64');
    await page.waitForTimeout(100);
    check(V, (await page.locator('#foe0-hp').inputValue()) === '64', 'opponent HP slider syncs the number');
    await cbPick(P, 'foe0-move0', koMove('Extreme Speed'), { value: 'Extreme Speed' }, { what: 'opponent revealed move' });
    await click(P, '#foe0-move1');
    const allMv = await cbItems(P, 'foe0-move1');
    check(V, allMv.length === Math.min(150, SP.get('Dragonite').moves.length) && allMv.every((x) => SP.get('Dragonite').moves.includes(x.sub || x.main)),
      'opponent move dropdown lists exactly the species\' legal moves', `${allMv.length} options vs ${SP.get('Dragonite').moves.length} legal`);
    await page.locator('#foe0-move1').fill('shadow ball');
    check(V, (await page.locator('#foe0-move1-list li.cb-opt').count()) === 0, 'opponent move dropdown rejects moves outside its learnset');
    await page.locator('#foe0-move1').press('Escape');
    await blur(P);
    await click(P, '#foe0-item');
    const its = await cbItems(P, 'foe0-item');
    check(V, its[0] && its[0].main === '모름' && its[1] && /없음/.test(its[1].main), 'opponent item dropdown starts with "모름" and "없음/소모"', its.slice(0, 3));
    await page.locator('#foe0-item').press('Escape');
    await cbPick(P, 'foe0-item', koItem('Leftovers'), { value: 'Leftovers' }, { what: 'opponent item' });
    await click(P, '#foe0-ability');
    const abs = await cbItems(P, 'foe0-ability');
    check(V, abs.length === 1 + SP.get('Dragonite').abilities.length && abs[0].main === '모름', 'opponent ability dropdown = "모름" + the species\' abilities', abs);
    await page.locator('#foe0-ability').press('Escape');
    await blur(P);
    check(V, (await page.locator('#foe0-mega').count()) === 1, 'opponent "메가진화함" checkbox shown for a species with a mega (Dragonite)');
    await select(P, '#foe0-boost-atk', '1');
    await click(P, '#foe0-vol-confusion', 'opponent confusion chip');
    await click(P, '#foe0-faster-1', 'speed observation chip');
    check(V, (await page.locator('#foe0-faster-1').getAttribute('aria-pressed')) === 'true', 'speed observation "내 ○○보다 빠름" toggles on',
      await page.locator('#foe0-faster-1').innerText());
    check(V, (await page.locator('#foe0-faster-1').innerText()).includes(mainSp(brought[1])), 'speed chip names my Pokemon in Korean', await page.locator('#foe0-faster-1').innerText());
    // mark Tyranitar (item none) and Gholdengo (unknown) as seen; a 4th is refused
    await click(P, '#foe-roster-4', 'roster Tyranitar');
    await click(P, '#foe-roster-1', 'roster Gholdengo');
    await click(P, '#foe-roster-2', 'roster Hippowdon');
    await toastSeen(P, /3마리만/, 'a 4th seen opponent is refused');
    st = await appState(P);
    check(V, deepEq(st.battle.foe.seen, ['Dragonite', 'Tyranitar', 'Gholdengo']), 'roster chips mark opponents as seen', st.battle.foe.seen);
    await click(P, '#d-foe-Tyranitar > summary', 'open Tyranitar');
    await cbPick(P, 'foe4-item', 'none', { label: '없음 / 소모됨' }, { what: 'opponent item none' });
    check(V, (await page.locator('#foe4-boost-atk').count()) === 0, 'non-active opponent has no boost selects');
    check(V, (await page.locator('#foe4-mega').count()) === 1 && (await page.locator('#foe1-mega').count()) === 0,
      'mega checkbox only for opponents that have a mega (Tyranitar yes, Gholdengo no)');
    await click(P, 'label[for="foe4-mega"]', 'Tyranitar mega evolved');
    st = await appState(P);
    check(V, st.battle.foe.mons.Tyranitar.mega === true && st.battle.foe.megaUsed === true, 'opponent "메가진화함" marks the mega and the side\'s mega as used',
      { mega: st.battle.foe.mons.Tyranitar.mega, used: st.battle.foe.megaUsed });
    check(V, (await page.locator('#foe0-mega').count()) === 0, 'once one opponent has Mega Evolved, the others lose the mega checkbox');
    check(V, (await page.locator('#sc-foe-megaused').isChecked()), 'field card "메가진화 사용함" (opponent) follows the mega checkbox');
    await shot(P, 'battle_foe_side', { sel: '#foe-side' });
    await audit(P, 'battle tab (sides)');
  });

  await step('turn-entry', async () => {
    const brought = ctx.brought;
    const lead = brought[0];
    const mx0 = maxHp(lead);
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    const qOpts = async () => { await click(P, '#q-foe-move'); const x = await cbItems(P, 'q-foe-move'); await page.locator('#q-foe-move').press('Escape'); await blur(P); return x; };
    const qm = await qOpts();
    check(V, qm[0] && qm[0].sub === 'Extreme Speed' && /공개/.test(qm[0].hint), 'per-turn move dropdown lists the revealed move first', qm.slice(0, 2));
    check(V, qm.length === Math.min(150, SP.get('Dragonite').moves.length), 'per-turn move dropdown = the active opponent\'s legal moves', qm.length);
    await cbPick(P, 'q-foe-move', koMove('Dragon Dance'), { value: 'Dragon Dance' }, { what: 'per-turn opponent move' });
    await select(P, '#q-my-move', ctx.myMove);
    await click(P, '#q-order-0', '내가 먼저');
    check(V, (await page.locator('#q-order-0').getAttribute('aria-checked')) === 'true', '"내가 먼저" is selected');
    await cbPick(P, 'q-foe-ability', koAb('Multiscale'), { value: 'Multiscale' }, { what: 'per-turn ability reveal' });
    await setNumber(P, '#q-my-hp', mx0 - 60);
    await setNumber(P, '#q-foe-hp', 55);
    check(V, (await page.locator('#q-foe-hp-r').inputValue()) === '55', 'per-turn opponent HP number syncs its slider');
    const autoOn = await page.locator('#q-auto').isChecked();
    check(V, autoOn, '"기록 후 바로 AI 계산" is on by default');
    if (!P.phone) await click(P, 'label[for="q-auto"]', 'turn off auto advise');
    await shot(P, 'battle_turn_entry', { sel: '#quick-card' });
    const before = adviseCount;
    const release = P.phone ? await holdAdvise() : null;
    const autoResp = P.phone ? page.waitForResponse((r) => r.url().endsWith('/api/advise'), { timeout: T_AI }) : null;
    await click(P, '#next-turn', '다음 턴');
    if (release) {
      await runInView('"다음 턴" with auto-advise');
      release();
    }
    await toastSeen(P, /턴 2 기록 완료/, '"다음 턴" confirms the recorded turn');
    const st = await appState(P);
    const b = st.battle;
    const fm = b.foe.mons.Dragonite;
    check(V, b.turn === 3, '"다음 턴" increments the turn', b.turn);
    check(V, deepEq(fm.moves.filter(Boolean), ['Extreme Speed', 'Dragon Dance']), 'the move used this turn is added to the revealed moves', fm.moves);
    check(V, fm.ability === 'Multiscale', 'the revealed ability is stored', fm.ability);
    check(V, fm.hp === 55 && b.me.mons[lead].hp === mx0 - 60, 'HP after the turn is applied (mine exact, opponent %)', { foe: fm.hp, me: b.me.mons[lead].hp });
    check(V, (fm.slower || []).includes(lead) && (fm.faster || []).includes(brought[1]), '"내가 먼저" adds my active to the opponent\'s slower_than', { faster: fm.faster, slower: fm.slower });
    check(V, b.quick.foeMove === '' && b.quick.order === '', 'the per-turn panel is cleared for the next turn', b.quick);
    if (ctx.locked) check(V, b.me.mons[lead].locked === ctx.locked, 'choice lock is kept', b.me.mons[lead].locked);
    if (P.phone) {
      const r = await autoResp;
      check(V, r.status() === 200, '"다음 턴" with auto-advise runs the AI calculation by itself', r.status());
      await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI });
    } else {
      await page.waitForTimeout(600);
      check(V, adviseCount === before, 'with auto-advise off, "다음 턴" does not start a calculation', adviseCount - before);
    }
  });

  await step('field', async () => {
    await page.locator('#field-card').scrollIntoViewIfNeeded();
    await select(P, '#f-weather', 'sandstorm');
    check(V, (await page.locator('#f-weather-turns').count()) === 1, 'weather turns select appears once a weather is chosen');
    await select(P, '#f-weather-turns', '3');
    await click(P, 'label[for="sc-foe-stealthrock"]', 'opponent stealth rock');
    await select(P, '#sc-me-reflect', '5');
    await select(P, '#sc-foe-spikes', '2');
    await select(P, '#sc-foe-spikes', '0');
    check(V, (await page.locator('#sc-me-megaused').count()) === 1 && (await page.locator('#sc-foe-megaused').count()) === 1, '"메가진화 사용함" flag for both sides');
    for (const id of ['f-terrain', 'f-trickroom', 'f-gravity', 'f-magicroom', 'f-wonderroom', 'sc-me-lightscreen', 'sc-me-auroraveil', 'sc-me-tailwind', 'sc-me-safeguard', 'sc-me-stickyweb', 'sc-me-spikes', 'sc-me-toxicspikes']) {
      if (!(await page.locator(`#${id}`).count())) fail(V, `field control #${id} exists`);
    }
    const st = await appState(P);
    check(V, st.battle.field.weather === 'sandstorm' && st.battle.field.weather_turns === 3 && st.battle.foe.side.stealthrock === 1 && st.battle.me.side.reflect === 5,
      'field selections are stored', { field: st.battle.field, me: st.battle.me.side, foe: st.battle.foe.side });
    await shot(P, 'battle_field', { sel: '#field-card' });
  });

  await step('advise', async () => {
    const brought = ctx.brought;
    const lead = brought[0];
    await click(P, '#samples-0', '빠름 6');
    check(V, (await page.locator('#samples-0').getAttribute('aria-checked')) === 'true', 'speed choice "빠름 6" is selected');
    await blur(P);
    await page.locator('.action-bar').waitFor({ state: 'visible' });
    const segTxt = (await page.locator('#samples').innerText()).replace(/\s+/g, ' ');
    check(V, /빠름/.test(segTxt) && /보통/.test(segTxt) && /정밀/.test(segTxt), 'the bar offers the 빠름 / 보통 / 정밀 presets', segTxt);
    if (!P.phone) check(V, /2~5초/.test(segTxt) && /10~30초/.test(segTxt), 'the presets show their expected time on a wide screen', segTxt);
    const bar = await page.locator('.action-bar').boundingBox();
    check(V, bar && Math.abs(bar.y + bar.height - vp.ctx.viewport.height) < 3, 'the "최선의 행동 계산" bar sticks to the bottom of the screen', bar);
    const respP = page.waitForResponse((r) => r.url().endsWith('/api/advise') && r.request().method() === 'POST', { timeout: T_AI });
    const t0 = Date.now();
    await click(P, '#advise-run', '최선의 행동 계산');
    await page.waitForTimeout(300);
    check(V, await page.locator('#advise-run').isDisabled(), '"최선의 행동 계산" is disabled while running');
    check(V, await page.locator('#next-turn').isDisabled(), '"다음 턴" is disabled while the AI runs');
    const prog = await page.locator('#results-card .running').innerText().catch(() => '');
    check(V, /계산 중/.test(prog) && /초/.test(prog), 'progress with elapsed time is shown', prog);
    check(V, /샘플 6개/.test(prog) && /1턴 앞/.test(prog) && /2~5초/.test(prog) && !/1분/.test(prog),
      'the progress line names the preset (샘플 6 · 1턴 앞) and a realistic time (2~5초)', prog);
    await page.waitForTimeout(1200);
    await shot(P, 'advise_running');
    const resp = await respP;
    R.timings[`${V} advise`] = ((Date.now() - t0) / 1000).toFixed(1);
    const body = resp.request().postDataJSON();
    R.bodies[`${V} advise`] = body;
    const res = await resp.json();
    check(V, resp.status() === 200 && !res.error && Array.isArray(res.recommendations) && res.recommendations.length > 0,
      'the server accepts the page\'s <state> (/api/advise 200 with recommendations)', res.error || resp.status());
    check(V, body.samples === 6 && body.depth === 1, '/api/advise body of 빠름 = samples 6, depth 1', { samples: body.samples, depth: body.depth });
    const mx = Object.fromEntries(brought.map((n) => [n, maxHp(n)]));
    const exp = {
      turn: 3, team: ctx.team, brought, myActive: lead, maxHp: mx,
      mePokemon: {
        [lead]: { hp: `${mx[lead] - 60}/${mx[lead]}`, status: '', boosts: { atk: 1, def: -1 }, volatiles: ['substitute'], substitute_hp: 0.25,
          locked_move: ctx.locked || ABSENT, fainted: ABSENT },
        [brought[1]]: { hp: `${Math.round(0.5 * mx[brought[1]])}/${mx[brought[1]]}`, status: 'brn', boosts: ABSENT },
        [brought[2]]: { hp: `${mx[brought[2]]}/${mx[brought[2]]}`, status: 'slp', sleep_turns: 1 },
      },
      meSide: { reflect: 5 }, foeSide: { stealthrock: 1 },
      foeSeen: ['Dragonite', 'Tyranitar', 'Gholdengo'], foeActive: 'Dragonite',
      foePokemon: {
        Dragonite: { hp: '55%', status: '', item: 'Leftovers', ability: 'Multiscale', moves: ['Extreme Speed', 'Dragon Dance'],
          faster_than: [brought[1]], slower_than: [lead], boosts: { atk: 1 }, volatiles: ['confusion'] },
        Tyranitar: { hp: '100%', item: '', ability: ABSENT, moves: ABSENT, fainted: ABSENT, mega: true },
        Gholdengo: { hp: '100%', item: ABSENT, ability: ABSENT },
      },
      field: { weather: 'sandstorm', weather_turns: 3, terrain: '', terrain_turns: 0, trickroom: 0, gravity: 0, magicroom: 0, wonderroom: 0 },
      meMegaUsed: false, foeMegaUsed: true,
    };
    const errs = validateState(body.state, exp);
    check(V, !errs.length, '<state> sent to /api/advise matches API.md and what was selected', errs.join('\n'));
    // results table
    const rows = page.locator('#results-card table.recs tbody tr');
    await rows.first().waitFor({ timeout: 10000 });
    const recs = res.recommendations;
    check(V, (await rows.count()) === recs.length, 'one result row per recommended action', `${await rows.count()} rows / ${recs.length}`);
    check(V, (await rows.first().getAttribute('class')).includes('best') && (await rows.first().innerText()).includes('추천'), 'the #1 action is highlighted with "추천"');
    const rowProbs = [];
    for (let i = 0; i < Math.min(recs.length, await rows.count()); i++) {
      const r = recs[i];
      const row = rows.nth(i);
      const main = (await row.locator('.act-main').innerText()).trim();
      const en = (await row.locator('.act-en').innerText()).trim();
      const txt = (await row.innerText()).replace(/\s+/g, ' ');
      if (main !== r.label_ko) rowProbs.push(`row ${i + 1}: Korean label "${main}" != label_ko "${r.label_ko}"`);
      if (en !== r.label) rowProbs.push(`row ${i + 1}: English label "${en}" != "${r.label}"`);
      if (!txt.includes(pctTxt(r.win_rate, 1))) rowProbs.push(`row ${i + 1}: win % ${pctTxt(r.win_rate, 1)} missing in "${txt}"`);
      if (!txt.includes(`±${((r.stderr || 0) * 100).toFixed(1)}`)) rowProbs.push(`row ${i + 1}: ± stderr missing in "${txt}"`);
      if ((await row.locator('.winbar-fill').count()) !== 1) rowProbs.push(`row ${i + 1}: no win bar`);
      if (r.meaning && r.meaning[0] && r.meaning[0][0] === 'move' && !hasHangul(main)) rowProbs.push(`row ${i + 1}: move label not Korean "${main}"`);
    }
    check(V, !rowProbs.length, 'result rows show Korean action + English label, win % ± stderr, win bar, policy', rowProbs.join('\n'));
    const rates = recs.map((r) => r.win_rate);
    check(V, rates.every((x, i) => i === 0 || x <= rates[i - 1]), 'actions are sorted by win rate', rates);
    if (res.value !== null && res.value !== undefined) {
      const vr = (await page.locator('#results-card .value-row').innerText().catch(() => '')).replace(/\s+/g, ' ');
      check(V, vr.includes(pctTxt(res.value, 1)) && /신경망 즉석 추정/.test(vr) && /시뮬레이션 승률이 더 정확/.test(vr),
        'the value-network estimate is shown and labelled as a rough instant estimate', vr);
      const lay = await page.evaluate(() => {
        const t = document.querySelector('#results-card table.recs');
        const v = document.querySelector('#results-card .value-row');
        const best = document.querySelector('#results-card tr.best .win strong');
        return { after: !!(t && v && (t.compareDocumentPosition(v) & Node.DOCUMENT_POSITION_FOLLOWING)),
          vSize: v ? parseFloat(getComputedStyle(v.querySelector('.value-num')).fontSize) : 0,
          bestSize: best ? parseFloat(getComputedStyle(best).fontSize) : 0 };
      });
      check(V, lay.after && lay.vSize < lay.bestSize, 'the instant estimate is secondary: below the action table, in smaller type', lay);
    }
    const nRep = await page.locator('#results-card .replies li').count();
    check(V, nRep === (res.foe_replies || []).length && nRep > 0, 'the opponent\'s likely actions are listed', `${nRep} vs ${(res.foe_replies || []).length}`);
    if (nRep) {
      const rep = (await page.locator('#results-card .replies li').first().innerText()).replace(/\s+/g, ' ');
      check(V, hasHangul(rep), 'opponent reply labels are Korean', rep);
    }
    const bel = page.locator('#results-card .belief');
    const nb = Object.keys(res.beliefs || {}).length;
    check(V, (await bel.count()) === nb && nb > 0, 'inferred opponent stats are shown', `${await bel.count()} vs ${nb}`);
    if (nb) {
      const bt = (await bel.first().innerText()).replace(/\s+/g, ' ');
      check(V, bt.includes('스피드') && bt.includes('구애스카프') && bt.includes('평균 실능'), 'beliefs show speed 10-90% range, Choice Scarf probability and mean stats', bt);
    }
    await toastSeen(P, /추천:/, 'best action toast');
    await shot(P, 'advise_results', { sel: '#results-card' });
    await shot(P, 'advise_results_screen');
    await audit(P, 'battle tab (results)');
    if (P.phone) {
      await page.emulateMedia({ colorScheme: 'dark' });
      await page.waitForTimeout(300);
      const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      const lum = (bg.match(/\d+/g) || [255, 255, 255]).slice(0, 3).map(Number).reduce((a, x) => a + x, 0) / 3;
      check(V, lum < 60, 'dark mode (prefers-color-scheme) gives a dark page', bg);
      await page.locator('#results-card').scrollIntoViewIfNeeded();
      await shot(P, 'dark_results');
      await page.emulateMedia({ colorScheme: 'light' });
    }
  });

  await step('turn-switch', async () => {
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await select(P, '#q-foe-switch', 'Gholdengo');
    const qt = (await page.locator('#quick-card').innerText()).replace(/\s+/g, ' ');
    check(V, qt.includes(`상대 ${mainSp('Gholdengo')} HP`), 'after picking an opponent switch the HP field targets the new Pokemon', qt.slice(0, 300));
    await setNumber(P, '#q-foe-hp', 80);
    // a switch-in reveal (e.g. Sand Stream / Intimidate) belongs to the incoming Pokemon
    await click(P, '#q-foe-ability');
    const abOpts = (await cbItems(P, 'q-foe-ability')).map((o) => o.sub || o.main);
    await page.locator('#q-foe-ability').press('Escape');
    await blur(P);
    const gAbs = SP.get('Gholdengo').abilities;
    check(V, gAbs.every((a) => abOpts.includes(a)) && abOpts.every((a) => a === '변경 없음' || gAbs.includes(a)),
      'with "상대 교체" selected, "상대 특성 공개" offers the incoming Pokemon\'s abilities', `incoming Gholdengo ${JSON.stringify(gAbs)}; offered ${JSON.stringify(abOpts)}`);
    check(V, qt.includes(`상대 ${mainSp('Gholdengo')} 도구 공개`) && qt.includes(`상대 ${mainSp('Gholdengo')} 특성 공개`),
      'the reveal fields name the Pokemon they apply to', qt.slice(0, 400));
    await cbPick(P, 'q-foe-item', 'knocked', { label: '없음 / 소모됨' }, { what: 'per-turn item reveal (knocked off)' });
    const brought = ctx.brought;
    await select(P, '#q-my-switch', brought[1]);
    const qt2 = (await page.locator('#quick-card').innerText()).replace(/\s+/g, ' ');
    check(V, qt2.includes(`내 ${mainSp(brought[1])} HP`) && qt2.includes(`/ ${maxHp(brought[1])}`), 'after picking my switch the HP field targets my new Pokemon (with its max HP)', qt2.slice(0, 400));
    const autoResp = P.phone ? page.waitForResponse((r) => r.url().endsWith('/api/advise'), { timeout: T_AI }) : null;
    await click(P, '#next-turn', '다음 턴 (switch)');
    await toastSeen(P, /턴 3 기록 완료/, 'turn 3 recorded');
    const st = await appState(P);
    const b = st.battle;
    check(V, b.foe.active === 'Gholdengo' && b.foe.mons.Gholdengo.hp === 80 && deepEq(b.foe.mons.Dragonite.boosts, {}) && !b.foe.mons.Dragonite.volatiles.length,
      'opponent switch: new active, its HP, and the old one loses boosts/volatiles', { active: b.foe.active, g: b.foe.mons.Gholdengo, d: b.foe.mons.Dragonite });
    check(V, b.turn === 4, 'turn 4', b.turn);
    check(V, b.foe.mons.Gholdengo.item === '' && b.foe.mons.Dragonite.item === 'Leftovers',
      'on a switch turn the item reveal "없음/소모" is applied to the incoming opponent (the outgoing one keeps its item)',
      { gholdengo: b.foe.mons.Gholdengo.item, dragonite: b.foe.mons.Dragonite.item });
    const old = b.me.mons[brought[0]];
    check(V, b.me.active === brought[1] && b.me.mons[brought[1]].fresh === true && deepEq(old.boosts, {}) && !old.volatiles.length && !old.locked,
      'my switch: new active (fresh), the old one loses boosts / substitute / choice lock', { active: b.me.active, old });
    if (P.phone) { await autoResp; await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI }); }
    const stale = await page.locator('#results-card .msg-warn').innerText().catch(() => '');
    if (!P.phone) check(V, /다시 계산/.test(stale), 'old results are marked stale after the situation changes', stale);
  });

  await step('faint', async () => {
    const brought = ctx.brought;
    let st = await appState(P);
    const act = st.battle.me.active;
    const k = brought.indexOf(act);
    await click(P, `label[for="me${k}-fainted"]`, 'my active fainted');
    await page.locator('#faint-card').waitFor({ timeout: 5000 });
    check(V, await page.locator('#faint-card').isVisible(), 'marking my active fainted shows "누구를 내보낼까?" with 교체 추천');
    const respP = page.waitForResponse((r) => r.url().endsWith('/api/advise_switch'), { timeout: T_AI });
    const t0 = Date.now();
    await click(P, '#switch-advice', '교체 추천');
    await page.waitForTimeout(300);
    check(V, await page.locator('#switch-advice').isDisabled(), '"교체 추천" is disabled while running');
    const resp = await respP;
    R.timings[`${V} advise_switch`] = ((Date.now() - t0) / 1000).toFixed(1);
    const body = resp.request().postDataJSON();
    R.bodies[`${V} advise_switch`] = body;
    check(V, body.samples === 6 && body.depth === 1, '/api/advise_switch body = samples 6, depth 1', { samples: body.samples, depth: body.depth });
    const res = await resp.json();
    check(V, resp.status() === 200 && !res.error && res.options && res.options.length === 2, '/api/advise_switch succeeds with one option per remaining Pokemon', res.error || res.options);
    const s = body.state;
    const mon = s && s.me && s.me.pokemon && s.me.pokemon[act];
    const mx = maxHp(act);
    check(V, mon && mon.fainted === true && mon.hp === `0/${mx}` && !mon.boosts && !mon.volatiles,
      'switch request marks the fainted Pokemon (fainted: true, hp "0/max", no boosts)', mon);
    check(V, s && deepEq(s.foe.active, ['Gholdengo']) && s.foe.pokemon.Gholdengo.hp === '80%' && !s.foe.pokemon.Dragonite.boosts
      && s.foe.pokemon.Gholdengo.item === '' && s.foe.pokemon.Dragonite.item === 'Leftovers', 'switch request carries the opponent switch and the knocked-off item', s && s.foe);
    const errs = validateState(s, {
      turn: 4, team: ctx.team, brought, myActive: act, maxHp: Object.fromEntries(brought.map((n) => [n, maxHp(n)])),
      mePokemon: { [act]: { fainted: true, hp: `0/${mx}` } }, meSide: { reflect: 4 }, foeSide: { stealthrock: 1 },
      foeSeen: ['Dragonite', 'Tyranitar', 'Gholdengo'], foeActive: 'Gholdengo',
      field: { weather: 'sandstorm', weather_turns: 2 }, foeMegaUsed: true,
    });
    check(V, !errs.length, '<state> sent to /api/advise_switch matches API.md (field / screens ticked down one turn)', errs.join('\n'));
    await page.locator('#send-0').waitFor({ timeout: 10000 });
    const items = page.locator('#faint-card li.opt');
    check(V, (await items.count()) === res.options.length, 'each switch option is listed', await items.count());
    const t1 = (await items.first().innerText()).replace(/\s+/g, ' ');
    check(V, t1.includes(mainSp(res.options[0].switch_to)) && t1.includes(pctTxt(res.options[0].win_rate)), 'best switch shows Korean name and win %', t1);
    await shot(P, 'faint_switch', { sel: '#faint-card' });
    await audit(P, 'battle tab (faint)');
    const autoResp = P.phone ? page.waitForResponse((r) => r.url().endsWith('/api/advise'), { timeout: T_AI }) : null;
    await click(P, '#send-0', '이 포켓몬 내보내기');
    st = await appState(P);
    check(V, st.battle.me.active === res.options[0].switch_to && st.battle.me.mons[res.options[0].switch_to].fresh === true,
      'one click sends in the recommended Pokemon (active + fresh)', { active: st.battle.me.active });
    check(V, (await page.locator('#faint-card').count()) === 0, 'the faint card goes away after sending a Pokemon in');
    if (P.phone) { await autoResp; await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI }); }
  });

  await step('reload', async () => {
    const before = await appState(P);
    await page.reload();
    await page.locator('#main').waitFor({ state: 'visible', timeout: 30000 });
    await page.waitForTimeout(500);
    const after = await appState(P);
    check(V, (await page.locator('#tab-battle').getAttribute('aria-selected')) === 'true', 'after reload the battle tab is still open');
    check(V, after && deepEq(after.battle, before.battle) && deepEq(after.team, before.team) && deepEq(after.foeTeam, before.foeTeam),
      'after reload the battle state is unchanged');
    const turn = await page.locator('#matchup .turn strong').innerText().catch(() => '');
    check(V, turn === String(before.battle.turn), 'after reload the turn number is shown', turn);
    check(V, (await page.locator('#f-weather').inputValue()) === 'sandstorm', 'after reload the field is restored');
    const mv = await page.locator('[id="foe0-move1"]').inputValue().catch(() => '(not rendered)');
    check(V, mv === shown(koMove('Dragon Dance'), 'Dragon Dance'), 'after reload the opponent\'s revealed moves are shown', mv);
    check(V, (await page.locator('#results-card table.recs').count()) === 1, 'after reload the last AI result is still shown');
    await shot(P, 'after_reload');
    // a state saved before the presets had a depth ("samples": 24) opens with 정밀 selected (separate browser context)
    const old = JSON.parse(JSON.stringify(after));
    delete old.preset;
    old.samples = 24;
    const ctx2 = await browser.newContext(vp.ctx);
    try {
      await ctx2.addInitScript(([k, v]) => { if (!sessionStorage.getItem('seeded')) { localStorage.setItem(k, v); sessionStorage.setItem('seeded', '1'); } }, [STORE_KEY, JSON.stringify(old)]);
      const p2 = await ctx2.newPage();
      await p2.goto(`${BASE}/`);
      await p2.locator('#samples-2').waitFor({ timeout: 30000 });
      const checked = await p2.locator('#samples-2').getAttribute('aria-checked');
      await p2.waitForTimeout(400);
      const saved = await p2.evaluate((k) => JSON.parse(localStorage.getItem(k)), STORE_KEY);
      check(V, checked === 'true' && saved.preset === 'precise' && !('samples' in saved),
        'an old saved preference (samples 24) is kept as the 정밀 preset', { checked, preset: saved.preset, samples: saved.samples });
    } finally {
      await ctx2.close();
    }
  });

  await step('api-error', async () => {
    await page.route('**/api/advise', (route) => route.fulfill({
      status: 400, contentType: 'application/json; charset=utf-8', body: JSON.stringify({ error: 'E2E 테스트 오류: bad state' }),
    }), { times: 1 });
    P.expectBad += 1;
    P.expectConsole += 1; // Chromium logs "Failed to load resource: ... 400" for it
    await blur(P);
    await click(P, '#advise-run', '최선의 행동 계산 (error)');
    const eb = page.locator('#results-card .msg-err');
    await eb.waitFor({ timeout: 10000 }).catch(() => {});
    check(V, (await eb.count()) && (await eb.innerText()).includes('E2E 테스트 오류'), 'an API {"error"} is shown in the results card', await eb.innerText().catch(() => ''));
    await toastSeen(P, /E2E 테스트 오류/, 'API error toast');
    check(V, !(await page.locator('#advise-run').isDisabled()), 'the button is enabled again after an error');
    await shot(P, 'api_error', { sel: '#results-card' });
    await page.unroute('**/api/advise');
  });

  await step('help', async () => {
    await click(P, '#tab-help');
    await page.locator('#status-card dl').waitFor({ timeout: 10000 });
    const t = (await page.locator('#status-card').innerText()).replace(/\s+/g, ' ');
    const status = (await getJSON('/api/status')).json;
    check(V, t.includes(DATA.format) && (!status.model || t.includes(status.model)) && (!status.library || t.includes(status.library)),
      'help shows format, model and library from /api/status', t);
    check(V, status.korean_names ? t.includes('사용 가능') : true, 'help shows Korean names are available', t);
    const steps = await page.locator('.help-steps li').count();
    check(V, steps >= 4, 'help has a step-by-step Korean how-to', steps);
    const hp = (await page.locator('#help-presets').innerText().catch(() => '')).replace(/\s+/g, ' ');
    check(V, /빠름: 샘플 6개 · 1턴 앞까지 시뮬레이션 · 약 2~5초/.test(hp) && /보통: 샘플 12개 · 1턴 앞까지 시뮬레이션 · 약 4~10초/.test(hp)
      && /정밀: 샘플 24개 · 2턴 앞까지 시뮬레이션 · 약 10~30초/.test(hp) && /몇 턴 앞까지/.test(hp) && !/1분/.test(hp),
    'help explains the presets with samples, depth (몇 턴 앞까지) and realistic times', hp);
    check(V, /신경망 즉석 추정/.test(hp), 'help explains that the position win rate is a rough instant estimate', hp);
    const ht = (await page.locator('#panel-help').innerText()).replace(/\s+/g, ' ');
    check(V, /필드 자동 반영/.test(ht) && /트릭룸/.test(ht), 'help explains the automatic field effects', ht.slice(0, 200));
    check(V, ht.includes('안개제거·고속스핀·정리정돈은') && !ht.includes('정리정돈는'), 'help uses the right particle after 정리정돈 (은, not 는)',
      (ht.match(/.{0,20}정리정돈.{0,6}/) || [''])[0]);
    check(V, /1턴의 선봉/.test(ht) && /입력하는 즉시 반영/.test(ht), 'help says a fresh Pokemon\'s weather / terrain ability counts as soon as it is entered',
      (ht.match(/막 나온 포켓몬.{0,80}/) || [''])[0]);
    await shot(P, 'help');
    await audit(P, 'help tab');
  });

  await step('team-guard', async () => {
    await click(P, '#tab-team');
    const before = await appState(P);
    await page.locator('#use-team-1').waitFor();
    await click(P, '#use-team-1', '이 팀으로 배틀 (mid-battle)');
    const txt = (await page.locator('#use-team-1').innerText()).trim();
    const st = await appState(P);
    check(V, /다시 누르면/.test(txt) && st.battle && st.battle.turn === before.battle.turn && st.tab === 'team',
      'mid-battle, one tap on another team\'s "이 팀으로 배틀" only asks for confirmation and keeps the battle', { txt, battle: !!st.battle, tab: st.tab });
    await click(P, '#tab-battle');
  });

  await step('new-battle', async () => {
    await click(P, '#tab-battle');
    const before = await appState(P);
    await click(P, '#new-battle', '새 배틀');
    const armed = (await page.locator('#new-battle').innerText()).trim();
    check(V, /다시/.test(armed), '"새 배틀" asks for a second tap while a battle is in progress', armed);
    await click(P, '#new-battle', '새 배틀 (confirm)');
    await toastSeen(P, /새 배틀을 준비/, 'new battle prepared');
    const undo = page.locator('#toasts .toast-act', { hasText: '되돌리기' });
    check(V, await undo.isVisible().catch(() => false), 'the reset message offers "되돌리기" (undo)');
    if (await undo.count()) {
      await click(P, undo, '되돌리기');
      const back = await appState(P);
      check(V, back.battle && deepEq(back.battle, before.battle) && deepEq(back.foeTeam, before.foeTeam) && back.tab === 'battle',
        '"되돌리기" restores the battle that was just reset', { turn: back.battle && back.battle.turn, tab: back.tab });
      await click(P, '#new-battle', '새 배틀 (again)');
      await click(P, '#new-battle', '새 배틀 (confirm again)');
      await toastSeen(P, /새 배틀을 준비/, 'new battle prepared again');
    }
    await page.waitForTimeout(4500); // longer than the 4 s auto-disarm of confirm buttons
    const hdr = (await page.locator('#new-battle').innerText()).trim();
    const hdrCls = (await page.locator('#new-battle').getAttribute('class')) || '';
    check(V, hdr === '새 배틀' && !/armed/.test(hdrCls), 'after the reset the header button reads "새 배틀" again (not still armed)', `text "${hdr}", class "${hdrCls}"`);
    const st = await appState(P);
    check(V, st.battle === null && st.foeTeam.every((x) => !x) && st.preview === null && st.team && st.team.sets.length === 6 && st.tab === 'preview',
      '"새 배틀" clears the battle and opponent entry, keeps my team, goes to 선출', { battle: st.battle, foeTeam: st.foeTeam, tab: st.tab });
    check(V, (await page.locator('#foe-pv-0').inputValue()) === '', 'opponent entry boxes are empty again');
    await click(P, '#tab-battle');
    const bt = (await page.locator('#panel-battle').innerText()).replace(/\s+/g, ' ');
    check(V, /선출/.test(bt) && (await page.locator('#matchup').count()) === 0, 'battle tab asks to go to 선출 first', bt.slice(0, 120));
    await shot(P, 'new_battle');
  });

  // 6) second battle: IME partial input, duplicate warning, manual pick, Mega Evolution, 취소, speed notes, opponent faint
  const FOE2 = ['Garchomp', 'Dragonite', 'Kingambit', 'Hippowdon', 'Gholdengo', 'Corviknight'];
  const B2 = ['Gengar', 'Garchomp', 'Rotom-Wash'];
  await step('battle2-setup', async () => {
    await click(P, '#tab-preview');
    await page.locator('#foe-pv-0').waitFor();
    await click(P, '#foe-pv-0');
    for (const q of ['한ㅋ', '한칼']) {
      await page.locator('#foe-pv-0').fill(q);
      await page.waitForTimeout(100);
      const items = await cbItems(P, 'foe-pv-0');
      check(V, items[0] && items[0].sub === 'Garchomp', `Korean IME intermediate input "${q}" (typing 한카리아스) already finds 한카리아스 first`,
        items.slice(0, 4).map((t) => t.main));
    }
    await page.locator('#foe-pv-0').press('Escape');
    await blur(P);
    await cbPick(P, 'foe-pv-0', '한칼', { value: 'Garchomp' }, { what: 'battle 2 opponent 1 (IME partial)' });
    await cbPick(P, 'foe-pv-1', 'dragonite', { value: 'Dragonite' }, { what: 'battle 2 opponent 2' });
    await cbPick(P, 'foe-pv-2', koSp('Dragonite'), { value: 'Dragonite' }, { what: 'battle 2 duplicate opponent' });
    await blur(P);
    const dup = await page.locator('#foe-entry .msg-warn').innerText().catch(() => '');
    check(V, dup.includes('중복') && dup.includes(mainSp('Dragonite')), 'a duplicate opponent species is flagged', dup);
    await click(P, '#foe-pv-3');
    const taken = (await cbItems(P, 'foe-pv-3')).find((t) => t.sub === 'Garchomp');
    await page.locator('#foe-pv-3').press('Escape');
    await blur(P);
    check(V, taken && /이미 입력/.test(taken.hint), 'species already entered are marked in the other opponent dropdowns', taken);
    await cbPick(P, 'foe-pv-2', 'kingambit', { value: 'Kingambit' }, { what: 'battle 2 opponent 3' });
    await cbPick(P, 'foe-pv-3', initials(koSp('Hippowdon')), { value: 'Hippowdon' }, { what: 'battle 2 opponent 4', rankWarnOnly: true });
    await cbPick(P, 'foe-pv-4', 'gholdengo', { value: 'Gholdengo' }, { what: 'battle 2 opponent 5' });
    await cbPick(P, 'foe-pv-5', 'corviknight', { value: 'Corviknight' }, { what: 'battle 2 opponent 6' });
    await blur(P);
    // with a duplicate the preview / start buttons stay off
    await cbPick(P, 'foe-pv-3', koSp('Garchomp'), { value: 'Garchomp' }, { what: 'battle 2 duplicate again', rankWarnOnly: true });
    await blur(P);
    check(V, await page.locator('#preview-run').isDisabled(), '"AI 선출 추천" is disabled while the opponent entry has a duplicate');
    await cbPick(P, 'foe-pv-3', initials(koSp('Hippowdon')), { value: 'Hippowdon' }, { what: 'battle 2 opponent 4 (fixed)', rankWarnOnly: true });
    await blur(P);
    let st = await appState(P);
    check(V, deepEq(st.foeTeam, FOE2), 'battle 2 opponent entry stored', st.foeTeam);
    if (!(await page.locator('#d-manual-pick').evaluate((e) => e.open))) await click(P, '#d-manual-pick > summary');
    const idx = (n) => ctx.team.findIndex((x) => x.species === n);
    for (const n of B2) await click(P, `#mp-${idx(n)}`, `manual pick ${n}`);
    check(V, (await page.locator('#mp-lead-0').getAttribute('aria-checked')) === 'true', 'the first manual pick is the default lead');
    await click(P, '#mp-start', '이 선출로 시작 (manual)');
    await toastSeen(P, /배틀을 시작/, 'manual battle started');
    st = await appState(P);
    check(V, st.battle && deepEq(st.battle.me.brought, B2) && st.battle.me.active === 'Gengar' && st.tab === 'battle',
      'manual "이 선출로 시작" starts the battle with my 3 and the chosen lead', st.battle && st.battle.me);
    await click(P, '#foe-lead-1', 'opponent lead Dragonite');
  });

  await step('battle2-mega', async () => {
    check(V, (await page.locator('#me0-mega').count()) === 1, '"메가진화함" is offered for Gengar holding Gengarite');
    check(V, (await page.locator('#me0-itemlost').count()) === 0, 'no "도구 사용/잃음" for Gengar holding its Mega Stone (it cannot be lost)');
    await click(P, '#samples-0', '빠름 6');
    await blur(P);
    const respP = page.waitForResponse((r) => r.url().endsWith('/api/advise') && r.request().method() === 'POST', { timeout: T_AI });
    await click(P, '#advise-run', '최선의 행동 계산 (mega available)');
    const resp = await respP;
    const res = await resp.json();
    check(V, resp.status() === 200 && !res.error, 'advise with Mega available succeeds', res.error);
    const megaRecs = (res.recommendations || []).filter((r) => /Mega Evolve/.test(r.label));
    check(V, megaRecs.length > 0, 'with a Mega Stone and Mega unused the AI offers "Mega Evolve + move" actions', (res.recommendations || []).map((r) => r.label));
    await page.locator('#results-card table.recs tbody tr').first().waitFor({ timeout: 10000 });
    const mains = await page.locator('#results-card table.recs .act-main').allInnerTexts();
    check(V, mains.some((t) => t.startsWith('메가진화 + ')), 'mega actions are shown as "메가진화 + 기술"', mains);
    const belTxt = (await page.locator('#beliefs-sec').innerText().catch(() => '')).replace(/\s+/g, ' ');
    check(V, !Object.keys(res.beliefs || {}).length && belTxt.includes('아직 관측 없음 — 같은 우선도 기술의 행동 순서를 기록하면 상대 스피드/구애스카프를 추정합니다'),
      'with no speed observation yet (no beliefs) the "상대 능력치 추정" area explains how to get one', { beliefs: res.beliefs, text: belTxt });
    await shot(P, 'battle2_mega_options', { sel: '#results-card' });
    await click(P, 'label[for="me0-mega"]', '메가진화함');
    const st = await appState(P);
    check(V, st.battle.me.mons.Gengar.mega === true && st.battle.me.megaUsed === true, '"메가진화함" marks my Pokemon and my side\'s Mega as used', { mega: st.battle.me.mons.Gengar.mega, used: st.battle.me.megaUsed });
    check(V, await page.locator('#sc-me-megaused').isChecked(), 'field card "메가진화 사용함" (mine) follows');
    check(V, (await page.locator('#q-my-mega').count()) === 0, 'per-turn "내가 메가진화" disappears once Mega is used');
  });

  await step('battle2-cancel', async () => {
    let cancelCalls = 0;
    const onReq = (r) => { if (r.url().endsWith('/api/cancel')) cancelCalls += 1; };
    page.on('request', onReq);
    await click(P, '#samples-2', '정밀 24');
    await blur(P);
    P.expectAbort += 1;
    const preciseReq = page.waitForRequest((r) => r.url().endsWith('/api/advise') && r.method() === 'POST', { timeout: 15000 });
    await click(P, '#advise-run', '최선의 행동 계산 (to cancel)');
    const pb = (await preciseReq).postDataJSON();
    check(V, pb.samples === 24 && pb.depth === 2, '/api/advise body of 정밀 = samples 24, depth 2', { samples: pb.samples, depth: pb.depth });
    const prog2 = await page.locator('#results-card .running').innerText().catch(() => '');
    check(V, /정밀/.test(prog2) && /2턴 앞/.test(prog2) && /10~30초/.test(prog2), 'the progress line of 정밀 says 2턴 앞 and 10~30초', prog2);
    await page.waitForTimeout(700);
    const cancelBtn = page.locator('.action-bar').getByRole('button', { name: '취소' });
    check(V, await cancelBtn.isVisible().catch(() => false), 'a "취소" button is shown while calculating');
    await click(P, cancelBtn, '취소');
    await toastSeen(P, /취소했습니다/, 'cancel confirmed');
    await page.waitForTimeout(300);
    check(V, !(await page.locator('#advise-run').isDisabled()), 'after 취소 the calculate button is enabled again');
    await click(P, '#samples-0', '빠름 6');
    await blur(P);
    const respP = page.waitForResponse((r) => r.url().endsWith('/api/advise') && r.request().method() === 'POST', { timeout: T_AI });
    const t0 = Date.now();
    await click(P, '#advise-run', '최선의 행동 계산 (after cancel)');
    const resp = await respP;
    const secs = (Date.now() - t0) / 1000;
    R.timings[`${V} advise after cancel`] = secs.toFixed(1);
    const res = await resp.json();
    check(V, resp.status() === 200 && !res.error && (res.recommendations || []).length > 0, 'the calculation after 취소 succeeds', res.error);
    check(V, secs < 45, 'the calculation after 취소 is not stuck behind the cancelled one', `${secs.toFixed(1)} s`);
    check(V, cancelCalls >= 1, '"취소" also tells the server to stop computing (POST /api/cancel)', `${cancelCalls} requests to /api/cancel`);
    const body = resp.request().postDataJSON();
    check(V, body.samples === 6 && body.depth === 1, '/api/advise body of 빠름 (again) = samples 6, depth 1', { samples: body.samples, depth: body.depth });
    const g = body.state.me.pokemon.Gengar;
    check(V, g && g.mega === true && body.state.me.mega_used === true, 'request after Mega: me.pokemon.Gengar.mega = true and me.mega_used = true', { g, mega_used: body.state.me.mega_used });
    check(V, !(res.recommendations || []).some((r) => /Mega Evolve/.test(r.label)), 'after Mega Evolution no more "Mega Evolve" actions are offered', (res.recommendations || []).map((r) => r.label));
    page.off('request', onReq);
  });

  await step('battle2-speed', async () => {
    const dn = SP.get('Dragonite').moves;
    const dc = ['Dragon Claw', 'Outrage', 'Earthquake', 'Fire Punch'].find((m) => dn.includes(m) && !(DATA.moves[m].priority || 0));
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await cbPick(P, 'q-foe-move', 'extreme speed', { value: 'Extreme Speed' }, { what: 'battle 2 opponent priority move' });
    await select(P, '#q-my-move', 'Shadow Ball');
    await click(P, '#q-order-1', '상대가 먼저');
    const auto1 = P.phone ? page.waitForResponse((r) => r.url().endsWith('/api/advise'), { timeout: T_AI }) : null;
    await click(P, '#next-turn', '다음 턴 (priority)');
    await toastSeen(P, /우선도가 달라/, 'different priority: speed comparison not recorded (note)');
    let st = await appState(P);
    let d = st.battle.foe.mons.Dragonite;
    check(V, !(d.faster || []).length && !(d.slower || []).length, 'no speed observation is recorded when the priorities differ', { faster: d.faster, slower: d.slower });
    if (P.phone) { await auto1; await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI }); }
    await select(P, '#f-trickroom', '3');
    // Mega Gengar: the AI compares base-forme Speed stats, so this order must not be recorded
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await cbPick(P, 'q-foe-move', koMove(dc), { value: dc }, { what: 'battle 2 opponent move under Trick Room' });
    await select(P, '#q-my-move', 'Shadow Ball');
    await click(P, '#q-order-1', '상대가 먼저 (Mega Gengar)');
    const auto2 = P.phone ? page.waitForResponse((r) => r.url().endsWith('/api/advise'), { timeout: T_AI }) : null;
    await click(P, '#next-turn', '다음 턴 (Mega)');
    await toastSeen(P, /메가진화 때문에/, 'a speed order involving a Mega Evolved Pokemon is not recorded (note)');
    st = await appState(P);
    d = st.battle.foe.mons.Dragonite;
    check(V, !(d.faster || []).length && !(d.slower || []).length, 'no speed observation is recorded while my Pokemon is Mega Evolved', { faster: d.faster, slower: d.slower });
    if (P.phone) { await auto2; await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI }); }
    // Rotom-Wash (no Mega) on the field: under Trick Room "상대가 먼저" means the opponent is slower
    await click(P, 'label[for="me2-active"]', 'Rotom-Wash active');
    const rw = setOf('Rotom-Wash').moves.find((m) => !(DATA.moves[m].priority || 0) && DATA.moves[m].category !== 'Status');
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await cbPick(P, 'q-foe-move', koMove(dc), { value: dc }, { what: 'battle 2 revealed move again' });
    await select(P, '#q-my-move', rw);
    await click(P, '#q-order-1', '상대가 먼저 (Trick Room)');
    await setNumber(P, '#q-foe-hp', 0);
    await click(P, '#next-turn', '다음 턴 (Trick Room, KO)');
    await toastSeen(P, /턴 3 기록 완료/, 'turn 3 recorded');
    st = await appState(P);
    d = st.battle.foe.mons.Dragonite;
    check(V, (d.slower || []).includes('Rotom-Wash') && !(d.faster || []).includes('Rotom-Wash'),
      'under Trick Room "상대가 먼저" records the opponent as slower than my Pokemon', { faster: d.faster, slower: d.slower });
    check(V, d.fainted === true && d.hp === 0, 'opponent HP 0 % after the turn marks it fainted', { fainted: d.fainted, hp: d.hp });
    check(V, await page.locator('#foe-replace').isVisible().catch(() => false), '"상대 ○○ 쓰러짐 — 다음 포켓몬은?" card is shown');
    await shot(P, 'battle2_foe_fainted', { sel: '#foe-replace' });
    const pick = page.locator('#foe-replace button', { hasText: mainSp('Kingambit') });
    await click(P, pick, 'opponent sends in Kingambit');
    st = await appState(P);
    check(V, st.battle.foe.active === 'Kingambit' && deepEq(st.battle.foe.seen, ['Dragonite', 'Kingambit']) && !(await page.locator('#foe-replace').count()),
      'picking the opponent\'s next Pokemon makes it active and seen', st.battle.foe);
    await blur(P);
    await click(P, '#samples-1', '보통');
    check(V, (await page.locator('#samples-1').getAttribute('aria-checked')) === 'true', 'preset "보통" is selected');
    const respP = page.waitForResponse((r) => r.url().endsWith('/api/advise') && r.request().method() === 'POST', { timeout: T_AI });
    await click(P, '#advise-run', '최선의 행동 계산 (after KO)');
    const resp = await respP;
    const res = await resp.json();
    const nb = resp.request().postDataJSON();
    check(V, nb.samples === 12 && nb.depth === 1, '/api/advise body of 보통 = samples 12, depth 1', { samples: nb.samples, depth: nb.depth });
    check(V, resp.status() === 200 && !res.error && (res.recommendations || []).length > 0, 'advise with a fainted opponent in the state succeeds', res.error);
    const errs = validateState(resp.request().postDataJSON().state, {
      turn: 4, team: ctx.team, brought: B2, myActive: 'Rotom-Wash', maxHp: Object.fromEntries(B2.map((n) => [n, maxHp(n)])),
      mePokemon: { Gengar: { mega: true, hp: `${maxHp('Gengar')}/${maxHp('Gengar')}` } }, meSide: {}, foeSide: {}, foeTeam: FOE2,
      foeSeen: ['Dragonite', 'Kingambit'], foeActive: 'Kingambit',
      foePokemon: { Dragonite: { fainted: true, hp: '0%', moves: { $sorted: ['Extreme Speed', dc] }, slower_than: ['Rotom-Wash'], faster_than: ABSENT } },
      field: { trickroom: 1, weather: '', weather_turns: 0 }, meMegaUsed: true, foeMegaUsed: false,
    });
    check(V, !errs.length, 'battle 2 <state> (mega, fainted opponent, Trick Room) matches API.md', errs.join('\n'));
    await page.locator('#results-card table.recs tbody tr').first().waitFor({ timeout: 10000 });
    await shot(P, 'battle2_results');
    await audit(P, 'battle 2 (after KO)');
  });

  await step('battle2-mega-forme', async () => {
    // Garchomp has two Megas: the opponent's Mega control asks which one (and sets its Mega Stone)
    await click(P, '#foe-roster-0', 'roster Garchomp');
    await click(P, '#d-foe-Garchomp > summary', 'open Garchomp');
    const ctl = page.locator('#foe0-mega');
    const tag = await ctl.evaluate((e) => e.tagName).catch(() => '');
    const formes = SP.get('Garchomp').megas;
    check(V, tag === 'SELECT', 'an opponent with two Megas gets a forme select instead of a plain checkbox', tag);
    if (tag === 'SELECT') {
      const z = formes[formes.length - 1];
      await select(P, '#foe0-mega', z.forme);
      const st = await appState(P);
      const g = st.battle.foe.mons.Garchomp;
      check(V, g.mega === true && g.item === z.stone && st.battle.foe.megaUsed === true,
        'choosing the Mega forme marks it Mega Evolved with that Mega Stone', { mega: g.mega, item: g.item, used: st.battle.foe.megaUsed });
      await audit(P, 'battle 2 (mega forme)');
    }
  });

  // 7) field effects of recorded moves: applied by themselves, counted as "turns left at the next decision"
  await step('battle2-field-auto', async () => {
    let st = await appState(P);
    check(V, st.battle.turn === 4 && st.battle.me.active === 'Rotom-Wash' && st.battle.foe.active === 'Kingambit' && st.battle.field.trickroom === 1,
      'field-effect test starts at turn 4: Rotom-Wash vs Kingambit, Trick Room 1 turn left', { turn: st.battle.turn, me: st.battle.me.active, foe: st.battle.foe.active, field: st.battle.field });
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    if (await page.locator('#q-auto').isChecked()) await click(P, 'label[for="q-auto"]', 'turn off auto advise');
    const fieldNow = async () => {
      const x = await appState(P);
      return { turn: x.battle.turn, f: x.battle.field, me: x.battle.me.side, foe: x.battle.foe.side, log: x.battle.fieldLog };
    };
    const live = (o) => Object.fromEntries(Object.entries(o || {}).filter(([, v]) => Number(v) > 0));
    // turn 4: I use Reflect, the opponent uses Sandstorm (item unknown: 5 turns)
    await select(P, '#q-my-move', 'Reflect');
    await cbPick(P, 'q-foe-move', 'sandstorm', { value: 'Sandstorm' }, { what: 'opponent weather move', rankWarnOnly: true });
    await blur(P);
    const hint = await page.locator('#q-field-hint').innerText().catch(() => '');
    check(V, hint.includes(koMove('Reflect')) && hint.includes(koMove('Sandstorm')), 'the per-turn panel says the field moves will be applied to 「필드」', hint);
    await click(P, '#next-turn', '다음 턴 (Reflect / Sandstorm)');
    await toastSeen(P, /턴 4 기록 완료.*필드 자동 반영/, 'the turn toast lists the applied field effects');
    const t4 = (await page.locator('#toasts').innerText().catch(() => '')).replace(/\s+/g, ' ');
    check(V, t4.includes('모래바람 4턴') && t4.includes('내 필드 리플렉터 4턴'), 'the toast names the effects with their turns left on the next turn', t4);
    let x = await fieldNow();
    check(V, x.turn === 5 && x.f.weather === 'sandstorm' && x.f.weather_turns === 4 && x.f.trickroom === 0 && deepEq(live(x.me), { reflect: 4 }) && deepEq(live(x.foe), {}),
      'turn 5: Sandstorm (5 turns) shows 4 left, my Reflect 4 left, the old Trick Room (1 left) ended', x);
    const note = (await page.locator('#field-note').innerText().catch(() => '')).replace(/\s+/g, ' ');
    check(V, /턴 4 기록에서 자동 반영/.test(note) && note.includes('리플렉터') && note.includes('모래바람'), 'the 필드 card shows what was applied automatically', note);
    check(V, (await page.locator('#f-weather').inputValue()) === 'sandstorm' && (await page.locator('#f-weather-turns').inputValue()) === '4'
      && (await page.locator('#sc-me-reflect').inputValue()) === '4', 'the 필드 card controls show the applied effects');
    // turn 5: I switch to Gengar, the opponent uses Stealth Rock (on my side)
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await select(P, '#q-my-switch', 'Gengar');
    await cbPick(P, 'q-foe-move', 'stealth rock', { value: 'Stealth Rock' }, { what: 'opponent hazard move' });
    await blur(P);
    await click(P, '#next-turn', '다음 턴 (switch / Stealth Rock)');
    await toastSeen(P, /턴 5 기록 완료/, 'turn 5 recorded');
    x = await fieldNow();
    check(V, x.turn === 6 && x.f.weather === 'sandstorm' && x.f.weather_turns === 3 && deepEq(live(x.me), { reflect: 3, stealthrock: 1 }) && deepEq(live(x.foe), {}),
      'turn 6: the opponent\'s Stealth Rock lands on my side; Sandstorm 3, Reflect 3', x);
    // turn 6: Gengar uses Trick Room; the opponent's second Stealth Rock changes nothing
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await select(P, '#q-my-move', 'Trick Room');
    await cbPick(P, 'q-foe-move', 'stealth rock', { value: 'Stealth Rock' }, { what: 'opponent hazard move again' });
    await blur(P);
    await click(P, '#next-turn', '다음 턴 (Trick Room)');
    await toastSeen(P, /턴 6 기록 완료.*트릭룸 4턴/, 'the toast says Trick Room has 4 turns left');
    x = await fieldNow();
    check(V, x.turn === 7 && x.f.trickroom === 4 && x.f.weather_turns === 2 && deepEq(live(x.me), { reflect: 2, stealthrock: 1 }),
      'turn 7: my Trick Room (5 turns) shows 4 left; a second Stealth Rock adds nothing', x);
    check(V, (await page.locator('#f-trickroom').inputValue()) === '4', 'the 필드 card shows Trick Room 4');
    // turn 7: Trick Room again ends it; the opponent's Rain Dance replaces the sandstorm
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await select(P, '#q-my-move', 'Trick Room');
    await cbPick(P, 'q-foe-move', koMove('Rain Dance'), { value: 'Rain Dance' }, { what: 'opponent second weather move', rankWarnOnly: true });
    await blur(P);
    await click(P, '#next-turn', '다음 턴 (Trick Room again / Rain Dance)');
    await toastSeen(P, /턴 7 기록 완료.*트릭룸 해제/, 'the toast says Trick Room was ended');
    x = await fieldNow();
    check(V, x.turn === 8 && x.f.trickroom === 0 && x.f.weather === 'raindance' && x.f.weather_turns === 4 && deepEq(live(x.me), { reflect: 1, stealthrock: 1 }),
      'turn 8: Trick Room used during Trick Room ends it; Rain Dance replaces Sandstorm with 4 turns left', x);
    await page.locator('#field-card').scrollIntoViewIfNeeded();
    await shot(P, 'battle2_field_auto', { sel: '#field-card' });
    // the server accepts the resulting state
    await click(P, '#samples-0', '빠름 6');
    await blur(P);
    const respP = page.waitForResponse((r) => r.url().endsWith('/api/advise') && r.request().method() === 'POST', { timeout: T_AI });
    await click(P, '#advise-run', '최선의 행동 계산 (after field effects)');
    const resp = await respP;
    const res = await resp.json();
    check(V, resp.status() === 200 && !res.error && (res.recommendations || []).length > 0, 'advise with the automatically applied field succeeds', res.error);
    const errs = validateState(resp.request().postDataJSON().state, {
      turn: 8, team: ctx.team, brought: B2, myActive: 'Gengar', maxHp: Object.fromEntries(B2.map((n) => [n, maxHp(n)])),
      meSide: { reflect: 1, stealthrock: 1 }, foeSide: {}, foeTeam: FOE2, foeSeen: ['Dragonite', 'Kingambit', 'Garchomp'], foeActive: 'Kingambit',
      foePokemon: { Kingambit: { moves: { $sorted: ['Sandstorm', 'Stealth Rock', 'Rain Dance'] } } },
      field: { weather: 'raindance', weather_turns: 4, terrain: '', terrain_turns: 0, trickroom: 0 }, meMegaUsed: true,
    });
    check(V, !errs.length, '<state> after the field effects (rain 4, Reflect 1, Stealth Rock on my side) matches API.md', errs.join('\n'));
    await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI });
  });

  // 8) phone: the bar hidden while typing must not swallow a tap on a button near the bottom of the screen
  if (P.phone) {
    await step('phone-keyboard', async () => {
      const toBottom = async (sel, gap) => page.evaluate(([s2, g]) => {
        const r = document.querySelector(s2).getBoundingClientRect();
        window.scrollBy(0, r.bottom - (window.innerHeight - g));
        return document.querySelector(s2).getBoundingClientRect().bottom;
      }, [sel, gap]);
      const barState = () => page.evaluate(() => ({
        display: getComputedStyle(document.querySelector('.action-bar')).display,
        kb: document.body.classList.contains('kb-open'), focus: (document.activeElement || {}).id || '',
      }));
      // a tap at fixed screen coordinates, as a finger does (no auto-scrolling to an unobscured spot)
      const tapAt = async (sel) => {
        const r = await page.locator(sel).evaluate((e) => { const b = e.getBoundingClientRect(); return { x: b.left + b.width / 2, y: b.top + b.height / 2 }; });
        await page.touchscreen.tap(r.x, r.y);
        return r;
      };
      // a message on screen while typing (here: tapping the active opponent in the roster, a 6 s error) must neither drop
      // into the bottom band where the bar comes back nor take the tap meant for the button under it
      const raiseToast = async () => {
        await dismissToasts();
        const chip = page.locator(`#foe-roster-${FOE2.indexOf(st.battle.foe.active)}`);
        await chip.scrollIntoViewIfNeeded();
        await chip.tap();
        await page.locator('#toasts .toast-error').waitFor({ timeout: 3000 });
      };
      const toastVsTap = async (sel, what) => {
        const r = await page.evaluate(([s2, h0]) => {
          const b = document.querySelector(s2).getBoundingClientRect();
          const x = b.left + b.width / 2;
          const y = b.top + b.height / 2;
          const t = document.querySelector('#toasts .toast');
          const hit = document.elementFromPoint(x, y);
          return { tap: [Math.round(x), Math.round(y)], toast: t ? [Math.round(t.getBoundingClientRect().top), Math.round(t.getBoundingClientRect().bottom)] : null,
            limit: window.innerHeight - h0, hit: hit ? (hit.id || hit.className) : '', onTarget: !!(hit && hit.closest(s2)),
            kb: document.body.classList.contains('kb-open') };
        }, [sel, barH0]);
        check(V, r.toast && r.kb && r.onTarget && r.toast[1] <= r.limit + 1,
          `${what}: with a message on screen while typing, the message stays above the bar's band and the tap point is the button`, r);
      };
      await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI });
      await dismissToasts();
      const barH0 = await page.evaluate(() => document.querySelector('.action-bar').offsetHeight);
      let st = await appState(P);
      const me = st.battle.me.active;
      const turn0 = st.battle.turn;
      await raiseToast();
      await page.locator('#q-foe-hp').scrollIntoViewIfNeeded();
      await page.locator('#q-foe-hp').tap();
      await page.keyboard.type('40');
      let bs = await barState();
      check(V, bs.display === 'none' && bs.kb && bs.focus === 'q-foe-hp', 'while a number field has the focus (keyboard up) the "최선의 행동 계산" bar is hidden', bs);
      const y = await toBottom('#q-my-faint', 12);
      await page.waitForTimeout(150);
      bs = await barState();
      check(V, bs.focus === 'q-foe-hp' && y > 844 - 80, '"기절" sits in the bottom 80px (where the bar reappears) and the number field still has the focus', { y, ...bs });
      const fy0 = await page.locator('#q-my-faint').evaluate((e) => e.getBoundingClientRect().top);
      await toastVsTap('#q-my-faint', '"기절"');
      const at = await tapAt('#q-my-faint');
      check(V, at.y > 844 - 80, 'the finger lands in the bottom 80px of the screen', at);
      await page.waitForTimeout(250);
      const fy1 = await page.locator('#q-my-faint').evaluate((e) => e.getBoundingClientRect().top);
      check(V, Math.abs(fy1 - fy0) < 3, 'the page does not jump after the tap ("기절" stays where it was tapped)', { before: fy0, after: fy1 });
      st = await appState(P);
      check(V, st.battle.quick.myHp === 0 && (await page.locator('#q-my-hp').inputValue()) === '0' && st.battle.quick.foeHp === 40,
        'tapping "기절" right after typing the opponent\'s HP records the faint (the tap is not lost to the reappearing bar)', { quick: st.battle.quick });
      await shot(P, 'phone_kb_after_faint_tap');
      // "다음 턴" near the bottom while the opponent's HP field still has the focus (and a message is on screen)
      await raiseToast();
      await page.locator('#q-foe-hp').scrollIntoViewIfNeeded();
      await page.locator('#q-foe-hp').tap();
      await page.locator('#q-foe-hp').fill('35');
      const y2 = await toBottom('#next-turn', 10);
      await page.waitForTimeout(150);
      bs = await barState();
      check(V, bs.focus === 'q-foe-hp' && bs.display === 'none' && y2 > 844 - 80, '"다음 턴" sits in the bottom 80px with the keyboard focus in a number field', { y: y2, ...bs });
      await toastVsTap('#next-turn', '"다음 턴"');
      await tapAt('#next-turn');
      await toastSeen(P, new RegExp(`턴 ${turn0} 기록 완료`), '"다음 턴" tapped with the keyboard up is recorded');
      st = await appState(P);
      check(V, st.battle.turn === turn0 + 1 && st.battle.me.mons[me].fainted === true && st.battle.foe.mons.Kingambit.hp === 35,
        'tapping "다음 턴" with the focus still in a number field records the turn (my Pokemon fainted, opponent HP 35%)',
        { turn: st.battle.turn, fainted: st.battle.me.mons[me].fainted, foeHp: st.battle.foe.mons.Kingambit.hp });
      await page.waitForTimeout(900);
      bs = await barState();
      const settle = await page.evaluate(() => document.body.classList.contains('bar-settle'));
      check(V, bs.display !== 'none' && !bs.kb && !settle, 'the bar comes back (and takes taps again) once the typing is over', { ...bs, settle });
      check(V, await page.locator('#faint-card').isVisible().catch(() => false), 'the faint card asks who to send in next');
      await shot(P, 'phone_kb_after_next_turn');
    });
  }

  // 9) third battle: weather abilities of the leads. Mine (known from my set) and the opponent's (entered in
  //    「상대 포켓몬」 on turn 1) are on the field for the turn-1 calculation; revealing it again does not apply it twice
  const FOE3 = ['Hippowdon', 'Dragonite', 'Kingambit', 'Tyranitar', 'Gholdengo', 'Corviknight'];
  const B3 = ['Torkoal', 'Garchomp', 'Rotom-Wash'];
  await step('battle3-entry-weather', async () => {
    if (!TEAM3_TXT.includes('Torkoal')) throw new Error('could not build the battle 3 team from examples/team_singles.txt');
    if (await page.locator('#advise-run').count()) await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI });
    await click(P, '#new-battle', '새 배틀 (battle 3)');
    await click(P, '#new-battle', '새 배틀 (battle 3, confirm)');
    await toastSeen(P, /새 배틀을 준비/, 'battle 3: new battle prepared');
    await click(P, '#tab-team');
    if (!(await page.locator('#d-paste').evaluate((e) => e.open))) await click(P, '#d-paste > summary', 'paste section');
    await page.fill('#paste-text', TEAM3_TXT);
    const pr = page.waitForResponse((r) => r.url().endsWith('/api/team/parse'));
    await click(P, '#paste-parse', '팀 확인 (battle 3 team)');
    await pr;
    // the result of the earlier paste is still shown until this one has been rendered
    await page.locator('#paste-parse:not([disabled])').waitFor({ timeout: 10000 });
    await page.locator('#d-paste .parse-res .mon-h .d-main', { hasText: mainSp('Torkoal') }).first().waitFor({ timeout: 10000 });
    check(V, (await page.locator('#d-paste .parse-res .msg-ok').count()) === 1, 'the battle 3 team (Torkoal instead of Talonflame) parses without problems',
      await page.locator('#d-paste .parse-res').innerText().catch(() => ''));
    await click(P, page.locator('#d-paste .parse-res').getByRole('button', { name: '이 팀 사용' }), '이 팀 사용 (battle 3 team)');
    await page.locator('#tab-preview[aria-selected="true"]').waitFor({ timeout: 8000 });
    for (let i = 0; i < 6; i++) await cbPick(P, `foe-pv-${i}`, FOE3[i].toLowerCase(), { value: FOE3[i] }, { what: `battle 3 opponent ${i + 1}`, rankWarnOnly: true });
    await blur(P);
    let st = await appState(P);
    const team3 = st.team.sets;
    const hp3 = (n) => (team3.find((s) => s.species === n) || { stats: { hp: 0 } }).stats.hp;
    check(V, deepEq(st.foeTeam, FOE3) && team3.some((s) => s.species === 'Torkoal' && s.ability === 'Drought' && s.item === 'Heat Rock'),
      'battle 3: my team with Torkoal (Drought @ Heat Rock) and the opponent entry are set', { foe: st.foeTeam, team: team3.map((s) => s.species) });
    if (!(await page.locator('#d-manual-pick').evaluate((e) => e.open))) await click(P, '#d-manual-pick > summary');
    for (const n of B3) await click(P, `#mp-${team3.findIndex((s) => s.species === n)}`, `manual pick ${n}`);
    await click(P, '#mp-start', '이 선출로 시작 (Torkoal lead)');
    await toastSeen(P, /배틀을 시작.*필드 자동 반영: 쾌청 8턴/, 'battle 3 start: the message says my lead\'s Drought set the sun (8 turns)');
    st = await appState(P);
    check(V, st.battle.me.active === 'Torkoal' && st.battle.field.weather === 'sunnyday' && st.battle.field.weather_turns === 8,
      'my lead with Drought sets sun (8 turns: its Heat Rock is known from my set)', st.battle.field);
    // the opponent's lead: its ability is not known yet, so nothing changes
    await click(P, '#foe-lead-0', 'opponent lead Hippowdon');
    st = await appState(P);
    check(V, st.battle.foe.active === 'Hippowdon' && st.battle.foe.mons.Hippowdon.fresh === true && st.battle.field.weather === 'sunnyday',
      'picking the opponent lead (ability unknown) leaves the field as it is', { foe: st.battle.foe.active, field: st.battle.field });
    // its Sand Stream entered in 「상대 포켓몬」 on turn 1: both leads came in together and Hippowdon (base Spe 47) is
    // surely faster than my Torkoal (Spe 36), so its sandstorm came first and my sun replaced it: the sun stays
    await dismissToasts();
    await cbPick(P, 'foe0-ability', koAb('Sand Stream'), { value: 'Sand Stream' }, { what: 'opponent lead ability (editor)' });
    await blur(P);
    await toastSeen(P, /필드 자동 반영: .*더 빨라 먼저 발동 → 쾌청 유지/, 'Sand Stream of the faster opponent lead: the notice says the sun stays');
    st = await appState(P);
    check(V, st.battle.field.weather === 'sunnyday' && st.battle.field.weather_turns === 8 && st.battle.foe.mons.Hippowdon.ability === 'Sand Stream'
      && (st.battle.fieldLog || {}).label === `상대 ${mainSp('Hippowdon')} 등장 (${koAb('Sand Stream')})`,
    'the faster opponent lead\'s Sand Stream entered on turn 1 leaves my slower lead\'s sun (8 turns)', { field: st.battle.field, log: st.battle.fieldLog });
    check(V, (await page.locator('#f-weather').inputValue()) === 'sunnyday' && (await page.locator('#f-weather-turns').inputValue()) === '8',
      'the 필드 card shows sun 8');
    // the turn-1 calculation sees it
    await click(P, '#samples-0', '빠름 6');
    await blur(P);
    const respP = page.waitForResponse((r) => r.url().endsWith('/api/advise') && r.request().method() === 'POST', { timeout: T_AI });
    await click(P, '#advise-run', '최선의 행동 계산 (battle 3, turn 1)');
    const resp = await respP;
    const res = await resp.json();
    check(V, resp.status() === 200 && !res.error && (res.recommendations || []).length > 0, 'battle 3 turn-1 advise succeeds', res.error);
    const errs = validateState(resp.request().postDataJSON().state, {
      turn: 1, team: team3, brought: B3, myActive: 'Torkoal', maxHp: Object.fromEntries(B3.map((n) => [n, hp3(n)])),
      mePokemon: { Torkoal: { fresh: true } }, meSide: {}, foeSide: {}, foeTeam: FOE3, foeSeen: ['Hippowdon'], foeActive: 'Hippowdon',
      foePokemon: { Hippowdon: { ability: 'Sand Stream', fresh: true } },
      field: { weather: 'sunnyday', weather_turns: 8, terrain: '', terrain_turns: 0 },
    });
    check(V, !errs.length, 'turn-1 /api/advise <state> has field.weather "sunnyday" with weather_turns 8 (and Hippowdon\'s ability)', errs.join('\n'));
    await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI });
    // turn 1 recorded with Sand Stream revealed again in 이번 턴 기록, auto-advise on: the page scrolls up while it runs
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await cbPick(P, 'q-foe-ability', koAb('Sand Stream'), { value: 'Sand Stream' }, { what: 'per-turn reveal of the same ability' });
    await blur(P);
    st = await appState(P);
    check(V, st.battle.field.weather === 'sunnyday' && st.battle.field.weather_turns === 8, 'revealing the ability that already took effect changes nothing right away', st.battle.field);
    await select(P, '#q-my-move', 'Protect');
    if (!(await page.locator('#q-auto').isChecked())) await click(P, 'label[for="q-auto"]', 'turn auto advise on');
    await dismissToasts();
    const release = await holdAdvise();
    const autoResp = page.waitForResponse((r) => r.url().endsWith('/api/advise'), { timeout: T_AI });
    await page.locator('#next-turn').scrollIntoViewIfNeeded();
    await click(P, '#next-turn', '다음 턴 (battle 3, turn 1, auto-advise)');
    await toastSeen(P, /턴 1 기록 완료/, 'battle 3 turn 1 recorded');
    const t1 = (await page.locator('#toasts').innerText().catch(() => '')).replace(/\s+/g, ' ');
    await runInView('battle 3 "다음 턴" with auto-advise');
    release();
    await autoResp;
    await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI });
    st = await appState(P);
    check(V, st.battle.turn === 2 && st.battle.field.weather === 'sunnyday' && st.battle.field.weather_turns === 7 && !st.battle.fieldLog && !/모래바람/.test(t1),
      'revealing the same Sand Stream again in 이번 턴 기록 does not apply it now (turn 2: sun 7, no field notice)', { field: st.battle.field, log: st.battle.fieldLog, toast: t1 });
    // turn 2: the opponent faints and sends in Tyranitar (ability unknown)
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    if (await page.locator('#q-auto').isChecked()) await click(P, 'label[for="q-auto"]', 'turn auto advise off');
    await setNumber(P, '#q-foe-hp', 0);
    await click(P, '#next-turn', '다음 턴 (battle 3, opponent faints)');
    await toastSeen(P, /턴 2 기록 완료/, 'battle 3 turn 2 recorded');
    await click(P, page.locator('#foe-replace button', { hasText: mainSp('Tyranitar') }), 'opponent sends in Tyranitar');
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await click(P, '#next-turn', '다음 턴 (battle 3, turn 3)');
    await toastSeen(P, /턴 3 기록 완료/, 'battle 3 turn 3 recorded');
    // turn 4: say the weather is gone (cleared by hand); Tyranitar came in earlier, so its Sand Stream entered in
    // 「상대 포켓몬」 now does not change the field ...
    await page.locator('#field-card').scrollIntoViewIfNeeded();
    await select(P, '#f-weather', '');
    await dismissToasts();
    await cbPick(P, `foe${FOE3.indexOf('Tyranitar')}-ability`, koAb('Sand Stream'), { value: 'Sand Stream' }, { what: 'opponent Tyranitar ability (not fresh)' });
    await blur(P);
    st = await appState(P);
    const notice = await page.locator('#toasts .toast', { hasText: '필드 자동 반영' }).count();
    check(V, st.battle.turn === 4 && st.battle.foe.mons.Tyranitar.fresh === false && st.battle.foe.mons.Tyranitar.ability === 'Sand Stream'
      && st.battle.field.weather === '' && !notice, 'the ability of an opponent that came in on an earlier turn does not change the field when entered in 「상대 포켓몬」',
    { turn: st.battle.turn, fresh: st.battle.foe.mons.Tyranitar.fresh, field: st.battle.field, notice });
    // ... but revealed in 이번 턴 기록 while no sandstorm is up, it counts although the ability was already known
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await cbPick(P, 'q-foe-ability', koAb('Sand Stream'), { value: 'Sand Stream' }, { what: 'per-turn reveal (no sandstorm up)' });
    await blur(P);
    await click(P, '#next-turn', '다음 턴 (battle 3, turn 4, reveal)');
    await toastSeen(P, /턴 4 기록 완료.*모래바람 4턴/, 'a reveal of the known Sand Stream applies it when no sandstorm is up');
    st = await appState(P);
    check(V, st.battle.turn === 5 && st.battle.field.weather === 'sandstorm' && st.battle.field.weather_turns === 4,
      'turn 5: the revealed Sand Stream (set during turn 4) shows 4 turns left', st.battle.field);
    await page.locator('#field-card').scrollIntoViewIfNeeded();
    await shot(P, 'battle3_entry_weather', { sel: '#field-card' });
    await audit(P, 'battle 3 (entry weather)');
  });

  // battle 4: my Pelipper (Drizzle @ Damp Rock, Spe 128) leads against Torkoal (at most 118 even with a Choice Scarf)
  const FOE4 = ['Torkoal', 'Politoed', 'Hippowdon', 'Tyranitar', 'Dragonite', 'Gholdengo'];
  const B4 = ['Pelipper', 'Garchomp', 'Rotom-Wash'];
  await step('battle4-entry-order', async () => {
    if (await page.locator('#advise-run').count()) await page.locator('#advise-run:not([disabled])').waitFor({ timeout: T_AI });
    await click(P, '#new-battle', '새 배틀 (battle 4)');
    await click(P, '#new-battle', '새 배틀 (battle 4, confirm)');
    await toastSeen(P, /새 배틀을 준비/, 'battle 4: new battle prepared');
    await click(P, '#tab-team');
    if (!(await page.locator('#d-paste').evaluate((e) => e.open))) await click(P, '#d-paste > summary', 'paste section');
    await page.fill('#paste-text', TEAM4_TXT);
    const pr = page.waitForResponse((r) => r.url().endsWith('/api/team/parse'));
    await click(P, '#paste-parse', '팀 확인 (battle 4 team)');
    await pr;
    await page.locator('#paste-parse:not([disabled])').waitFor({ timeout: 10000 });
    await page.locator('#d-paste .parse-res .mon-h .d-main', { hasText: mainSp('Pelipper') }).first().waitFor({ timeout: 10000 });
    check(V, (await page.locator('#d-paste .parse-res .msg-ok').count()) === 1, 'the battle 4 team (Pelipper instead of Talonflame) parses without problems',
      await page.locator('#d-paste .parse-res').innerText().catch(() => ''));
    await click(P, page.locator('#d-paste .parse-res').getByRole('button', { name: '이 팀 사용' }), '이 팀 사용 (battle 4 team)');
    await page.locator('#tab-preview[aria-selected="true"]').waitFor({ timeout: 8000 });
    for (let i = 0; i < 6; i++) await cbPick(P, `foe-pv-${i}`, FOE4[i].toLowerCase(), { value: FOE4[i] }, { what: `battle 4 opponent ${i + 1}`, rankWarnOnly: true });
    await blur(P);
    let st = await appState(P);
    const team4 = st.team.sets;
    if (!(await page.locator('#d-manual-pick').evaluate((e) => e.open))) await click(P, '#d-manual-pick > summary');
    for (const n of B4) await click(P, `#mp-${team4.findIndex((s) => s.species === n)}`, `manual pick ${n}`);
    await click(P, '#mp-start', '이 선출로 시작 (Pelipper lead)');
    await toastSeen(P, /배틀을 시작.*필드 자동 반영: 비 8턴/, 'battle 4 start: my lead\'s Drizzle sets rain (8 turns, Damp Rock)');
    await click(P, '#foe-lead-0', 'opponent lead Torkoal');
    const field = async () => (await appState(P)).battle.field;
    const fieldIs = (f, w, n) => f.weather === w && f.weather_turns === n;
    // Drought revealed in 이번 턴 기록: the slower Torkoal's sun replaces my rain at once
    await page.locator('#quick-card').scrollIntoViewIfNeeded();
    await dismissToasts();
    await cbPick(P, 'q-foe-ability', koAb('Drought'), { value: 'Drought' }, { what: 'battle 4 reveal Drought' });
    await blur(P);
    await toastSeen(P, /필드 자동 반영: 쾌청 5턴/, 'the slower opponent lead\'s Drought replaces my rain (sun 5)');
    let f = await field();
    check(V, fieldIs(f, 'sunnyday', 5), 'turn 1: the slower lead\'s Drought wins over my faster lead\'s Drizzle (sun 5)', f);
    // its Heat Rock revealed afterwards: the sun lasts 8 turns
    await dismissToasts();
    await cbPick(P, 'q-foe-item', koItem('Heat Rock'), { value: 'Heat Rock' }, { what: 'battle 4 reveal Heat Rock' });
    await blur(P);
    await toastSeen(P, /필드 자동 반영: 쾌청 8턴/, 'Heat Rock revealed after Drought: the notice says sun 8 turns');
    f = await field();
    check(V, fieldIs(f, 'sunnyday', 8), 'a Heat Rock known after the ability took effect makes the sun last 8 turns', f);
    // the ability was a mistake: White Smoke takes the sun back and my rain comes back
    await dismissToasts();
    await cbPick(P, 'q-foe-ability', koAb('White Smoke'), { value: 'White Smoke' }, { what: 'battle 4 corrected ability' });
    await blur(P);
    await toastSeen(P, /필드 자동 반영: 쾌청 취소 → 비 8턴 복원/, 'a corrected ability takes the sun back and restores the rain');
    f = await field();
    check(V, fieldIs(f, 'raindance', 8), 'correcting the mistaken Drought restores my rain (8 turns)', f);
    // Drought again (Heat Rock still revealed): sun 8 at once; recorded turn: sun 7 on turn 2, not applied twice
    await dismissToasts();
    await cbPick(P, 'q-foe-ability', koAb('Drought'), { value: 'Drought' }, { what: 'battle 4 Drought again' });
    await blur(P);
    await toastSeen(P, /필드 자동 반영: 쾌청 8턴/, 'Drought entered again with Heat Rock known: sun 8');
    await select(P, '#q-my-move', 'Roost');
    await page.locator('#next-turn').scrollIntoViewIfNeeded();
    await click(P, '#next-turn', '다음 턴 (battle 4, turn 1)');
    await toastSeen(P, /턴 1 기록 완료/, 'battle 4 turn 1 recorded');
    st = await appState(P);
    check(V, st.battle.turn === 2 && fieldIs(st.battle.field, 'sunnyday', 7) && st.battle.foe.mons.Torkoal.item === 'Heat Rock'
      && st.battle.foe.mons.Torkoal.ability === 'Drought', 'turn 2: sun 7, Torkoal\'s Drought and Heat Rock are recorded', { field: st.battle.field, foe: st.battle.foe.mons.Torkoal });
    // 「상대 포켓몬」: the item corrected to 모름 and back counts from when the sun came (4 turns left / 7 turns left)
    await dismissToasts();
    await cbPick(P, 'foe0-item', '모름', { label: '모름' }, { what: 'battle 4 Torkoal item unknown (editor)', rankWarnOnly: true });
    await blur(P);
    await toastSeen(P, /필드 자동 반영: 쾌청 4턴/, 'the item corrected to unknown: sun 4 turns left');
    f = await field();
    check(V, fieldIs(f, 'sunnyday', 4), 'turn 2, item unknown: the sun set on turn 1 has 4 turns left', f);
    await dismissToasts();
    await cbPick(P, 'foe0-item', koItem('Heat Rock'), { value: 'Heat Rock' }, { what: 'battle 4 Torkoal Heat Rock (editor)' });
    await blur(P);
    await toastSeen(P, /필드 자동 반영: 쾌청 7턴/, 'Heat Rock entered in 「상대 포켓몬」 on turn 2: sun 7 turns left');
    f = await field();
    check(V, fieldIs(f, 'sunnyday', 7), 'turn 2, Heat Rock entered in 「상대 포켓몬」: sun 7', f);
    await page.locator('#field-card').scrollIntoViewIfNeeded();
    await shot(P, 'battle4_entry_order', { sel: '#field-card' });
    await audit(P, 'battle 4 (entry order)');
  });

  // collect page-level findings
  R.timings[`${V} steps`] = T;
  R.console[V] = P.console;
  R.requestFailures[V] = P.reqFail;
  R.badResponses[V] = P.bad;
  R.external[V] = [...new Set(P.external)];
  R.forcedClicks[V] = P.forced;
  R.layout[V] = P.layout;
  check(V, !P.console.length, 'no console errors / page errors', P.console.join('\n'));
  check(V, !P.reqFail.length, 'no failed requests', P.reqFail.join('\n'));
  check(V, !P.bad.length, 'no unexpected HTTP error responses', P.bad.join('\n'));
  check(V, !P.external.length, 'no requests to external hosts (no CDN)', [...new Set(P.external)].join('\n'));
  check(V, !P.forced.length, 'every control could be clicked without being covered/hidden', P.forced.join('\n'));
  for (const a of P.layout) {
    check(V, a.docOverflow <= 0 && !a.overflow.length, `no horizontal overflow on "${a.step}"`, `page overflow ${a.docOverflow}px; ${a.overflow.join('; ')}`);
    check(V, !a.unlabeled.length, `all form controls have accessible labels on "${a.step}"`, a.unlabeled.join('; '));
    check(V, !a.unnamed.length, `all buttons have accessible names on "${a.step}"`, a.unnamed.join('; '));
    check(V, !a.wrapped.length, `short labels (rank, tags, buttons) stay on one line on "${a.step}"`, a.wrapped.join('; '));
    if (a.truncated.length) warn(V, `clipped text on "${a.step}"`, a.truncated.join('; '));
    if (a.tiny.length) warn(V, `text smaller than 11px on "${a.step}"`, a.tiny.join('; '));
    if (a.small.length) warn(V, `small touch targets (<32px) on "${a.step}"`, a.small.join('; '));
  }
  await P.ctx.close();
}

// ---------------------------------------------------------------------------

async function main() {
  console.log(`output: ${OUT}`);
  await startServer();
  let browser = null;
  try {
    DATA = (await getJSON('/api/data')).json;
    REC = (await getJSON('/api/recommended_teams')).json;
    SP = new Map(DATA.species.map((s) => [s.name, s]));
    ITEMS = new Map(DATA.items.map((i) => [i.name, i]));
    const status = (await getJSON('/api/status')).json;
    console.log('status', JSON.stringify(status));
    check('server', status.korean_names === true, '/api/status reports Korean names available', status);
    check('server', REC.teams && REC.teams.length > 0, '/api/recommended_teams returns teams', REC.note);
    const idx = await fetch(`${BASE}/`);
    const html = await idx.text();
    check('server', idx.ok && /lang="ko"/.test(html) && !/https?:\/\//.test(html.replace(/<!--[\s\S]*?-->/g, '')), 'index.html is served, Korean, and references no external URLs');
    for (const f of ['app.js', 'combobox.js', 'style.css']) {
      const r = await fetch(`${BASE}/static/${f}`);
      const body = await r.text();
      check('server', r.ok && !/(?:src|href)\s*=\s*["']https?:|@import\s+url\(["']?https?:|fetch\(["']https?:/.test(body), `/static/${f} is served and loads nothing external`);
    }
    browser = await chromium.launch({ headless: true });
    for (const name of RUN_VPS) {
      if (!VIEWPORTS[name]) throw new Error(`unknown viewport ${name}`);
      await runViewport(browser, VIEWPORTS[name]);
    }
  } finally {
    if (browser) await browser.close();
    await stopServer();
  }
  R.finished = new Date().toISOString();
  fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify(R, null, 2));
  console.log(`\n===== ${R.passed.length} passed, ${R.failed.length} failed, ${R.warnings.length} warnings =====`);
  for (const f of R.failed) console.log(`FAIL [${f.vp}] ${f.msg}${f.evidence ? `\n     ${f.evidence.replace(/\n/g, '\n     ').slice(0, 1200)}` : ''}`);
  for (const w of R.warnings) console.log(`warn [${w.vp}] ${w.msg}${w.evidence ? ` -- ${w.evidence.slice(0, 400)}` : ''}`);
  console.log(`timings: ${JSON.stringify(R.timings)}`);
  console.log(`results: ${path.join(OUT, 'results.json')}  screenshots: ${R.screenshots.length} in ${OUT}`);
  process.exitCode = R.failed.length ? 1 : 0;
}

main().catch(async (e) => {
  console.error('E2E aborted:', e);
  await stopServer();
  try { fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify({ ...R, aborted: String(e && e.stack) }, null, 2)); } catch (x) { /* ignore */ }
  process.exit(2);
});
