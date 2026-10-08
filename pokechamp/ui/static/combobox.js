/* Searchable dropdown (combobox) used for species / moves / items / abilities / natures.
 *
 *   const cb = new Combobox({id, options, value, onChange, placeholder, clearable, emptyValue, label});
 *   container.appendChild(cb.el);
 *
 * options: array (or function returning an array) of
 *   {value, label, sub?, hint?, badge?: {text, cls}, keys?: [extra search strings], cls?: extra class of the row}
 * Typing filters by Korean or English name (case-insensitive, ignores spaces / hyphens) and by
 * Korean initial consonants ("ㅎㅋㄹ" -> 한카리아스). Keyboard: up/down/enter/escape/tab; touch friendly.
 * The input's value is "label (sub)"; while it is not focused, a two-line overlay shows the label with the
 * sub name (English) under it, so a narrow field never cuts the Korean name off.
 */
'use strict';

(function (global) {
  const CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ';
  const JONG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
    'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
  // compound final consonants: [final that stays, initial that moves to the next syllable]
  const JONG_SPLIT = {
    'ㄳ': ['ㄱ', 'ㅅ'], 'ㄵ': ['ㄴ', 'ㅈ'], 'ㄶ': ['ㄴ', 'ㅎ'], 'ㄺ': ['ㄹ', 'ㄱ'], 'ㄻ': ['ㄹ', 'ㅁ'], 'ㄼ': ['ㄹ', 'ㅂ'],
    'ㄽ': ['ㄹ', 'ㅅ'], 'ㄾ': ['ㄹ', 'ㅌ'], 'ㄿ': ['ㄹ', 'ㅍ'], 'ㅀ': ['ㄹ', 'ㅎ'], 'ㅄ': ['ㅂ', 'ㅅ'],
  };
  const BASE = 0xAC00;
  const isSyl = (ch) => { const c = ch.charCodeAt(0); return c >= BASE && c <= 0xD7A3; };
  const choOf = (ch) => (isSyl(ch) ? CHO[Math.floor((ch.charCodeAt(0) - BASE) / 588)] : ch);
  const isCho = (ch) => CHO.indexOf(ch) >= 0;
  const hasKorean = (s) => /[ㄱ-ㆎ가-힣]/.test(s);

  function charMatch(q, t, isLast) {
    if (q === t) return true;
    if (isCho(q)) return choOf(t) === q;
    // while the IME is still composing the last syllable ("하" before "한"), match syllables sharing
    // the same initial consonant + vowel
    if (isLast && isSyl(q) && isSyl(t) && (q.charCodeAt(0) - BASE) % 28 === 0) {
      return Math.floor((q.charCodeAt(0) - BASE) / 28) === Math.floor((t.charCodeAt(0) - BASE) / 28);
    }
    return false;
  }

  function koIndex(target, query) {
    if (!query || !target) return -1;
    outer:
    for (let i = 0; i + query.length <= target.length; i++) {
      for (let j = 0; j < query.length; j++) {
        if (!charMatch(query[j], target[i + j], j === query.length - 1)) continue outer;
      }
      return i;
    }
    return -1;
  }

  // "한칼" (IME mid-composition of "한카리...") -> also try "한카ㄹ"
  function koVariants(q) {
    const out = [q];
    const last = q[q.length - 1];
    if (last && isSyl(last)) {
      const code = last.charCodeAt(0) - BASE;
      const jong = JONG[code % 28];
      if (jong) {
        const bare = String.fromCharCode(BASE + code - (code % 28));
        if (JONG_SPLIT[jong]) {
          const [stay, move] = JONG_SPLIT[jong];
          const withStay = String.fromCharCode(BASE + code - (code % 28) + JONG.indexOf(stay));
          out.push(q.slice(0, -1) + withStay + move);
        } else if (isCho(jong)) {
          out.push(q.slice(0, -1) + bare + jong);
        }
      }
    }
    return out;
  }

  const norm = (s) => String(s || '').toLowerCase().replace(/\s+/g, '');
  const idOf = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9ㄱ-ㆎ가-힣]/g, '');

  function prepare(opt) {
    if (opt._prep) return opt;
    const texts = [opt.label, opt.sub].concat(opt.keys || []).filter(Boolean).map(String);
    opt._en = texts.filter((t) => !hasKorean(t)).map((t) => t.toLowerCase());
    opt._enId = opt._en.map(idOf);
    opt._ko = texts.filter(hasKorean).map(norm);
    opt._prep = true;
    return opt;
  }

  /** Match score of an option for a query (0 = no match, higher = better). */
  function score(opt, query) {
    prepare(opt);
    const q = query.trim().toLowerCase();
    if (!q) return 1;
    let best = 0;
    if (hasKorean(q)) {
      const variants = koVariants(norm(q));
      for (const t of opt._ko) {
        for (const v of variants) {
          const i = koIndex(t, v);
          if (i === 0) return 4;
          if (i > 0) best = Math.max(best, 2);
        }
      }
      return best;
    }
    const qid = idOf(q);
    for (let k = 0; k < opt._en.length; k++) {
      const t = opt._en[k];
      const i = t.indexOf(q);
      if (i === 0) return 4;
      if (i > 0) best = Math.max(best, /[\s\-(]/.test(t[i - 1]) ? 3 : 2);
      else if (qid && opt._enId[k].indexOf(qid) >= 0) best = Math.max(best, opt._enId[k].indexOf(qid) === 0 ? 3 : 2);
    }
    return best;
  }

  function filterOptions(options, query, limit) {
    const scored = [];
    for (let i = 0; i < options.length; i++) {
      const s = score(options[i], query);
      if (s > 0) scored.push([s, i]);
    }
    scored.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    return { items: scored.slice(0, limit).map((x) => options[x[1]]), total: scored.length };
  }

  let uid = 0;

  class Combobox {
    constructor(cfg) {
      this.cfg = Object.assign({ placeholder: '검색 / 선택', clearable: false, emptyValue: '', limit: 150 }, cfg);
      this.id = cfg.id || `cb${++uid}`;
      this.value = cfg.value == null ? this.cfg.emptyValue : cfg.value;
      this.open = false;
      this.items = [];
      this.active = -1;
      this.typed = false;
      this._build();
      this._showValue();
    }

    get options() {
      const o = this.cfg.options;
      return (typeof o === 'function' ? o() : o) || [];
    }

    _build() {
      const el = document.createElement('div');
      el.className = 'cb';
      const input = document.createElement('input');
      input.type = 'text';
      input.id = this.id;
      input.className = 'cb-input';
      input.setAttribute('role', 'combobox');
      input.setAttribute('aria-autocomplete', 'list');
      input.setAttribute('aria-expanded', 'false');
      input.setAttribute('aria-controls', `${this.id}-list`);
      input.autocomplete = 'off';
      input.autocapitalize = 'off';
      input.spellcheck = false;
      input.placeholder = this.cfg.placeholder;
      if (this.cfg.label) input.setAttribute('aria-label', this.cfg.label);
      if (this.cfg.disabled) input.disabled = true;

      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'cb-toggle';
      toggle.tabIndex = -1;
      toggle.setAttribute('aria-label', '목록 열기');
      toggle.textContent = '▾';
      if (this.cfg.disabled) toggle.disabled = true;

      const list = document.createElement('ul');
      list.id = `${this.id}-list`;
      list.className = 'cb-list';
      list.setAttribute('role', 'listbox');
      if (this.cfg.label) list.setAttribute('aria-label', this.cfg.label);
      list.hidden = true;

      // the chosen value, shown over the (transparent) input text while the input is not focused
      const show = document.createElement('div');
      show.className = 'cb-show';
      show.setAttribute('aria-hidden', 'true');
      const showMain = document.createElement('span');
      showMain.className = 'cb-show-main';
      const showSub = document.createElement('span');
      showSub.className = 'cb-show-sub';
      show.append(showMain, showSub);
      this.showMain = showMain;
      this.showSub = showSub;

      el.append(input, show, toggle, list);
      if (this.cfg.clearable) {
        const clear = document.createElement('button');
        clear.type = 'button';
        clear.className = 'cb-clear';
        clear.setAttribute('aria-label', '지우기');
        clear.textContent = '×';
        if (this.cfg.disabled) clear.disabled = true;
        clear.addEventListener('mousedown', (e) => e.preventDefault());
        clear.addEventListener('click', () => { this._choose(this.cfg.emptyValue, false); });
        el.appendChild(clear);
        this.clearBtn = clear;
        el.classList.add('cb-clearable');
      }
      this.el = el;
      this.input = input;
      this.list = list;

      input.addEventListener('focus', () => {
        if (Combobox.suppressOpen) return;
        this.typed = false;
        this._openList('');
        // select the shown text so typing replaces it
        setTimeout(() => { if (document.activeElement === input) input.select(); }, 0);
      });
      input.addEventListener('click', () => { if (!this.open) this._openList(''); });
      input.addEventListener('input', () => {
        this.typed = true;
        this._openList(input.value);
      });
      input.addEventListener('keydown', (e) => this._onKey(e));
      input.addEventListener('blur', () => this._onBlur());
      toggle.addEventListener('mousedown', (e) => e.preventDefault());
      toggle.addEventListener('click', () => {
        if (this.open) { this._close(); this._showValue(); } else { input.focus(); if (!this.open) this._openList(''); }
      });
      // keep the focus in the input while picking with the mouse / finger
      list.addEventListener('mousedown', (e) => e.preventDefault());
      list.addEventListener('click', (e) => {
        const li = e.target.closest('li[data-i]');
        if (!li) return;
        const opt = this.items[Number(li.dataset.i)];
        if (opt) this._choose(opt.value, true);
      });
    }

    _display(value) {
      if (value === this.cfg.emptyValue && !this.options.some((o) => o.value === value)) return '';
      const opt = this.options.find((o) => o.value === value);
      if (!opt) return value == null ? '' : String(value);
      return opt.sub ? `${opt.label} (${opt.sub})` : opt.label;
    }

    _showValue() {
      this.input.value = this._display(this.value);
      this.input.title = this.input.value;
      const opt = this.input.value ? this.options.find((o) => o.value === this.value) : null;
      this.showMain.textContent = opt ? opt.label : this.input.value;
      this.showSub.textContent = opt && opt.sub ? opt.sub : '';
      this.el.classList.toggle('cb-shown', !!this.input.value);
      if (this.clearBtn) this.clearBtn.hidden = this.value === this.cfg.emptyValue || this.value == null;
      this.el.classList.toggle('cb-empty', this.value === this.cfg.emptyValue || this.value == null);
    }

    _openList(query) {
      if (this.input.disabled) return;
      const { items, total } = filterOptions(this.options, query, this.cfg.limit);
      this.items = items;
      this.total = total;
      const list = this.list;
      list.textContent = '';
      const frag = document.createDocumentFragment();
      items.forEach((opt, i) => {
        const li = document.createElement('li');
        li.id = `${this.id}-o${i}`;
        li.setAttribute('role', 'option');
        li.dataset.i = String(i);
        li.className = opt.cls ? `cb-opt ${opt.cls}` : 'cb-opt';
        if (opt.value === this.value) { li.classList.add('cb-selected'); li.setAttribute('aria-selected', 'true'); } else {
          li.setAttribute('aria-selected', 'false');
        }
        if (opt.badge) {
          const b = document.createElement('span');
          b.className = `cb-badge ${opt.badge.cls || ''}`;
          b.textContent = opt.badge.text;
          li.appendChild(b);
        }
        const main = document.createElement('span');
        main.className = 'cb-main';
        main.textContent = opt.label;
        li.appendChild(main);
        if (opt.sub) {
          const sub = document.createElement('span');
          sub.className = 'cb-sub';
          sub.textContent = opt.sub;
          li.appendChild(sub);
        }
        if (opt.hint) {
          const hint = document.createElement('span');
          hint.className = 'cb-hint';
          hint.textContent = opt.hint;
          li.appendChild(hint);
        }
        frag.appendChild(li);
      });
      // message rows are not options (role=presentation keeps the listbox made of options only)
      if (!items.length) {
        const li = document.createElement('li');
        li.className = 'cb-none';
        li.setAttribute('role', 'presentation');
        li.textContent = '일치하는 항목이 없습니다';
        frag.appendChild(li);
      } else if (total > items.length) {
        const li = document.createElement('li');
        li.className = 'cb-none';
        li.setAttribute('role', 'presentation');
        li.textContent = `… 외 ${total - items.length}개 (검색어를 더 입력하세요)`;
        frag.appendChild(li);
      }
      list.appendChild(frag);
      list.hidden = false;
      this.open = true;
      this.input.setAttribute('aria-expanded', 'true');
      this.el.classList.add('cb-open');
      // flip upwards when there is not enough room below
      const r = this.input.getBoundingClientRect();
      const below = window.innerHeight - r.bottom;
      this.el.classList.toggle('cb-up', below < 220 && r.top > below);
      let idx = query ? 0 : items.findIndex((o) => o.value === this.value);
      if (!items.length) idx = -1;
      this._setActive(idx, true);
    }

    _setActive(i, center) {
      const prev = this.list.querySelector('.cb-active');
      if (prev) prev.classList.remove('cb-active');
      this.active = i;
      if (i < 0 || i >= this.items.length) {
        this.input.removeAttribute('aria-activedescendant');
        return;
      }
      const li = document.getElementById(`${this.id}-o${i}`);
      if (li) {
        li.classList.add('cb-active');
        this.input.setAttribute('aria-activedescendant', li.id);
        const lt = li.offsetTop;
        const lb = lt + li.offsetHeight;
        const L = this.list;
        if (center) L.scrollTop = Math.max(0, lt - L.clientHeight / 2 + li.offsetHeight / 2);
        else if (lt < L.scrollTop) L.scrollTop = lt;
        else if (lb > L.scrollTop + L.clientHeight) L.scrollTop = lb - L.clientHeight;
      }
    }

    _close() {
      this.open = false;
      this.list.hidden = true;
      this.input.setAttribute('aria-expanded', 'false');
      this.input.removeAttribute('aria-activedescendant');
      this.el.classList.remove('cb-open', 'cb-up');
    }

    _onKey(e) {
      if (e.isComposing || e.keyCode === 229) return;
      const n = this.items.length;
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          if (!this.open) this._openList(this.typed ? this.input.value : '');
          else if (n) this._setActive((this.active + 1) % n);
          break;
        case 'ArrowUp':
          e.preventDefault();
          if (!this.open) this._openList(this.typed ? this.input.value : '');
          else if (n) this._setActive(this.active <= 0 ? n - 1 : this.active - 1);
          break;
        case 'PageDown':
          if (this.open && n) { e.preventDefault(); this._setActive(Math.min(n - 1, this.active + 8)); }
          break;
        case 'PageUp':
          if (this.open && n) { e.preventDefault(); this._setActive(Math.max(0, this.active - 8)); }
          break;
        case 'Enter':
          if (this.open && this.active >= 0 && this.items[this.active]) {
            e.preventDefault();
            this._choose(this.items[this.active].value, false);
          } else if (!this.open) {
            e.preventDefault();
            this._openList('');
          }
          break;
        case 'Escape':
          if (this.open) {
            e.preventDefault();
            e.stopPropagation();
            this._close();
            this.typed = false;
            this._showValue();
            this.input.select();
          }
          break;
        case 'Tab':
          if (this.open && this.typed && this.input.value.trim() && this.active >= 0 && this.items[this.active]) {
            this._choose(this.items[this.active].value, false);
          }
          break;
        default:
      }
    }

    _onBlur() {
      if (!this.open && !this.typed) { this._showValue(); return; }
      const text = this.input.value.trim();
      this._close();
      if (this.typed) {
        if (!text && this.cfg.clearable) {
          this.typed = false;
          this._choose(this.cfg.emptyValue, false, true);
          return;
        }
        // exact name typed -> take it
        const t = text.toLowerCase();
        const exact = this.options.find((o) => [o.label, o.sub, this._display(o.value)].some(
          (x) => x && String(x).toLowerCase() === t));
        if (exact) {
          this.typed = false;
          this._choose(exact.value, false, true);
          return;
        }
      }
      this.typed = false;
      this._showValue();
    }

    _choose(value, fromPointer, fromBlur) {
      const changed = value !== this.value;
      this.value = value;
      this.typed = false;
      this._close();
      this._showValue();
      if (fromPointer && global.matchMedia && global.matchMedia('(pointer: coarse)').matches) this.input.blur();
      if (changed && this.cfg.onChange) this.cfg.onChange(value, this);
      return fromBlur;
    }

    setValue(v) {
      this.value = v == null ? this.cfg.emptyValue : v;
      this._showValue();
    }

    setDisabled(d) {
      this.input.disabled = !!d;
      this.el.querySelectorAll('button').forEach((b) => { b.disabled = !!d; });
    }
  }

  Combobox.suppressOpen = false;
  Combobox.score = score;
  Combobox.filter = filterOptions;
  global.Combobox = Combobox;
}(window));
