"""Data access layer (port of Showdown's ``sim/dex*.ts`` for the champions mod).

All battle data is loaded from the JSON files in ``pokechamp/data`` (exported
from Pokemon Showdown's ``champions`` mod by ``tools/export_showdown_data.js``).
Function-valued fields of the original data (event handlers such as
``onBasePower``) are exported as ``{"__fn__": true}`` placeholders and are
filled in with the hand-ported Python implementations from
``pokechamp.sim.data``.
"""
from __future__ import annotations

import inspect
import json
import os
from typing import Any, Callable

from .js import NULL, Obj, deep_clone, obj_hook, to_id, trunc as js_trunc

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

STAT_IDS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')


def _load(name: str) -> Any:
    with open(os.path.join(DATA_DIR, name), encoding='utf-8') as f:
        return json.load(f, object_hook=obj_hook)


# ---------------------------------------------------------------------------
# Effect classes


class Effect(Obj):
    """Base class of all effects (moves, abilities, items, conditions, ...)."""

    __slots__ = ()

    def __str__(self):
        return self.get('name') or ''

    def __repr__(self):
        return f"<{self.get('effectType')}:{self.get('id')}>"


class Species(Effect):
    __slots__ = ()


class Move(Effect):
    __slots__ = ()


class Ability(Effect):
    __slots__ = ()


class Item(Effect):
    __slots__ = ()


class Condition(Effect):
    __slots__ = ()


class Format(Effect):
    __slots__ = ()


class Nature(Effect):
    __slots__ = ()


class TypeInfo(Obj):
    __slots__ = ()


def _make_empty(cls, effect_type: str, **extra) -> Effect:
    e = cls(name='', id='', fullname='', effectType=effect_type, exists=False, num=0, gen=0,
            isNonstandard=None, noCopy=False, affectsFainted=False, sourceEffect='', flags=Obj(), **extra)
    return e


# ---------------------------------------------------------------------------
# Handler attachment


class _Missing:
    """Placeholder for a callback that has not been ported to Python."""

    def __init__(self, where: str):
        self.where = where

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"Showdown callback not ported: {self.where}")


def adapt(fn: Callable) -> Callable:
    """Wrap a handler so it can be called with more positional args than it declares.

    JavaScript silently drops extra arguments; Showdown's event system relies on
    that (handlers are always called with ``(relayVar?, target, source, effect)``).
    Handlers are written ``def onX(self, a, b)`` with ``self`` being the Battle.
    """
    if getattr(fn, '_adapted', False):
        return fn
    code = fn.__code__
    if code.co_flags & inspect.CO_VARARGS:
        fn._adapted = True
        return fn
    n = code.co_argcount - 1

    if n == 0:
        def w(battle, *args):
            return fn(battle)
    elif n == 1:
        def w(battle, a=None, *args):
            return fn(battle, a)
    elif n == 2:
        def w(battle, a=None, b=None, *args):
            return fn(battle, a, b)
    elif n == 3:
        def w(battle, a=None, b=None, c=None, *args):
            return fn(battle, a, b, c)
    else:
        def w(battle, *args):
            args = args[:n]
            return fn(battle, *args)
    w._adapted = True
    w.__name__ = fn.__name__
    w.__qualname__ = getattr(fn, '__qualname__', fn.__name__)
    w.__wrapped__ = fn
    return w


def _attach(data: Obj, handler_cls: type | None, where: str, missing: list[str]) -> None:
    """Replace ``{"__fn__": true}`` placeholders in ``data`` by Python handlers."""
    impl = {}
    if handler_cls is not None:
        for klass in reversed(handler_cls.__mro__[:-1]):
            for k, v in vars(klass).items():
                if k.startswith('__'):
                    continue
                impl[k.rstrip('_')] = v
    for key in list(data.keys()):
        value = data[key]
        if isinstance(value, Obj) and value.get('__fn__') is True:
            fn = impl.get(key)
            if fn is None:
                data[key] = _Missing(f"{where}.{key}")
                missing.append(f"{where}.{key}")
            elif isinstance(fn, (staticmethod, classmethod)):
                data[key] = adapt(fn.__func__)
            else:
                data[key] = adapt(fn)
        elif isinstance(value, Obj):
            sub = impl.get(key)
            _attach(value, sub if isinstance(sub, type) else None, f"{where}.{key}", missing)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, Obj):
                    sub = impl.get(f"{key}_{i}")
                    if sub is None and key == 'secondaries' and i == 0:
                        sub = impl.get('secondary')
                    _attach(item, sub if isinstance(sub, type) else None, f"{where}.{key}[{i}]", missing)
    # handlers defined in Python that have no placeholder in the data
    # (e.g. engine-level additions); attach them as well
    for k, v in impl.items():
        if k in data or isinstance(v, type):
            continue
        if callable(v) and (k.startswith('on') or k.endswith('Callback')):
            data[k] = adapt(v)


def _handler_table(module) -> dict[str, type]:
    table = {}
    for name, value in vars(module).items():
        if isinstance(value, type) and not name.startswith('_'):
            effect_id = getattr(value, 'ID', None) or name.rstrip('_')
            table[effect_id] = value
    return table


# ---------------------------------------------------------------------------
# Dex


class _Table:
    def __init__(self, dex: 'ModdedDex', data: dict, cls: type, empty: Effect):
        self.dex = dex
        self.data = data
        self.cls = cls
        self.empty = empty
        self.cache: dict[str, Effect] = {}

    def get(self, name=None):
        if name is None or name is NULL or name == '':
            return self.empty
        if not isinstance(name, str):
            return name
        return self.getByID(to_id(name))

    def getByID(self, id_: str):
        if not id_:
            return self.empty
        e = self.data.get(id_)
        if e is None:
            alias = ALIASES.get(id_)
            if alias:
                e = self.data.get(to_id(alias))
        if e is None:
            e = self.cls(deep_clone(self.empty))
            e.name = id_
            e.id = id_
            e.exists = False
        return e

    def all(self):
        return list(self.data.values())

    def __contains__(self, id_):
        return id_ in self.data


ALIASES = {
    'rotomw': 'Rotom-Wash', 'rotomh': 'Rotom-Heat', 'rotomf': 'Rotom-Frost', 'rotoms': 'Rotom-Fan',
    'rotomc': 'Rotom-Mow', 'megacharizardx': 'Charizard-Mega-X', 'megacharizardy': 'Charizard-Mega-Y',
}


class TypeTable:
    def __init__(self, chart: dict):
        self.chart = chart
        self.by_id = {to_id(k): k for k in chart}

    def get(self, name):
        if isinstance(name, TypeInfo):
            return name
        tname = self.by_id.get(to_id(name))
        if tname is None:
            return TypeInfo(name=name, id=to_id(name), exists=False, damageTaken=Obj())
        data = self.chart[tname]
        return TypeInfo(name=tname, id=to_id(tname), exists=True, damageTaken=data['damageTaken'],
                        HPivs=data.get('HPivs'), HPdvs=data.get('HPdvs'))

    def isName(self, name):
        return name in self.chart

    def names(self):
        return list(self.chart.keys())

    def all(self):
        return [self.get(n) for n in self.chart]


class ConditionTable:
    def __init__(self, dex: 'ModdedDex'):
        self.dex = dex
        self.cache: dict[str, Condition] = {}
        self.empty = _make_empty(Condition, 'Condition')

    def get(self, name=None):
        if not name:
            return self.empty
        if not isinstance(name, str):
            return name
        if name.startswith('item:') or name.startswith('ability:'):
            return self.getByID(name)
        return self.getByID(to_id(name))

    def getByID(self, id_: str):
        if not id_:
            return self.empty
        c = self.cache.get(id_)
        if c is not None:
            return c
        dex = self.dex
        found = None
        if id_.startswith('item:'):
            item = dex.items.getByID(id_[5:])
            c = Condition(item)
            c.id = 'item:' + item.id
        elif id_.startswith('ability:'):
            ability = dex.abilities.getByID(id_[8:])
            c = Condition(ability)
            c.id = 'ability:' + ability.id
        elif id_ in dex.rulesets:
            c = dex.rulesets[id_]
            self.cache[id_] = c
            return c
        elif id_ in dex.condition_data:
            c = _new_condition(Obj({'name': id_, **dex.condition_data[id_]}))
        elif ((id_ in dex.move_data and (found := dex.move_data[id_]).get('condition') is not None) or
              (id_ in dex.ability_data and (found := dex.ability_data[id_]).get('condition') is not None) or
              (id_ in dex.item_data and (found := dex.item_data[id_]).get('condition') is not None)):
            c = _new_condition(Obj({'name': found.get('name') or id_, **found['condition']}))
        elif id_ == 'recoil':
            c = _new_condition(Obj(name='Recoil', effectType='Recoil'))
        elif id_ == 'drain':
            c = _new_condition(Obj(name='Drain', effectType='Drain'))
        else:
            c = _new_condition(Obj(name=id_, exists=False))
        self.cache[id_] = c
        return c


def _new_condition(data: Obj) -> Condition:
    """Equivalent of ``new Condition(data)`` (BasicEffect constructor semantics)."""
    c = Condition()
    name = (data.get('name') or '').strip() if isinstance(data.get('name'), str) else ''
    c['name'] = name
    c['id'] = to_id(name)
    c['fullname'] = data.get('fullname') or name
    et = data.get('effectType')
    c['effectType'] = et if et in ('Weather', 'Status', 'Terrain') else 'Condition'
    exists = data.get('exists')
    c['exists'] = exists if exists is not None else bool(c['id'])
    c['num'] = data.get('num') or 0
    c['gen'] = data.get('gen') or 0
    c['isNonstandard'] = data.get('isNonstandard') or None
    c['duration'] = data.get('duration')
    c['noCopy'] = bool(data.get('noCopy'))
    c['affectsFainted'] = bool(data.get('affectsFainted'))
    c['status'] = data.get('status') or None
    c['weather'] = data.get('weather') or None
    c['sourceEffect'] = data.get('sourceEffect') or ''
    for k, v in data.items():
        if k not in c:
            c[k] = v
    return c


class RuleTable:
    def __init__(self, fmt: Obj):
        self.rules = dict(fmt.get('rules') or {})
        self.valueRules = dict(fmt.get('valueRules') or {})
        self.pickedTeamSize = fmt.get('pickedTeamSize')
        self.minTeamSize = fmt.get('minTeamSize')
        self.maxTeamSize = fmt.get('maxTeamSize')
        self.adjustLevel = fmt.get('adjustLevel')
        self.defaultLevel = fmt.get('defaultLevel')
        self.maxLevel = fmt.get('maxLevel')
        self.evLimit = fmt.get('evLimit')
        self.maxMoveCount = fmt.get('maxMoveCount')

    def has(self, rule: str) -> bool:
        return rule in self.rules

    def keys(self):
        return list(self.rules.keys())


class ModdedDex:
    """The champions-mod dex. A single shared instance is available as ``Dex``."""

    gen = 9
    currentMod = 'champions'

    def __init__(self):
        from .data import moves as moves_mod, abilities as abilities_mod, items as items_mod
        from .data import conditions as conditions_mod, rulesets as rulesets_mod, species as species_mod

        self.missing: list[str] = []
        self.move_data = _load('moves.json')
        self.ability_data = _load('abilities.json')
        self.item_data = _load('items.json')
        self.species_data = _load('species.json')
        self.condition_data = _load('conditions.json')
        self.typechart = _load('typechart.json')
        self.nature_data = _load('natures.json')
        self.learnsets = _load('learnsets.json')
        self.format_data = _load('formats.json')
        self.meta = _load('meta.json')

        move_h = _handler_table(moves_mod)
        abil_h = _handler_table(abilities_mod)
        item_h = _handler_table(items_mod)
        cond_h = _handler_table(conditions_mod)
        spec_h = _handler_table(species_mod)

        moves = {}
        for id_, d in self.move_data.items():
            _attach(d, move_h.get(id_), f"move:{id_}", self.missing)
            moves[id_] = Move(d)
            moves[id_]['fullname'] = f"move: {d['name']}"
            moves[id_]['effectType'] = 'Move'
        abilities = {}
        for id_, d in self.ability_data.items():
            _attach(d, abil_h.get(id_), f"ability:{id_}", self.missing)
            a = Ability(d)
            a['fullname'] = f"ability: {d['name']}"
            a['effectType'] = 'Ability'
            abilities[id_] = a
        items = {}
        for id_, d in self.item_data.items():
            _attach(d, item_h.get(id_), f"item:{id_}", self.missing)
            i = Item(d)
            i['fullname'] = f"item: {d['name']}"
            i['effectType'] = 'Item'
            items[id_] = i
        for id_, d in self.condition_data.items():
            _attach(d, cond_h.get(id_), f"condition:{id_}", self.missing)
        species = {}
        for id_, d in self.species_data.items():
            _attach(d, spec_h.get(id_), f"species:{id_}", self.missing)
            s = Species(d)
            s['fullname'] = f"pokemon: {d['name']}"
            s['effectType'] = 'Pokemon'
            species[id_] = s
        # keep the raw tables pointing at the effect objects
        self.move_data = moves
        self.ability_data = abilities
        self.item_data = items

        self.moves = _Table(self, moves, Move, _make_empty(Move, 'Move', basePower=0, category='Physical'))
        self.abilities = _Table(self, abilities, Ability, _make_empty(Ability, 'Ability'))
        self.items = _Table(self, items, Item, _make_empty(Item, 'Item'))
        self.species = _Table(self, species, Species, _make_empty(Species, 'Pokemon'))
        self.types = TypeTable(self.typechart)
        self.natures = _Table(self, {k: Nature(v, id=k, effectType='Nature', fullname=f"nature: {v['name']}",
                                               exists=True) for k, v in self.nature_data.items()},
                              Nature, _make_empty(Nature, 'Nature'))

        # rulesets / formats
        self.rulesets: dict[str, Format] = {}
        rules_h = _handler_table(rulesets_mod)
        for rid, cls in rules_h.items():
            f = Format(id=rid, name=getattr(cls, 'NAME', rid), fullname=getattr(cls, 'NAME', rid),
                       effectType=getattr(cls, 'EFFECT_TYPE', 'Rule'), exists=True, num=0, gen=0)
            for k, v in vars(cls).items():
                if k.startswith('on') and callable(v):
                    f[k] = adapt(v)
            self.rulesets[rid] = f
        self.formats = FormatTable(self)
        self.conditions = ConditionTable(self)

    # --- helpers ported from ModdedDex ---------------------------------------

    @staticmethod
    def trunc(num, bits=0):
        return js_trunc(num, bits)

    def getActiveMove(self, move) -> Move:
        if move is not None and isinstance(move, Move) and isinstance(move.get('hit'), int) \
                and not isinstance(move.get('hit'), bool):
            return move
        move = self.moves.get(move)
        move_copy = Move({k: deep_clone(v) for k, v in move.items()})
        move_copy['hit'] = 0
        return move_copy

    def getImmunity(self, source, target) -> bool:
        source_type = source if isinstance(source, str) else source.type
        if isinstance(target, str):
            target_typing = target
        elif isinstance(target, list):
            target_typing = target
        elif hasattr(target, 'getTypes'):
            target_typing = target.getTypes()
        else:
            target_typing = target.types
        if isinstance(target_typing, list):
            for t in target_typing:
                if not self.getImmunity(source_type, t):
                    return False
            return True
        data = self.typechart.get(target_typing)
        if data is not None and data['damageTaken'].get(source_type) == 3:
            return False
        return True

    def getEffectiveness(self, source, target) -> int:
        source_type = source if isinstance(source, str) else source.type
        if isinstance(target, str):
            target_typing = target
        elif isinstance(target, list):
            target_typing = target
        elif hasattr(target, 'getTypes'):
            target_typing = target.getTypes()
        else:
            target_typing = target.types
        if isinstance(target_typing, list):
            total = 0
            for t in target_typing:
                total += self.getEffectiveness(source_type, t)
            return total
        data = self.typechart.get(target_typing)
        if data is None:
            return 0
        v = data['damageTaken'].get(source_type)
        if v == 1:
            return 1
        if v == 2:
            return -1
        return 0

    def getHiddenPower(self, ivs):
        hp_types = ['Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel',
                    'Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark']
        hp_type_x = 0
        i = 1
        for s in ('hp', 'atk', 'def', 'spe', 'spa', 'spd'):
            hp_type_x += i * (ivs[s] % 2)
            i *= 2
        return Obj(type=hp_types[js_trunc(hp_type_x * 15 / 63)], power=60)

    def learnset(self, species_id: str) -> list[str]:
        """Learnable moves of a species (falls back to its base forme)."""
        s = self.species.get(species_id)
        seen = set()
        while s.exists and s.id not in seen:
            seen.add(s.id)
            if s.id in self.learnsets:
                return list(self.learnsets[s.id])
            nxt = s.changesFrom or s.baseSpecies
            if not nxt:
                break
            s = self.species.get(nxt)
        return []


class FormatTable:
    def __init__(self, dex: ModdedDex):
        self.dex = dex
        self.cache: dict[str, Format] = {}

    def get(self, name, is_trusted=False):
        id_ = to_id(name)
        if id_ in self.cache:
            return self.cache[id_]
        if id_ in self.dex.format_data:
            d = self.dex.format_data[id_]
            f = Format(d)
            f['id'] = id_
            f['effectType'] = 'Format'
            f['fullname'] = f"format: {d['name']}"
            f['exists'] = True
            f['ruleTable'] = RuleTable(d)
            self.cache[id_] = f
            return f
        if id_ in self.dex.rulesets:
            return self.dex.rulesets[id_]
        return Format(id=id_, name=name, exists=False, effectType='Format')

    def getRuleTable(self, fmt):
        return fmt.ruleTable


_DEX: ModdedDex | None = None


def get_dex() -> ModdedDex:
    global _DEX
    if _DEX is None:
        _DEX = ModdedDex()
    return _DEX


def make_move(data) -> Move:
    """``new Dex.Move(data)``: build a move from raw data applying DataMove defaults."""
    d = Obj(deep_clone(data))
    name = (d.get('name') or '').strip()
    m = Move(d)
    m['name'] = name
    m['id'] = to_id(d.get('placeholderFor') or name) if d.get('placeholderFor') else (d.get('id') or to_id(name))
    m['fullname'] = f"move: {name}"
    m['effectType'] = 'Move'
    m['exists'] = d.get('exists') if d.get('exists') is not None else bool(m['id'])
    m['type'] = d.get('type') or ''
    m['basePower'] = d.get('basePower') or 0
    m['critRatio'] = d.get('critRatio') or 1
    if d.get('secondaries') is None and d.get('secondary'):
        m['secondaries'] = [d['secondary']]
    m['priority'] = d.get('priority') or 0
    m['ignoreImmunity'] = d['ignoreImmunity'] if d.get('ignoreImmunity') is not None else d.get('category') == 'Status'
    m['flags'] = d.get('flags') or Obj()
    m['noPPBoosts'] = bool(d.get('noPPBoosts'))
    return m
