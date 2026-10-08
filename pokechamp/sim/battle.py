"""Port of Showdown's ``sim/battle.ts`` (champions mod, gen 9 code paths).

The most important part is the event system (:meth:`Battle.runEvent` and
:meth:`Battle.singleEvent`). Method names deliberately mirror Showdown's so
that the hand-ported effect handlers in ``pokechamp.sim.data`` read like the
original TypeScript (``this.`` -> ``self.``).
"""
from __future__ import annotations

import functools
import json
import math
import re
import time
from typing import Any, Callable

from .dex import Condition, Effect, Format, ModdedDex, get_dex
from .js import (NULL, Obj, clamp_int_range, deep_clone, is_number, join_part, js_round,
                 js_str, js_typeof, to_id, to_int32, trunc, truthy)
from .prng import PRNG
from .battle_actions import BattleActions
from .battle_queue import BattleQueue
from .field import Field
from .pokemon import RESTORATIVE_BERRIES, Pokemon
from .side import Side

__all__ = ['Battle', 'BattleError']


class BattleError(Exception):
    pass


def _end_clear_status(holder, *args):
    return holder.clearStatus()


def _end_remove_volatile(holder, effect_id=None, *args):
    return holder.removeVolatile(effect_id)


def _end_clear_ability(holder, *args):
    return holder.clearAbility()


def _end_clear_item(holder, *args):
    return holder.clearItem()


def _end_noop(*args):
    return None


def _end_remove_side_condition(holder, effect_id=None, *args):
    return holder.removeSideCondition(effect_id)


def _end_remove_slot_condition(side, pokemon, effect_id, *args):
    return side.removeSlotCondition(pokemon, effect_id)


def _end_remove_pseudo_weather(holder, effect_id=None, *args):
    return holder.removePseudoWeather(effect_id)


def _end_clear_weather(holder, *args):
    return holder.clearWeather()


def _end_clear_terrain(holder, *args):
    return holder.clearTerrain()


_EFFECT_TYPE_ORDER = {
    'Condition': 2,
    'Weather': 5,
    'Format': 5,
    'Rule': 5,
    'Ruleset': 5,
    'Ability': 7,
    'Item': 8,
}

_ATTACKING_EVENTS = {
    'BeforeMove', 'BasePower', 'Immunity', 'RedirectTarget', 'Heal', 'SetStatus', 'CriticalHit',
    'ModifyAtk', 'ModifyDef', 'ModifySpA', 'ModifySpD', 'ModifySpe', 'ModifyAccuracy',
    'ModifyBoost', 'ModifyDamage', 'ModifySecondaries', 'ModifyWeight', 'TryAddVolatile',
    'TryHit', 'TryHitSide', 'TryMove', 'Boost', 'DragOut', 'Effectiveness',
}

_NON_PREFIXED_EVENTS = {'BeforeTurn', 'Update', 'Weather', 'WeatherChange', 'TerrainChange'}

_ORDER_DEFAULT = 4294967296


def _or0(v):
    return v if v else 0


class Handler:
    """An event listener (Showdown's EventListener)."""

    __slots__ = ('effect', 'callback', 'state', 'end', 'endCallArgs', 'effectHolder', 'order', 'priority',
                 'subOrder', 'effectOrder', 'speed', 'index', 'target')

    def __init__(self, effect, callback, state, end, effectHolder, endCallArgs=None):
        self.effect = effect
        self.callback = callback
        self.state = state
        self.end = end
        self.endCallArgs = endCallArgs
        self.effectHolder = effectHolder
        self.order = False
        self.priority = 0
        self.subOrder = 0
        self.effectOrder = None
        self.speed = None
        self.index = None
        self.target = None

    def get(self, name, default=None):
        return getattr(self, name, default)


def compare_priority(a, b):
    """``Battle.comparePriority`` (order asc, priority desc, speed desc, subOrder asc, effectOrder asc)."""
    ao = a.get('order') or _ORDER_DEFAULT
    bo = b.get('order') or _ORDER_DEFAULT
    r = -(bo - ao)
    if r:
        return r
    r = _or0(b.get('priority')) - _or0(a.get('priority'))
    if r:
        return r
    r = _or0(b.get('speed')) - _or0(a.get('speed'))
    if r:
        return r
    r = -(_or0(b.get('subOrder')) - _or0(a.get('subOrder')))
    if r:
        return r
    r = -(_or0(b.get('effectOrder')) - _or0(a.get('effectOrder')))
    if r:
        return r
    return 0


def compare_redirect_order(a, b):
    r = _or0(b.get('priority')) - _or0(a.get('priority'))
    if r:
        return r
    r = _or0(b.get('speed')) - _or0(a.get('speed'))
    if r:
        return r
    ah = a.get('effectHolder')
    bh = b.get('effectHolder')
    a_state = getattr(ah, 'abilityState', None) if ah is not None else None
    b_state = getattr(bh, 'abilityState', None) if bh is not None else None
    if a_state is not None and b_state is not None:
        r = -(b_state.effectOrder - a_state.effectOrder)
        if r:
            return r
    return 0


def compare_left_to_right_order(a, b):
    ao = a.get('order') or _ORDER_DEFAULT
    bo = b.get('order') or _ORDER_DEFAULT
    r = -(bo - ao)
    if r:
        return r
    r = _or0(b.get('priority')) - _or0(a.get('priority'))
    if r:
        return r
    r = -(_or0(b.get('index')) - _or0(a.get('index')))
    if r:
        return r
    return 0


def _stable_sort(lst, cmp):
    """``Array#sort`` with a comparator (V8's TimSort is stable)."""
    lst.sort(key=functools.cmp_to_key(cmp))


class Battle:
    NOT_FAIL = ''
    HIT_SUBSTITUTE = 0
    FAIL = False
    SILENT_FAIL = NULL

    def __init__(self, formatid: str = 'gen9championsbssregmc', seed=None, p1=None, p2=None,
                 send: Callable | None = None, strict_choices: bool = False, debug: bool = False,
                 timestamp: int | None = None, dex: ModdedDex | None = None):
        from .battle_actions import BattleActions
        from .battle_queue import BattleQueue

        self.log: list[str] = []
        self._timestamp = timestamp
        self.add('t:', self._now())

        self.dex = dex or get_dex()
        fmt = self.dex.formats.get(formatid)
        if not fmt.exists:
            raise BattleError(f"Unknown format {formatid}")
        self.format: Format = fmt
        self.gen = 9
        self.ruleTable = fmt.ruleTable
        self.id = ''
        self.debugMode = bool(fmt.get('debug')) or debug
        self.forceRandomChance = None
        self.deserialized = False
        self.strictChoices = strict_choices
        self.formatData = self.initEffectState(Obj(id=fmt.id))
        self.gameType = fmt.get('gameType') or 'singles'
        self.field = Field(self)
        self.sides: list = [None, None]
        self.activePerHalf = 2 if self.gameType == 'doubles' else (3 if self.gameType == 'triples' else 1)
        self.prng = seed if isinstance(seed, PRNG) else PRNG(seed or None)
        self.prngSeed = self.prng.starting_seed
        self.rated = False
        self.reportExactHP = bool(fmt.get('debug'))
        self.reportPercentages = False
        self.supportCancel = False

        self.queue = BattleQueue(self)
        self.actions = BattleActions(self)
        self.faintQueue: list[Obj] = []

        self.inputLog: list[str] = []
        self.messageLog: list[str] = []
        self.sentLogPos = 0
        self.sentEnd = False
        self.sentRequests = True

        self.requestState = ''
        self.turn = 0
        self.midTurn = False
        self.started = False
        self.ended = False
        self.winner = None

        self.effect = Obj(id='')
        self.effectState = self.initEffectState(Obj(id=''))
        self.event = Obj(id='')
        self.events = None
        self.eventDepth = 0

        self.activeMove = None
        self.activePokemon = None
        self.activeTarget = None
        self.lastMove = None
        self.lastMoveLine = -1
        self.lastSuccessfulMoveThisTurn = None
        self.lastDamage = 0
        self.effectOrder = 0
        self.quickClawRoll = False
        self.speedOrder = list(range(self.activePerHalf * 2))
        self.teamGenerator = None
        self.hints: set[str] = set()
        self.send = send or (lambda type_, data: None)
        self.trunc = trunc
        self.clampIntRange = clamp_int_range
        self.toID = to_id

        self.inputLog.append('>start ' + json.dumps({'formatid': formatid, 'seed': self.prngSeed},
                                                    separators=(',', ':')))
        self.add('gametype', self.gameType)

        # rules with battle event handlers are attached as pseudo-weathers
        for rule in self.ruleTable.keys():
            if rule[:1] in '+*-!':
                continue
            sub = self.dex.formats.get(rule)
            if sub.exists:
                has_handler = any(
                    k.startswith('on') and k not in (
                        'onBegin', 'onTeamPreview', 'onBattleStart', 'onValidateRule', 'onValidateTeam',
                        'onChangeSet', 'onValidateSet')
                    for k in sub.keys())
                if has_handler:
                    self.field.addPseudoWeather(rule)

        if p1 is not None:
            self.setPlayer('p1', p1)
        if p2 is not None:
            self.setPlayer('p2', p2)

    # ------------------------------------------------------------------
    def _now(self):
        return self._timestamp if self._timestamp is not None else int(time.time())

    @property
    def p1(self):
        return self.sides[0]

    @property
    def p2(self):
        return self.sides[1]

    def __str__(self):
        return f"Battle: {self.format.name}"

    def random(self, m=None, n=None):
        return self.prng.random(m, n)

    def randomChance(self, numerator, denominator):
        if self.forceRandomChance is not None:
            return self.forceRandomChance
        return self.prng.random_chance(numerator, denominator)

    def sample(self, items):
        return self.prng.sample(items)

    def resetRNG(self, seed=None):
        self.prng = PRNG(seed if seed is not None else self.prngSeed)
        self.add('message', "The battle's RNG was reset.")

    def suppressingAbility(self, target=None):
        ap = self.activePokemon
        am = self.activeMove
        return bool(ap and ap.isActive and (ap is not target) and am and am.ignoreAbility and
                    not (target is not None and target.hasItem('Ability Shield')))

    def suppressingSecondaries(self):
        am = self.activeMove
        return bool(am and am.hasSheerForce and self.activePokemon and
                    self.activePokemon.hasAbility('sheerforce'))

    def setActiveMove(self, move=None, pokemon=None, target=None):
        self.activeMove = move or None
        self.activePokemon = pokemon or None
        self.activeTarget = target or pokemon or None

    def clearActiveMove(self, failed=False):
        if self.activeMove:
            if not failed:
                self.lastMove = self.activeMove
            self.activeMove = None
            self.activePokemon = None
            self.activeTarget = None

    def updateSpeed(self):
        for pokemon in self.getAllActive():
            pokemon.updateSpeed()

    comparePriority = staticmethod(compare_priority)

    def speedSort(self, lst: list, comparator: Callable = compare_priority):
        if len(lst) < 2:
            return
        sorted_ = 0
        n = len(lst)
        while sorted_ + 1 < n:
            next_indexes = [sorted_]
            for i in range(sorted_ + 1, n):
                delta = comparator(lst[next_indexes[0]], lst[i])
                if delta < 0:
                    continue
                if delta > 0:
                    next_indexes = [i]
                if delta == 0:
                    next_indexes.append(i)
            for i, index in enumerate(next_indexes):
                if index != sorted_ + i:
                    lst[sorted_ + i], lst[index] = lst[index], lst[sorted_ + i]
            if len(next_indexes) > 1:
                self.prng.shuffle(lst, sorted_, sorted_ + len(next_indexes))
            sorted_ += len(next_indexes)

    def eachEvent(self, eventid: str, effect=None, relayVar=None):
        actives = self.getAllActive()
        if not effect and self.effect:
            effect = self.effect
        self.speedSort(actives, lambda a, b: b.speed - a.speed)
        for pokemon in actives:
            self.runEvent(eventid, pokemon, None, effect, relayVar)
        if eventid == 'Weather':
            self.eachEvent('Update')

    def fieldEvent(self, eventid: str, targets=None):
        callback_name = 'on' + eventid
        get_key = 'duration' if eventid == 'Residual' else None
        handlers = self.findFieldEventHandlers(self.field, f"onField{eventid}", get_key)
        for side in self.sides:
            if side.n < 2 or not side.allySide:
                handlers += self.findSideEventHandlers(side, f"onSide{eventid}", get_key)
            for active in side.active:
                if not active:
                    continue
                if eventid == 'SwitchIn':
                    handlers += self.findPokemonEventHandlers(active, f"onAny{eventid}")
                if targets is not None and active not in targets:
                    continue
                handlers += self.findPokemonEventHandlers(active, callback_name, get_key)
                handlers += self.findSideEventHandlers(side, callback_name, None, active)
                handlers += self.findFieldEventHandlers(self.field, callback_name, None, active)
                handlers += self.findBattleEventHandlers(callback_name, get_key, active)
        self.speedSort(handlers)
        while handlers:
            handler = handlers.pop(0)
            effect = handler.effect
            holder = handler.effectHolder
            state = handler.state
            if isinstance(holder, Pokemon) and holder.fainted:
                if not (state is not None and state.isSlotCondition):
                    continue
            if eventid == 'Residual' and handler.end and state is not None and state.duration:
                state.duration -= 1
                if not state.duration:
                    end_args = handler.endCallArgs or [holder, effect.id]
                    handler.end(*end_args)
                    if self.ended:
                        return
                    continue
            # effect may have been removed by a prior handler
            st_target = state.target if state is not None else None
            if isinstance(st_target, Pokemon):
                et = effect.effectType
                if et == 'Ability' and not state.id.startswith('ability:'):
                    expected = st_target.abilityState
                elif et == 'Item' and not state.id.startswith('item:'):
                    expected = st_target.itemState
                elif et == 'Status':
                    expected = st_target.statusState
                else:
                    expected = st_target.volatiles.get(effect.id)
                if expected is not state:
                    continue
            elif isinstance(st_target, Side) and not state.isSlotCondition:
                if st_target.sideConditions.get(effect.id) is not state:
                    continue
            elif isinstance(st_target, Field):
                et = effect.effectType
                if et == 'Weather':
                    expected = st_target.weatherState
                elif et == 'Terrain':
                    expected = st_target.terrainState
                else:
                    expected = st_target.pseudoWeather.get(effect.id)
                if expected is not state:
                    continue
            handler_eventid = eventid
            if isinstance(holder, Side):
                handler_eventid = 'Side' + eventid
            if isinstance(holder, Field):
                handler_eventid = 'Field' + eventid
            if handler.callback is not None:
                self.singleEvent(handler_eventid, effect, state, holder, None, None, None, handler.callback)
            self.faintMessages()
            if self.ended:
                return

    def singleEvent(self, eventid: str, effect, state, target, source=None, sourceEffect=None,
                    relayVar=None, customCallback=None):
        if self.eventDepth >= 8:
            self.add('message', 'STACK LIMIT EXCEEDED')
            self.add('message', 'PLEASE REPORT IN BUG THREAD')
            self.add('message', 'Event: ' + eventid)
            self.add('message', 'Parent event: ' + js_str(self.event.id))
            raise BattleError('Stack overflow')
        if len(self.log) - self.sentLogPos > 1000:
            self.add('message', 'LINE LIMIT EXCEEDED')
            self.add('message', 'PLEASE REPORT IN BUG THREAD')
            self.add('message', 'Event: ' + eventid)
            self.add('message', 'Parent event: ' + js_str(self.event.id))
            raise BattleError('Infinite loop')
        has_relay_var = True
        if relayVar is None:
            relayVar = True
            has_relay_var = False

        et = effect.effectType
        if et == 'Status' and isinstance(target, Pokemon) and target.status != effect.id:
            return relayVar
        if (eventid == 'SwitchIn' and et == 'Ability' and effect.flags and effect.flags.get('breakable') and
                self.suppressingAbility(target)):
            self.debug(eventid + ' handler suppressed by Mold Breaker')
            return relayVar
        if (eventid not in ('Start', 'TakeItem', 'SetAbility') and et == 'Item' and
                isinstance(target, Pokemon) and target.ignoringItem()):
            self.debug(eventid + ' handler suppressed by Embargo, Klutz or Magic Room')
            return relayVar
        if eventid != 'End' and et == 'Ability' and isinstance(target, Pokemon) and target.ignoringAbility():
            self.debug(eventid + ' handler suppressed by Gastro Acid or Neutralizing Gas')
            return relayVar
        if (et == 'Weather' and eventid not in ('FieldStart', 'FieldResidual', 'FieldEnd') and
                self.field.suppressingWeather()):
            self.debug(eventid + ' handler suppressed by Air Lock')
            return relayVar

        callback = customCallback if customCallback is not None else effect.get('on' + eventid)
        if callback is None:
            return relayVar

        parent_effect = self.effect
        parent_effect_state = self.effectState
        parent_event = self.event

        self.effect = effect
        self.effectState = state if state is not None else self.initEffectState(Obj())
        self.event = Obj(id=eventid, target=target, source=source, effect=sourceEffect)
        self.eventDepth += 1

        try:
            if callable(callback):
                if has_relay_var:
                    return_val = callback(self, relayVar, target, source, sourceEffect)
                else:
                    return_val = callback(self, target, source, sourceEffect)
            else:
                return_val = callback
        finally:
            self.eventDepth -= 1
            self.effect = parent_effect
            self.effectState = parent_effect_state
            self.event = parent_event

        return relayVar if return_val is None else return_val

    def runEvent(self, eventid: str, target=None, source=None, sourceEffect=None, relayVar=None,
                 onEffect=False, fastExit=False):
        if self.eventDepth >= 8:
            self.add('message', 'STACK LIMIT EXCEEDED')
            self.add('message', 'PLEASE REPORT IN BUG THREAD')
            self.add('message', 'Event: ' + eventid)
            self.add('message', 'Parent event: ' + js_str(self.event.id))
            raise BattleError('Stack overflow')
        if not target:
            target = self
        effect_source = source if isinstance(source, Pokemon) else None
        handlers = self.findEventHandlers(target, eventid, effect_source)
        if onEffect:
            if not sourceEffect:
                raise BattleError('onEffect passed without an effect')
            callback = sourceEffect.get('on' + eventid)
            if callback is not None:
                if isinstance(target, list):
                    raise BattleError('')
                handlers.insert(0, self.resolvePriority(Handler(
                    sourceEffect, callback, self.initEffectState(Obj()), None, target), 'on' + eventid))

        if eventid in ('Invulnerability', 'TryHit', 'DamagingHit', 'EntryHazard'):
            _stable_sort(handlers, compare_left_to_right_order)
        elif fastExit:
            _stable_sort(handlers, compare_redirect_order)
        else:
            self.speedSort(handlers)

        has_relay_var = 1
        args = [target, source, sourceEffect]
        if relayVar is None or relayVar is NULL:
            relayVar = True
            has_relay_var = 0
        else:
            args.insert(0, relayVar)

        parent_event = self.event
        self.event = Obj(id=eventid, target=target, source=source, effect=sourceEffect, modifier=1)
        self.eventDepth += 1

        target_relay_vars = []
        if isinstance(target, list):
            if isinstance(relayVar, list):
                target_relay_vars = relayVar
            else:
                target_relay_vars = [True] * len(target)

        try:
            for handler in handlers:
                if handler.index is not None:
                    trv = target_relay_vars[handler.index]
                    if not trv and not (is_number(trv) and trv == 0 and eventid == 'DamagingHit'):
                        continue
                    if handler.target:
                        args[has_relay_var] = handler.target
                        self.event.target = handler.target
                    if has_relay_var:
                        args[0] = trv
                effect = handler.effect
                effect_holder = handler.effectHolder
                et = effect.effectType
                if et == 'Status' and effect_holder.status != effect.id:
                    continue
                if et == 'Ability' and effect.flags and effect.flags.get('breakable') and \
                        self.suppressingAbility(effect_holder):
                    self.debug(eventid + ' handler suppressed by Mold Breaker')
                    continue
                if (eventid not in ('Start', 'SwitchIn', 'TakeItem') and et == 'Item' and
                        isinstance(effect_holder, Pokemon) and effect_holder.ignoringItem()):
                    if eventid != 'Update':
                        self.debug(eventid + ' handler suppressed by Embargo, Klutz or Magic Room')
                    continue
                elif (eventid != 'End' and et == 'Ability' and isinstance(effect_holder, Pokemon) and
                      effect_holder.ignoringAbility()):
                    if eventid != 'Update':
                        self.debug(eventid + ' handler suppressed by Gastro Acid or Neutralizing Gas')
                    continue
                if ((et == 'Weather' or eventid == 'Weather') and eventid not in ('Residual', 'End') and
                        self.field.suppressingWeather()):
                    self.debug(eventid + ' handler suppressed by Air Lock')
                    continue
                callback = handler.callback
                if callable(callback):
                    parent_effect = self.effect
                    parent_effect_state = self.effectState
                    self.effect = handler.effect
                    self.effectState = handler.state if handler.state is not None else self.initEffectState(Obj())
                    self.effectState.target = effect_holder
                    try:
                        return_val = callback(self, *args)
                    finally:
                        self.effect = parent_effect
                        self.effectState = parent_effect_state
                else:
                    return_val = callback

                if return_val is not None:
                    relayVar = return_val
                    if not truthy(relayVar) or fastExit:
                        if handler.index is not None:
                            target_relay_vars[handler.index] = relayVar
                            if all(not truthy(v) for v in target_relay_vars):
                                break
                        else:
                            break
                    if has_relay_var:
                        args[0] = relayVar
        finally:
            self.eventDepth -= 1
        if is_number(relayVar) and relayVar == abs(math.floor(relayVar)):
            relayVar = self.modify(relayVar, self.event.modifier)
        self.event = parent_event

        return target_relay_vars if isinstance(target, list) else relayVar

    def priorityEvent(self, eventid, target, source=None, effect=None, relayVar=None, onEffect=False):
        return self.runEvent(eventid, target, source, effect, relayVar, onEffect, True)

    def resolvePriority(self, handler: Obj, callbackName: str) -> Obj:
        effect = handler.effect
        handler.order = effect.get(callbackName + 'Order') or False
        handler.priority = effect.get(callbackName + 'Priority') or 0
        handler.subOrder = effect.get(callbackName + 'SubOrder') or 0
        if not handler.subOrder:
            et = effect.effectType
            handler.subOrder = _EFFECT_TYPE_ORDER.get(et, 0)
            if et == 'Condition':
                st = handler.state
                st_target = st.target if st is not None else None
                if isinstance(st_target, Side):
                    handler.subOrder = 3 if st.isSlotCondition else 4
                elif isinstance(st_target, Field):
                    handler.subOrder = 5
            elif et == 'Ability':
                if effect.name in ('Poison Touch', 'Perish Body'):
                    handler.subOrder = 6
                elif effect.name == 'Stall':
                    handler.subOrder = 9
        if callbackName.endswith('SwitchIn') or callbackName.endswith('RedirectTarget'):
            handler.effectOrder = handler.state.effectOrder if handler.state is not None else None
        holder = handler.effectHolder
        if isinstance(holder, Pokemon):
            handler.speed = holder.speed
            if effect.effectType == 'Ability' and effect.name == 'Magic Bounce' and \
                    callbackName == 'onAllyTryHitSide':
                handler.speed = holder.getStat('spe', True, True)
            if callbackName.endswith('SwitchIn'):
                handler.speed -= self.speedOrder.index(holder.getFieldPositionValue()) / (self.activePerHalf * 2)
        return handler

    def getCallback(self, target, effect, callbackName: str):
        callback = effect.get(callbackName)
        if (callback is None and callbackName == 'onSwitchIn' and isinstance(target, Pokemon) and
                effect.get('onAnySwitchIn') is None and
                (effect.get('effectType') in ('Ability', 'Item') or
                 (effect.get('effectType') == 'Status' and effect.id.split(':')[0] in ('ability', 'item')))):
            callback = effect.get('onStart')
        return callback

    def findEventHandlers(self, target, eventName: str, source=None) -> list:
        handlers = []
        if isinstance(target, list):
            for i, pokemon in enumerate(target):
                cur = self.findEventHandlers(pokemon, eventName, source)
                for h in cur:
                    h.target = pokemon
                    h.index = i
                handlers += cur
            return handlers
        should_bubble_down = isinstance(target, Side)
        prefixed = eventName not in _NON_PREFIXED_EVENTS
        if isinstance(target, Pokemon) and (target.isActive or (source is not None and source.isActive)):
            handlers = self.findPokemonEventHandlers(target, 'on' + eventName)
            if prefixed:
                for ally in target.alliesAndSelf():
                    handlers += self.findPokemonEventHandlers(ally, 'onAlly' + eventName)
                    handlers += self.findPokemonEventHandlers(ally, 'onAny' + eventName)
                for foe in target.foes():
                    handlers += self.findPokemonEventHandlers(foe, 'onFoe' + eventName)
                    handlers += self.findPokemonEventHandlers(foe, 'onAny' + eventName)
            target = target.side
        if source is not None and prefixed:
            handlers += self.findPokemonEventHandlers(source, 'onSource' + eventName)
        if isinstance(target, Side):
            for side in self.sides:
                if should_bubble_down:
                    for active in side.active:
                        if side is target or side is target.allySide:
                            handlers += self.findPokemonEventHandlers(active, 'on' + eventName)
                        elif prefixed:
                            handlers += self.findPokemonEventHandlers(active, 'onFoe' + eventName)
                        if prefixed:
                            handlers += self.findPokemonEventHandlers(active, 'onAny' + eventName)
                if side.n < 2 or not side.allySide:
                    if side is target or side is target.allySide:
                        handlers += self.findSideEventHandlers(side, 'on' + eventName)
                    elif prefixed:
                        handlers += self.findSideEventHandlers(side, 'onFoe' + eventName)
                    if prefixed:
                        handlers += self.findSideEventHandlers(side, 'onAny' + eventName)
        handlers += self.findFieldEventHandlers(self.field, 'on' + eventName)
        handlers += self.findBattleEventHandlers('on' + eventName)
        return handlers

    def findPokemonEventHandlers(self, pokemon, callbackName: str, getKey=None) -> list:
        handlers = []
        # Fast path: no effect anywhere defines this callback (and no Residual duration bookkeeping).
        if (getKey is None and callbackName not in self.dex.callback_names and
                not (callbackName == 'onSwitchIn')):
            return handlers
        getCallback = self.getCallback
        resolve = self.resolvePriority
        status = pokemon.getStatus()
        callback = getCallback(pokemon, status, callbackName)
        if callback is not None or (getKey and pokemon.statusState.get(getKey)):
            handlers.append(resolve(Handler(status, callback, pokemon.statusState, _end_clear_status, pokemon),
                                    callbackName))
        if pokemon.volatiles:
            get_condition = self.dex.conditions.getByID
            for id_, volatile_state in list(pokemon.volatiles.items()):
                volatile = get_condition(id_)
                callback = getCallback(pokemon, volatile, callbackName)
                if callback is not None or (getKey and volatile_state.get(getKey)):
                    handlers.append(resolve(Handler(volatile, callback, volatile_state, _end_remove_volatile,
                                                    pokemon), callbackName))
        ability = pokemon.getAbility()
        callback = getCallback(pokemon, ability, callbackName)
        if callback is not None or (getKey and pokemon.abilityState.get(getKey)):
            handlers.append(resolve(Handler(ability, callback, pokemon.abilityState, _end_clear_ability, pokemon),
                                    callbackName))
        item = pokemon.getItem()
        callback = getCallback(pokemon, item, callbackName)
        if callback is not None or (getKey and pokemon.itemState.get(getKey)):
            handlers.append(resolve(Handler(item, callback, pokemon.itemState, _end_clear_item, pokemon),
                                    callbackName))
        species = pokemon.baseSpecies
        callback = getCallback(pokemon, species, callbackName)
        if callback is not None:
            handlers.append(resolve(Handler(species, callback, pokemon.speciesState, _end_noop, pokemon),
                                    callbackName))
        side = pokemon.side
        # JS: `for (id in side.slotConditions[pokemon.position])` is a no-op for benched Pokemon
        if pokemon.position < len(side.slotConditions):
            slot_conds = side.slotConditions[pokemon.position]
            if slot_conds:
                for cond_id, slot_state in list(slot_conds.items()):
                    slot_condition = self.dex.conditions.getByID(cond_id)
                    callback = getCallback(pokemon, slot_condition, callbackName)
                    if callback is not None or (getKey and slot_state.get(getKey)):
                        handlers.append(resolve(Handler(slot_condition, callback, slot_state,
                                                        _end_remove_slot_condition, pokemon,
                                                        [side, pokemon, slot_condition.id]), callbackName))
        return handlers

    def findBattleEventHandlers(self, callbackName: str, getKey=None, customHolder=None) -> list:
        handlers = []
        if getKey is None and not self.events and callbackName not in self.dex.callback_names:
            return handlers
        fmt = self.format
        callback = self.getCallback(self, fmt, callbackName)
        if callback is not None or (getKey and self.formatData.get(getKey)):
            handlers.append(self.resolvePriority(Handler(
                fmt, callback, self.formatData, None, customHolder or self), callbackName))
        if self.events and self.events.get(callbackName) is not None:
            for handler in self.events[callbackName]:
                state = self.formatData if handler.target.effectType == 'Format' else None
                h = Handler(handler.target, handler.callback, state, None, customHolder or self)
                h.priority = handler.priority
                h.order = handler.order
                h.subOrder = handler.subOrder
                handlers.append(h)
        return handlers

    def findFieldEventHandlers(self, field, callbackName: str, getKey=None, customHolder=None) -> list:
        handlers = []
        if getKey is None and callbackName not in self.dex.callback_names:
            return handlers
        for id_, pw_state in list(field.pseudoWeather.items()):
            pseudo_weather = self.dex.conditions.getByID(id_)
            callback = self.getCallback(field, pseudo_weather, callbackName)
            if callback is not None or (getKey and pw_state.get(getKey)):
                handlers.append(self.resolvePriority(Handler(
                    pseudo_weather, callback, pw_state, None if customHolder else _end_remove_pseudo_weather,
                    customHolder or field), callbackName))
        weather = field.getWeather()
        callback = self.getCallback(field, weather, callbackName)
        if callback is not None or (getKey and self.field.weatherState.get(getKey)):
            handlers.append(self.resolvePriority(Handler(
                weather, callback, self.field.weatherState, None if customHolder else _end_clear_weather,
                customHolder or field), callbackName))
        terrain = field.getTerrain()
        callback = self.getCallback(field, terrain, callbackName)
        if callback is not None or (getKey and field.terrainState.get(getKey)):
            handlers.append(self.resolvePriority(Handler(
                terrain, callback, field.terrainState, None if customHolder else _end_clear_terrain,
                customHolder or field), callbackName))
        return handlers

    def findSideEventHandlers(self, side, callbackName: str, getKey=None, customHolder=None) -> list:
        handlers = []
        if not side.sideConditions or (getKey is None and callbackName not in self.dex.callback_names):
            return handlers
        for id_, data in list(side.sideConditions.items()):
            side_condition = self.dex.conditions.getByID(id_)
            callback = self.getCallback(side, side_condition, callbackName)
            if callback is not None or (getKey and data.get(getKey)):
                handlers.append(self.resolvePriority(Handler(
                    side_condition, callback, data, None if customHolder else _end_remove_side_condition,
                    customHolder or side), callbackName))
        return handlers

    def onEvent(self, eventid: str, target, *rest):
        if not rest:
            raise TypeError('Event handlers must have a callback')
        if len(rest) == 1:
            callback, = rest
            priority, order, sub_order = 0, False, 0
        else:
            data, callback = rest
            if isinstance(data, dict):
                priority = data.get('priority') or 0
                order = data.get('order') or False
                sub_order = data.get('subOrder') or 0
            else:
                priority, order, sub_order = data or 0, False, 0
        if self.events is None:
            self.events = Obj()
        name = 'on' + eventid
        h = Obj(callback=callback, target=target, priority=priority, order=order, subOrder=sub_order)
        self.events.setdefault(name, []).append(h)

    def checkMoveMakesContact(self, move, attacker, defender, announcePads=False):
        if move.flags.get('contact') and attacker.hasItem('protectivepads'):
            if announcePads:
                self.add('-activate', defender, self.effect.fullname)
                self.add('-activate', attacker, 'item: Protective Pads')
            return False
        return bool(move.flags.get('contact'))

    def checkMoveBypassesProtect(self, move, attacker, defender, blockStatus=True):
        if ((move.category != 'Status' or blockStatus) and move.flags.get('protect') and
                self.runEvent('HitProtect', attacker, defender, move)):
            return False
        if move.isZOrMaxPowered and move.id not in ('gmaxoneblow', 'gmaxrapidflow'):
            defender.getMoveHitData(move).bypassProtect = True
        return True

    def skillSwap(self, source, target):
        if source.fainted or target.fainted:
            return False
        if source.volatiles.get('dynamax') or target.volatiles.get('dynamax'):
            return False
        source_ability = source.getAbility()
        target_ability = target.getAbility()
        if source_ability.flags.get('failskillswap') or target_ability.flags.get('failskillswap'):
            return False
        source_effect = self.dex.conditions.get('skillswap')
        target_can_be_set = self.runEvent('SetAbility', target, source, source_effect, source_ability)
        if not target_can_be_set:
            return target_can_be_set
        source_can_be_set = self.runEvent('SetAbility', source, source, source_effect, target_ability)
        if not source_can_be_set:
            return source_can_be_set
        if source.isAlly(target):
            self.add('-activate', source, 'Skill Swap', '', '', f"[of] {target}")
            self.hint("Skill Swap does not announce the abilities of the Pokémon when used between allies.")
        else:
            self.add('-activate', source, 'Skill Swap', target_ability.name, source_ability.name, f"[of] {target}")
        self.singleEvent('End', source_ability, source.abilityState, source)
        self.singleEvent('End', target_ability, target.abilityState, target)
        source.ability = target_ability.id
        target.ability = source_ability.id
        source.abilityState = self.initEffectState(Obj(id=to_id(source.ability), target=source))
        target.abilityState = self.initEffectState(Obj(id=to_id(target.ability), target=target))
        source.volatileStaleness = None
        if not source.isAlly(target):
            target.volatileStaleness = 'external'
        self.singleEvent('Start', source_ability, target.abilityState, target)
        self.singleEvent('Start', target_ability, source.abilityState, source)

    # ------------------------------------------------------------------
    def getPokemon(self, fullname):
        if not isinstance(fullname, str):
            fullname = fullname.fullname
        for side in self.sides:
            for pokemon in side.pokemon:
                if pokemon.fullname == fullname:
                    return pokemon
        return None

    def getAllPokemon(self):
        out = []
        for side in self.sides:
            out += side.pokemon
        return out

    def getAllActive(self, includeFainted=False):
        out = []
        for side in self.sides:
            for pokemon in side.active:
                if pokemon and (includeFainted or not pokemon.fainted):
                    out.append(pokemon)
        return out

    def makeRequest(self, type_=None):
        if type_:
            self.requestState = type_
            for side in self.sides:
                side.clearChoice()
        else:
            type_ = self.requestState
        for side in self.sides:
            side.activeRequest = None
        if type_ == 'teampreview':
            picked = self.ruleTable.pickedTeamSize
            self.add('teampreview' + (f"|{picked}" if picked else ''))
        requests = self.getRequests(type_)
        for i, side in enumerate(self.sides):
            side.activeRequest = requests[i]
        self.sentRequests = False
        if all(side.isChoiceDone() for side in self.sides):
            raise BattleError('Choices are done immediately after a request')

    def clearRequest(self):
        self.requestState = ''
        for side in self.sides:
            side.activeRequest = None
            side.clearChoice()

    def getRequests(self, type_):
        requests = [None] * len(self.sides)
        if type_ == 'switch':
            for i, side in enumerate(self.sides):
                if not side.pokemonLeft:
                    continue
                switch_table = [bool(p and p.switchFlag) for p in side.active]
                if any(switch_table):
                    requests[i] = {'forceSwitch': switch_table, 'side': side.getRequestData()}
        elif type_ == 'teampreview':
            for i, side in enumerate(self.sides):
                max_chosen = self.ruleTable.pickedTeamSize or None
                req = {'teamPreview': True}
                if max_chosen is not None:
                    req['maxChosenTeamSize'] = max_chosen
                req['side'] = side.getRequestData()
                requests[i] = req
        else:
            for i, side in enumerate(self.sides):
                if not side.pokemonLeft:
                    continue
                active_data = [p.getMoveRequestData() if p else None for p in side.active]
                requests[i] = {'active': active_data, 'side': side.getRequestData()}
        multiple = len([r for r in requests if r]) >= 2
        for i, side in enumerate(self.sides):
            if requests[i]:
                if not self.supportCancel or not multiple:
                    requests[i]['noCancel'] = True
            else:
                requests[i] = {'wait': True, 'side': side.getRequestData()}
        return requests

    def tiebreak(self):
        if self.ended:
            return False
        self.inputLog.append('>tiebreak')
        self.add('message', "Time's up! Going to tiebreaker...")
        not_fainted = [len([p for p in side.pokemon if not p.fainted]) for side in self.sides]
        self.add('-message', '; '.join(f"{side.name}: {not_fainted[i]} Pokemon left"
                                       for i, side in enumerate(self.sides)))
        max_nf = max(not_fainted)
        tied = [side for i, side in enumerate(self.sides) if not_fainted[i] == max_nf]
        if len(tied) <= 1:
            return self.win(tied[0])
        hp_pct = [sum(p.hp / p.maxhp for p in side.pokemon) * 100 / 6 for side in tied]
        self.add('-message', '; '.join(f"{side.name}: {js_round(hp_pct[i])}% total HP left"
                                       for i, side in enumerate(tied)))
        max_pct = max(hp_pct)
        tied = [side for i, side in enumerate(tied) if hp_pct[i] == max_pct]
        if len(tied) <= 1:
            return self.win(tied[0])
        hp_total = [sum(p.hp for p in side.pokemon) for side in tied]
        self.add('-message', '; '.join(f"{side.name}: {js_round(hp_total[i])} total HP left"
                                       for i, side in enumerate(tied)))
        max_total = max(hp_total)
        tied = [side for i, side in enumerate(tied) if hp_total[i] == max_total]
        if len(tied) <= 1:
            return self.win(tied[0])
        return self.tie()

    def forceWin(self, side=None):
        if self.ended:
            return False
        self.inputLog.append(f">forcewin {side}" if side else '>forcetie')
        return self.win(side)

    def tie(self):
        return self.win()

    def win(self, side=None):
        if self.ended:
            return False
        if side and isinstance(side, str):
            side = self.getSide(side)
        elif not side or not isinstance(side, Side) or side not in self.sides:
            side = None
        self.winner = side.name if side else ''
        self.add('')
        if side is not None and side.allySide:
            self.add('win', side.name + ' & ' + side.allySide.name)
        elif side is not None:
            self.add('win', side.name)
        else:
            self.add('tie')
        self.ended = True
        self.requestState = ''
        for s in self.sides:
            if s:
                s.activeRequest = None
        return True

    def lose(self, side):
        if isinstance(side, str):
            side = self.getSide(side)
        if not side:
            return None
        return self.win(side.foe)

    def canSwitch(self, side):
        return len(self.possibleSwitches(side))

    def getRandomSwitchable(self, side):
        can = self.possibleSwitches(side)
        return self.sample(can) if can else None

    def possibleSwitches(self, side):
        if not side.pokemonLeft:
            return []
        out = []
        for i in range(len(side.active), len(side.pokemon)):
            pokemon = side.pokemon[i]
            if not pokemon.fainted:
                out.append(pokemon)
        return out

    def swapPosition(self, pokemon, newPosition, attributes=None):
        if newPosition >= len(pokemon.side.active):
            raise BattleError('Invalid swap position')
        target = pokemon.side.active[newPosition]
        if newPosition != 1 and (not target or target.fainted):
            return False
        self.add('swap', pokemon, newPosition, attributes or '')
        side = pokemon.side
        side.pokemon[pokemon.position] = target
        side.pokemon[newPosition] = pokemon
        side.active[pokemon.position] = side.pokemon[pokemon.position]
        side.active[newPosition] = side.pokemon[newPosition]
        if target:
            target.position = pokemon.position
        pokemon.position = newPosition
        self.runEvent('Swap', target, pokemon)
        self.runEvent('Swap', pokemon, target)
        return True

    def getAtSlot(self, slot):
        if not slot:
            return None
        side = self.sides[ord(slot[1]) - 49]
        position = ord(slot[2]) - 97
        offset = (side.n // 2) * len(side.active)
        return side.active[position - offset]

    def faint(self, pokemon, source=None, effect=None):
        pokemon.faint(source, effect)

    def endTurn(self):
        self.turn += 1
        self.lastSuccessfulMoveThisTurn = None

        trapped_by_side = []
        staleness_by_side = []
        for side in self.sides:
            side_trapped = True
            side_staleness = None
            for pokemon in side.active:
                if not pokemon:
                    continue
                pokemon.moveThisTurn = ''
                pokemon.newlySwitched = False
                pokemon.moveLastTurnResult = pokemon.moveThisTurnResult
                pokemon.moveThisTurnResult = None
                if self.turn != 1:
                    pokemon.usedItemThisTurn = False
                    pokemon.statsRaisedThisTurn = False
                    pokemon.statsLoweredThisTurn = False
                    pokemon.hurtThisTurn = None

                pokemon.maybeDisabled = False
                pokemon.maybeLocked = False
                for move_slot in pokemon.moveSlots:
                    move_slot.disabled = False
                    move_slot.disabledSource = ''
                self.runEvent('DisableMove', pokemon)
                for move_slot in pokemon.moveSlots:
                    base_move = self.dex.moves.getByID(move_slot.id)
                    if base_move.get('onDisableMove') is not None:
                        active_move = self.dex.getActiveMove(move_slot.id)
                        self.singleEvent('DisableMove', active_move, None, pokemon)
                    if base_move.flags.get('cantusetwice') and pokemon.lastMove and \
                            pokemon.lastMove.id == move_slot.id:
                        pokemon.disableMove(pokemon.lastMove.id)

                if pokemon.getLastAttackedBy():
                    pokemon.knownType = True

                for i in range(len(pokemon.attackedBy) - 1, -1, -1):
                    attack = pokemon.attackedBy[i]
                    if attack.source.isActive:
                        attack.thisTurn = False
                    else:
                        pokemon.attackedBy.pop(pokemon.attackedBy.index(attack))

                if not pokemon.terastallized:
                    seen = pokemon.illusion or pokemon
                    real_type_string = '/'.join(seen.getTypes(True))
                    if real_type_string != seen.apparentType:
                        self.add('-start', pokemon, 'typechange', real_type_string, '[silent]')
                        seen.apparentType = real_type_string
                        if pokemon.addedType:
                            self.add('-start', pokemon, 'typeadd', pokemon.addedType, '[silent]')

                pokemon.trapped = pokemon.maybeTrapped = False
                self.runEvent('TrapPokemon', pokemon)
                if not pokemon.knownType or self.dex.getImmunity('trapped', pokemon):
                    self.runEvent('MaybeTrapPokemon', pokemon)
                for source in pokemon.foes():
                    species = (source.illusion or source).species
                    if not species.abilities:
                        continue
                    for ability_slot, ability_name in species.abilities.items():
                        if ability_name == source.ability:
                            continue
                        rt = self.ruleTable
                        if (rt.has('+hackmons') or not rt.has('obtainableabilities')) and not self.format.get('team'):
                            continue
                        elif ability_slot == 'H' and species.unreleasedHidden:
                            continue
                        ability = self.dex.abilities.get(ability_name)
                        if rt.has('-ability:' + ability.id):
                            continue
                        if pokemon.knownType and not self.dex.getImmunity('trapped', pokemon):
                            continue
                        self.singleEvent('FoeMaybeTrapPokemon', ability, Obj(), pokemon, source)

                if pokemon.fainted:
                    continue
                side_trapped = side_trapped and bool(pokemon.trapped)
                staleness = pokemon.volatileStaleness or pokemon.staleness
                if staleness:
                    side_staleness = side_staleness if side_staleness == 'external' else staleness
                pokemon.activeTurns += 1
            trapped_by_side.append(side_trapped)
            staleness_by_side.append(side_staleness)
            side.faintedLastTurn = side.faintedThisTurn
            side.faintedThisTurn = None

        if self.maybeTriggerEndlessBattleClause(trapped_by_side, staleness_by_side):
            return

        self.add('turn', self.turn)
        self.makeRequest('move')

    def maybeTriggerEndlessBattleClause(self, trappedBySide, stalenessBySide):
        if self.turn <= 100:
            return None
        if self.turn > 1000:
            self.add('message', 'It is turn 1000. You have hit the turn limit!')
            self.tie()
            return True
        if ((self.turn >= 500 and self.turn % 100 == 0) or (self.turn >= 900 and self.turn % 10 == 0) or
                self.turn >= 990):
            turns_left = 1000 - self.turn
            text = '1 turn' if turns_left == 1 else f"{turns_left} turns"
            self.add('bigerror', f"You will auto-tie if the battle doesn't end in {text} (on turn 1000).")
        if not self.ruleTable.has('endlessbattleclause'):
            return None
        if not all(stalenessBySide) or 'external' not in stalenessBySide:
            return None
        can_switch = []
        for i, trapped in enumerate(trappedBySide):
            can_switch.append(False)
            if trapped:
                break
            side = self.sides[i]
            for pokemon in side.pokemon:
                if not pokemon.fainted and not (pokemon.volatileStaleness or pokemon.staleness):
                    can_switch[i] = True
                    break
        if len(can_switch) == len(trappedBySide) and all(can_switch):
            return None
        losers = []
        for side in self.sides:
            berry = False
            cycle = False
            for pokemon in side.pokemon:
                berry = to_id(pokemon.set.get('item')) in RESTORATIVE_BERRIES
                if to_id(pokemon.set.get('ability')) in ('harvest', 'pickup') or \
                        'recycle' in [to_id(m) for m in pokemon.set.get('moves', [])]:
                    cycle = True
                if berry and cycle:
                    break
            if berry and cycle:
                losers.append(side)
        if len(losers) == 1:
            loser = losers[0]
            self.add('-message', f"{loser.name}'s team started with the rudimentary means to perform "
                                 "restorative berry-cycling and thus loses.")
            return self.win(loser.foe)
        if len(losers) == len(self.sides):
            self.add('-message', "Each side's team started with the rudimentary means to perform "
                                 "restorative berry-cycling.")
        return self.tie()

    def start(self):
        if not all(self.sides):
            raise BattleError(f"Missing sides")
        if self.started:
            raise BattleError('Battle already started')
        fmt = self.format
        self.started = True
        self.sides[1].foe = self.sides[0]
        self.sides[0].foe = self.sides[1]

        self.add('gen', self.gen)
        self.add('tier', fmt.name)
        if self.rated:
            self.add('rated', self.rated if isinstance(self.rated, str) else '')

        if fmt.get('onBegin'):
            fmt.onBegin(self)
        for rule in self.ruleTable.keys():
            if rule[:1] in '+*-!':
                continue
            sub = self.dex.formats.get(rule)
            if sub.get('onBegin'):
                sub.onBegin(self)

        if any(not side.pokemon[0] for side in self.sides):
            raise BattleError('Battle not started: A player has an empty team.')

        self.runPickTeam()
        self.queue.addChoice(Obj(choice='start'))
        self.midTurn = True
        if not self.requestState:
            self.turnLoop()

    def runPickTeam(self):
        if self.format.get('onTeamPreview'):
            self.format.onTeamPreview(self)
        for rule in self.ruleTable.keys():
            if rule[:1] in '+*-!':
                continue
            sub = self.dex.formats.get(rule)
            if sub.get('onTeamPreview'):
                sub.onTeamPreview(self)
        if self.requestState == 'teampreview':
            return
        if self.ruleTable.pickedTeamSize:
            self.add('clearpoke')
            for pokemon in self.getAllPokemon():
                details = pokemon.details.replace(', shiny', '')
                details = re.sub(r'(Zacian|Zamazenta)(?!-Crowned)', r'\1-*', details)
                details = re.sub(r'(Xerneas)(-[a-zA-Z?-]+)?', r'\1-*', details)
                self.addSplit(pokemon.side.id, ['poke', pokemon.side.id, details, ''])
            self.makeRequest('teampreview')

    def boost(self, boost, target=None, source=None, effect=None, isSecondary=False, isSelf=False):
        if self.event:
            target = target or self.event.target
            source = source or self.event.source
            effect = effect or self.effect
        if not target or not target.hp:
            return 0
        if not target.isActive:
            return False
        if not target.side.foePokemonLeft():
            return False
        boost = self.runEvent('ChangeBoost', target, source, effect, Obj(boost))
        boost = target.getCappedBoost(boost)
        boost = self.runEvent('TryBoost', target, source, effect, Obj(boost))
        success = NULL
        boosted = isSecondary
        for boost_name in list(boost.keys()):
            current_boost = Obj({boost_name: boost[boost_name]})
            boost_by = target.boostBy(current_boost)
            msg = '-boost'
            if boost[boost_name] < 0 or target.boosts[boost_name] == -6:
                msg = '-unboost'
                boost_by = -boost_by
            if boost_by:
                success = True
                eid = effect.id if effect else None
                if eid in ('bellydrum', 'angerpoint'):
                    self.add('-setboost', target, 'atk', target.boosts['atk'], '[from] ' + effect.fullname)
                elif eid == 'zpower':
                    self.add(msg, target, boost_name, boost_by, '[zeffect]')
                elif effect:
                    if effect.effectType == 'Move':
                        self.add(msg, target, boost_name, boost_by)
                    elif effect.effectType == 'Item':
                        self.add(msg, target, boost_name, boost_by, '[from] item: ' + effect.name)
                    else:
                        if effect.effectType == 'Ability' and not boosted:
                            self.add('-ability', target, effect.name, 'boost')
                            boosted = True
                        self.add(msg, target, boost_name, boost_by)
                self.runEvent('AfterEachBoost', target, source, effect, current_boost)
            elif effect and effect.effectType == 'Ability':
                if isSecondary or isSelf:
                    self.add(msg, target, boost_name, boost_by)
            elif not isSecondary and not isSelf:
                self.add(msg, target, boost_name, boost_by)
        self.runEvent('AfterBoost', target, source, effect, boost)
        if success:
            if any(x > 0 for x in boost.values()):
                target.statsRaisedThisTurn = True
            if any(x < 0 for x in boost.values()):
                target.statsLoweredThisTurn = True
        return success

    def spreadDamage(self, damage, targetArray=None, source=None, effect=None, instafaint=False):
        if not targetArray:
            return [0]
        ret_vals = []
        if isinstance(effect, str) or not effect:
            effect = self.dex.conditions.getByID(effect or '')
        for i, cur_damage in enumerate(damage):
            target = targetArray[i]
            target_damage = cur_damage
            if not (target_damage or (is_number(target_damage) and target_damage == 0)):
                ret_vals.append(target_damage)
                continue
            if not target or not target.hp:
                ret_vals.append(0)
                continue
            if not target.isActive:
                ret_vals.append(False)
                continue
            if target_damage != 0 or target_damage is True:
                target_damage = clamp_int_range(target_damage, 1)
            if effect.id != 'strugglerecoil':
                if effect.effectType == 'Weather' and not target.runStatusImmunity(effect.id):
                    self.debug('weather immunity')
                    ret_vals.append(0)
                    continue
                target_damage = self.runEvent('Damage', target, source, effect, target_damage, True)
                if not (target_damage or (is_number(target_damage) and target_damage == 0)):
                    self.debug('damage event failed')
                    ret_vals.append(None if cur_damage is True else target_damage)
                    continue
            if target_damage != 0 or target_damage is True:
                target_damage = clamp_int_range(target_damage, 1)

            target_damage = target.damage(target_damage, source, effect)
            ret_vals.append(target_damage)
            if target_damage != 0:
                target.hurtThisTurn = target.hp
            if source and effect.effectType == 'Move':
                source.lastDamage = target_damage

            name = 'psn' if effect.fullname == 'tox' else effect.fullname
            eid = effect.id
            if eid == 'partiallytrapped':
                self.add('-damage', target, target.getHealth,
                         '[from] ' + target.volatiles['partiallytrapped'].sourceEffect.fullname, '[partiallytrapped]')
            elif eid == 'powder':
                self.add('-damage', target, target.getHealth, '[silent]')
            elif eid == 'confused':
                self.add('-damage', target, target.getHealth, '[from] confusion')
            else:
                if effect.effectType == 'Move' or not name:
                    self.add('-damage', target, target.getHealth)
                elif source and (source is not target or effect.effectType == 'Ability'):
                    self.add('-damage', target, target.getHealth, f"[from] {name}", f"[of] {source}")
                else:
                    self.add('-damage', target, target.getHealth, f"[from] {name}")

            if target_damage and effect.effectType == 'Move':
                if effect.drain and source:
                    amount = js_round(target_damage * effect.drain[0] / effect.drain[1])
                    self.heal(amount, source, target, 'drain')

        if instafaint:
            for i, target in enumerate(targetArray):
                if not ret_vals[i] or not target:
                    continue
                if target.hp <= 0:
                    self.debug('instafaint')
                    self.faintMessages(True)
        return ret_vals

    def damage(self, damage, target=None, source=None, effect=None, instafaint=False):
        if self.event:
            target = target or self.event.target
            source = source or self.event.source
            effect = effect or self.effect
        return self.spreadDamage([damage], [target], source, effect, instafaint)[0]

    def directDamage(self, damage, target=None, source=None, effect=None):
        if self.event:
            target = target or self.event.target
            source = source or self.event.source
            effect = effect or self.effect
        if not target or not target.hp:
            return 0
        if not damage:
            return 0
        damage = clamp_int_range(damage, 1)
        if isinstance(effect, str) or not effect:
            effect = self.dex.conditions.getByID(effect or '')
        damage = target.damage(damage, source, effect)
        if effect.id == 'strugglerecoil':
            self.add('-damage', target, target.getHealth, '[from] recoil')
        elif effect.id == 'confusion':
            self.add('-damage', target, target.getHealth, '[from] confusion')
        else:
            self.add('-damage', target, target.getHealth)
        if target.fainted:
            self.faint(target)
        return damage

    def heal(self, damage, target=None, source=None, effect=None):
        if self.event:
            target = target or self.event.target
            source = source or self.event.source
            effect = effect or self.effect
        if effect == 'drain':
            effect = self.dex.conditions.getByID('drain')
        if damage and damage <= 1:
            damage = 1
        damage = trunc(damage)
        damage = self.runEvent('TryHeal', target, source, effect, damage)
        if not damage:
            return damage
        if not target or not target.hp:
            return False
        if not target.isActive:
            return False
        if target.hp >= target.maxhp:
            return False
        final_damage = target.heal(damage, source, effect)
        eid = effect.id if effect else None
        if eid in ('leechseed', 'rest'):
            self.add('-heal', target, target.getHealth, '[silent]')
        elif eid == 'drain':
            self.add('-heal', target, target.getHealth, '[from] drain', f"[of] {source}")
        elif eid == 'wish':
            pass
        elif eid == 'zpower':
            self.add('-heal', target, target.getHealth, '[zeffect]')
        elif effect:
            if effect.effectType == 'Move':
                self.add('-heal', target, target.getHealth)
            elif source and source is not target:
                self.add('-heal', target, target.getHealth, f"[from] {effect.fullname}", f"[of] {source}")
            else:
                self.add('-heal', target, target.getHealth, f"[from] {effect.fullname}")
        self.runEvent('Heal', target, source, effect, final_damage)
        return final_damage

    def chain(self, previousMod, nextMod):
        if isinstance(previousMod, (list, tuple)):
            previousMod = trunc(previousMod[0] * 4096 / previousMod[1])
        else:
            previousMod = trunc(previousMod * 4096)
        if isinstance(nextMod, (list, tuple)):
            nextMod = trunc(nextMod[0] * 4096 / nextMod[1])
        else:
            nextMod = trunc(nextMod * 4096)
        return (to_int32(previousMod * nextMod + 2048) >> 12) / 4096

    def chainModify(self, numerator, denominator=1):
        previous_mod = trunc(self.event.modifier * 4096)
        if isinstance(numerator, (list, tuple)):
            denominator = numerator[1]
            numerator = numerator[0]
        next_mod = trunc(numerator * 4096 / denominator)
        self.event.modifier = (to_int32(previous_mod * next_mod + 2048) >> 12) / 4096

    def modify(self, value, numerator, denominator=1):
        if isinstance(numerator, (list, tuple)):
            denominator = numerator[1]
            numerator = numerator[0]
        modifier = trunc(numerator * 4096 / denominator)
        return trunc((trunc(value * modifier) + 2048 - 1) / 4096)

    def spreadModify(self, baseStats, set_):
        mod_stats = Obj()
        for stat_name in baseStats:
            mod_stats[stat_name] = self.statModify(baseStats, set_, stat_name)
        return mod_stats

    def statModify(self, baseStats, set_, statName):
        """Champions stat formula: stats depend on Stat Points (stored as EVs), level 50."""
        stat = baseStats[statName]
        evs = set_['evs'][statName]
        if statName == 'hp':
            return stat + evs + 75
        stat = stat + evs + 20
        nature = self.dex.natures.get(set_.get('nature'))
        if nature.plus == statName:
            stat = trunc(trunc(stat * 110, 16) / 100)
        elif nature.minus == statName:
            stat = trunc(trunc(stat * 90, 16) / 100)
        return stat

    def calculatePP(self, move, ppUps=3):
        if move.noPPBoosts:
            return move.pp
        return (move.pp / 5 + 1) * 4

    def finalModify(self, relayVar):
        relayVar = self.modify(relayVar, self.event.modifier)
        self.event.modifier = 1
        return relayVar

    def getCategory(self, move):
        return self.dex.moves.get(move).category or 'Physical'

    def randomizer(self, baseDamage):
        return trunc(trunc(baseDamage * (100 - self.random(16))) / 100)

    def validTargetLoc(self, targetLoc, source, targetType):
        if targetLoc == 0:
            return True
        if targetLoc is None or targetLoc is NULL:
            # JS: an undefined location makes every comparison NaN/false; only 'any' (!isSelf) passes
            return targetType == 'any'
        num_slots = self.activePerHalf
        source_loc = source.getLocOf(source)
        if abs(targetLoc) > num_slots:
            return False
        is_self = source_loc == targetLoc
        is_foe = targetLoc > 0
        across = -(num_slots + 1 - targetLoc)
        is_adjacent = (abs(across - source_loc) <= 1) if targetLoc > 0 else (abs(targetLoc - source_loc) == 1)
        if targetType in ('randomNormal', 'scripted', 'normal'):
            return is_adjacent
        if targetType == 'adjacentAlly':
            return is_adjacent and not is_foe
        if targetType == 'adjacentAllyOrSelf':
            return (is_adjacent and not is_foe) or is_self
        if targetType == 'adjacentFoe':
            return is_adjacent and is_foe
        if targetType == 'any':
            return not is_self
        return False

    def validTarget(self, target, source, targetType):
        return self.validTargetLoc(source.getLocOf(target), source, targetType)

    def getTarget(self, pokemon, move, targetLoc, originalTarget=None):
        move = self.dex.moves.get(move)
        tracks_target = move.tracksTarget
        if pokemon.hasAbility(['stalwart', 'propellertail']):
            tracks_target = True
        if tracks_target and originalTarget is not None and originalTarget.isActive:
            return originalTarget
        if move.smartTarget:
            cur = pokemon.getAtLoc(targetLoc)
            return cur if cur and not cur.fainted else self.getRandomTarget(pokemon, move)
        self_loc = pokemon.getLocOf(pokemon)
        if (move.target in ('adjacentAlly', 'any', 'normal') and targetLoc == self_loc and
                not pokemon.volatiles.get('twoturnmove') and not pokemon.volatiles.get('iceball') and
                not pokemon.volatiles.get('rollout')):
            return pokemon if move.flags.get('futuremove') else None
        if move.target != 'randomNormal' and self.validTargetLoc(targetLoc, pokemon, move.target):
            target = pokemon.getAtLoc(targetLoc)
            if target and target.fainted:
                if target.isAlly(pokemon):
                    if move.target == 'adjacentAllyOrSelf':
                        return pokemon
                    return target
            if target and not target.fainted:
                return target
        return self.getRandomTarget(pokemon, move)

    def getRandomTarget(self, pokemon, move):
        move = self.dex.moves.get(move)
        if move.target in ('self', 'all', 'allySide', 'allyTeam', 'adjacentAllyOrSelf'):
            return pokemon
        elif move.target == 'adjacentAlly':
            if self.gameType == 'singles':
                return None
            adjacent_allies = pokemon.adjacentAllies()
            return self.sample(adjacent_allies) if adjacent_allies else None
        if self.gameType == 'singles':
            return pokemon.side.foe.active[0]
        return pokemon.side.randomFoe() or pokemon.side.foe.active[0]

    def checkFainted(self):
        for side in self.sides:
            for pokemon in side.active:
                if pokemon.fainted:
                    pokemon.status = 'fnt'
                    pokemon.switchFlag = True

    def faintMessages(self, lastFirst=False, forceCheck=False, checkWin=True):
        if self.ended:
            return None
        length = len(self.faintQueue)
        if not length:
            if forceCheck and self.checkWin():
                return True
            return False
        if lastFirst:
            self.faintQueue.insert(0, self.faintQueue[-1])
            self.faintQueue.pop()
        faint_data = None
        while self.faintQueue:
            faint_queue_left = len(self.faintQueue)
            faint_data = self.faintQueue.pop(0)
            pokemon = faint_data.target
            if not pokemon.fainted and self.runEvent('BeforeFaint', pokemon, faint_data.source, faint_data.effect):
                self.add('faint', pokemon)
                if pokemon.side.pokemonLeft:
                    pokemon.side.pokemonLeft -= 1
                if pokemon.side.totalFainted < 100:
                    pokemon.side.totalFainted += 1
                self.runEvent('Faint', pokemon, faint_data.source, faint_data.effect)
                self.singleEvent('End', pokemon.getAbility(), pokemon.abilityState, pokemon)
                self.singleEvent('End', pokemon.getItem(), pokemon.itemState, pokemon)
                if pokemon.formeRegression and not pokemon.transformed:
                    pokemon.baseSpecies = self.dex.species.get(pokemon.set.get('species') or pokemon.set.get('name'))
                    pokemon.baseAbility = to_id(pokemon.set.get('ability'))
                pokemon.clearVolatile(False)
                pokemon.fainted = True
                pokemon.illusion = None
                pokemon.isActive = False
                pokemon.isStarted = False
                pokemon.terastallized = None
                if pokemon.formeRegression:
                    pokemon.details = pokemon.getUpdatedDetails()
                    self.add('detailschange', pokemon, pokemon.details, '[silent]')
                    pokemon.updateMaxHp()
                    pokemon.formeRegression = False
                pokemon.side.faintedThisTurn = pokemon
                if len(self.faintQueue) >= faint_queue_left:
                    checkWin = True
        if checkWin and self.checkWin(faint_data):
            return True
        if faint_data and length:
            self.runEvent('AfterFaint', faint_data.target, faint_data.source, faint_data.effect, length)
        return False

    def checkWin(self, faintData=None):
        if all(not side.pokemonLeft for side in self.sides):
            self.win(faintData.target.side if faintData else None)
            return True
        for side in self.sides:
            if not side.foePokemonLeft():
                self.win(side)
                return True
        return None

    def getActionSpeed(self, action):
        if action.choice == 'move':
            move = action.move
            priority = self.dex.moves.get(move.id).priority
            target = self.getTarget(action.pokemon, action.move, action.targetLoc)
            priority = self.singleEvent('ModifyPriority', move, None, action.pokemon, target, None, priority)
            priority = self.runEvent('ModifyPriority', action.pokemon, target, move, priority)
            action.priority = priority + action.fractionalPriority
            action.move.priority = priority
        if not action.pokemon:
            action.speed = 1
        else:
            action.speed = action.pokemon.getActionSpeed()

    def runAction(self, action):
        pokemon_original_hp = action.pokemon.hp if action.pokemon else None
        residual_pokemon = []
        choice = action.choice
        if choice == 'start':
            for side in self.sides:
                if side.pokemonLeft:
                    side.pokemonLeft = len(side.pokemon)
                self.add('teamsize', side.id, len(side.pokemon))
            self.add('start')
            for pokemon in self.getAllPokemon():
                self.singleEvent('BattleStart', self.dex.conditions.getByID(pokemon.species.id),
                                 pokemon.speciesState, pokemon)
            if self.format.get('onBattleStart'):
                self.format.onBattleStart(self)
            for rule in self.ruleTable.keys():
                if rule[:1] in '+*-!':
                    continue
                sub = self.dex.formats.get(rule)
                if sub.get('onBattleStart'):
                    sub.onBattleStart(self)
            for side in self.sides:
                for i in range(len(side.active)):
                    if not side.pokemonLeft:
                        side.active[i] = side.pokemon[i]
                        side.active[i].fainted = True
                        side.active[i].hp = 0
                    else:
                        self.actions.switchIn(side.pokemon[i], i)
            self.midTurn = True
        elif choice == 'move':
            if not action.pokemon.isActive:
                return False
            if action.pokemon.fainted:
                return False
            self.actions.runMove(action.move, action.pokemon, action.targetLoc, Obj(
                sourceEffect=action.sourceEffect, zMove=action.zmove, maxMove=action.maxMove,
                originalTarget=action.originalTarget))
        elif choice == 'megaEvo':
            self.actions.runMegaEvo(action.pokemon)
        elif choice == 'beforeTurnMove':
            if not action.pokemon.isActive:
                return False
            if action.pokemon.fainted:
                return False
            self.debug('before turn callback: ' + action.move.id)
            target = self.getTarget(action.pokemon, action.move, action.targetLoc)
            if not target:
                return False
            if not action.move.beforeTurnCallback:
                raise BattleError('beforeTurnMove has no beforeTurnCallback')
            action.move.beforeTurnCallback(self, action.pokemon, target)
        elif choice == 'priorityChargeMove':
            if not action.pokemon.isActive:
                return False
            if action.pokemon.fainted:
                return False
            self.debug('priority charge callback: ' + action.move.id)
            if not action.move.priorityChargeCallback:
                raise BattleError('priorityChargeMove has no priorityChargeCallback')
            action.move.priorityChargeCallback(self, action.pokemon)
        elif choice == 'event':
            self.runEvent(action.event, action.pokemon)
        elif choice == 'team':
            if action.index == 0:
                action.pokemon.side.pokemon = []
            action.pokemon.side.pokemon.append(action.pokemon)
            action.pokemon.position = action.index
            return None
        elif choice == 'pass':
            return None
        elif choice in ('instaswitch', 'switch'):
            if choice == 'switch' and action.pokemon.status:
                self.singleEvent('CheckShow', self.dex.abilities.getByID('naturalcure'), None, action.pokemon)
            if self.actions.switchIn(action.target, action.pokemon.position, action.sourceEffect) == 'pursuitfaint':
                self.hint("A Pokemon can't switch between when it runs out of HP and when it faints")
        elif choice == 'revivalblessing':
            action.pokemon.side.pokemonLeft += 1
            if action.target.position < len(action.pokemon.side.active):
                self.queue.addChoice(Obj(choice='instaswitch', pokemon=action.target, target=action.target))
            action.target.fainted = False
            action.target.faintQueued = False
            action.target.subFainted = False
            action.target.status = ''
            action.target.hp = 1
            action.target.sethp(action.target.maxhp / 2)
            self.add('-heal', action.target, action.target.getHealth, '[from] move: Revival Blessing')
            action.pokemon.side.removeSlotCondition(action.pokemon, 'revivalblessing')
        elif choice == 'runSwitch':
            self.actions.runSwitch(action.pokemon)
        elif choice == 'shift':
            if not action.pokemon.isActive:
                return False
            if action.pokemon.fainted:
                return False
            self.swapPosition(action.pokemon, 1)
        elif choice == 'beforeTurn':
            self.eachEvent('BeforeTurn')
        elif choice == 'residual':
            self.add('')
            self.clearActiveMove(True)
            self.updateSpeed()
            residual_pokemon = [(p, p.hp) for p in self.getAllActive()]
            self.fieldEvent('Residual')
            if not self.ended:
                self.add('upkeep')

        # phazing (Roar, etc)
        for side in self.sides:
            for pokemon in side.active:
                if pokemon.forceSwitchFlag:
                    if pokemon.hp:
                        self.actions.dragIn(pokemon.side, pokemon.position)
                    pokemon.forceSwitchFlag = False

        self.clearActiveMove()

        self.faintMessages()
        if self.ended:
            return True

        nxt = self.queue.peek()
        if not nxt:
            self.checkFainted()
        elif nxt.choice == 'instaswitch':
            return False

        if choice != 'start':
            self.eachEvent('Update')
            for pokemon, original_hp in residual_pokemon:
                self.runEvent('EmergencyExit', pokemon, None, None, original_hp)

        if choice == 'runSwitch':
            self.runEvent('EmergencyExit', action.pokemon, None, None, pokemon_original_hp)

        switches = [any(p and p.switchFlag for p in side.active) for side in self.sides]

        for i, side in enumerate(self.sides):
            revive_switch = False
            if switches[i] and not self.canSwitch(side):
                for pokemon in side.active:
                    if side.slotConditions[pokemon.position].get('revivalblessing'):
                        revive_switch = True
                        continue
                    pokemon.switchFlag = False
                if not revive_switch:
                    switches[i] = False
            elif switches[i]:
                for pokemon in side.active:
                    if (pokemon.hp and pokemon.switchFlag and pokemon.switchFlag != 'revivalblessing' and
                            not pokemon.skipBeforeSwitchOutEventFlag):
                        self.runEvent('BeforeSwitchOut', pokemon)
                        pokemon.skipBeforeSwitchOutEventFlag = True
                        self.faintMessages()
                        if self.ended:
                            return True
                        if pokemon.fainted:
                            switches[i] = any(p and p.switchFlag for p in side.active)

        for player_switch in switches:
            if player_switch:
                self.makeRequest('switch')
                return True

        nxt = self.queue.peek()
        if nxt and nxt.choice in ('move', 'runDynamax'):
            self.updateSpeed()
            for queue_action in self.queue.list:
                if queue_action.pokemon:
                    self.getActionSpeed(queue_action)
            self.queue.sort()
        return False

    def turnLoop(self):
        self.add('')
        self.add('t:', self._now())
        if self.requestState:
            self.requestState = ''
        if not self.midTurn:
            self.queue.insertChoice(Obj(choice='beforeTurn'))
            self.queue.addChoice(Obj(choice='residual'))
            self.midTurn = True
        while True:
            action = self.queue.shift()
            if action is None:
                break
            self.runAction(action)
            if self.requestState or self.ended:
                return
        self.endTurn()
        self.midTurn = False
        self.queue.clear()

    def choose(self, sideid: str, input_: str) -> bool:
        side = self.getSide(sideid)
        if not side.choose(input_):
            if not side.choice.error:
                side.emitChoiceError(f"Unknown error for choice: {input_}. If you're not using a custom "
                                     "client, please report this as a bug.")
            return False
        if not side.isChoiceDone():
            side.emitChoiceError(f"Incomplete choice: {input_} - missing other pokemon")
            return False
        if self.allChoicesDone():
            self.commitChoices()
        return True

    def makeChoices(self, *inputs):
        if inputs:
            for i, inp in enumerate(inputs):
                if inp:
                    self.sides[i].choose(inp)
        else:
            for side in self.sides:
                side.autoChoose()
        self.commitChoices()

    def commitChoices(self):
        self.updateSpeed()
        old_queue = self.queue.list
        self.queue.clear()
        if not self.allChoicesDone():
            raise BattleError('Not all choices done')
        for side in self.sides:
            choice = side.getChoice()
            if choice:
                self.inputLog.append(f">{side.id} {choice}")
        for side in self.sides:
            side.commitChoices()
        self.clearRequest()
        self.queue.sort()
        self.queue.list.extend(old_queue)
        self.requestState = ''
        for side in self.sides:
            side.activeRequest = None
        self.turnLoop()
        # workaround in Showdown for tests (affects the line-limit safety check)
        if len(self.log) - self.sentLogPos > 500:
            self.sendUpdates()

    def sendUpdates(self):
        if self.sentLogPos >= len(self.log):
            return
        self.send('update', self.log[self.sentLogPos:])
        self.sentRequests = True
        self.sentLogPos = len(self.log)

    def undoChoice(self, sideid):
        side = self.getSide(sideid)
        if not side.requestState:
            return
        if side.choice.cantUndo:
            side.emitChoiceError("Can't undo: A trapping/disabling effect would cause undo to leak information")
            return
        side.clearChoice()

    def allChoicesDone(self):
        total = 0
        for side in self.sides:
            if side.isChoiceDone():
                if not self.supportCancel:
                    side.choice.cantUndo = True
                total += 1
        return total >= len(self.sides)

    def hint(self, hint, once=False, side=None):
        key = f"{side.id}|{hint}" if side else hint
        if key in self.hints:
            return
        if side:
            self.addSplit(side.id, ['-hint', hint])
        else:
            self.add('-hint', hint)
        if once:
            self.hints.add(key)

    def addSplit(self, side, secret, shared=None):
        self.log.append(f"|split|{side}")
        self.add(*secret)
        if shared:
            self.add(*shared)
        else:
            self.log.append('')

    def add(self, *parts):
        if not any(callable(p) for p in parts):
            self.log.append('|' + '|'.join(join_part(p) for p in parts))
            return
        side = None
        secret = []
        shared = []
        for part in parts:
            if callable(part):
                split = part()
                if side and side != split['side']:
                    raise BattleError('Multiple sides passed to add')
                side = split['side']
                secret.append(split['secret'])
                shared.append(split['shared'])
            else:
                secret.append(part)
                shared.append(part)
        self.addSplit(side, secret, shared)

    def addMove(self, *args):
        self.lastMoveLine = len(self.log)
        self.log.append('|' + '|'.join(join_part(a) for a in args))

    def attrLastMove(self, *args):
        if self.lastMoveLine < 0:
            return
        if self.log[self.lastMoveLine].startswith('|-anim|'):
            if '[still]' in args:
                self.log.pop(self.lastMoveLine)
                self.lastMoveLine = -1
                return
        elif '[still]' in args:
            parts = self.log[self.lastMoveLine].split('|')
            if len(parts) > 4:
                parts[4] = ''
            else:
                while len(parts) < 4:
                    parts.append('')
                parts.append('')
            self.log[self.lastMoveLine] = '|'.join(parts)
        self.log[self.lastMoveLine] += '|' + '|'.join(join_part(a) for a in args)

    def retargetLastMove(self, newTarget):
        if self.lastMoveLine < 0:
            return
        parts = self.log[self.lastMoveLine].split('|')
        while len(parts) < 5:
            parts.append('')
        parts[4] = str(newTarget)
        self.log[self.lastMoveLine] = '|'.join(parts)

    def debug(self, *activity):
        if self.debugMode:
            self.add('debug', *activity)

    def debugError(self, activity):
        self.add('debug', activity)

    def showOpenTeamSheets(self):
        from .teams import pack_team
        if self.turn != 0:
            return
        for side in self.sides:
            team = []
            for pokemon in side.pokemon:
                s = pokemon.set
                team.append({'name': '', 'species': s.get('species'), 'item': s.get('item'),
                             'ability': s.get('ability'), 'moves': s.get('moves'), 'nature': s.get('nature'),
                             'gender': pokemon.gender, 'evs': None, 'ivs': None, 'level': s.get('level')})
            self.add('showteam', side.id, pack_team(team))

    def setPlayer(self, slot: str, options: dict):
        from .teams import pack_team, unpack_team
        slot_num = int(slot[1]) - 1
        if self.sides[slot_num] is None:
            team = options.get('team')
            if isinstance(team, str):
                team = unpack_team(team)
            team = [Obj(deep_clone(Obj(s) if not isinstance(s, Obj) else s)) for s in team]
            side = Side(options.get('name') or f"Player {slot_num + 1}", self, slot_num, team)
            if options.get('avatar'):
                side.avatar = str(options['avatar'])
            self.sides[slot_num] = side
        else:
            raise BattleError(f"Player {slot} already exists")
        opts = dict(options)
        if opts.get('team') is not None and not isinstance(opts['team'], str):
            opts['team'] = pack_team(opts['team'])
        self.inputLog.append(f">player {slot} " + json.dumps(opts, separators=(',', ':'), ensure_ascii=False))
        self.add('player', side.id, side.name, side.avatar, options.get('rating') or '')
        if all(self.sides) and not self.started:
            self.start()

    def getSide(self, sideid: str):
        return self.sides[int(sideid[1]) - 1]

    def getOverflowedTurnCount(self):
        return trunc(self.turn - 1, 8)

    def initEffectState(self, obj: Obj, effectOrder=None) -> Obj:
        if not isinstance(obj, Obj):
            obj = Obj(obj)
        if not obj.get('id'):
            obj['id'] = ''
        if effectOrder is not None:
            obj['effectOrder'] = effectOrder
        elif obj.get('id') and obj.get('target') is not None and \
                (not isinstance(obj['target'], Pokemon) or obj['target'].isActive):
            obj['effectOrder'] = self.effectOrder
            self.effectOrder += 1
        else:
            obj['effectOrder'] = 0
        return obj

    def clearEffectState(self, state: Obj):
        state['id'] = ''
        for k in list(state.keys()):
            if k in ('id', 'target'):
                continue
            elif k == 'effectOrder':
                state['effectOrder'] = 0
            else:
                del state[k]

    # convenience -----------------------------------------------------------
    def get_log(self, include_timestamps=False) -> list[str]:
        if include_timestamps:
            return list(self.log)
        return [l for l in self.log if not l.startswith('|t:|')]
