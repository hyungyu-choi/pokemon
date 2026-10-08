"""Port of Showdown's ``sim/battle-actions.ts`` with the champions mod overrides.

Z-Moves, Dynamax and Terastallization do not exist in Pokemon Champions and are
not implemented; Mega Evolution is.
"""
from __future__ import annotations

import math

from .js import NULL, Obj, clamp_int_range, is_number, js_round, js_typeof, to_id, trunc

CHOOSABLE_TARGETS = {'normal', 'any', 'adjacentAlly', 'adjacentAllyOrSelf', 'adjacentFoe'}

_MULTIHIT_2_5 = [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5]
_ACC_BOOST_TABLE = [1, 4 / 3, 5 / 3, 2, 7 / 3, 8 / 3, 3]
_RESULT_PRIORITIES = ['undefined', 'string', 'object', 'boolean', 'number']
_STAT_TABLE = {'atk': 'Atk', 'def': 'Def', 'spa': 'SpA', 'spd': 'SpD', 'spe': 'Spe'}


def _ok(v):
    """JS ``v || v === 0`` (truthy or the number zero)."""
    return bool(v) or (is_number(v) and v == 0)


class BattleActions:
    def __init__(self, battle):
        self.battle = battle
        self.dex = battle.dex

    # ------------------------------------------------------------------ SWITCH
    def switchIn(self, pokemon, pos, sourceEffect=None, isDrag=False):
        battle = self.battle
        if not pokemon or pokemon.isActive:
            battle.hint("A switch failed because the Pokémon trying to switch in is already in.")
            return False
        side = pokemon.side
        if pos >= len(side.active):
            raise ValueError(f"Invalid switch position {pos} / {len(side.active)}")
        old_active = side.active[pos]
        unfainted_active = old_active if (old_active and old_active.hp) else None
        if unfainted_active:
            old_active.beingCalledBack = True
            switch_copy_flag = False
            if sourceEffect and isinstance(sourceEffect.get('selfSwitch'), str):
                switch_copy_flag = sourceEffect.selfSwitch
            if not old_active.skipBeforeSwitchOutEventFlag and not isDrag:
                battle.runEvent('BeforeSwitchOut', old_active)
                battle.eachEvent('Update')
            old_active.skipBeforeSwitchOutEventFlag = False
            if not battle.runEvent('SwitchOut', old_active):
                return False
            if not old_active.hp:
                return 'pursuitfaint'
            battle.singleEvent('End', old_active.getAbility(), old_active.abilityState, old_active)
            battle.singleEvent('End', old_active.getItem(), old_active.itemState, old_active)
            battle.queue.cancelAction(old_active)
            if switch_copy_flag:
                pokemon.copyVolatileFrom(old_active, switch_copy_flag)
            old_active.clearVolatile()
        if old_active:
            old_active.isActive = False
            old_active.isStarted = False
            old_active.usedItemThisTurn = False
            old_active.statsRaisedThisTurn = False
            old_active.statsLoweredThisTurn = False
            old_active.position = pokemon.position
            if old_active.fainted:
                old_active.status = ''
            pokemon.position = pos
            side.pokemon[pokemon.position] = pokemon
            side.pokemon[old_active.position] = old_active
        pokemon.isActive = True
        side.active[pos] = pokemon
        pokemon.activeTurns = 0
        pokemon.activeMoveActions = 0
        for ms in pokemon.moveSlots:
            ms.used = False
        pokemon.abilityState = battle.initEffectState(Obj(id=pokemon.ability, target=pokemon))
        pokemon.itemState = battle.initEffectState(Obj(id=pokemon.item, target=pokemon))
        battle.runEvent('BeforeSwitchIn', pokemon)
        if sourceEffect:
            battle.add('drag' if isDrag else 'switch', pokemon, pokemon.getFullDetails, f"[from] {sourceEffect}")
        else:
            battle.add('drag' if isDrag else 'switch', pokemon, pokemon.getFullDetails)
        pokemon.previouslySwitchedIn += 1
        if isDrag:
            self.runSwitch(pokemon)
        else:
            battle.queue.insertChoice(Obj(choice='runSwitch', pokemon=pokemon))
        return True

    def dragIn(self, side, pos):
        battle = self.battle
        pokemon = battle.getRandomSwitchable(side)
        if not pokemon or pokemon.isActive:
            return False
        old_active = side.active[pos]
        if not old_active:
            raise ValueError('nothing to drag out')
        if not old_active.hp:
            return False
        if not battle.runEvent('DragOut', old_active):
            return False
        if not self.switchIn(pokemon, pos, None, True):
            return False
        return True

    def runSwitch(self, pokemon):
        battle = self.battle
        switchers_in = [pokemon]
        while battle.queue.peek() and battle.queue.peek().choice == 'runSwitch':
            nxt = battle.queue.shift()
            switchers_in.append(nxt.pokemon)
        all_active = battle.getAllActive(True)
        battle.speedSort(all_active)
        battle.speedOrder = [a.getFieldPositionValue() for a in all_active]
        battle.fieldEvent('SwitchIn', switchers_in)
        for poke in switchers_in:
            if not poke.hp:
                continue
            poke.isStarted = True
            poke.draggedIn = None
        return True

    # ------------------------------------------------------------------- MOVES
    def runMove(self, moveOrMoveName, pokemon, targetLoc, options=None):
        battle = self.battle
        options = options or Obj()
        pokemon.activeMoveActions += 1
        external_move = options.externalMove
        original_target = options.originalTarget
        source_effect = options.sourceEffect
        target = battle.getTarget(pokemon, moveOrMoveName, targetLoc, original_target)
        base_move = self.dex.getActiveMove(moveOrMoveName)
        priority = base_move.priority
        prankster_boosted = base_move.pranksterBoosted
        if base_move.id != 'struggle' and not external_move:
            changed_move = battle.runEvent('OverrideAction', pokemon, target, base_move)
            if changed_move and changed_move is not True:
                base_move = self.dex.getActiveMove(changed_move)
                base_move.priority = priority
                if prankster_boosted:
                    base_move.pranksterBoosted = prankster_boosted
                target = battle.getRandomTarget(pokemon, base_move)
        move = base_move
        move.isExternal = external_move
        battle.setActiveMove(move, pokemon, target)

        will_try_move = battle.runEvent('BeforeMove', pokemon, target, move)
        if not will_try_move:
            battle.runEvent('MoveAborted', pokemon, target, move)
            battle.clearActiveMove(True)
            pokemon.moveThisTurnResult = will_try_move
            return

        if move.flags.get('cantusetwice') and pokemon.lastMove and pokemon.lastMove.id == move.id:
            pokemon.addVolatile(move.id)

        if move.beforeMoveCallback:
            if move.beforeMoveCallback(battle, pokemon, target, move):
                battle.clearActiveMove(True)
                pokemon.moveThisTurnResult = False
                return
        pokemon.lastDamage = 0
        if not external_move:
            locked_move = pokemon.getLockedMove()
            if not locked_move:
                if not pokemon.deductPP(base_move, None, target) and move.id != 'struggle':
                    battle.add('cant', pokemon, 'nopp', move)
                    battle.clearActiveMove(True)
                    pokemon.moveThisTurnResult = False
                    return
            else:
                source_effect = self.dex.conditions.get('lockedmove')
            pokemon.moveUsed(move, targetLoc)

        no_lock = external_move and not pokemon.volatiles.get('lockedmove')

        move_did_something = self.useMove(base_move, pokemon, Obj(target=NULL if target is None else target,
                                                                  sourceEffect=source_effect))
        battle.lastSuccessfulMoveThisTurn = (battle.activeMove and battle.activeMove.id) if move_did_something \
            else None
        if battle.activeMove:
            move = battle.activeMove
        battle.singleEvent('AfterMove', move, None, pokemon, target, move)
        battle.runEvent('AfterMove', pokemon, target, move)
        if move.flags.get('cantusetwice') and pokemon.removeVolatile(move.id):
            battle.add('-hint', f"Some effects can force a Pokemon to use {move.name} again in a row.")

        if move.flags.get('dance') and move_did_something and not move.isExternal:
            dancers = []
            for cur in battle.getAllActive():
                if pokemon is cur:
                    continue
                if cur.hasAbility('dancer') and not cur.isSemiInvulnerable():
                    dancers.append(cur)
            import functools
            dancers.sort(key=functools.cmp_to_key(
                lambda a, b: (-(b.storedStats['spe'] - a.storedStats['spe'])) or
                             (b.abilityState.effectOrder - a.abilityState.effectOrder)))
            target_of_1st_dance = battle.activeTarget
            for dancer in dancers:
                if battle.faintMessages():
                    break
                if dancer.fainted:
                    continue
                battle.add('-activate', dancer, 'ability: Dancer')
                dancers_target = (target_of_1st_dance if (not target_of_1st_dance.isAlly(dancer) and
                                                          pokemon.isAlly(dancer)) else pokemon)
                dancers_target_loc = dancer.getLocOf(dancers_target)
                self.runMove(move.id, dancer, dancers_target_loc,
                             Obj(sourceEffect=self.dex.abilities.get('dancer'), externalMove=True))
        if no_lock and pokemon.volatiles.get('lockedmove'):
            pokemon.volatiles.pop('lockedmove', None)
        battle.faintMessages()
        battle.checkWin()

    def useMove(self, move, pokemon, options=None):
        pokemon.moveThisTurnResult = None
        old_move_result = pokemon.moveThisTurnResult
        move_result = self.useMoveInner(move, pokemon, options)
        if pokemon.moveThisTurnResult is old_move_result:
            pokemon.moveThisTurnResult = move_result
        return move_result

    def useMoveInner(self, moveOrMoveName, pokemon, options=None):
        battle = self.battle
        options = options or Obj()
        # options.target: None = JS undefined (not given), NULL = JS null (no valid target)
        target = options.get('target')
        source_effect = options.sourceEffect
        if not source_effect and battle.effect.id:
            source_effect = battle.effect
        if source_effect and source_effect.id in ('instruct', 'custapberry'):
            source_effect = None

        move = self.dex.getActiveMove(moveOrMoveName)
        pokemon.lastMoveUsed = move

        if battle.activeMove:
            move.priority = battle.activeMove.priority
            if not move.hasBounced:
                move.pranksterBoosted = battle.activeMove.pranksterBoosted
        base_target = move.target
        target_relay_var = Obj(target=target)
        target_relay_var = battle.runEvent('ModifyTarget', pokemon, None if target is NULL else target, move,
                                           target_relay_var, True)
        if target_relay_var.target is not None:
            target = target_relay_var.target
        if target is None:
            target = battle.getRandomTarget(pokemon, move)
        if target is NULL:
            target = None
        if move.target in ('self', 'allies'):
            target = pokemon
        if source_effect:
            move.sourceEffect = source_effect.id
            move.ignoreAbility = source_effect.ignoreAbility
        move_result = False

        battle.setActiveMove(move, pokemon, target)

        battle.singleEvent('ModifyType', move, None, pokemon, target, move, move)
        battle.singleEvent('ModifyMove', move, None, pokemon, target, move, move)
        if base_target != move.target:
            target = battle.getRandomTarget(pokemon, move)
        move = battle.runEvent('ModifyType', pokemon, target, move, move)
        move = battle.runEvent('ModifyMove', pokemon, target, move, move)
        if base_target != move.target:
            target = battle.getRandomTarget(pokemon, move)
        if not move or pokemon.fainted:
            return False

        attrs = ''
        movename = move.name
        if move.id == 'hiddenpower':
            movename = 'Hidden Power'
        if source_effect:
            attrs += f"|[from] {source_effect.fullname}"
        target_str = 'null' if target is None else str(target)
        battle.addMove('move', pokemon, movename, f"{target_str}{attrs}")

        if not target:
            battle.attrLastMove('[notarget]')
            battle.add('-fail', pokemon)
            return False

        targets, pressure_targets = pokemon.getMoveTargets(move, target)
        if targets:
            target = targets[-1]

        caller_move_for_pressure = source_effect if (source_effect and source_effect.get('pp')) else None
        if not source_effect or caller_move_for_pressure:
            extra_pp = 0
            for source in pressure_targets:
                pp_drop = battle.runEvent('DeductPP', source, pokemon, move)
                if pp_drop is not True:
                    extra_pp += pp_drop or 0
            if extra_pp > 0:
                pokemon.deductPP(caller_move_for_pressure or moveOrMoveName, extra_pp)

        try_move_result = battle.singleEvent('TryMove', move, None, pokemon, target, move)
        if try_move_result:
            try_move_result = battle.runEvent('TryMove', pokemon, target, move)
        if not try_move_result:
            return try_move_result

        battle.singleEvent('UseMoveMessage', move, None, pokemon, target, move)

        if move.ignoreImmunity is None:
            move.ignoreImmunity = move.category == 'Status'

        if move.selfdestruct == 'always':
            battle.faint(pokemon, pokemon, move)

        damage = False
        if move.target in ('all', 'foeSide', 'allySide', 'allyTeam'):
            damage = self.tryMoveHit(targets, pokemon, move)
            if damage == '' and isinstance(damage, str):
                pokemon.moveThisTurnResult = NULL
            if damage or (is_number(damage) and damage == 0) or damage is None:
                move_result = True
        else:
            if not targets:
                battle.attrLastMove('[notarget]')
                battle.add('-fail', pokemon)
                return False
            move_result = self.trySpreadMoveHit(targets, pokemon, move)
        if move.selfBoost and move_result:
            self.moveHit(pokemon, pokemon, move, move.selfBoost, False, True)
        if not pokemon.hp:
            battle.faint(pokemon, pokemon, move)

        if not move_result:
            original_hp = pokemon.hp
            battle.singleEvent('MoveFail', move, None, target, pokemon, move)
            if pokemon and pokemon is not target and move.category != 'Status':
                battle.runEvent('EmergencyExit', pokemon, pokemon, None, original_hp)
            return False

        if not battle.suppressingSecondaries() and not move.flags.get('futuremove'):
            original_hp = pokemon.hp
            battle.singleEvent('AfterMoveSecondarySelf', move, None, pokemon, target, move)
            battle.runEvent('AfterMoveSecondarySelf', pokemon, target, move)
            if pokemon and pokemon is not target and move.category != 'Status':
                battle.runEvent('EmergencyExit', pokemon, pokemon, None, original_hp)
        return True

    def trySpreadMoveHit(self, targets, pokemon, move, notActive=False):
        battle = self.battle
        if len(targets) > 1 and not move.smartTarget:
            move.spreadHit = True
        move_steps = [
            self.hitStepInvulnerabilityEvent,
            self.hitStepTryHitEvent,
            self.hitStepTypeImmunity,
            self.hitStepTryImmunity,
            self.hitStepAccuracy,
            self.hitStepBreakProtect,
            self.hitStepStealBoosts,
            self.hitStepMoveHitLoop,
        ]
        if notActive:
            battle.setActiveMove(move, pokemon, targets[0])
        hit_result = battle.singleEvent('Try', move, None, pokemon, targets[0], move)
        if hit_result:
            hit_result = battle.singleEvent('PrepareHit', move, Obj(), targets[0], pokemon, move)
        if hit_result:
            hit_result = battle.runEvent('PrepareHit', pokemon, targets[0], move)
        if not hit_result:
            if hit_result is False:
                battle.add('-fail', pokemon)
                battle.attrLastMove('[still]')
            return isinstance(hit_result, str) and hit_result == ''

        at_least_one_failure = False
        for step in move_steps:
            hit_results = step(targets, pokemon, move)
            if hit_results is None:
                continue
            targets = [t for i, t in enumerate(targets) if _ok(hit_results[i])]
            at_least_one_failure = at_least_one_failure or any(v is False for v in hit_results)
            if move.smartTarget and at_least_one_failure:
                move.smartTarget = False
            if not targets:
                break
        move.hitTargets = targets
        move_result = bool(targets)
        if not move_result and not at_least_one_failure:
            pokemon.moveThisTurnResult = NULL
        hit_slot = [p.getSlot() for p in targets]
        if move.spreadHit:
            battle.attrLastMove('[spread] ' + ','.join(hit_slot))
        return move_result

    def hitStepInvulnerabilityEvent(self, targets, pokemon, move):
        battle = self.battle
        if move.id == 'helpinghand':
            return [True] * len(targets)
        hit_results = []
        for i, target in enumerate(targets):
            if target.volatiles.get('commanding'):
                r = False
            elif move.id == 'toxic' and pokemon.hasType('Poison'):
                r = True
            else:
                r = battle.runEvent('Invulnerability', target, pokemon, move)
            hit_results.append(r)
            if r is False:
                if move.smartTarget:
                    move.smartTarget = False
                else:
                    if not move.spreadHit:
                        battle.attrLastMove('[miss]')
                    battle.add('-miss', pokemon, target)
        return hit_results

    def hitStepTryHitEvent(self, targets, pokemon, move):
        battle = self.battle
        hit_results = battle.runEvent('TryHit', targets, pokemon, move)
        if True not in [r is True for r in hit_results] and any(r is False for r in hit_results):
            battle.add('-fail', pokemon)
            battle.attrLastMove('[still]')
        for i in range(len(targets)):
            r = hit_results[i]
            if not (isinstance(r, str) and r == ''):
                hit_results[i] = r or False
        return hit_results

    def hitStepTypeImmunity(self, targets, pokemon, move):
        if move.ignoreImmunity is None:
            move.ignoreImmunity = move.category == 'Status'
        return [t.runImmunity(move, not move.smartTarget) for t in targets]

    def hitStepTryImmunity(self, targets, pokemon, move):
        battle = self.battle
        hit_results = []
        for i, target in enumerate(targets):
            if move.flags.get('powder') and target is not pokemon and not self.dex.getImmunity('powder', target):
                battle.debug('natural powder immunity')
                battle.add('-immune', target)
                hit_results.append(False)
            elif not battle.singleEvent('TryImmunity', move, Obj(), target, pokemon, move):
                battle.add('-immune', target)
                hit_results.append(False)
            elif (move.pranksterBoosted and pokemon.hasAbility('prankster') and not targets[i].isAlly(pokemon) and
                  not self.dex.getImmunity('prankster', target)):
                battle.debug('natural prankster immunity')
                if target.illusion or not (move.status and not self.dex.getImmunity(move.status, target)):
                    battle.hint("Since gen 7, Dark is immune to Prankster moves.")
                battle.add('-immune', target)
                hit_results.append(False)
            else:
                hit_results.append(True)
        return hit_results

    def hitStepAccuracy(self, targets, pokemon, move):
        battle = self.battle
        hit_results = []
        for i, target in enumerate(targets):
            battle.activeTarget = target
            accuracy = move.accuracy
            if move.ohko:
                if not target.isSemiInvulnerable():
                    accuracy = 30
                    if move.ohko == 'Ice' and not pokemon.hasType('Ice'):
                        accuracy = 20
                    if pokemon.level >= target.level and (move.ohko is True or not target.hasType(move.ohko)):
                        accuracy += pokemon.level - target.level
                    else:
                        battle.add('-immune', target, '[ohko]')
                        hit_results.append(False)
                        continue
            else:
                accuracy = battle.runEvent('ModifyAccuracy', target, pokemon, move, accuracy)
                if accuracy is not True:
                    boost = 0
                    if not move.ignoreAccuracy:
                        boosts = battle.runEvent('ModifyBoost', pokemon, None, None, Obj(pokemon.boosts))
                        boost = clamp_int_range(boosts['accuracy'], -6, 6)
                    if not move.ignoreEvasion:
                        boosts = battle.runEvent('ModifyBoost', target, None, None, Obj(target.boosts))
                        boost = clamp_int_range(boost - boosts['evasion'], -6, 6)
                    if boost > 0:
                        accuracy = trunc(accuracy * (3 + boost) / 3)
                    elif boost < 0:
                        accuracy = trunc(accuracy * 3 / (3 - boost))
            if (move.alwaysHit or (move.id == 'toxic' and pokemon.hasType('Poison')) or
                    (move.target == 'self' and move.category == 'Status' and not target.isSemiInvulnerable())):
                accuracy = True
            else:
                accuracy = battle.runEvent('Accuracy', target, pokemon, move, accuracy)
            if accuracy is not True and not battle.randomChance(accuracy, 100):
                if move.smartTarget:
                    move.smartTarget = False
                else:
                    if not move.spreadHit:
                        battle.attrLastMove('[miss]')
                    battle.add('-miss', pokemon, target)
                if not move.ohko and pokemon.hasItem('blunderpolicy') and pokemon.useItem():
                    battle.boost(Obj(spe=2), pokemon)
                hit_results.append(False)
                continue
            hit_results.append(True)
        return hit_results

    def hitStepBreakProtect(self, targets, pokemon, move):
        battle = self.battle
        if move.breaksProtect:
            for target in targets:
                broke = False
                for effectid in ('banefulbunker', 'burningbulwark', 'kingsshield', 'obstruct', 'protect',
                                 'silktrap', 'spikyshield'):
                    if target.removeVolatile(effectid):
                        broke = True
                for effectid in ('craftyshield', 'matblock', 'quickguard', 'wideguard'):
                    if target.side.removeSideCondition(effectid):
                        broke = True
                if broke:
                    if move.id == 'feint':
                        battle.add('-activate', target, 'move: Feint')
                    else:
                        battle.add('-activate', target, f"move: {move.name}", '[broken]')
                    target.volatiles.pop('stall', None)
        return None

    def hitStepStealBoosts(self, targets, pokemon, move):
        battle = self.battle
        target = targets[0]
        if move.stealsBoosts:
            boosts = Obj()
            stolen = False
            for stat_name, stage in target.boosts.items():
                if stage > 0:
                    boosts[stat_name] = stage
                    stolen = True
            if stolen:
                battle.attrLastMove('[still]')
                battle.add('-clearpositiveboost', target, pokemon, 'move: ' + move.name)
                battle.boost(boosts, pokemon, pokemon)
                for stat_name in boosts:
                    boosts[stat_name] = 0
                target.setBoost(boosts)
                if move.id == 'spectralthief':
                    battle.addMove('-anim', pokemon, 'Spectral Thief', target)
        return None

    def afterMoveSecondaryEvent(self, targets, pokemon, move):
        battle = self.battle
        battle.singleEvent('AfterMoveSecondary', move, None, targets[0] if targets else None, pokemon, move)
        battle.runEvent('AfterMoveSecondary', targets, pokemon, move)
        return None

    def tryMoveHit(self, targetOrTargets, pokemon, move):
        battle = self.battle
        target = targetOrTargets[0] if isinstance(targetOrTargets, list) else targetOrTargets
        targets = targetOrTargets if isinstance(targetOrTargets, list) else [target]
        battle.setActiveMove(move, pokemon, targets[0] if targets else None)
        hit_result = battle.singleEvent('Try', move, None, pokemon, target, move)
        if hit_result:
            hit_result = battle.singleEvent('PrepareHit', move, Obj(), target, pokemon, move)
        if hit_result:
            hit_result = battle.runEvent('PrepareHit', pokemon, target, move)
        if not hit_result:
            if hit_result is False:
                battle.add('-fail', pokemon)
                battle.attrLastMove('[still]')
            return False
        if move.target == 'all':
            hit_result = battle.runEvent('TryHitField', target, pokemon, move)
        else:
            hit_result = battle.runEvent('TryHitSide', target, pokemon, move)
        if not hit_result:
            if hit_result is False:
                battle.add('-fail', pokemon)
                battle.attrLastMove('[still]')
            return False
        return self.moveHit(target, pokemon, move)

    def hitStepMoveHitLoop(self, targets, pokemon, move):
        """Champions override: AfterMoveSecondary events always run (Sheer Force doesn't suppress them)."""
        battle = self.battle
        damage = [0] * len(targets)
        move.totalDamage = 0
        pokemon.lastDamage = 0
        target_hits = move.multihit or 1
        if isinstance(target_hits, list):
            if target_hits[0] == 2 and target_hits[1] == 5:
                target_hits = battle.sample(_MULTIHIT_2_5)
                if target_hits < 4 and pokemon.hasItem('loadeddice'):
                    target_hits = 5 - battle.random(2)
            else:
                target_hits = battle.random(target_hits[0], target_hits[1] + 1)
        if target_hits == 10 and pokemon.hasItem('loadeddice'):
            target_hits -= battle.random(7)
        target_hits = math.floor(target_hits)
        null_damage = True
        move_damage = []
        is_sleep_usable = move.sleepUsable or self.dex.moves.get(move.sourceEffect).sleepUsable

        targets_copy = list(targets)
        hit = 1
        while hit <= target_hits:
            if any(d is False for d in damage):
                break
            if hit > 1 and pokemon.status == 'slp' and not is_sleep_usable:
                break
            if all(not (t and t.hp) for t in targets):
                break
            move.hit = hit
            move.lastHit = move.hit == target_hits
            if move.smartTarget and len(targets) > 1:
                targets_copy = [targets[hit - 1]]
                damage = [damage[hit - 1]]
            else:
                targets_copy = list(targets)
            target = targets_copy[0]
            if target and isinstance(move.smartTarget, bool):
                if hit > 1:
                    battle.addMove('-anim', pokemon, move.name, target)
                else:
                    battle.retargetLastMove(target)

            if target and move.multiaccuracy and hit > 1:
                accuracy = move.accuracy
                if accuracy is not True:
                    if not move.ignoreAccuracy:
                        boosts = battle.runEvent('ModifyBoost', pokemon, None, None, Obj(pokemon.boosts))
                        boost = clamp_int_range(boosts['accuracy'], -6, 6)
                        if boost > 0:
                            accuracy *= _ACC_BOOST_TABLE[boost]
                        else:
                            accuracy /= _ACC_BOOST_TABLE[-boost]
                    if not move.ignoreEvasion:
                        boosts = battle.runEvent('ModifyBoost', target, None, None, Obj(target.boosts))
                        boost = clamp_int_range(boosts['evasion'], -6, 6)
                        if boost > 0:
                            accuracy /= _ACC_BOOST_TABLE[boost]
                        elif boost < 0:
                            accuracy *= _ACC_BOOST_TABLE[-boost]
                accuracy = battle.runEvent('ModifyAccuracy', target, pokemon, move, accuracy)
                if not move.alwaysHit:
                    accuracy = battle.runEvent('Accuracy', target, pokemon, move, accuracy)
                    if accuracy is not True and not battle.randomChance(accuracy, 100):
                        break

            move_data = move
            if not move_data.flags:
                move_data.flags = Obj()
            move_damage_this_hit, targets_copy = self.spreadMoveHit(targets_copy, pokemon, move, move_data)
            if move.smartTarget:
                move_damage.extend(move_damage_this_hit)
            else:
                move_damage = move_damage_this_hit
            if not any(v is not False for v in move_damage):
                break
            null_damage = False
            for i, md in enumerate(move_damage):
                if move.smartTarget and i != hit - 1:
                    continue
                damage[i] = 0 if (md is True or not md) else md
                move.totalDamage += damage[i]
            battle.eachEvent('Update')
            if not pokemon.hp and len(targets) == 1:
                hit += 1
                break
            hit += 1
        if hit == 1:
            return [False] * len(damage)
        if null_damage:
            damage = [False] * len(damage)
        battle.faintMessages(False, False, not pokemon.hp)
        if move.multihit and not isinstance(move.smartTarget, bool) and \
                not (move.hit == 1 and move.multihitType == 'parentalbond'):
            battle.add('-hitcount', targets[0], hit - 1)

        if move.totalDamage:
            self.applyRecoilDamage(move.totalDamage, move, pokemon)

        if move.smartTarget:
            targets_copy = list(targets)

        for i, target in enumerate(targets_copy):
            if target and pokemon is not target:
                md = move_damage[i] if i < len(move_damage) else None
                target.gotAttacked(move, md, pokemon)
                if is_number(md):
                    target.timesAttacked += 1 if move.smartTarget else hit - 1

        if move.ohko and not targets[0].hp:
            battle.add('-ohko')

        if not any(bool(v) or (is_number(v) and v == 0) for v in damage):
            return damage

        battle.eachEvent('Update')

        self.afterMoveSecondaryEvent([t for t in targets_copy if t], pokemon, move)

        for i, d in enumerate(damage):
            cur_damage = move.totalDamage if len(targets) == 1 else d
            if is_number(cur_damage) and targets[i].hp:
                target_hp_before = (targets[i].hurtThisTurn or 0) + cur_damage
                battle.runEvent('EmergencyExit', targets[i], pokemon, None, target_hp_before)
        return damage

    def spreadMoveHit(self, targets, pokemon, moveOrMoveName, hitEffect=None, isSecondary=False, isSelf=False):
        """Champions override: AfterHit runs even if the source fainted."""
        battle = self.battle
        target = targets[0] if targets else None
        damage = [True] * len(targets)
        move = self.dex.getActiveMove(moveOrMoveName)
        hit_result = True
        move_data = hitEffect if hitEffect else move
        if not move_data.flags:
            move_data.flags = Obj()
        if move.target == 'all' and not isSelf:
            hit_result = battle.singleEvent('TryHitField', move_data, Obj(), target or None, pokemon, move)
        elif move.target in ('foeSide', 'allySide', 'allyTeam') and not isSelf:
            hit_result = battle.singleEvent('TryHitSide', move_data, Obj(), target or None, pokemon, move)
        elif target:
            hit_result = battle.singleEvent('TryHit', move_data, Obj(), target, pokemon, move)
        if not hit_result:
            if hit_result is False:
                battle.add('-fail', pokemon)
                battle.attrLastMove('[still]')
            return [False], targets

        if not isSecondary and not isSelf:
            if move.target not in ('all', 'allyTeam', 'allySide', 'foeSide'):
                damage = self.tryPrimaryHitEvent(damage, targets, pokemon, move, move_data, isSecondary)

        for i in range(len(targets)):
            if is_number(damage[i]) and damage[i] == 0 and damage[i] is not False:
                # HIT_SUBSTITUTE
                damage[i] = True
                targets[i] = NULL
            if targets[i] and isSecondary and not move_data.self:
                damage[i] = True
            if not damage[i]:
                targets[i] = False

        damage = self.getSpreadDamage(damage, targets, pokemon, move, move_data, isSecondary, isSelf)
        for i in range(len(targets)):
            if damage[i] is False:
                targets[i] = False

        damage = battle.spreadDamage(damage, targets, pokemon, move)
        for i in range(len(targets)):
            if damage[i] is False:
                targets[i] = False

        damage = self.runMoveEffects(damage, targets, pokemon, move, move_data, isSecondary, isSelf)
        for i in range(len(targets)):
            if not _ok(damage[i]):
                targets[i] = False

        active_target = battle.activeTarget
        if move_data.self and not move.selfDropped:
            self.selfDrops(targets, pokemon, move, move_data, isSecondary)
        if move_data.secondaries:
            self.secondaries(targets, pokemon, move, move_data, isSelf)
        battle.activeTarget = active_target

        if move_data.forceSwitch:
            damage = self.forceSwitch(damage, targets, pokemon, move)

        for i in range(len(targets)):
            if not _ok(damage[i]):
                targets[i] = False

        damaged_targets = []
        damaged_damage = []
        for i, t in enumerate(targets):
            if is_number(damage[i]) and t:
                damaged_targets.append(t)
                damaged_damage.append(damage[i])
        pokemon_original_hp = pokemon.hp
        if damaged_damage and not isSecondary and not isSelf:
            battle.runEvent('DamagingHit', damaged_targets, pokemon, move, damaged_damage)
            if move_data.onAfterHit:
                for t in damaged_targets:
                    battle.singleEvent('AfterHit', move_data, Obj(), t, pokemon, move)
            battle.runEvent('EmergencyExit', pokemon, None, None, pokemon_original_hp)
        return damage, targets

    def tryPrimaryHitEvent(self, damage, targets, pokemon, move, moveData, isSecondary=False):
        battle = self.battle
        for i, target in enumerate(targets):
            if not target:
                continue
            damage[i] = battle.runEvent('TryPrimaryHit', target, pokemon, moveData)
        return damage

    def getSpreadDamage(self, damage, targets, source, move, moveData, isSecondary=False, isSelf=False):
        battle = self.battle
        for i, target in enumerate(targets):
            if not target:
                continue
            battle.activeTarget = target
            damage[i] = None
            cur_damage = self.getDamage(source, target, moveData)
            if cur_damage is False or cur_damage is NULL:
                if damage[i] is False and not isSecondary and not isSelf:
                    battle.add('-fail', source)
                    battle.attrLastMove('[still]')
                battle.debug('damage calculation interrupted')
                damage[i] = False
                continue
            damage[i] = cur_damage
        return damage

    def runMoveEffects(self, damage, targets, source, move, moveData, isSecondary=False, isSelf=False):
        battle = self.battle
        did_anything = damage[0] if damage else None
        for d in damage[1:]:
            did_anything = self.combineResults(did_anything, d)
        if not damage:
            did_anything = None
        for i, target in enumerate(targets):
            if target is False:
                continue
            did_something = None
            if target:
                if moveData.boosts and not target.fainted:
                    hit_result = battle.boost(moveData.boosts, target, source, move, isSecondary, isSelf)
                    did_something = self.combineResults(did_something, hit_result)
                if moveData.heal and not target.fainted:
                    if target.hp >= target.maxhp:
                        battle.add('-fail', target, 'heal')
                        battle.attrLastMove('[still]')
                        damage[i] = self.combineResults(damage[i], False)
                        did_anything = self.combineResults(did_anything, NULL)
                        continue
                    amount = target.baseMaxhp * moveData.heal[0] / moveData.heal[1]
                    d = battle.heal(js_round(amount), target, source, move)
                    if not _ok(d):
                        if d is not NULL:
                            battle.add('-fail', source)
                            battle.attrLastMove('[still]')
                        battle.debug('heal interrupted')
                        damage[i] = self.combineResults(damage[i], False)
                        did_anything = self.combineResults(did_anything, NULL)
                        continue
                    did_something = True
                if moveData.status:
                    hit_result = target.trySetStatus(moveData.status, source,
                                                     moveData.ability if moveData.ability else move)
                    if not hit_result and move.status:
                        damage[i] = self.combineResults(damage[i], False)
                        did_anything = self.combineResults(did_anything, NULL)
                        continue
                    did_something = self.combineResults(did_something, hit_result)
                if moveData.forceStatus:
                    hit_result = target.setStatus(moveData.forceStatus, source, move)
                    did_something = self.combineResults(did_something, hit_result)
                if moveData.volatileStatus:
                    hit_result = target.addVolatile(moveData.volatileStatus, source, move)
                    did_something = self.combineResults(did_something, hit_result)
                if moveData.sideCondition:
                    hit_result = target.side.addSideCondition(moveData.sideCondition, source, move)
                    did_something = self.combineResults(did_something, hit_result)
                if moveData.slotCondition:
                    hit_result = target.side.addSlotCondition(target, moveData.slotCondition, source, move)
                    did_something = self.combineResults(did_something, hit_result)
                if moveData.weather:
                    hit_result = battle.field.setWeather(moveData.weather, source, move)
                    did_something = self.combineResults(did_something, hit_result)
                if moveData.terrain:
                    hit_result = battle.field.setTerrain(moveData.terrain, source, move)
                    did_something = self.combineResults(did_something, hit_result)
                if moveData.pseudoWeather:
                    hit_result = battle.field.addPseudoWeather(moveData.pseudoWeather, source, move)
                    did_something = self.combineResults(did_something, hit_result)
                if moveData.forceSwitch:
                    hit_result = bool(battle.canSwitch(target.side))
                    did_something = self.combineResults(did_something, hit_result)
                if move.target == 'all' and not isSelf:
                    if moveData.onHitField:
                        hit_result = battle.singleEvent('HitField', moveData, Obj(), target, source, move)
                        did_something = self.combineResults(did_something, hit_result)
                elif move.target in ('foeSide', 'allySide') and not isSelf:
                    if moveData.onHitSide:
                        hit_result = battle.singleEvent('HitSide', moveData, Obj(), target.side, source, move)
                        did_something = self.combineResults(did_something, hit_result)
                else:
                    if moveData.onHit:
                        hit_result = battle.singleEvent('Hit', moveData, Obj(), target, source, move)
                        did_something = self.combineResults(did_something, hit_result)
                    if not isSelf and not isSecondary:
                        battle.runEvent('Hit', target, source, move)
            if moveData.selfdestruct == 'ifHit' and damage[i] is not False:
                battle.faint(source, source, move)
            if moveData.selfSwitch:
                if battle.canSwitch(source.side) and not source.volatiles.get('commanded'):
                    did_something = True
                else:
                    did_something = self.combineResults(did_something, False)
            if did_something is None:
                did_something = True
            damage[i] = self.combineResults(damage[i], False if did_something is NULL else did_something)
            did_anything = self.combineResults(did_anything, did_something)

        if not did_anything and not (is_number(did_anything) and did_anything == 0) and \
                not moveData.self and not moveData.selfdestruct:
            if not isSelf and not isSecondary:
                if did_anything is False:
                    battle.add('-fail', source)
                    battle.attrLastMove('[still]')
            battle.debug('move failed because it did nothing')
        elif move.selfSwitch and source.hp and not source.volatiles.get('commanded'):
            source.switchFlag = move.id
        return damage

    def selfDrops(self, targets, source, move, moveData, isSecondary=False):
        battle = self.battle
        for target in targets:
            if target is False:
                continue
            if moveData.self and not move.selfDropped:
                if not isSecondary and moveData.self.boosts:
                    secondary_roll = battle.random(100)
                    if moveData.self.chance is None or secondary_roll < moveData.self.chance:
                        self.moveHit(source, source, move, moveData.self, isSecondary, True)
                    if not move.multihit:
                        move.selfDropped = True
                else:
                    self.moveHit(source, source, move, moveData.self, isSecondary, True)

    def secondaries(self, targets, source, move, moveData, isSelf=False):
        battle = self.battle
        if not moveData.secondaries:
            return
        for target in targets:
            if target is False:
                continue
            secondaries = battle.runEvent('ModifySecondaries', target, source, moveData,
                                          list(moveData.secondaries))
            for secondary in secondaries:
                secondary_roll = battle.random(100)
                if secondary.chance is None or secondary_roll < secondary.chance:
                    self.moveHit(target, source, move, secondary, True, isSelf)

    def forceSwitch(self, damage, targets, source, move):
        battle = self.battle
        for i, target in enumerate(targets):
            if target and target.hp > 0 and source.hp > 0 and battle.canSwitch(target.side):
                hit_result = battle.runEvent('DragOut', target, source, move)
                if hit_result:
                    target.forceSwitchFlag = True
                elif hit_result is False and move.category == 'Status':
                    battle.add('-fail', source)
                    battle.attrLastMove('[still]')
                    damage[i] = False
        return damage

    def moveHit(self, targets, pokemon, moveOrMoveName, moveData=None, isSecondary=False, isSelf=False):
        if not isinstance(targets, list):
            targets = [targets]
        ret = self.spreadMoveHit(targets, pokemon, moveOrMoveName, moveData, isSecondary, isSelf)[0][0]
        return None if ret is True else ret

    def applyRecoilDamage(self, damageDealt, move, pokemon):
        battle = self.battle
        if move.struggleRecoil:
            recoil_damage = clamp_int_range(js_round(pokemon.baseMaxhp / 4), 1)
        elif move.mindBlownRecoil or move.chloroblastRecoil:
            recoil_damage = js_round(pokemon.maxhp / 2)
        elif move.recoil:
            recoil_damage = clamp_int_range(js_round(damageDealt * move.recoil[0] / move.recoil[1]), 1)
        else:
            return None
        hp_before = pokemon.hp
        if move.struggleRecoil:
            battle.directDamage(recoil_damage, pokemon, pokemon, Obj(id='strugglerecoil'))
        else:
            effect = self.dex.conditions.get(move.name) if move.mindBlownRecoil else 'recoil'
            battle.damage(recoil_damage, pokemon, pokemon, effect)
        battle.runEvent('EmergencyExit', pokemon, pokemon, None, hp_before)
        return recoil_damage

    def targetTypeChoices(self, targetType):
        return targetType in CHOOSABLE_TARGETS

    def combineResults(self, left, right):
        tl = js_typeof(left)
        tr = js_typeof(right)
        li = _RESULT_PRIORITIES.index(tl) if tl in _RESULT_PRIORITIES else -1
        ri = _RESULT_PRIORITIES.index(tr) if tr in _RESULT_PRIORITIES else -1
        if li > ri:
            return left
        elif left and not right and not (is_number(right) and right == 0):
            return left
        elif is_number(left) and is_number(right):
            return left + right
        else:
            return right

    def getDamage(self, source, target, move, suppressMessages=False):
        battle = self.battle
        if isinstance(move, str):
            move = self.dex.getActiveMove(move)
        if is_number(move):
            from .dex import Move
            base_power = move
            move = Move(basePower=base_power, type='???', category='Physical', willCrit=False, flags=Obj(),
                        id='', name='', effectType='Move', critRatio=1, priority=0)
            move.hit = 0

        if not target.runImmunity(move, not suppressMessages):
            return False
        if move.ohko:
            return target.maxhp
        if move.damageCallback:
            return move.damageCallback(battle, source, target)
        if move.damage == 'level':
            return source.level
        elif move.damage:
            return move.damage

        category = battle.getCategory(move) if move.id else (move.category or 'Physical')
        base_power = move.basePower
        if move.basePowerCallback:
            base_power = move.basePowerCallback(battle, source, target, move)
        if not base_power:
            return None if (is_number(base_power) and base_power == 0) else base_power
        base_power = clamp_int_range(base_power, 1)

        crit_ratio = battle.runEvent('ModifyCritRatio', source, target, move, move.critRatio or 0)
        crit_ratio = clamp_int_range(crit_ratio, 0, 4)
        crit_mult = [0, 24, 8, 2, 1]

        move_hit = target.getMoveHitData(move)
        move_hit.crit = move.willCrit or False
        if move.willCrit is None:
            if crit_ratio:
                move_hit.crit = battle.randomChance(1, crit_mult[crit_ratio])
        if move_hit.crit:
            move_hit.crit = battle.runEvent('CriticalHit', target, None, move)

        base_power = battle.runEvent('BasePower', source, target, move, base_power, True)
        if not base_power:
            return 0
        base_power = clamp_int_range(base_power, 1)

        level = source.level
        attacker = target if move.overrideOffensivePokemon == 'target' else source
        defender = source if move.overrideDefensivePokemon == 'source' else target
        is_physical = move.category == 'Physical'
        attack_stat = move.overrideOffensiveStat or ('atk' if is_physical else 'spa')
        defense_stat = move.overrideDefensiveStat or ('def' if is_physical else 'spd')

        atk_boosts = attacker.boosts[attack_stat]
        def_boosts = defender.boosts[defense_stat]
        ignore_negative_offensive = bool(move.ignoreNegativeOffensive)
        ignore_positive_defensive = bool(move.ignorePositiveDefensive)
        if move_hit.crit:
            ignore_negative_offensive = True
            ignore_positive_defensive = True
        ignore_offensive = bool(move.ignoreOffensive or (ignore_negative_offensive and atk_boosts < 0))
        ignore_defensive = bool(move.ignoreDefensive or (ignore_positive_defensive and def_boosts > 0))
        if ignore_offensive:
            battle.debug('Negating (sp)atk boost/penalty.')
            atk_boosts = 0
        if ignore_defensive:
            battle.debug('Negating (sp)def boost/penalty.')
            def_boosts = 0

        attack = attacker.calculateStat(attack_stat, atk_boosts, 1, source)
        defense = defender.calculateStat(defense_stat, def_boosts, 1, target)

        attack_stat = 'atk' if category == 'Physical' else 'spa'
        attack = battle.runEvent('Modify' + _STAT_TABLE[attack_stat], source, target, move, attack)
        defense = battle.runEvent('Modify' + _STAT_TABLE[defense_stat], target, source, move, defense)

        base_damage = trunc(trunc(trunc(trunc(2 * level / 5 + 2) * base_power * attack) / defense) / 50)
        return self.modifyDamage(base_damage, source, target, move, suppressMessages)

    def modifyDamage(self, baseDamage, pokemon, target, move, suppressMessages=False):
        """Champions override: announces 4x / 0.25x effectiveness."""
        battle = self.battle
        if not move.type:
            move.type = '???'
        type_ = move.type
        baseDamage += 2
        if move.spreadHit:
            spread_modifier = 0.75
            battle.debug(f"Spread modifier: {spread_modifier}")
            baseDamage = battle.modify(baseDamage, spread_modifier)
        elif move.multihitType == 'parentalbond' and move.hit > 1:
            baseDamage = battle.modify(baseDamage, 0.25)

        baseDamage = battle.priorityEvent('WeatherModifyDamage', pokemon, target, move, baseDamage)

        is_crit = target.getMoveHitData(move).crit
        if is_crit:
            baseDamage = trunc(baseDamage * (move.critModifier or 1.5))

        baseDamage = battle.randomizer(baseDamage)

        if type_ != '???':
            stab = 1
            is_stab = move.forceSTAB or pokemon.hasType(type_) or type_ in pokemon.getTypes(False, True)
            if is_stab:
                stab = 1.5
            if pokemon.terastallized == 'Stellar':
                pass  # no terastallization in champions
            else:
                if pokemon.terastallized == type_ and type_ in pokemon.getTypes(False, True):
                    stab = 2
                stab = battle.runEvent('ModifySTAB', pokemon, target, move, stab)
            baseDamage = battle.modify(baseDamage, stab)

        type_mod = target.runEffectiveness(move)
        type_mod = clamp_int_range(type_mod, -6, 6)
        target.getMoveHitData(move).typeMod = type_mod
        if type_mod > 0:
            if not suppressMessages:
                battle.add('-supereffective', target, min(type_mod, 2))
            for _ in range(type_mod):
                baseDamage *= 2
        if type_mod < 0:
            if not suppressMessages:
                battle.add('-resisted', target, min(-type_mod, 2))
            for _ in range(-type_mod):
                baseDamage = trunc(baseDamage / 2)

        if is_crit and not suppressMessages:
            battle.add('-crit', target)

        if pokemon.status == 'brn' and move.category == 'Physical' and not pokemon.hasAbility('guts'):
            if move.id != 'facade':
                baseDamage = battle.modify(baseDamage, 0.5)

        baseDamage = battle.runEvent('ModifyDamage', pokemon, target, move, baseDamage)

        bypass_protect = target.getMoveHitData(move).bypassProtect
        if bypass_protect:
            baseDamage = battle.modify(baseDamage, 0.25)
            if bypass_protect is not True and bypass_protect.effectType == 'Ability':
                battle.add('-ability', pokemon, bypass_protect.name)
            battle.add('-zbroken', target)

        if not baseDamage:
            return 1
        return trunc(baseDamage, 16)

    def getConfusionDamage(self, pokemon, basePower):
        battle = self.battle
        attack = pokemon.calculateStat('atk', pokemon.boosts['atk'])
        defense = pokemon.calculateStat('def', pokemon.boosts['def'])
        level = pokemon.level
        base_damage = trunc(trunc(trunc(trunc(2 * level / 5 + 2) * basePower * attack) / defense) / 50) + 2
        damage = trunc(base_damage, 16)
        damage = battle.randomizer(damage)
        return max(1, damage)

    # ----------------------------------------------------------- MEGA EVOLUTION
    def canMegaEvo(self, pokemon):
        """Champions override."""
        species = pokemon.baseSpecies
        item = pokemon.getItem()
        mega_stone = item.megaStone
        if not mega_stone:
            return None
        return mega_stone.get(species.name) or None

    def canUltraBurst(self, pokemon):
        return None

    def runMegaEvo(self, pokemon):
        battle = self.battle
        speciesid = pokemon.canMegaEvo or pokemon.canUltraBurst
        if not speciesid:
            return False
        pokemon.formeChange(speciesid, pokemon.getItem(), True)
        was_mega = pokemon.canMegaEvo
        for ally in pokemon.side.pokemon:
            if was_mega:
                ally.canMegaEvo = False
            else:
                ally.canUltraBurst = None
        battle.runEvent('AfterMega', pokemon)
        return True

    def canTerastallize(self, pokemon):
        return None
