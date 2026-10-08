"""Generic conditions (``data/conditions.ts`` + champions ``conditions.ts``), ported to Python.

Each class is a condition id; its methods are the condition's event handlers.
Handlers receive the Battle as ``self`` (Showdown's ``this``).
"""
from __future__ import annotations

import math

from ..js import NULL, Obj, js_round, to_id


class brn:
    def onStart(self, target, source, sourceEffect):
        if sourceEffect and sourceEffect.id == 'flameorb':
            self.add('-status', target, 'brn', '[from] item: Flame Orb')
        elif sourceEffect and sourceEffect.effectType == 'Ability':
            self.add('-status', target, 'brn', '[from] ability: ' + sourceEffect.name, f"[of] {source}")
        else:
            self.add('-status', target, 'brn')

    def onResidual(self, pokemon):
        self.damage(pokemon.baseMaxhp / 16)


class par:
    def onStart(self, target, source, sourceEffect):
        if sourceEffect and sourceEffect.effectType == 'Ability':
            self.add('-status', target, 'par', '[from] ability: ' + sourceEffect.name, f"[of] {source}")
        else:
            self.add('-status', target, 'par')

    def onModifySpe(self, spe, pokemon):
        spe = self.finalModify(spe)
        if not pokemon.hasAbility('quickfeet'):
            spe = math.floor(spe * 50 / 100)
        return spe

    def onBeforeMove(self, pokemon):
        # Champions: 1/8 chance of full paralysis
        if self.randomChance(1, 8):
            self.add('cant', pokemon, 'par')
            return False


class slp:
    def onStart(self, target, source, sourceEffect):
        if sourceEffect and sourceEffect.effectType == 'Ability':
            self.add('-status', target, 'slp', '[from] ability: ' + sourceEffect.name, f"[of] {source}")
        elif sourceEffect and sourceEffect.effectType == 'Move':
            self.add('-status', target, 'slp', f"[from] move: {sourceEffect.name}")
        else:
            self.add('-status', target, 'slp')
        # Champions: 1/3 chance to wake up on turn 2, otherwise on turn 3
        self.effectState.startTime = self.sample([2, 3, 3])
        self.effectState.time = self.effectState.startTime
        if target.removeVolatile('nightmare'):
            self.add('-end', target, 'Nightmare', '[silent]')

    def onBeforeMove(self, pokemon, target, move):
        if pokemon.hasAbility('earlybird'):
            pokemon.statusState.time -= 1
        pokemon.statusState.time -= 1
        if pokemon.statusState.time <= 0:
            pokemon.cureStatus()
            return None
        self.add('cant', pokemon, 'slp')
        if move.sleepUsable:
            return None
        return False


class frz:
    def onStart(self, target, source, sourceEffect):
        if sourceEffect and sourceEffect.effectType == 'Ability':
            self.add('-status', target, 'frz', '[from] ability: ' + sourceEffect.name, f"[of] {source}")
        else:
            self.add('-status', target, 'frz')
        if target.species.name == 'Shaymin-Sky' and target.baseSpecies.baseSpecies == 'Shaymin':
            target.formeChange('Shaymin', self.effect, True)
        # Champions: thaws after at most 3 turns
        self.effectState.startTime = 3
        self.effectState.time = self.effectState.startTime

    def onBeforeMove(self, pokemon, target, move):
        if move.flags.get('defrost') and not (move.id == 'burnup' and not pokemon.hasType('Fire')):
            return None
        pokemon.statusState.time -= 1
        if pokemon.statusState.time <= 0 or self.randomChance(1, 4):
            pokemon.cureStatus()
            return None
        self.add('cant', pokemon, 'frz')
        return False

    def onModifyMove(self, move, pokemon):
        if move.flags.get('defrost'):
            self.add('-curestatus', pokemon, 'frz', f"[from] move: {move}")
            pokemon.clearStatus()

    def onAfterMoveSecondary(self, target, source, move):
        if move.thawsTarget:
            target.cureStatus()

    def onDamagingHit(self, damage, target, source, move):
        if move.type == 'Fire' and move.category != 'Status' and move.id != 'polarflare':
            target.cureStatus()


class psn:
    def onStart(self, target, source, sourceEffect):
        if sourceEffect and sourceEffect.effectType == 'Ability':
            self.add('-status', target, 'psn', '[from] ability: ' + sourceEffect.name, f"[of] {source}")
        else:
            self.add('-status', target, 'psn')

    def onResidual(self, pokemon):
        self.damage(pokemon.baseMaxhp / 8)


class tox:
    def onStart(self, target, source, sourceEffect):
        self.effectState.stage = 0
        if sourceEffect and sourceEffect.id == 'toxicorb':
            self.add('-status', target, 'tox', '[from] item: Toxic Orb')
        elif sourceEffect and sourceEffect.effectType == 'Ability':
            self.add('-status', target, 'tox', '[from] ability: ' + sourceEffect.name, f"[of] {source}")
        else:
            self.add('-status', target, 'tox')

    def onSwitchIn(self):
        self.effectState.stage = 0

    def onResidual(self, pokemon):
        if self.effectState.stage < 15:
            self.effectState.stage += 1
        self.damage(self.clampIntRange(pokemon.baseMaxhp / 16, 1) * self.effectState.stage)


class confusion:
    def onStart(self, target, source, sourceEffect):
        if sourceEffect and sourceEffect.id == 'lockedmove':
            self.add('-start', target, 'confusion', '[fatigue]')
        elif sourceEffect and sourceEffect.effectType == 'Ability':
            self.add('-start', target, 'confusion', '[from] ability: ' + sourceEffect.name, f"[of] {source}")
        else:
            self.add('-start', target, 'confusion')
        mn = 3 if (sourceEffect and sourceEffect.id == 'axekick') else 2
        self.effectState.time = self.random(mn, 6)

    def onEnd(self, target):
        self.add('-end', target, 'confusion')

    def onBeforeMove(self, pokemon):
        pokemon.volatiles['confusion'].time -= 1
        if not pokemon.volatiles['confusion'].time:
            pokemon.removeVolatile('confusion')
            return None
        self.add('-activate', pokemon, 'confusion')
        if not self.randomChance(33, 100):
            return None
        self.activeTarget = pokemon
        damage = self.actions.getConfusionDamage(pokemon, 40)
        active_move = Obj(id='confused', effectType='Move', type='???')
        self.damage(damage, pokemon, pokemon, active_move)
        return False


class flinch:
    def onBeforeMove(self, pokemon):
        self.add('cant', pokemon, 'flinch')
        self.runEvent('Flinch', pokemon)
        return False


class trapped:
    def onTrapPokemon(self, pokemon):
        pokemon.tryTrap()

    def onStart(self, target):
        self.add('-activate', target, 'trapped')


class partiallytrapped:
    def durationCallback(self, target, source):
        if source and source.hasItem('gripclaw'):
            return 8
        return self.random(5, 7)

    def onStart(self, pokemon, source):
        self.add('-activate', pokemon, 'move: ' + str(self.effectState.sourceEffect), f"[of] {source}")
        self.effectState.boundDivisor = 6 if source.hasItem('bindingband') else 8

    def onResidual(self, pokemon):
        source = self.effectState.source
        gmax = self.effectState.sourceEffect.id in ('gmaxcentiferno', 'gmaxsandblast')
        if source and (not source.isActive or source.hp <= 0 or not source.activeTurns) and not gmax:
            pokemon.volatiles.pop('partiallytrapped', None)
            self.add('-end', pokemon, self.effectState.sourceEffect, '[partiallytrapped]', '[silent]')
            return None
        self.damage(pokemon.baseMaxhp / self.effectState.boundDivisor)

    def onEnd(self, pokemon):
        self.add('-end', pokemon, self.effectState.sourceEffect, '[partiallytrapped]')

    def onTrapPokemon(self, pokemon):
        gmax = self.effectState.sourceEffect.id in ('gmaxcentiferno', 'gmaxsandblast')
        src = self.effectState.source
        if (src and src.isActive) or gmax:
            pokemon.tryTrap()


class lockedmove:
    def onResidual(self, target):
        if target.status == 'slp':
            target.volatiles.pop('lockedmove', None)
        self.effectState.trueDuration -= 1

    def onStart(self, target, source, effect):
        self.effectState.trueDuration = self.random(2, 4)
        self.effectState.move = effect.id

    def onRestart(self):
        if self.effectState.trueDuration >= 2:
            self.effectState.duration = 2

    def onAfterMove(self, pokemon):
        if self.effectState.duration == 1:
            pokemon.removeVolatile('lockedmove')

    def onEnd(self, target):
        if self.effectState.trueDuration > 1:
            return None
        target.addVolatile('confusion')

    def onLockMove(self, pokemon):
        if pokemon.volatiles.get('dynamax'):
            return None
        return self.effectState.move


class twoturnmove:
    def onStart(self, attacker, defender, effect):
        self.effectState.move = effect.id
        attacker.addVolatile(effect.id)
        move_target_loc = attacker.lastMoveTargetLoc
        if effect.sourceEffect and self.dex.moves.get(effect.id).target != 'self':
            if defender.fainted:
                defender = self.sample(attacker.foes(True))
            move_target_loc = attacker.getLocOf(defender)
        attacker.volatiles[effect.id].targetLoc = move_target_loc
        self.attrLastMove('[still]')
        self.runEvent('PrepareHit', attacker, defender, effect)

    def onEnd(self, target):
        target.removeVolatile(self.effectState.move)

    def onLockMove(self):
        return self.effectState.move

    def onMoveAborted(self, pokemon):
        pokemon.removeVolatile('twoturnmove')


class choicelock:
    def onStart(self, pokemon):
        if not self.activeMove:
            raise ValueError('Battle.activeMove is null')
        if not self.activeMove.id or self.activeMove.hasBounced or self.activeMove.sourceEffect == 'snatch':
            return False
        self.effectState.move = self.activeMove.id

    def onBeforeMove(self, pokemon, target, move):
        if not pokemon.getItem().isChoice:
            pokemon.removeVolatile('choicelock')
            return None
        if (not pokemon.ignoringItem() and not pokemon.volatiles.get('dynamax') and
                move.id != self.effectState.move and move.id != 'struggle'):
            self.addMove('move', pokemon, move.name)
            self.attrLastMove('[still]')
            self.debug('Disabled by Choice item lock')
            self.add('-fail', pokemon)
            return False

    def onDisableMove(self, pokemon):
        if not pokemon.getItem().isChoice or not pokemon.hasMove(self.effectState.move):
            pokemon.removeVolatile('choicelock')
            return None
        if pokemon.ignoringItem() or pokemon.volatiles.get('dynamax'):
            return None
        for ms in pokemon.moveSlots:
            if ms.id != self.effectState.move:
                pokemon.disableMove(ms.id, False, self.effectState.sourceEffect)


class mustrecharge:
    def onBeforeMove(self, pokemon):
        self.add('cant', pokemon, 'recharge')
        pokemon.removeVolatile('mustrecharge')
        pokemon.removeVolatile('truant')
        return NULL

    def onStart(self, pokemon):
        self.add('-mustrecharge', pokemon)


class futuremove:
    def onStart(self, target):
        self.effectState.targetSlot = target.getSlot()
        self.effectState.endingTurn = self.turn - 1 + 2
        if self.effectState.endingTurn >= 254:
            self.hint('In Gen 8+, Future attacks will never resolve when used on the 255th turn or later.')

    def onResidual(self, target):
        if self.getOverflowedTurnCount() < self.effectState.endingTurn:
            return None
        target.side.removeSlotCondition(self.getAtSlot(self.effectState.targetSlot), 'futuremove')

    def onEnd(self, target):
        from ..dex import make_move
        data = self.effectState
        move = self.dex.moves.get(data.move)
        if target.fainted or target is data.source:
            self.hint(f"{move.name} did not hit because the target is {'fainted' if target.fainted else 'the user'}.")
            return None
        self.add('-end', target, 'move: ' + move.name)
        if data.source.hasAbility('infiltrator'):
            data.moveData.infiltrates = True
        if data.source.hasAbility('normalize'):
            data.moveData.type = 'Normal'
        hit_move = make_move(data.moveData)
        self.actions.trySpreadMoveHit([target], data.source, hit_move, True)
        if data.source.isActive and data.source.hasItem('lifeorb'):
            self.singleEvent('AfterMoveSecondarySelf', data.source.getItem(), data.source.itemState, data.source,
                             target, data.source.getItem())
        self.activeMove = None
        self.checkWin()


class healreplacement:
    def onStart(self, target, source, sourceEffect):
        self.effectState.sourceEffect = sourceEffect
        self.add('-activate', source, 'healreplacement')

    def onSwitchIn(self, target):
        if not target.fainted:
            target.heal(target.maxhp)
            self.add('-heal', target, target.getHealth, '[from] move: ' + str(self.effectState.sourceEffect),
                     '[zeffect]')
            target.side.removeSlotCondition(target, 'healreplacement')


class stall:
    def onStart(self):
        self.effectState.counter = 3

    def onStallMove(self, pokemon):
        counter = self.effectState.counter or 1
        self.debug(f"Success chance: {js_round(100 / counter)}%")
        success = self.randomChance(1, counter)
        if not success:
            pokemon.volatiles.pop('stall', None)
        return success

    def onRestart(self):
        if self.effectState.counter < self.effect.counterMax:
            self.effectState.counter *= 3
        self.effectState.duration = 2


class gem:
    def onBasePower(self, basePower, user, target, move):
        self.debug('Gem Boost')
        return self.chainModify([5325, 4096])


class raindance:
    def durationCallback(self, source, effect):
        if source and source.hasItem('damprock'):
            return 8
        return 5

    def onWeatherModifyDamage(self, damage, attacker, defender, move):
        if defender.effectiveWeather() != 'raindance':
            return None
        if move.type == 'Water':
            self.debug('rain water boost')
            return self.chainModify(1.5)
        if move.type == 'Fire':
            self.debug('rain fire suppress')
            return self.chainModify(0.5)

    def onFieldStart(self, field, source, effect):
        if effect and effect.effectType == 'Ability':
            self.add('-weather', 'RainDance', '[from] ability: ' + effect.name, f"[of] {source}")
        else:
            self.add('-weather', 'RainDance')

    def onFieldResidual(self):
        self.add('-weather', 'RainDance', '[upkeep]')
        self.eachEvent('Weather')

    def onFieldEnd(self):
        self.add('-weather', 'none')


class primordialsea:
    def onTryMove(self, attacker, defender, move):
        if move.type == 'Fire' and move.category != 'Status':
            self.debug('Primordial Sea fire suppress')
            self.add('-fail', attacker, move, '[from] Primordial Sea')
            self.attrLastMove('[still]')
            return NULL

    def onWeatherModifyDamage(self, damage, attacker, defender, move):
        if defender.effectiveWeather() != 'primordialsea':
            return None
        if move.type == 'Water':
            self.debug('Rain water boost')
            return self.chainModify(1.5)

    def onFieldStart(self, field, source, effect):
        self.add('-weather', 'PrimordialSea', '[from] ability: ' + effect.name, f"[of] {source}")

    def onFieldResidual(self):
        self.add('-weather', 'PrimordialSea', '[upkeep]')
        self.eachEvent('Weather')

    def onFieldEnd(self):
        self.add('-weather', 'none')


class sunnyday:
    def durationCallback(self, source, effect):
        if source and source.hasItem('heatrock'):
            return 8
        return 5

    def onWeatherModifyDamage(self, damage, attacker, defender, move):
        if move.id == 'hydrosteam' and attacker.effectiveWeather() == 'sunnyday':
            self.debug('Sunny Day Hydro Steam boost')
            return self.chainModify(1.5)
        if defender.effectiveWeather() != 'sunnyday':
            return None
        if move.type == 'Fire':
            self.debug('Sunny Day fire boost')
            return self.chainModify(1.5)
        if move.type == 'Water':
            self.debug('Sunny Day water suppress')
            return self.chainModify(0.5)

    def onFieldStart(self, battle, source, effect):
        if effect and effect.effectType == 'Ability':
            self.add('-weather', 'SunnyDay', '[from] ability: ' + effect.name, f"[of] {source}")
        else:
            self.add('-weather', 'SunnyDay')

    def onImmunity(self, type_, pokemon):
        if pokemon.effectiveWeather() != 'sunnyday':
            return None
        if type_ == 'frz':
            return False

    def onFieldResidual(self):
        self.add('-weather', 'SunnyDay', '[upkeep]')
        self.eachEvent('Weather')

    def onFieldEnd(self):
        self.add('-weather', 'none')


class desolateland:
    def onTryMove(self, attacker, defender, move):
        if move.type == 'Water' and move.category != 'Status':
            self.debug('Desolate Land water suppress')
            self.add('-fail', attacker, move, '[from] Desolate Land')
            self.attrLastMove('[still]')
            return NULL

    def onWeatherModifyDamage(self, damage, attacker, defender, move):
        if defender.effectiveWeather() != 'desolateland':
            return None
        if move.type == 'Fire':
            self.debug('Desolate Land fire boost')
            return self.chainModify(1.5)

    def onFieldStart(self, field, source, effect):
        self.add('-weather', 'DesolateLand', '[from] ability: ' + effect.name, f"[of] {source}")

    def onImmunity(self, type_, pokemon):
        if pokemon.effectiveWeather() != 'desolateland':
            return None
        if type_ == 'frz':
            return False

    def onFieldResidual(self):
        self.add('-weather', 'DesolateLand', '[upkeep]')
        self.eachEvent('Weather')

    def onFieldEnd(self):
        self.add('-weather', 'none')


class sandstorm:
    def durationCallback(self, source, effect):
        if source and source.hasItem('smoothrock'):
            return 8
        return 5

    def onModifySpD(self, spd, pokemon):
        if pokemon.hasType('Rock') and pokemon.effectiveWeather() == 'sandstorm':
            return self.modify(spd, 1.5)

    def onFieldStart(self, field, source, effect):
        if effect and effect.effectType == 'Ability':
            self.add('-weather', 'Sandstorm', '[from] ability: ' + effect.name, f"[of] {source}")
        else:
            self.add('-weather', 'Sandstorm')

    def onFieldResidual(self):
        self.add('-weather', 'Sandstorm', '[upkeep]')
        if self.field.isWeather('sandstorm'):
            self.eachEvent('Weather')

    def onWeather(self, target):
        self.damage(target.baseMaxhp / 16)

    def onFieldEnd(self):
        self.add('-weather', 'none')


class hail:
    def durationCallback(self, source, effect):
        if source and source.hasItem('icyrock'):
            return 8
        return 5

    def onFieldStart(self, field, source, effect):
        if effect and effect.effectType == 'Ability':
            self.add('-weather', 'Hail', '[from] ability: ' + effect.name, f"[of] {source}")
        else:
            self.add('-weather', 'Hail')

    def onFieldResidual(self):
        self.add('-weather', 'Hail', '[upkeep]')
        if self.field.isWeather('hail'):
            self.eachEvent('Weather')

    def onWeather(self, target):
        self.damage(target.baseMaxhp / 16)

    def onFieldEnd(self):
        self.add('-weather', 'none')


class snowscape:
    def durationCallback(self, source, effect):
        if source and source.hasItem('icyrock'):
            return 8
        return 5

    def onModifyDef(self, def_, pokemon):
        if pokemon.hasType('Ice') and pokemon.effectiveWeather() == 'snowscape':
            return self.modify(def_, 1.5)

    def onFieldStart(self, field, source, effect):
        if effect and effect.effectType == 'Ability':
            self.add('-weather', 'Snowscape', '[from] ability: ' + effect.name, f"[of] {source}")
        else:
            self.add('-weather', 'Snowscape')

    def onFieldResidual(self):
        self.add('-weather', 'Snowscape', '[upkeep]')
        if self.field.isWeather('snowscape'):
            self.eachEvent('Weather')

    def onFieldEnd(self):
        self.add('-weather', 'none')


class deltastream:
    def onEffectiveness(self, typeMod, target, type_, move):
        if move and move.effectType == 'Move' and move.category != 'Status' and type_ == 'Flying' and typeMod > 0:
            self.add('-fieldactivate', 'Delta Stream')
            return 0

    def onFieldStart(self, field, source, effect):
        self.add('-weather', 'DeltaStream', '[from] ability: ' + effect.name, f"[of] {source}")

    def onFieldResidual(self):
        self.add('-weather', 'DeltaStream', '[upkeep]')
        self.eachEvent('Weather')

    def onFieldEnd(self):
        self.add('-weather', 'none')


class commanded:
    def onStart(self, pokemon):
        self.boost(Obj(atk=2, spa=2, spe=2, **{'def': 2}, spd=2), pokemon)

    def onDragOut(self):
        return False

    def onTrapPokemon(self, pokemon):
        pokemon.trapped = True


class commanding:
    def onDragOut(self):
        return False

    def onTrapPokemon(self, pokemon):
        pokemon.trapped = True

    def onBeforeTurn(self, pokemon):
        self.queue.cancelAction(pokemon)


class rolloutstorage:
    def onBasePower(self, relayVar, source, target, move):
        bp = max(1, move.basePower)
        bp *= 2 ** source.volatiles['rolloutstorage'].contactHitCount
        if source.volatiles.get('defensecurl'):
            bp *= 2
        source.removeVolatile('rolloutstorage')
        return bp


class arceus:
    def onType(self, types, pokemon):
        if pokemon.transformed or pokemon.ability != 'multitype':
            return types
        t = pokemon.getItem().onPlate or 'Normal'
        return [t]


class silvally:
    def onType(self, types, pokemon):
        if pokemon.transformed or pokemon.ability != 'rkssystem':
            return types
        t = pokemon.getItem().onMemory or 'Normal'
        return [t]


class dynamax:
    """Dynamax does not exist in Champions; kept only for data completeness."""

    def onStart(self, pokemon):
        self.effectState.turns = 3

    def onTryAddVolatile(self, status, pokemon):
        if status.id == 'flinch':
            return NULL

    def onBeforeSwitchOut(self, pokemon):
        pokemon.removeVolatile('dynamax')

    def onSourceModifyDamage(self, damage, source, target, move):
        if move.id in ('behemothbash', 'behemothblade', 'dynamaxcannon'):
            return self.chainModify(2)

    def onDragOut(self, pokemon):
        self.add('-block', pokemon, 'Dynamax')
        return NULL

    def onResidual(self):
        self.effectState.turns -= 1

    def onEnd(self, pokemon):
        self.add('-end', pokemon, 'Dynamax')


def _crowned(species_name, signature, item):
    def onBattleStart(self, pokemon):
        if pokemon.item != item:
            return None
        raw = self.dex.species.get(species_name)
        species = pokemon.setSpecies(raw)
        if not species:
            return None
        pokemon.baseSpecies = raw
        pokemon.details = pokemon.getUpdatedDetails()
        pokemon.setAbility(species.abilities['0'], None, None, True)
        pokemon.baseAbility = pokemon.ability
        if 'ironhead' in pokemon.baseMoves:
            idx = pokemon.baseMoves.index('ironhead')
            move = self.dex.moves.get(signature)
            pp = self.calculatePP(move, pokemon.ppUps[idx])
            pokemon.baseMoveSlots[idx] = Obj(move=move.name, id=move.id, pp=pp, maxpp=pp, target=move.target,
                                             disabled=False, disabledSource='', used=False)
            pokemon.moveSlots = list(pokemon.baseMoveSlots)
    return onBattleStart


class zacian:
    onBattleStart = _crowned('Zacian-Crowned', 'behemothblade', 'rustedsword')


class zamazenta:
    onBattleStart = _crowned('Zamazenta-Crowned', 'behemothbash', 'rustedshield')


class trapper:
    pass
