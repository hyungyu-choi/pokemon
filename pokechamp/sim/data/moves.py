"""Move handlers (Showdown ``data/moves.ts`` + champions ``moves.ts``), ported to Python.

Each class is a move id. Methods are the move's callbacks; nested classes
``condition``, ``secondary`` and ``self_`` hold the callbacks of the move's
condition / secondary effect / self effect. ``self`` inside a handler is the
Battle (Showdown's ``this``).
"""
from __future__ import annotations

import math

from ..js import NULL, Obj, clamp_int_range, deep_clone, is_number, js_round, to_id, trunc


def _protect_prepare_hit(self, pokemon):
    return bool(self.queue.willAct()) and self.runEvent('StallMove', pokemon)


def _protect_hit(self, pokemon):
    pokemon.addVolatile('stall')


def _two_turn_try_move(self, attacker, defender, move):
    if attacker.removeVolatile(move.id):
        return None
    self.add('-prepare', attacker, move.name)
    if not self.runEvent('ChargeMove', attacker, defender, move):
        return None
    attacker.addVolatile('twoturnmove', defender)
    return NULL


def _protect_try_hit_common(self, target, source, move):
    """Shared start of the Protect-like onTryHit handlers. Returns True if blocked."""
    if self.checkMoveBypassesProtect(move, source, target):
        return False
    if move.smartTarget:
        move.smartTarget = False
    else:
        self.add('-activate', target, 'move: Protect')
    lockedmove = source.getVolatile('lockedmove')
    if lockedmove:
        if source.volatiles['lockedmove'].duration == 2:
            source.volatiles.pop('lockedmove', None)
    return True


class acrobatics:
    def basePowerCallback(self, pokemon, target, move):
        if not pokemon.item:
            self.debug('BP doubled for no item')
            return move.basePower * 2
        return move.basePower


class acupressure:
    def onHit(self, target):
        stats = [stat for stat in target.boosts if target.boosts[stat] < 6]
        if stats:
            random_stat = self.sample(stats)
            self.boost(Obj({random_stat: 2}))
        else:
            return False


class afteryou:
    def onHit(self, target):
        if self.activePerHalf == 1:
            return False
        action = self.queue.willMove(target)
        if action:
            self.queue.prioritizeAction(action)
            self.add('-activate', target, 'move: After You')
        else:
            return False


class alluringvoice:
    class secondary:
        def onHit(self, target, source, move):
            if target and target.statsRaisedThisTurn:
                target.addVolatile('confusion', source, move)


class allyswitch:
    def onPrepareHit(self, pokemon):
        return pokemon.addVolatile('allyswitch')

    def onHit(self, pokemon):
        success = True
        if self.gameType not in ('doubles', 'triples'):
            success = False
        if len(pokemon.side.active) == 3 and pokemon.position == 1:
            success = False
        new_position = len(pokemon.side.active) - 1 if pokemon.position == 0 else 0
        if not pokemon.side.active[new_position]:
            success = False
        if pokemon.side.active[new_position].fainted:
            success = False
        if not success:
            self.add('-fail', pokemon, 'move: Ally Switch')
            self.attrLastMove('[still]')
            return self.NOT_FAIL
        self.swapPosition(pokemon, new_position, '[from] move: Ally Switch')

    class condition:
        def onStart(self):
            self.effectState.counter = 3

        def onRestart(self, pokemon):
            counter = self.effectState.counter or 1
            self.debug(f"Ally Switch success chance: {js_round(100 / counter)}%")
            success = self.randomChance(1, counter)
            if not success:
                pokemon.volatiles.pop('allyswitch', None)
                return False
            if self.effectState.counter < self.effect.counterMax:
                self.effectState.counter *= 3
            self.effectState.duration = 2


class aquaring:
    class condition:
        def onStart(self, pokemon):
            self.add('-start', pokemon, 'Aqua Ring')

        def onResidual(self, pokemon):
            self.heal(pokemon.baseMaxhp / 16)


class assurance:
    def basePowerCallback(self, pokemon, target, move):
        if target.hurtThisTurn:
            self.debug('BP doubled on damaged target')
            return move.basePower * 2
        return move.basePower


class attract:
    class condition:
        def onStart(self, pokemon, source, effect):
            if not (pokemon.gender == 'M' and source.gender == 'F') and not (pokemon.gender == 'F' and source.gender == 'M'):
                self.debug('incompatible gender')
                return False
            if not self.runEvent('Attract', pokemon, source):
                self.debug('Attract event failed')
                return False
            if effect.name == 'Cute Charm':
                self.add('-start', pokemon, 'Attract', '[from] ability: Cute Charm', f"[of] {source}")
            elif effect.name == 'Destiny Knot':
                self.add('-start', pokemon, 'Attract', '[from] item: Destiny Knot', f"[of] {source}")
            else:
                self.add('-start', pokemon, 'Attract')

        def onUpdate(self, pokemon):
            if self.effectState.source and not self.effectState.source.isActive and pokemon.volatiles.get('attract'):
                self.debug(f"Removing Attract volatile on {pokemon}")
                pokemon.removeVolatile('attract')

        def onBeforeMove(self, pokemon, target, move):
            self.add('-activate', pokemon, 'move: Attract', '[of] ' + str(self.effectState.source))
            if self.randomChance(1, 2):
                self.add('cant', pokemon, 'Attract')
                return False

        def onEnd(self, pokemon):
            self.add('-end', pokemon, 'Attract', '[silent]')

    def onTryImmunity(self, target, source):
        return (target.gender == 'M' and source.gender == 'F') or (target.gender == 'F' and source.gender == 'M')


class aurawheel:
    def onTry(self, source):
        if source.species.baseSpecies == 'Morpeko':
            return None
        self.attrLastMove('[still]')
        self.add('-fail', source, 'move: Aura Wheel')
        self.hint('Only a Pokemon whose form is Morpeko or Morpeko-Hangry can use this move.')
        return NULL

    def onModifyType(self, move, pokemon):
        if pokemon.species.name == 'Morpeko-Hangry':
            move.type = 'Dark'
        else:
            move.type = 'Electric'


class auroraveil:
    def onTry(self):
        return self.field.isWeather(['hail', 'snowscape'])

    class condition:
        def durationCallback(self, target, source, effect):
            if source and source.hasItem('lightclay'):
                return 8
            return 5

        def onAnyModifyDamage(self, damage, source, target, move):
            if target is not source and self.effectState.target.hasAlly(target):
                if ((target.side.getSideCondition('reflect') and self.getCategory(move) == 'Physical') or
                        (target.side.getSideCondition('lightscreen') and self.getCategory(move) == 'Special')):
                    return None
                if not target.getMoveHitData(move).crit and not move.infiltrates:
                    self.debug('Aurora Veil weaken')
                    if self.activePerHalf > 1:
                        return self.chainModify([2732, 4096])
                    return self.chainModify(0.5)

        def onSideStart(self, side):
            self.add('-sidestart', side, 'move: Aurora Veil')

        def onSideEnd(self, side):
            self.add('-sideend', side, 'move: Aurora Veil')


class avalanche:
    def basePowerCallback(self, pokemon, target, move):
        damaged_by_target = any(p.source is target and p.damage > 0 and p.thisTurn for p in pokemon.attackedBy)
        if damaged_by_target:
            self.debug(f"BP doubled for getting hit by {target}")
            return move.basePower * 2
        return move.basePower


class axekick:
    def onMoveFail(self, target, source, move):
        self.damage(source.baseMaxhp / 2, source, source, self.dex.conditions.get('Axe Kick'))


class banefulbunker:
    onPrepareHit = _protect_prepare_hit
    onHit = _protect_hit

    class condition:
        def onStart(self, target):
            self.add('-singleturn', target, 'move: Protect')

        def onTryHit(self, target, source, move):
            if not _protect_try_hit_common(self, target, source, move):
                return None
            if self.checkMoveMakesContact(move, source, target):
                source.trySetStatus('psn', target, self.dex.getActiveMove('Baneful Bunker'))
            return self.NOT_FAIL

        def onHit(self, target, source, move):
            if move.isZOrMaxPowered and self.checkMoveMakesContact(move, source, target):
                source.trySetStatus('psn', target, self.dex.getActiveMove('Baneful Bunker'))


class barbbarrage:
    def onBasePower(self, basePower, pokemon, target):
        if target.status in ('psn', 'tox'):
            return self.chainModify(2)


class batonpass:
    def onHit(self, target):
        if not self.canSwitch(target.side) or target.volatiles.get('commanded'):
            self.attrLastMove('[still]')
            self.add('-fail', target)
            return self.NOT_FAIL

    class self_:
        def onHit(self, source):
            source.skipBeforeSwitchOutEventFlag = True


class beakblast:
    def priorityChargeCallback(self, pokemon):
        pokemon.addVolatile('beakblast')

    class condition:
        def onStart(self, pokemon):
            self.add('-singleturn', pokemon, 'move: Beak Blast')

        def onHit(self, target, source, move):
            if self.checkMoveMakesContact(move, source, target):
                source.trySetStatus('brn', target)

    def onAfterMove(self, pokemon):
        pokemon.removeVolatile('beakblast')


class beatup:
    def basePowerCallback(self, pokemon, target, move):
        set_species = self.dex.species.get(move.allies.pop(0).set.get('species'))
        bp = 5 + math.floor(set_species.baseStats['atk'] / 10)
        self.debug(f"BP for {set_species.name} hit: {bp}")
        return bp

    def onModifyMove(self, move, pokemon):
        move.allies = [a for a in pokemon.side.pokemon if a is pokemon or (not a.fainted and not a.status)]
        move.multihit = len(move.allies)


class belch:
    def onTry(self, source):
        return source.ateBerry


class bellydrum:
    def onHit(self, target):
        if target.hp <= target.maxhp / 2 or target.boosts['atk'] >= 6 or target.maxhp == 1:
            return False
        self.directDamage(target.maxhp / 2)
        self.boost(Obj(atk=12), target)


class blizzard:
    def onModifyMove(self, move):
        if self.field.isWeather(['hail', 'snowscape']):
            move.accuracy = True


class block:
    def onHit(self, target, source, move):
        return target.addVolatile('trapped', source, move, 'trapper')


class bounce:
    onTryMove = _two_turn_try_move

    class condition:
        def onInvulnerability(self, target, source, move):
            if move.id in ('gust', 'twister', 'skyuppercut', 'thunder', 'hurricane', 'smackdown', 'thousandarrows'):
                return None
            return False

        def onSourceBasePower(self, basePower, target, source, move):
            if move.id in ('gust', 'twister'):
                return self.chainModify(2)


class brickbreak:
    def onTryHit(self, pokemon):
        pokemon.side.removeSideCondition('reflect')
        pokemon.side.removeSideCondition('lightscreen')
        pokemon.side.removeSideCondition('auroraveil')


class bugbite:
    def onHit(self, target, source, move):
        item = target.getItem()
        if source.hp and item.isBerry and target.takeItem(source):
            self.add('-enditem', target, item.name, '[from] stealeat', '[move] Bug Bite', f"[of] {source}")
            if self.singleEvent('Eat', item, target.itemState, source, source, move):
                self.runEvent('EatItem', source, source, move, item)
                if item.id == 'leppaberry':
                    target.staleness = 'external'
            if item.onEat:
                source.ateBerry = True


class burningjealousy:
    class secondary:
        def onHit(self, target, source, move):
            if target and target.statsRaisedThisTurn:
                target.trySetStatus('brn', source, move)


class burnup:
    def onTryMove(self, pokemon, target, move):
        if pokemon.hasType('Fire'):
            return None
        self.add('-fail', pokemon, 'move: Burn Up')
        self.attrLastMove('[still]')
        return NULL

    class self_:
        def onHit(self, pokemon):
            pokemon.setType(['???' if t == 'Fire' else t for t in pokemon.getTypes(True)])
            self.add('-start', pokemon, 'typechange', '/'.join(pokemon.getTypes()), '[from] move: Burn Up')


def _add_spikes(self, source):
    for side in source.side.foeSidesWithConditions():
        side.addSideCondition('spikes')


class ceaselessedge:
    def onAfterHit(self, target, source, move):
        if not move.hasSheerForce:
            _add_spikes(self, source)

    def onAfterSubDamage(self, damage, target, source, move):
        if not move.hasSheerForce and source.hp:
            _add_spikes(self, source)


class charge:
    class condition:
        def onStart(self, pokemon, source, effect):
            if effect and effect.name in ('Electromorphosis', 'Wind Power'):
                self.add('-start', pokemon, 'Charge', self.activeMove.name, '[from] ability: ' + effect.name)
            else:
                self.add('-start', pokemon, 'Charge')

        def onRestart(self, pokemon, source, effect):
            if effect and effect.name in ('Electromorphosis', 'Wind Power'):
                self.add('-start', pokemon, 'Charge', self.activeMove.name, '[from] ability: ' + effect.name)
            else:
                self.add('-start', pokemon, 'Charge')

        def onBasePower(self, basePower, attacker, defender, move):
            if move.type == 'Electric':
                self.debug('charge boost')
                return self.chainModify(2)

        def onMoveAborted(self, pokemon, target, move):
            if move.type == 'Electric' and move.id != 'charge':
                pokemon.removeVolatile('charge')

        def onAfterMove(self, pokemon, target, move):
            if move.type == 'Electric' and move.id != 'charge':
                pokemon.removeVolatile('charge')

        def onEnd(self, pokemon):
            self.add('-end', pokemon, 'Charge', '[silent]')


class chillyreception:
    def priorityChargeCallback(self, source):
        source.addVolatile('chillyreception')

    class condition:
        def onBeforeMove(self, source, target, move):
            if move.id != 'chillyreception':
                return None
            self.add('-prepare', source, 'Chilly Reception', '[premajor]')


class clangoroussoul:
    def onTry(self, source):
        if source.hp <= source.maxhp * 33 / 100 or source.maxhp == 1:
            return False

    def onTryHit(self, pokemon, target, move):
        if not self.boost(move.boosts):
            return NULL
        move.pop('boosts', None)

    def onHit(self, pokemon):
        self.directDamage(pokemon.maxhp * 33 / 100)


class clearsmog:
    def onHit(self, target):
        target.clearBoosts()
        self.add('-clearboost', target)


class comeuppance:
    def damageCallback(self, pokemon):
        last = pokemon.getLastDamagedBy(True)
        if last is not None:
            return (last.damage * 1.5) or 1
        return 0

    def onTry(self, source):
        last = source.getLastDamagedBy(True)
        if not (last and last.thisTurn):
            return False

    def onModifyTarget(self, targetRelayVar, source, target, move):
        last = source.getLastDamagedBy(True)
        if last:
            targetRelayVar.target = self.getAtSlot(last.slot)


class copycat:
    def onHit(self, pokemon):
        move = self.lastMove
        if not move:
            return None
        if move.flags.get('failcopycat') or move.isZ or move.isMax:
            return False
        self.actions.useMove(move.id, pokemon)


class corrosivegas:
    def onHit(self, target, source):
        item = target.takeItem(source)
        if item:
            self.add('-enditem', target, item.name, '[from] move: Corrosive Gas', f"[of] {source}")
        else:
            self.add('-fail', target, 'move: Corrosive Gas')


class counter:
    def damageCallback(self, pokemon):
        if not pokemon.volatiles.get('counter'):
            return 0
        return pokemon.volatiles['counter'].damage or 1

    def beforeTurnCallback(self, pokemon):
        pokemon.addVolatile('counter')

    def onTry(self, source):
        if not source.volatiles.get('counter'):
            return False
        if source.volatiles['counter'].slot is NULL:
            return False

    class condition:
        def onStart(self, target, source, move):
            self.effectState.slot = NULL
            self.effectState.damage = 0

        def onRedirectTarget(self, target, source, source2, move):
            if move.id != 'counter':
                return None
            if source is not self.effectState.target or not self.effectState.slot:
                return None
            return self.getAtSlot(self.effectState.slot)

        def onDamagingHit(self, damage, target, source, move):
            if not source.isAlly(target) and self.getCategory(move) == 'Physical':
                self.effectState.slot = source.getSlot()
                self.effectState.damage = 2 * damage


_COURT_CHANGE_CONDITIONS = [
    'mist', 'lightscreen', 'reflect', 'spikes', 'safeguard', 'tailwind', 'toxicspikes', 'stealthrock', 'waterpledge',
    'firepledge', 'grasspledge', 'stickyweb', 'auroraveil', 'luckychant', 'gmaxsteelsurge', 'gmaxcannonade',
    'gmaxvinelash', 'gmaxwildfire', 'gmaxvolcalith',
]


class courtchange:
    def onHitField(self, target, source):
        success = False
        source_conds = source.side.sideConditions
        target_conds = source.side.foe.sideConditions
        source_temp = {}
        target_temp = {}
        for id_ in list(source_conds.keys()):
            if id_ not in _COURT_CHANGE_CONDITIONS:
                continue
            source_temp[id_] = source_conds.pop(id_)
            success = True
        for id_ in list(target_conds.keys()):
            if id_ not in _COURT_CHANGE_CONDITIONS:
                continue
            target_temp[id_] = target_conds.pop(id_)
            success = True
        for id_, v in source_temp.items():
            target_conds[id_] = v
            target_conds[id_].target = source.side.foe
        for id_, v in target_temp.items():
            source_conds[id_] = v
            source_conds[id_].target = source.side
        if not success:
            return False
        self.add('-swapsideconditions')
        self.add('-activate', source, 'move: Court Change')


class covet:
    def onAfterHit(self, target, source, move):
        if source.item or source.volatiles.get('gem'):
            return None
        your_item = target.takeItem(source)
        if not your_item:
            return None
        if (not self.singleEvent('TakeItem', your_item, target.itemState, source, target, move, your_item) or
                not source.setItem(your_item)):
            target.item = your_item.id
            return None
        self.add('-item', source, your_item, '[from] move: Covet', f"[of] {target}")


class curse:
    def onModifyMove(self, move, source, target):
        self.debug('Curse onModifyMove triggered')
        if not source.hasType('Ghost'):
            move.target = 'self'
        elif not target or (source is not target and source.isAlly(target)):
            move.target = 'randomNormal'

    def onTryHit(self, target, source, move):
        self.debug('Curse onTryHit triggered')
        if source.hasType('Ghost') and target.volatiles.get('curse'):
            return False

    def onHit(self, target, source):
        self.debug('Curse onHit triggered')
        if not source.hasType('Ghost'):
            return bool(self.boost(Obj(spe=-1, atk=1, **{'def': 1}), source, source))
        self.directDamage(source.maxhp / 2, source, source)
        if source.isAlly(target):
            random = self.getRandomTarget(source, 'Curse')
            if not random:
                return False
            target = random
        target.volatiles.pop('curse', None)
        target.addVolatile('curse')

    class condition:
        def onStart(self, pokemon, source):
            self.add('-start', pokemon, 'Curse', f"[of] {source}")

        def onResidual(self, pokemon):
            self.damage(pokemon.baseMaxhp / 4)


class defog:
    def onHit(self, target, source, move):
        success = False
        if not target.volatiles.get('substitute') or move.infiltrates:
            success = bool(self.boost(Obj(evasion=-1)))
        remove_all = ['spikes', 'toxicspikes', 'stealthrock', 'stickyweb', 'gmaxsteelsurge']
        remove_target = ['reflect', 'lightscreen', 'auroraveil', 'safeguard', 'mist'] + remove_all
        for cond in remove_target:
            if target.side.removeSideCondition(cond):
                if cond not in remove_all:
                    continue
                self.add('-sideend', target.side, self.dex.conditions.get(cond).name, '[from] move: Defog',
                         f"[of] {source}")
                success = True
        for cond in remove_all:
            if source.side.removeSideCondition(cond):
                self.add('-sideend', source.side, self.dex.conditions.get(cond).name, '[from] move: Defog',
                         f"[of] {source}")
                success = True
        self.field.clearTerrain()
        return success


class destinybond:
    def onPrepareHit(self, pokemon):
        return not pokemon.removeVolatile('destinybond')

    class condition:
        def onStart(self, pokemon):
            self.add('-singlemove', pokemon, 'Destiny Bond')

        def onFaint(self, target, source, effect):
            if not source or not effect or target.isAlly(source):
                return None
            if effect.effectType == 'Move' and not effect.flags.get('futuremove'):
                if source.volatiles.get('dynamax'):
                    self.add('-hint', 'Dynamaxed Pokémon are immune to Destiny Bond.')
                    return None
                self.add('-activate', target, 'move: Destiny Bond')
                source.faint()

        def onBeforeMove(self, pokemon, target, move):
            if move.id == 'destinybond':
                return None
            self.debug('removing Destiny Bond before attack')
            pokemon.removeVolatile('destinybond')

        def onMoveAborted(self, pokemon, target, move):
            pokemon.removeVolatile('destinybond')


class detect:
    onPrepareHit = _protect_prepare_hit
    onHit = _protect_hit


class dig:
    onTryMove = _two_turn_try_move

    class condition:
        def onImmunity(self, type_, pokemon):
            if type_ in ('sandstorm', 'hail'):
                return False

        def onInvulnerability(self, target, source, move):
            if move.id in ('earthquake', 'magnitude'):
                return None
            return False

        def onSourceModifyDamage(self, damage, source, target, move):
            if move.id in ('earthquake', 'magnitude'):
                return self.chainModify(2)


class direclaw:
    class secondary:
        def onHit(self, target, source):
            status = self.sample(['psn', 'par', 'slp'])
            target.trySetStatus(status, source)


class disable:
    def onTryHit(self, target):
        lm = target.lastMove
        if not lm or lm.isZOrMaxPowered or lm.isMax or lm.id == 'struggle':
            return False

    class condition:
        def onStart(self, pokemon, source, effect):
            if self.queue.willMove(pokemon) or (pokemon is self.activePokemon and self.activeMove and
                                                not self.activeMove.isExternal):
                self.effectState.duration -= 1
            if not pokemon.lastMove:
                self.debug("Pokemon hasn't moved yet")
                return False
            for ms in pokemon.moveSlots:
                if ms.id == pokemon.lastMove.id:
                    if not ms.pp:
                        self.debug('Move out of PP')
                        return False
            if effect.effectType == 'Ability':
                self.add('-start', pokemon, 'Disable', pokemon.lastMove.name, '[from] ability: ' + effect.name,
                         f"[of] {source}")
            else:
                self.add('-start', pokemon, 'Disable', pokemon.lastMove.name)
            self.effectState.move = pokemon.lastMove.id

        def onEnd(self, pokemon):
            self.add('-end', pokemon, 'Disable')

        def onBeforeMove(self, attacker, defender, move):
            if not (move.isZ and move.isZOrMaxPowered) and move.id == self.effectState.move and \
                    not move.flags.get('cantusetwice'):
                self.add('cant', attacker, 'Disable', move)
                return False

        def onDisableMove(self, pokemon):
            for ms in pokemon.moveSlots:
                if ms.id == self.effectState.move:
                    pokemon.disableMove(ms.id)


class dive:
    def onTryMove(self, attacker, defender, move):
        if attacker.removeVolatile(move.id):
            return None
        if attacker.hasAbility('gulpmissile') and attacker.species.name == 'Cramorant' and not attacker.transformed:
            forme = 'cramorantgorging' if attacker.hp <= attacker.maxhp / 2 else 'cramorantgulping'
            attacker.formeChange(forme, move)
        self.add('-prepare', attacker, move.name)
        if not self.runEvent('ChargeMove', attacker, defender, move):
            return None
        attacker.addVolatile('twoturnmove', defender)
        return NULL

    class condition:
        def onImmunity(self, type_, pokemon):
            if type_ in ('sandstorm', 'hail'):
                return False

        def onInvulnerability(self, target, source, move):
            if move.id in ('surf', 'whirlpool'):
                return None
            return False

        def onSourceModifyDamage(self, damage, source, target, move):
            if move.id in ('surf', 'whirlpool'):
                return self.chainModify(2)


class doubleshock:
    def onTryMove(self, pokemon, target, move):
        if pokemon.hasType('Electric'):
            return None
        self.add('-fail', pokemon, 'move: Double Shock')
        self.attrLastMove('[still]')
        return NULL

    class self_:
        def onHit(self, pokemon):
            pokemon.setType(['???' if t == 'Electric' else t for t in pokemon.getTypes(True)])
            self.add('-start', pokemon, 'typechange', '/'.join(pokemon.getTypes()), '[from] move: Double Shock')


class dragoncheer:
    class condition:
        def onStart(self, target, source, effect):
            if target.volatiles.get('focusenergy'):
                return False
            if effect and effect.id in ('costar', 'imposter', 'psychup', 'transform'):
                self.add('-start', target, 'move: Dragon Cheer', '[silent]')
            else:
                self.add('-start', target, 'move: Dragon Cheer')
            self.effectState.hasDragonType = target.hasType('Dragon')

        def onModifyCritRatio(self, critRatio, source):
            return critRatio + (2 if self.effectState.hasDragonType else 1)


class eeriespell:
    class secondary:
        def onHit(self, target):
            if not target.hp:
                return None
            move = target.lastMove
            if not move or move.isZ:
                return None
            if move.isMax and move.baseMove:
                move = self.dex.moves.get(move.baseMove)
            pp_deducted = target.deductPP(move.id, 3)
            if not pp_deducted:
                return None
            self.add('-activate', target, 'move: Eerie Spell', move.name, pp_deducted)


def _terrain_duration(self, source, effect):
    if source and source.hasItem('terrainextender'):
        return 8
    return 5


class electricterrain:
    class condition:
        durationCallback = _terrain_duration

        def onSetStatus(self, status, target, source, effect):
            if status.id == 'slp' and target.isGrounded() and not target.isSemiInvulnerable():
                if effect.id == 'yawn' or (effect.effectType == 'Move' and not effect.secondaries):
                    self.add('-activate', target, 'move: Electric Terrain')
                return False

        def onTryAddVolatile(self, status, target):
            if not target.isGrounded() or target.isSemiInvulnerable():
                return None
            if status.id == 'yawn':
                self.add('-activate', target, 'move: Electric Terrain')
                return NULL

        def onBasePower(self, basePower, attacker, defender, move):
            if move.type == 'Electric' and attacker.isGrounded() and not attacker.isSemiInvulnerable():
                self.debug('electric terrain boost')
                return self.chainModify([5325, 4096])

        def onFieldStart(self, field, source, effect):
            if effect and effect.effectType == 'Ability':
                self.add('-fieldstart', 'move: Electric Terrain', '[from] ability: ' + effect.name, f"[of] {source}")
            else:
                self.add('-fieldstart', 'move: Electric Terrain')

        def onFieldEnd(self):
            self.add('-fieldend', 'move: Electric Terrain')


class electrify:
    def onTryHit(self, target):
        if not self.queue.willMove(target) and target.activeTurns:
            return False

    class condition:
        def onStart(self, target):
            self.add('-singleturn', target, 'move: Electrify')

        def onModifyType(self, move):
            if move.id != 'struggle':
                self.debug('Electrify making move type electric')
                move.type = 'Electric'


class electroball:
    def basePowerCallback(self, pokemon, target):
        tspe = target.getStat('spe')
        try:
            ratio = math.floor(pokemon.getStat('spe') / tspe)
        except ZeroDivisionError:
            ratio = 0
        bp = [40, 60, 80, 120, 150][min(ratio, 4)]
        self.debug(f"BP: {bp}")
        return bp


class electroshot:
    def onTryMove(self, attacker, defender, move):
        if attacker.removeVolatile(move.id):
            return None
        self.add('-prepare', attacker, move.name)
        self.boost(Obj(spa=1), attacker, attacker, move)
        if attacker.effectiveWeather() in ('raindance', 'primordialsea'):
            self.attrLastMove('[still]')
            self.addMove('-anim', attacker, move.name, defender)
            return None
        if not self.runEvent('ChargeMove', attacker, defender, move):
            return None
        attacker.addVolatile('twoturnmove', defender)
        return NULL


class encore:
    class condition:
        def onStart(self, target):
            move = target.lastMove
            if not move or target.volatiles.get('dynamax'):
                return False
            if move.isMax and move.baseMove:
                move = self.dex.moves.get(move.baseMove)
            move_slot = target.getMoveData(move.id)
            if move.isZ or move.isMax or move.flags.get('failencore') or not move_slot or move_slot.pp <= 0:
                return False
            self.effectState.move = move.id
            self.add('-start', target, 'Encore')
            action = self.queue.willMove(target)
            if not action:
                self.effectState.duration += 1
            elif action.moveid != move.id and not target.hasItem('mentalherb'):
                self.queue.changeAction(target, Obj(choice='move', moveid=move.id))

        def onResidual(self, target):
            move_slot = target.getMoveData(self.effectState.move)
            if not move_slot or move_slot.pp <= 0:
                target.removeVolatile('encore')

        def onEnd(self, target):
            self.add('-end', target, 'Encore')

        def onDisableMove(self, pokemon):
            if not self.effectState.move or not pokemon.hasMove(self.effectState.move):
                return None
            for ms in pokemon.moveSlots:
                if ms.id != self.effectState.move:
                    pokemon.disableMove(ms.id)


class endeavor:
    def damageCallback(self, pokemon, target):
        return target.getUndynamaxedHP() - pokemon.hp

    def onTryImmunity(self, target, pokemon):
        return pokemon.hp < target.hp


class endure:
    onPrepareHit = _protect_prepare_hit
    onHit = _protect_hit

    class condition:
        def onStart(self, target):
            self.add('-singleturn', target, 'move: Endure')

        def onDamage(self, damage, target, source, effect):
            if effect and effect.effectType == 'Move' and damage >= target.hp:
                self.add('-activate', target, 'move: Endure')
                return target.hp - 1


class entrainment:
    def onTryHit(self, target, source):
        if target is source or target.volatiles.get('dynamax'):
            return False
        if (target.ability == source.ability or target.getAbility().flags.get('cantsuppress') or
                target.ability == 'truant' or source.getAbility().flags.get('noentrain')):
            return False

    def onHit(self, target, source):
        old_ability = target.setAbility(source.ability, source)
        if not old_ability:
            return old_ability
        if not target.isAlly(source):
            target.volatileStaleness = 'external'


class eruption:
    def basePowerCallback(self, pokemon, target, move):
        bp = move.basePower * pokemon.hp / pokemon.maxhp
        self.debug(f"BP: {bp}")
        return bp


class expandingforce:
    def onBasePower(self, basePower, source):
        if self.field.isTerrain('psychicterrain') and source.isGrounded():
            self.debug('terrain buff')
            return self.chainModify(1.5)

    def onModifyMove(self, move, source, target):
        if self.field.isTerrain('psychicterrain') and source.isGrounded():
            move.target = 'allAdjacentFoes'


class facade:
    def onBasePower(self, basePower, pokemon):
        if pokemon.status and pokemon.status != 'slp':
            return self.chainModify(2)


class fairylock:
    class condition:
        def onFieldStart(self, target):
            self.add('-fieldactivate', 'move: Fairy Lock')

        def onTrapPokemon(self, pokemon):
            pokemon.tryTrap()


class fakeout:
    def onTry(self, source):
        if source.activeMoveActions > 1:
            self.hint('Fake Out only works on your first turn out.')
            return False

    def onDisableMove(self, pokemon):
        if pokemon.activeMoveActions:
            pokemon.disableMove('fakeout')


class fellstinger:
    def onAfterMoveSecondarySelf(self, pokemon, target, move):
        if not target or target.fainted or target.hp <= 0:
            self.boost(Obj(atk=3), pokemon, pokemon, move)


class ficklebeam:
    def onBasePower(self, basePower, pokemon):
        if self.randomChance(3, 10):
            self.attrLastMove('[anim] Fickle Beam All Out')
            self.add('-activate', pokemon, 'move: Fickle Beam')
            return self.chainModify(2)


class finalgambit:
    def damageCallback(self, pokemon):
        damage = pokemon.hp
        pokemon.faint()
        return damage


class firstimpression:
    def onTry(self, source):
        if source.activeMoveActions > 1:
            self.hint('First Impression only works on your first turn out.')
            return False

    def onDisableMove(self, pokemon):
        if pokemon.activeMoveActions:
            pokemon.disableMove('firstimpression')


class flail:
    def basePowerCallback(self, pokemon):
        ratio = max(math.floor(pokemon.hp * 48 / pokemon.maxhp), 1)
        if ratio < 2:
            bp = 200
        elif ratio < 5:
            bp = 150
        elif ratio < 10:
            bp = 100
        elif ratio < 17:
            bp = 80
        elif ratio < 33:
            bp = 40
        else:
            bp = 20
        self.debug(f"BP: {bp}")
        return bp


class fling:
    def onPrepareHit(self, target, source, move):
        from ..dex import adapt
        if source.ignoringItem(True):
            return False
        item = source.getItem()
        if not self.singleEvent('TakeItem', item, source.itemState, source, source, move, item):
            return False
        if not item.fling:
            return False
        move.basePower = item.fling.basePower
        self.debug(f"BP: {move.basePower}")
        if item.isBerry:
            if source.hasAbility('cudchew'):
                self.singleEvent('EatItem', source.getAbility(), source.abilityState, source, source, move, item)

            def on_hit(self, foe):
                if self.singleEvent('Eat', item, source.itemState, foe, source, move):
                    self.runEvent('EatItem', foe, source, move, item)
                    if item.id == 'leppaberry':
                        foe.staleness = 'external'
                if item.onEat:
                    foe.ateBerry = True
            move.onHit = adapt(on_hit)
        elif item.fling.effect:
            move.onHit = item.fling.effect
        else:
            if move.secondaries is None:
                move.secondaries = []
            if item.fling.status:
                move.secondaries.append(Obj(status=item.fling.status))
            elif item.fling.volatileStatus:
                move.secondaries.append(Obj(volatileStatus=item.fling.volatileStatus))
        source.addVolatile('fling')

    class condition:
        def onUpdate(self, pokemon):
            item = pokemon.getItem()
            pokemon.setItem('')
            pokemon.lastItem = item.id
            pokemon.usedItemThisTurn = True
            self.add('-enditem', pokemon, item.name, '[from] move: Fling')
            self.runEvent('AfterUseItem', pokemon, None, None, item)
            pokemon.removeVolatile('fling')


class fly:
    onTryMove = _two_turn_try_move

    class condition:
        def onInvulnerability(self, target, source, move):
            if move.id in ('gust', 'twister', 'skyuppercut', 'thunder', 'hurricane', 'smackdown', 'thousandarrows'):
                return None
            return False

        def onSourceModifyDamage(self, damage, source, target, move):
            if move.id in ('gust', 'twister'):
                return self.chainModify(2)


class flyingpress:
    def onEffectiveness(self, typeMod, target, type_, move):
        return typeMod + self.dex.getEffectiveness('Flying', type_)


class focusenergy:
    class condition:
        def onStart(self, target, source, effect):
            if target.volatiles.get('dragoncheer'):
                return False
            if effect and effect.id == 'zpower':
                self.add('-start', target, 'move: Focus Energy', '[zeffect]')
            elif effect and effect.id in ('costar', 'imposter', 'psychup', 'transform'):
                self.add('-start', target, 'move: Focus Energy', '[silent]')
            else:
                self.add('-start', target, 'move: Focus Energy')

        def onModifyCritRatio(self, critRatio):
            return critRatio + 2


class focuspunch:
    def priorityChargeCallback(self, pokemon):
        pokemon.addVolatile('focuspunch')

    def beforeMoveCallback(self, pokemon):
        v = pokemon.volatiles.get('focuspunch')
        if v and v.lostFocus:
            self.add('cant', pokemon, 'Focus Punch', 'Focus Punch')
            return True

    class condition:
        def onStart(self, pokemon):
            self.add('-singleturn', pokemon, 'move: Focus Punch')

        def onHit(self, pokemon, source, move):
            if move.category != 'Status':
                self.effectState.lostFocus = True

        def onTryAddVolatile(self, status, pokemon):
            if status.id == 'flinch':
                return NULL


class followme:
    def onTry(self, source):
        return self.activePerHalf > 1

    class condition:
        def onStart(self, target, source, effect):
            if effect and effect.id == 'zpower':
                self.add('-singleturn', target, 'move: Follow Me', '[zeffect]')
            else:
                self.add('-singleturn', target, 'move: Follow Me')

        def onFoeRedirectTarget(self, target, source, source2, move):
            if not self.effectState.target.isSkyDropped() and \
                    self.validTarget(self.effectState.target, source, move.target):
                if move.smartTarget:
                    move.smartTarget = False
                self.debug('Follow Me redirected target of move')
                return self.effectState.target


class forestscurse:
    def onHit(self, target):
        if target.hasType('Grass'):
            return False
        if not target.addType('Grass'):
            return False
        self.add('-start', target, 'typeadd', 'Grass', "[from] move: Forest's Curse")


class freezedry:
    def onEffectiveness(self, typeMod, target, type_):
        if type_ == 'Water':
            return 1


class futuresight:
    def onTry(self, source, target):
        if not target.side.addSlotCondition(target, 'futuremove'):
            return False
        state = target.side.slotConditions[target.position]['futuremove']
        state.move = 'futuresight'
        state.source = source
        state.moveData = Obj(id='futuresight', name='Future Sight', accuracy=100, basePower=120,
                             category='Special', priority=0, flags=Obj(allyanim=1, metronome=1, futuremove=1),
                             ignoreImmunity=False, effectType='Move', type='Psychic')
        self.add('-start', source, 'move: Future Sight')
        return self.NOT_FAIL


class gastroacid:
    def onTryHit(self, target):
        if target.getAbility().flags.get('cantsuppress'):
            return False
        if target.hasItem('Ability Shield'):
            self.add('-block', target, 'item: Ability Shield')
            return NULL

    class condition:
        def onStart(self, pokemon):
            if pokemon.hasItem('Ability Shield'):
                return False
            self.add('-endability', pokemon)
            self.singleEvent('End', pokemon.getAbility(), pokemon.abilityState, pokemon, pokemon, 'gastroacid')

        def onCopy(self, pokemon):
            if pokemon.getAbility().flags.get('cantsuppress'):
                pokemon.removeVolatile('gastroacid')


class glaiverush:
    class condition:
        def onStart(self, pokemon):
            self.add('-singlemove', pokemon, 'Glaive Rush', '[silent]')

        def onAccuracy(self):
            return True

        def onSourceModifyDamage(self):
            return self.chainModify(2)

        def onBeforeMove(self, pokemon):
            self.debug('removing Glaive Rush drawback before attack')
            pokemon.removeVolatile('glaiverush')


def _weight_bp(target_weight):
    if target_weight >= 2000:
        return 120
    if target_weight >= 1000:
        return 100
    if target_weight >= 500:
        return 80
    if target_weight >= 250:
        return 60
    if target_weight >= 100:
        return 40
    return 20


class grassknot:
    def basePowerCallback(self, pokemon, target):
        bp = _weight_bp(target.getWeight())
        self.debug(f"BP: {bp}")
        return bp

    def onTryHit(self, target, source, move):
        if target.volatiles.get('dynamax'):
            self.add('-fail', source, 'move: Grass Knot', '[from] Dynamax')
            self.attrLastMove('[still]')
            return NULL


class lowkick:
    def basePowerCallback(self, pokemon, target):
        bp = _weight_bp(target.getWeight())
        self.debug(f"BP: {bp}")
        return bp

    def onTryHit(self, target, pokemon, move):
        if target.volatiles.get('dynamax'):
            self.add('-fail', pokemon, 'Dynamax')
            self.attrLastMove('[still]')
            return NULL


class grassyglide:
    def onModifyPriority(self, priority, source, target, move):
        if self.field.isTerrain('grassyterrain') and source.isGrounded():
            return priority + 1


class grassyterrain:
    class condition:
        durationCallback = _terrain_duration

        def onBasePower(self, basePower, attacker, defender, move):
            if move.id in ('earthquake', 'bulldoze', 'magnitude') and defender.isGrounded() and \
                    not defender.isSemiInvulnerable():
                self.debug('move weakened by grassy terrain')
                return self.chainModify(0.5)
            if move.type == 'Grass' and attacker.isGrounded():
                self.debug('grassy terrain boost')
                return self.chainModify([5325, 4096])

        def onFieldStart(self, field, source, effect):
            if effect and effect.effectType == 'Ability':
                self.add('-fieldstart', 'move: Grassy Terrain', '[from] ability: ' + effect.name, f"[of] {source}")
            else:
                self.add('-fieldstart', 'move: Grassy Terrain')

        def onResidual(self, pokemon):
            if pokemon.isGrounded() and not pokemon.isSemiInvulnerable():
                self.heal(pokemon.baseMaxhp / 16, pokemon, pokemon)
            else:
                self.debug('Pokemon semi-invuln or not grounded; Grassy Terrain skipped')

        def onFieldEnd(self):
            self.add('-fieldend', 'move: Grassy Terrain')


class gravapple:
    def onBasePower(self, basePower):
        if self.field.getPseudoWeather('gravity'):
            return self.chainModify(1.5)


class gravity:
    class condition:
        def durationCallback(self, source, effect):
            if source and source.hasAbility('persistent'):
                self.add('-activate', source, 'ability: Persistent', '[move] Gravity')
                return 7
            return 5

        def onFieldStart(self, target, source):
            if source and source.hasAbility('persistent'):
                self.add('-fieldstart', 'move: Gravity', '[persistent]')
            else:
                self.add('-fieldstart', 'move: Gravity')
            for pokemon in self.getAllActive():
                applies = False
                if pokemon.removeVolatile('bounce') or pokemon.removeVolatile('fly'):
                    applies = True
                    self.queue.cancelMove(pokemon)
                    pokemon.removeVolatile('twoturnmove')
                if pokemon.volatiles.get('skydrop'):
                    applies = True
                    self.queue.cancelMove(pokemon)
                    if pokemon.volatiles['skydrop'].source:
                        self.add('-end', pokemon.volatiles['twoturnmove'].source, 'Sky Drop', '[interrupt]')
                    pokemon.removeVolatile('skydrop')
                    pokemon.removeVolatile('twoturnmove')
                if pokemon.volatiles.get('magnetrise'):
                    applies = True
                    pokemon.volatiles.pop('magnetrise', None)
                if pokemon.volatiles.get('telekinesis'):
                    applies = True
                    pokemon.volatiles.pop('telekinesis', None)
                if applies:
                    self.add('-activate', pokemon, 'move: Gravity')

        def onModifyAccuracy(self, accuracy):
            if not is_number(accuracy):
                return None
            return self.chainModify([6840, 4096])

        def onDisableMove(self, pokemon):
            for ms in pokemon.moveSlots:
                if self.dex.moves.get(ms.id).flags.get('gravity'):
                    pokemon.disableMove(ms.id)

        def onBeforeMove(self, pokemon, target, move):
            if move.flags.get('gravity') and not move.isZ:
                self.add('cant', pokemon, 'move: Gravity', move)
                return False

        def onModifyMove(self, move, pokemon, target):
            if move.flags.get('gravity') and not move.isZ:
                self.add('cant', pokemon, 'move: Gravity', move)
                return False

        def onFieldEnd(self):
            self.add('-fieldend', 'move: Gravity')


class growth:
    def onModifyMove(self, move, pokemon):
        if pokemon.effectiveWeather() in ('sunnyday', 'desolateland'):
            move.boosts = Obj(atk=2, spa=2)


class guardsplit:
    def onHit(self, target, source):
        newdef = math.floor((target.storedStats['def'] + source.storedStats['def']) / 2)
        target.storedStats['def'] = newdef
        source.storedStats['def'] = newdef
        newspd = math.floor((target.storedStats['spd'] + source.storedStats['spd']) / 2)
        target.storedStats['spd'] = newspd
        source.storedStats['spd'] = newspd
        self.add('-activate', source, 'move: Guard Split', f"[of] {target}")


class guardswap:
    def onHit(self, target, source):
        target_boosts = Obj()
        source_boosts = Obj()
        for stat in ('def', 'spd'):
            target_boosts[stat] = target.boosts[stat]
            source_boosts[stat] = source.boosts[stat]
        source.setBoost(target_boosts)
        target.setBoost(source_boosts)
        self.add('-swapboost', source, target, 'def, spd', '[from] move: Guard Swap')


class gyroball:
    def basePowerCallback(self, pokemon, target):
        pspe = pokemon.getStat('spe')
        if pspe == 0:
            power = 1
        else:
            power = math.floor(25 * target.getStat('spe') / pspe) + 1
        if power > 150:
            power = 150
        self.debug(f"BP: {power}")
        return power


class hardpress:
    def basePowerCallback(self, pokemon, target):
        hp = target.hp
        max_hp = target.maxhp
        bp = math.floor(math.floor((100 * (100 * math.floor(hp * 4096 / max_hp)) + 2048 - 1) / 4096) / 100) or 1
        self.debug(f"BP for {hp}/{max_hp} HP: {bp}")
        return bp


class haze:
    def onHitField(self):
        self.add('-clearallboost')
        for pokemon in self.getAllActive():
            pokemon.clearBoosts()


class healbell:
    def onHit(self, target, source):
        self.add('-activate', source, 'move: Heal Bell')
        success = False
        allies = list(target.side.pokemon) + (list(target.side.allySide.pokemon) if target.side.allySide else [])
        for ally in allies:
            if ally is not source and not self.suppressingAbility(ally):
                if ally.hasAbility('soundproof'):
                    self.add('-immune', ally, '[from] ability: Soundproof')
                    continue
                if ally.hasAbility('goodasgold'):
                    self.add('-immune', ally, '[from] ability: Good as Gold')
                    continue
            if ally.cureStatus():
                success = True
        return success


class healingwish:
    def onTryHit(self, source):
        if not self.canSwitch(source.side):
            self.attrLastMove('[still]')
            self.add('-fail', source)
            return self.NOT_FAIL

    class condition:
        def onSwitchIn(self, target):
            self.singleEvent('Swap', self.effect, self.effectState, target)

        def onSwap(self, target):
            if not target.fainted and (target.hp < target.maxhp or target.status):
                target.heal(target.maxhp)
                target.clearStatus()
                self.add('-heal', target, target.getHealth, '[from] move: Healing Wish')
                target.side.removeSlotCondition(target, 'healingwish')


class healpulse:
    def onHit(self, target, source):
        if source.hasAbility('megalauncher'):
            success = bool(self.heal(self.modify(target.baseMaxhp, 0.75)))
        else:
            success = bool(self.heal(math.ceil(target.baseMaxhp * 0.5)))
        if success and not target.isAlly(source):
            target.staleness = 'external'
        if not success:
            self.add('-fail', target, 'heal')
            return self.NOT_FAIL
        return success


def _weight_ratio_bp(pokemon_weight, target_weight):
    if pokemon_weight >= target_weight * 5:
        return 120
    if pokemon_weight >= target_weight * 4:
        return 100
    if pokemon_weight >= target_weight * 3:
        return 80
    if pokemon_weight >= target_weight * 2:
        return 60
    return 40


class heatcrash:
    def basePowerCallback(self, pokemon, target):
        bp = _weight_ratio_bp(pokemon.getWeight(), target.getWeight())
        self.debug(f"BP: {bp}")
        return bp

    def onTryHit(self, target, pokemon, move):
        if target.volatiles.get('dynamax'):
            self.add('-fail', pokemon, 'Dynamax')
            self.attrLastMove('[still]')
            return NULL


class heavyslam(heatcrash):
    pass


class helpinghand:
    def onTryHit(self, target):
        if not target.newlySwitched and not self.queue.willMove(target):
            return False

    class condition:
        def onStart(self, target, source):
            self.effectState.multiplier = 1.5
            self.add('-singleturn', target, 'Helping Hand', f"[of] {source}")

        def onRestart(self, target, source):
            self.effectState.multiplier *= 1.5
            self.add('-singleturn', target, 'Helping Hand', f"[of] {source}")

        def onBasePower(self, basePower):
            self.debug('Boosting from Helping Hand: ' + str(self.effectState.multiplier))
            return self.chainModify(self.effectState.multiplier)


class hex:
    def basePowerCallback(self, pokemon, target, move):
        if target.status or target.hasAbility('comatose'):
            self.debug('BP doubled from status condition')
            return move.basePower * 2
        return move.basePower


class highjumpkick:
    def onMoveFail(self, target, source, move):
        self.damage(source.baseMaxhp / 2, source, source, self.dex.conditions.get('High Jump Kick'))


class hurricane:
    def onModifyMove(self, move, pokemon, target):
        w = target.effectiveWeather() if target else None
        if w in ('raindance', 'primordialsea'):
            move.accuracy = True
        elif w in ('sunnyday', 'desolateland'):
            move.accuracy = 50


class icespinner:
    def onAfterHit(self, target, source):
        self.field.clearTerrain()

    def onAfterSubDamage(self, damage, target, source):
        if source.hp:
            self.field.clearTerrain()


class imprison:
    class condition:
        def onStart(self, target):
            self.add('-start', target, 'move: Imprison')

        def onFoeDisableMove(self, pokemon):
            for ms in self.effectState.source.moveSlots:
                if ms.id == 'struggle':
                    continue
                pokemon.disableMove(ms.id, True)
            pokemon.maybeDisabled = True

        def onFoeBeforeMove(self, attacker, defender, move):
            if move.id != 'struggle' and self.effectState.source.hasMove(move.id) and not move.isZOrMaxPowered:
                self.add('cant', attacker, 'move: Imprison', move)
                return False


class infernalparade:
    def basePowerCallback(self, pokemon, target, move):
        if target.status or target.hasAbility('comatose'):
            return move.basePower * 2
        return move.basePower


class ingrain:
    class condition:
        def onStart(self, pokemon):
            self.add('-start', pokemon, 'move: Ingrain')

        def onResidual(self, pokemon):
            self.heal(pokemon.baseMaxhp / 16)

        def onTrapPokemon(self, pokemon):
            pokemon.tryTrap()

        def onDragOut(self, pokemon):
            self.add('-activate', pokemon, 'move: Ingrain')
            return NULL


class instruct:
    def onHit(self, target, source):
        if not target.lastMove or target.volatiles.get('dynamax'):
            return False
        last_move = target.lastMove
        move_slot = target.getMoveData(last_move.id)
        if (last_move.flags.get('failinstruct') or last_move.isZ or last_move.isMax or
                last_move.flags.get('charge') or last_move.flags.get('recharge') or
                target.volatiles.get('beakblast') or target.volatiles.get('focuspunch') or
                target.volatiles.get('shelltrap') or (move_slot and move_slot.pp <= 0)):
            return False
        self.add('-singleturn', target, 'move: Instruct', f"[of] {source}")
        self.queue.prioritizeAction(self.queue.resolveAction(Obj(
            choice='move', pokemon=target, moveid=target.lastMove.id, targetLoc=target.lastMoveTargetLoc))[0])


class jawlock:
    def onHit(self, target, source, move):
        source.addVolatile('trapped', target, move, 'trapper')
        target.addVolatile('trapped', source, move, 'trapper')


class kingsshield:
    onPrepareHit = _protect_prepare_hit
    onHit = _protect_hit

    class condition:
        def onStart(self, target):
            self.add('-singleturn', target, 'Protect')

        def onTryHit(self, target, source, move):
            if self.checkMoveBypassesProtect(move, source, target, False):
                return None
            if move.smartTarget:
                move.smartTarget = False
            else:
                self.add('-activate', target, 'move: Protect')
            if source.getVolatile('lockedmove'):
                if source.volatiles['lockedmove'].duration == 2:
                    source.volatiles.pop('lockedmove', None)
            if self.checkMoveMakesContact(move, source, target):
                self.boost(Obj(atk=-1), source, target, self.dex.getActiveMove("King's Shield"))
            return self.NOT_FAIL

        def onHit(self, target, source, move):
            if move.isZOrMaxPowered and self.checkMoveMakesContact(move, source, target):
                self.boost(Obj(atk=-1), source, target, self.dex.getActiveMove("King's Shield"))


class knockoff:
    def onBasePower(self, basePower, source, target, move):
        item = target.getItem()
        if not self.singleEvent('TakeItem', item, target.itemState, target, target, move, item):
            return None
        if item.id:
            return self.chainModify(1.5)

    def onAfterHit(self, target, source):
        item = target.takeItem()
        if item:
            self.add('-enditem', target, item.name, '[from] move: Knock Off', f"[of] {source}")


class lashout:
    def onBasePower(self, basePower, source):
        if source.statsLoweredThisTurn:
            self.debug('lashout buff')
            return self.chainModify(2)


class lastresort:
    def onTry(self, source):
        if len(source.moveSlots) < 2:
            return False
        has_last_resort = False
        for ms in source.moveSlots:
            if ms.id == 'lastresort':
                has_last_resort = True
                continue
            if not ms.used:
                return False
        return has_last_resort


class lastrespects:
    def basePowerCallback(self, pokemon, target, move):
        return 50 + 50 * pokemon.side.totalFainted


class leechseed:
    class condition:
        def onStart(self, target):
            self.add('-start', target, 'move: Leech Seed')

        def onResidual(self, pokemon):
            target = self.getAtSlot(pokemon.volatiles['leechseed'].sourceSlot)
            if not target or target.fainted or target.hp <= 0:
                self.debug('Nothing to leech into')
                return None
            damage = self.damage(pokemon.baseMaxhp / 8, pokemon, target)
            if damage:
                self.heal(damage, target, pokemon)

    def onTryImmunity(self, target):
        return not target.hasType('Grass')


def _screen_duration(self, target, source, effect):
    if source and source.hasItem('lightclay'):
        return 8
    return 5


class lightscreen:
    class condition:
        durationCallback = _screen_duration

        def onAnyModifyDamage(self, damage, source, target, move):
            if target is not source and self.effectState.target.hasAlly(target) and \
                    self.getCategory(move) == 'Special':
                if not target.getMoveHitData(move).crit and not move.infiltrates:
                    self.debug('Light Screen weaken')
                    if self.activePerHalf > 1:
                        return self.chainModify([2732, 4096])
                    return self.chainModify(0.5)

        def onSideStart(self, side):
            self.add('-sidestart', side, 'move: Light Screen')

        def onSideEnd(self, side):
            self.add('-sideend', side, 'move: Light Screen')


class lockon:
    def onTryHit(self, target, source):
        if source.volatiles.get('lockon'):
            return False

    def onHit(self, target, source):
        source.addVolatile('lockon', target)
        self.add('-activate', source, 'move: Lock-On', f"[of] {target}")

    class condition:
        def onSourceInvulnerability(self, target, source, move):
            if move and source is self.effectState.target and target is self.effectState.source:
                return 0

        def onSourceAccuracy(self, accuracy, target, source, move):
            if move and source is self.effectState.target and target is self.effectState.source:
                return True


class magicpowder:
    def onHit(self, target):
        if ','.join(target.getTypes()) == 'Psychic' or not target.setType('Psychic'):
            return False
        self.add('-start', target, 'typechange', 'Psychic')


def _persistent_room_duration(move_name):
    def durationCallback(self, source, effect):
        if source and source.hasAbility('persistent'):
            self.add('-activate', source, 'ability: Persistent', f"[move] {move_name}")
            return 7
        return 5
    return durationCallback


class magicroom:
    class condition:
        durationCallback = _persistent_room_duration('Magic Room')

        def onFieldStart(self, target, source):
            if source and source.hasAbility('persistent'):
                self.add('-fieldstart', 'move: Magic Room', f"[of] {source}", '[persistent]')
            else:
                self.add('-fieldstart', 'move: Magic Room', f"[of] {source}")
            for mon in self.getAllActive():
                self.singleEvent('End', mon.getItem(), mon.itemState, mon)

        def onFieldRestart(self, target, source):
            self.field.removePseudoWeather('magicroom')

        def onFieldEnd(self):
            self.add('-fieldend', 'move: Magic Room', '[of] ' + str(self.effectState.source))


class magneticflux:
    def onHitSide(self, side, source, move):
        targets = [a for a in side.allies() if a.hasAbility(['plus', 'minus']) and
                   (not a.volatiles.get('maxguard') or self.runEvent('TryHit', a, source, move))]
        if not targets:
            return False
        did_something = False
        for target in targets:
            did_something = self.boost(Obj({'def': 1, 'spd': 1}), target, source, move, False, True) or did_something
        return did_something


class magnetrise:
    def onTry(self, source, target, move):
        if target.volatiles.get('smackdown') or target.volatiles.get('ingrain'):
            return False
        if self.field.getPseudoWeather('Gravity'):
            self.add('cant', source, 'move: Gravity', move)
            return NULL

    class condition:
        def onStart(self, target):
            self.add('-start', target, 'Magnet Rise')

        def onImmunity(self, type_):
            if type_ == 'Ground':
                return False

        def onEnd(self, target):
            self.add('-end', target, 'Magnet Rise')


class meanlook:
    def onHit(self, target, source, move):
        return target.addVolatile('trapped', source, move, 'trapper')


class metalburst(comeuppance):
    pass


class meteorbeam:
    def onTryMove(self, attacker, defender, move):
        if attacker.removeVolatile(move.id):
            return None
        self.add('-prepare', attacker, move.name)
        self.boost(Obj(spa=1), attacker, attacker, move)
        if not self.runEvent('ChargeMove', attacker, defender, move):
            return None
        attacker.addVolatile('twoturnmove', defender)
        return NULL


class minimize:
    class condition:
        def onRestart(self):
            return NULL

        def onSourceModifyDamage(self, damage, source, target, move):
            if move.flags.get('minimize'):
                return self.chainModify(2)

        def onAccuracy(self, accuracy, target, source, move):
            if move.flags.get('minimize'):
                return True
            return accuracy


class mirrorcoat:
    def damageCallback(self, pokemon):
        if not pokemon.volatiles.get('mirrorcoat'):
            return 0
        return pokemon.volatiles['mirrorcoat'].damage or 1

    def beforeTurnCallback(self, pokemon):
        pokemon.addVolatile('mirrorcoat')

    def onTry(self, source):
        if not source.volatiles.get('mirrorcoat'):
            return False
        if source.volatiles['mirrorcoat'].slot is NULL:
            return False

    class condition:
        def onStart(self, target, source, move):
            self.effectState.slot = NULL
            self.effectState.damage = 0

        def onRedirectTarget(self, target, source, source2, move):
            if move.id != 'mirrorcoat':
                return None
            if source is not self.effectState.target or not self.effectState.slot:
                return None
            return self.getAtSlot(self.effectState.slot)

        def onDamagingHit(self, damage, target, source, move):
            if not source.isAlly(target) and self.getCategory(move) == 'Special':
                self.effectState.slot = source.getSlot()
                self.effectState.damage = 2 * damage


class mistyexplosion:
    def onBasePower(self, basePower, source):
        if self.field.isTerrain('mistyterrain') and source.isGrounded():
            self.debug('misty terrain boost')
            return self.chainModify(1.5)


class mistyterrain:
    class condition:
        durationCallback = _terrain_duration

        def onSetStatus(self, status, target, source, effect):
            if not target.isGrounded() or target.isSemiInvulnerable():
                return None
            if effect and (effect.status or effect.id == 'yawn'):
                self.add('-activate', target, 'move: Misty Terrain')
            return False

        def onTryAddVolatile(self, status, target, source, effect):
            if not target.isGrounded() or target.isSemiInvulnerable():
                return None
            if status.id == 'confusion':
                if effect.effectType == 'Move' and not effect.secondaries:
                    self.add('-activate', target, 'move: Misty Terrain')
                return NULL

        def onBasePower(self, basePower, attacker, defender, move):
            if move.type == 'Dragon' and defender.isGrounded() and not defender.isSemiInvulnerable():
                self.debug('misty terrain weaken')
                return self.chainModify(0.5)

        def onFieldStart(self, field, source, effect):
            if effect and effect.effectType == 'Ability':
                self.add('-fieldstart', 'move: Misty Terrain', '[from] ability: ' + effect.name, f"[of] {source}")
            else:
                self.add('-fieldstart', 'move: Misty Terrain')

        def onFieldEnd(self):
            self.add('-fieldend', 'Misty Terrain')


def _weather_heal(self, pokemon):
    factor = 0.5
    w = pokemon.effectiveWeather(None, True)
    if w in ('sunnyday', 'desolateland'):
        factor = 0.667
    elif w in ('raindance', 'primordialsea', 'sandstorm', 'hail', 'snowscape'):
        factor = 0.25
    success = bool(self.heal(self.modify(pokemon.maxhp, factor)))
    if not success:
        self.add('-fail', pokemon, 'heal')
        return self.NOT_FAIL
    return success


class moonlight:
    onHit = _weather_heal


class morningsun:
    onHit = _weather_heal


_HAZARDS = ['spikes', 'toxicspikes', 'stealthrock', 'stickyweb', 'gmaxsteelsurge']


class mortalspin:
    def onAfterHit(self, target, pokemon, move):
        if not move.hasSheerForce:
            if pokemon.removeVolatile('leechseed'):
                self.add('-end', pokemon, 'Leech Seed', '[from] move: Mortal Spin', f"[of] {pokemon}")
            for cond in _HAZARDS:
                if pokemon.side.removeSideCondition(cond):
                    self.add('-sideend', pokemon.side, self.dex.conditions.get(cond).name, '[from] move: Mortal Spin',
                             f"[of] {pokemon}")
            if pokemon.volatiles.get('partiallytrapped'):
                pokemon.removeVolatile('partiallytrapped')

    def onAfterSubDamage(self, damage, target, pokemon, move):
        if not move.hasSheerForce:
            if pokemon.hp and pokemon.removeVolatile('leechseed'):
                self.add('-end', pokemon, 'Leech Seed', '[from] move: Mortal Spin', f"[of] {pokemon}")
            for cond in _HAZARDS:
                if pokemon.hp and pokemon.side.removeSideCondition(cond):
                    self.add('-sideend', pokemon.side, self.dex.conditions.get(cond).name, '[from] move: Mortal Spin',
                             f"[of] {pokemon}")
            if pokemon.hp and pokemon.volatiles.get('partiallytrapped'):
                pokemon.removeVolatile('partiallytrapped')


class noretreat:
    def onTry(self, source, target, move):
        if source.volatiles.get('noretreat'):
            return False
        if source.volatiles.get('trapped'):
            move.pop('volatileStatus', None)

    class condition:
        def onStart(self, pokemon):
            self.add('-start', pokemon, 'move: No Retreat')

        def onTrapPokemon(self, pokemon):
            pokemon.tryTrap()


class octolock:
    def onTryImmunity(self, target):
        return self.dex.getImmunity('trapped', target)

    class condition:
        def onStart(self, pokemon, source):
            self.add('-start', pokemon, 'move: Octolock', f"[of] {source}")

        def onResidual(self, pokemon):
            source = self.effectState.source
            if source and (not source.isActive or source.hp <= 0 or not source.activeTurns):
                pokemon.volatiles.pop('octolock', None)
                self.add('-end', pokemon, 'Octolock', '[partiallytrapped]', '[silent]')
                return None
            self.boost(Obj({'def': -1, 'spd': -1}), pokemon, source, self.dex.getActiveMove('octolock'))

        def onTrapPokemon(self, pokemon):
            src = self.effectState.source
            if src and src.isActive:
                pokemon.tryTrap()


class painsplit:
    def onHit(self, target, pokemon):
        target_hp = target.getUndynamaxedHP()
        averagehp = math.floor((target_hp + pokemon.hp) / 2) or 1
        target_change = target_hp - averagehp
        target.sethp(target.hp - target_change)
        self.add('-sethp', target, target.getHealth, '[from] move: Pain Split', '[silent]')
        pokemon.sethp(averagehp)
        self.add('-sethp', pokemon, pokemon.getHealth, '[from] move: Pain Split')


class partingshot:
    def onHit(self, target, source, move):
        success = self.boost(Obj(atk=-1, spa=-1), target, source)
        if not success and not target.hasAbility('mirrorarmor'):
            move.pop('selfSwitch', None)


class payback:
    def basePowerCallback(self, pokemon, target, move):
        if target.newlySwitched or self.queue.willMove(target):
            self.debug('Payback NOT boosted')
            return move.basePower
        self.debug('Payback damage boost')
        return move.basePower * 2


class perishsong:
    def onHitField(self, target, source, move):
        result = False
        message = False
        for pokemon in self.getAllActive():
            if self.runEvent('Invulnerability', pokemon, source, move) is False:
                self.add('-miss', source, pokemon)
                result = True
            elif self.runEvent('TryHit', pokemon, source, move) is NULL:
                result = True
            elif not pokemon.volatiles.get('perishsong'):
                pokemon.addVolatile('perishsong')
                self.add('-start', pokemon, 'perish3', '[silent]')
                result = True
                message = True
        if not result:
            return False
        if message:
            self.add('-fieldactivate', 'move: Perish Song')

    class condition:
        def onEnd(self, target):
            self.add('-start', target, 'perish0')
            target.faint()

        def onResidual(self, pokemon):
            duration = pokemon.volatiles['perishsong'].duration
            self.add('-start', pokemon, f"perish{duration}")


class phantomforce:
    onTryMove = _two_turn_try_move


class pluck:
    def onHit(self, target, source, move):
        item = target.getItem()
        if source.hp and item.isBerry and target.takeItem(source):
            self.add('-enditem', target, item.name, '[from] stealeat', '[move] Pluck', f"[of] {source}")
            if self.singleEvent('Eat', item, target.itemState, source, source, move):
                self.runEvent('EatItem', source, source, move, item)
                if item.id == 'leppaberry':
                    target.staleness = 'external'
            if item.onEat:
                source.ateBerry = True


class pollenpuff:
    def onTryHit(self, target, source, move):
        if source.isAlly(target):
            move.basePower = 0
            move.infiltrates = True

    def onTryMove(self, source, target, move):
        if source.isAlly(target) and source.volatiles.get('healblock'):
            self.attrLastMove('[still]')
            self.add('cant', source, 'move: Heal Block', move)
            return False

    def onHit(self, target, source, move):
        if source.isAlly(target):
            if not self.heal(math.floor(target.baseMaxhp * 0.5)):
                return self.NOT_FAIL


class poltergeist:
    def onTry(self, source, target):
        return bool(target.item)

    def onTryHit(self, target, source, move):
        self.add('-activate', target, 'move: Poltergeist', self.dex.items.get(target.item).name)


def _swap_atk_def(pokemon):
    newatk = pokemon.storedStats['def']
    newdef = pokemon.storedStats['atk']
    pokemon.storedStats['atk'] = newatk
    pokemon.storedStats['def'] = newdef


def _atk_def_swap_condition(label):
    class condition:
        def onStart(self, pokemon):
            self.add('-start', pokemon, label)
            _swap_atk_def(pokemon)

        def onCopy(self, pokemon):
            _swap_atk_def(pokemon)

        def onEnd(self, pokemon):
            self.add('-end', pokemon, label)
            _swap_atk_def(pokemon)

        def onRestart(self, pokemon):
            pokemon.removeVolatile(label)
    return condition


class powershift:
    condition = _atk_def_swap_condition('Power Shift')


class powertrick:
    condition = _atk_def_swap_condition('Power Trick')


class powersplit:
    def onHit(self, target, source):
        newatk = math.floor((target.storedStats['atk'] + source.storedStats['atk']) / 2)
        target.storedStats['atk'] = newatk
        source.storedStats['atk'] = newatk
        newspa = math.floor((target.storedStats['spa'] + source.storedStats['spa']) / 2)
        target.storedStats['spa'] = newspa
        source.storedStats['spa'] = newspa
        self.add('-activate', source, 'move: Power Split', f"[of] {target}")


class powerswap:
    def onHit(self, target, source):
        target_boosts = Obj()
        source_boosts = Obj()
        for stat in ('atk', 'spa'):
            target_boosts[stat] = target.boosts[stat]
            source_boosts[stat] = source.boosts[stat]
        source.setBoost(target_boosts)
        target.setBoost(source_boosts)
        self.add('-swapboost', source, target, 'atk, spa', '[from] move: Power Swap')


class powertrip:
    def basePowerCallback(self, pokemon, target, move):
        bp = move.basePower + 20 * pokemon.positiveBoosts()
        self.debug(f"BP: {bp}")
        return bp


class storedpower(powertrip):
    pass


class protect:
    onPrepareHit = _protect_prepare_hit
    onHit = _protect_hit

    class condition:
        def onStart(self, target):
            self.add('-singleturn', target, 'Protect')

        def onTryHit(self, target, source, move):
            if not _protect_try_hit_common(self, target, source, move):
                return None
            return self.NOT_FAIL


class psychicfangs(brickbreak):
    pass


class psychicterrain:
    class condition:
        durationCallback = _terrain_duration

        def onTryHit(self, target, source, effect):
            if effect and (effect.priority <= 0.1 or effect.target == 'self'):
                return None
            if target.isSemiInvulnerable() or target.isAlly(source):
                return None
            if not target.isGrounded():
                base_move = self.dex.moves.get(effect.id)
                if base_move.priority > 0:
                    self.hint("Psychic Terrain doesn't affect airborne Pokémon.")
                return None
            self.add('-activate', target, 'move: Psychic Terrain')
            return NULL

        def onBasePower(self, basePower, attacker, defender, move):
            if move.type == 'Psychic' and attacker.isGrounded() and not attacker.isSemiInvulnerable():
                self.debug('psychic terrain boost')
                return self.chainModify([5325, 4096])

        def onFieldStart(self, field, source, effect):
            if effect and effect.effectType == 'Ability':
                self.add('-fieldstart', 'move: Psychic Terrain', '[from] ability: ' + effect.name, f"[of] {source}")
            else:
                self.add('-fieldstart', 'move: Psychic Terrain')

        def onFieldEnd(self):
            self.add('-fieldend', 'move: Psychic Terrain')


class psychup:
    def onHit(self, target, source):
        for i in target.boosts:
            source.boosts[i] = target.boosts[i]
        volatiles_to_copy = ['dragoncheer', 'focusenergy', 'gmaxchistrike', 'laserfocus']
        for v in volatiles_to_copy:
            source.removeVolatile(v)
        for v in volatiles_to_copy:
            if target.volatiles.get(v):
                source.addVolatile(v)
                if v == 'gmaxchistrike':
                    source.volatiles[v].layers = target.volatiles[v].layers
                if v == 'dragoncheer':
                    source.volatiles[v].hasDragonType = target.volatiles[v].hasDragonType
        self.add('-copyboost', source, target, '[from] move: Psych Up')


class quash:
    def onHit(self, target):
        if self.activePerHalf == 1:
            return False
        action = self.queue.willMove(target)
        if not action:
            return False
        action.order = 201
        self.add('-activate', target, 'move: Quash')


class quickguard:
    def onTry(self):
        return bool(self.queue.willAct())

    def onHitSide(self, side, source):
        source.addVolatile('stall')

    class condition:
        def onSideStart(self, target, source):
            self.add('-singleturn', source, 'Quick Guard')

        def onTryHit(self, target, source, move):
            if move.priority <= 0.1:
                return None
            if self.checkMoveBypassesProtect(move, source, target):
                return None
            self.add('-activate', target, 'move: Quick Guard')
            if source.getVolatile('lockedmove'):
                if source.volatiles['lockedmove'].duration == 2:
                    source.volatiles.pop('lockedmove', None)
            return self.NOT_FAIL


class ragefist:
    def basePowerCallback(self, pokemon):
        return min(350, 50 + 50 * pokemon.timesAttacked)


class ragepowder:
    def onTry(self, source):
        return self.activePerHalf > 1

    class condition:
        def onStart(self, pokemon):
            self.add('-singleturn', pokemon, 'move: Rage Powder')

        def onFoeRedirectTarget(self, target, source, source2, move):
            user = self.effectState.target
            if user.isSkyDropped():
                return None
            if source.runStatusImmunity('powder') and self.validTarget(user, source, move.target):
                if move.smartTarget:
                    move.smartTarget = False
                self.debug('Rage Powder redirected target of move')
                return user


class ragingbull(brickbreak):
    def onModifyType(self, move, pokemon):
        name = pokemon.species.name
        if name == 'Tauros-Paldea-Combat':
            move.type = 'Fighting'
        elif name == 'Tauros-Paldea-Blaze':
            move.type = 'Fire'
        elif name == 'Tauros-Paldea-Aqua':
            move.type = 'Water'


class rapidspin:
    def onAfterHit(self, target, pokemon, move):
        if not move.hasSheerForce:
            if pokemon.removeVolatile('leechseed'):
                self.add('-end', pokemon, 'Leech Seed', '[from] move: Rapid Spin', f"[of] {pokemon}")
            for cond in _HAZARDS:
                if pokemon.side.removeSideCondition(cond):
                    self.add('-sideend', pokemon.side, self.dex.conditions.get(cond).name, '[from] move: Rapid Spin',
                             f"[of] {pokemon}")
            if pokemon.volatiles.get('partiallytrapped'):
                pokemon.removeVolatile('partiallytrapped')

    def onAfterSubDamage(self, damage, target, pokemon, move):
        if not move.hasSheerForce:
            if pokemon.hp and pokemon.removeVolatile('leechseed'):
                self.add('-end', pokemon, 'Leech Seed', '[from] move: Rapid Spin', f"[of] {pokemon}")
            for cond in _HAZARDS:
                if pokemon.hp and pokemon.side.removeSideCondition(cond):
                    self.add('-sideend', pokemon.side, self.dex.conditions.get(cond).name, '[from] move: Rapid Spin',
                             f"[of] {pokemon}")
            if pokemon.hp and pokemon.volatiles.get('partiallytrapped'):
                pokemon.removeVolatile('partiallytrapped')


class recycle:
    def onHit(self, pokemon, source, move):
        if pokemon.item or not pokemon.lastItem:
            return False
        item = pokemon.lastItem
        pokemon.lastItem = ''
        self.add('-item', pokemon, self.dex.items.get(item), '[from] move: Recycle')
        pokemon.setItem(item, source, move)


class reflect:
    class condition:
        durationCallback = _screen_duration

        def onAnyModifyDamage(self, damage, source, target, move):
            if target is not source and self.effectState.target.hasAlly(target) and \
                    self.getCategory(move) == 'Physical':
                if not target.getMoveHitData(move).crit and not move.infiltrates:
                    self.debug('Reflect weaken')
                    if self.activePerHalf > 1:
                        return self.chainModify([2732, 4096])
                    return self.chainModify(0.5)

        def onSideStart(self, side):
            self.add('-sidestart', side, 'Reflect')

        def onSideEnd(self, side):
            self.add('-sideend', side, 'Reflect')


class reflecttype:
    def onHit(self, target, source):
        if source.species and source.species.num in (493, 773):
            return False
        if source.terastallized:
            return False
        old_apparent = source.apparentType
        new_types = [t for t in target.getTypes(True) if t != '???']
        if not new_types:
            if target.addedType:
                new_types = ['Normal']
            else:
                return False
        self.add('-start', source, 'typechange', '[from] move: Reflect Type', f"[of] {target}")
        source.setType(new_types)
        source.addedType = target.addedType
        source.knownType = target.isAlly(source) and target.knownType
        if not source.knownType:
            source.apparentType = old_apparent


class rest:
    def onTry(self, source):
        if source.status == 'slp' or source.hasAbility('comatose'):
            return False
        if source.hp == source.maxhp:
            self.add('-fail', source, 'heal')
            return NULL
        if source.hasAbility('insomnia'):
            self.add('-fail', source, '[from] ability: Insomnia', f"[of] {source}")
            return NULL
        if source.hasAbility('vitalspirit'):
            self.add('-fail', source, '[from] ability: Vital Spirit', f"[of] {source}")
            return NULL

    def onHit(self, target, source, move):
        result = target.setStatus('slp', source, move)
        if not result:
            return result
        target.statusState.time = 3
        target.statusState.startTime = 3
        self.heal(target.maxhp)


class reversal(flail):
    pass


class revivalblessing:
    def onTryHit(self, source):
        if not [a for a in source.side.pokemon if a.fainted]:
            return False


class risingvoltage:
    def basePowerCallback(self, source, target, move):
        if self.field.isTerrain('electricterrain') and target.isGrounded():
            if not source.isAlly(target):
                self.hint(f"{move.name}'s BP doubled on grounded target.")
            return move.basePower * 2
        return move.basePower


class roleplay:
    def onTryHit(self, target, source):
        if target.ability == source.ability:
            return False
        if target.getAbility().flags.get('failroleplay') or source.getAbility().flags.get('cantsuppress'):
            return False

    def onHit(self, target, source):
        old = source.setAbility(target.ability, target)
        if not old:
            return old


class roost:
    class condition:
        def onStart(self, target):
            if target.terastallized:
                if target.hasType('Flying'):
                    self.add('-hint', 'If a Terastallized Pokemon uses Roost, it remains Flying-type.')
                return False
            self.add('-singleturn', target, 'move: Roost')

        def onType(self, types, pokemon):
            self.effectState.typeWas = types
            return [t for t in types if t != 'Flying']


class round:
    def basePowerCallback(self, target, source, move):
        if move.sourceEffect == 'round':
            self.debug('BP doubled')
            return move.basePower * 2
        return move.basePower

    def onTry(self, source, target, move):
        for action in self.queue.list:
            if not action.pokemon or not action.move or action.maxMove or action.zmove:
                continue
            if action.move.id == 'round':
                self.queue.prioritizeAction(action, move)
                return None


class safeguard:
    class condition:
        def durationCallback(self, target, source, effect):
            if source and source.hasAbility('persistent'):
                self.add('-activate', source, 'ability: Persistent', '[move] Safeguard')
                return 7
            return 5

        def onSetStatus(self, status, target, source, effect):
            if not effect or not source:
                return None
            if effect.id == 'yawn':
                return None
            if effect.effectType == 'Move' and effect.infiltrates and not target.isAlly(source):
                return None
            if target is not source:
                self.debug('interrupting setStatus')
                if effect.id == 'synchronize' or (effect.effectType == 'Move' and not effect.secondaries):
                    self.add('-activate', target, 'move: Safeguard')
                return NULL

        def onTryAddVolatile(self, status, target, source, effect):
            if not effect or not source:
                return None
            if effect.effectType == 'Move' and effect.infiltrates and not target.isAlly(source):
                return None
            if status.id in ('confusion', 'yawn') and target is not source:
                if effect.effectType == 'Move' and not effect.secondaries:
                    self.add('-activate', target, 'move: Safeguard')
                return NULL

        def onSideStart(self, side, source):
            if source and source.hasAbility('persistent'):
                self.add('-sidestart', side, 'Safeguard', '[persistent]')
            else:
                self.add('-sidestart', side, 'Safeguard')

        def onSideEnd(self, side):
            self.add('-sideend', side, 'Safeguard')


class saltcure:
    class condition:
        def onStart(self, pokemon):
            self.add('-start', pokemon, 'Salt Cure')

        def onResidual(self, pokemon):
            # champions: 1/8 for Water/Steel types, 1/16 otherwise
            self.damage(pokemon.baseMaxhp / (8 if pokemon.hasType(['Water', 'Steel']) else 16))

        def onEnd(self, pokemon):
            self.add('-end', pokemon, 'Salt Cure')


class shedtail:
    def onTryHit(self, source):
        if not self.canSwitch(source.side) or source.volatiles.get('commanded'):
            self.add('-fail', source)
            return self.NOT_FAIL
        if source.volatiles.get('substitute'):
            self.add('-fail', source, 'move: Shed Tail')
            return self.NOT_FAIL
        if source.hp <= math.ceil(source.maxhp / 2):
            self.add('-fail', source, 'move: Shed Tail', '[weak]')
            return self.NOT_FAIL

    def onHit(self, target):
        self.directDamage(math.ceil(target.maxhp / 2))

    class self_:
        def onHit(self, source):
            source.skipBeforeSwitchOutEventFlag = True


class shellsidearm:
    def onPrepareHit(self, target, source, move):
        if not source.isAlly(target):
            self.attrLastMove('[anim] Shell Side Arm ' + move.category)

    def onModifyMove(self, move, pokemon, target):
        if not target:
            return None
        atk = pokemon.getStat('atk', False, True)
        spa = pokemon.getStat('spa', False, True)
        def_ = target.getStat('def', False, True)
        spd = target.getStat('spd', False, True)
        physical = math.floor(math.floor(math.floor(math.floor(2 * pokemon.level / 5 + 2) * 90 * atk) / def_) / 50)
        special = math.floor(math.floor(math.floor(math.floor(2 * pokemon.level / 5 + 2) * 90 * spa) / spd) / 50)
        if physical > special or (physical == special and self.randomChance(1, 2)):
            move.category = 'Physical'
            move.flags['contact'] = 1

    def onHit(self, target, source, move):
        if not source.isAlly(target):
            self.hint(move.category + ' Shell Side Arm')

    def onAfterSubDamage(self, damage, target, source, move):
        if not source.isAlly(target):
            self.hint(move.category + ' Shell Side Arm')


class simplebeam:
    def onTryHit(self, target):
        if target.getAbility().flags.get('cantsuppress') or target.ability in ('simple', 'truant'):
            return False

    def onHit(self, target, source):
        old = target.setAbility('simple')
        if not old:
            return old


class skillswap:
    def onHit(self, target, source, move):
        return self.skillSwap(source, target)


class skyattack:
    onTryMove = _two_turn_try_move


class sleeptalk:
    def onTry(self, source):
        return source.status == 'slp' or source.hasAbility('comatose')

    def onHit(self, pokemon):
        moves = []
        for ms in pokemon.moveSlots:
            moveid = ms.id
            if not moveid:
                continue
            move = self.dex.moves.get(moveid)
            if move.flags.get('nosleeptalk') or move.flags.get('charge') or (move.isZ and move.basePower != 1) or \
                    move.isMax:
                continue
            moves.append(moveid)
        random_move = ''
        if moves:
            random_move = self.sample(moves)
        if not random_move:
            return False
        self.actions.useMove(random_move, pokemon)


class smackdown:
    class condition:
        def onStart(self, pokemon):
            applies = False
            if pokemon.hasType('Flying') or pokemon.hasAbility(['levitate', 'eelevate']):
                applies = True
            if pokemon.hasItem('ironball') or pokemon.volatiles.get('ingrain') or \
                    self.field.getPseudoWeather('gravity'):
                applies = False
            if pokemon.removeVolatile('fly') or pokemon.removeVolatile('bounce'):
                applies = True
                self.queue.cancelMove(pokemon)
                pokemon.removeVolatile('twoturnmove')
            if pokemon.volatiles.get('magnetrise'):
                applies = True
                pokemon.volatiles.pop('magnetrise', None)
            if pokemon.volatiles.get('telekinesis'):
                applies = True
                pokemon.volatiles.pop('telekinesis', None)
            if not applies:
                return False
            self.add('-start', pokemon, 'Smack Down')

        def onRestart(self, pokemon):
            if pokemon.removeVolatile('fly') or pokemon.removeVolatile('bounce'):
                self.queue.cancelMove(pokemon)
                pokemon.removeVolatile('twoturnmove')
                self.add('-start', pokemon, 'Smack Down')


class snore:
    def onTry(self, source):
        return source.status == 'slp' or source.hasAbility('comatose')


class soak:
    def onHit(self, target):
        if ','.join(target.getTypes()) == 'Water' or not target.setType('Water'):
            self.add('-fail', target)
            return NULL
        self.add('-start', target, 'typechange', 'Water')


_WEAK_WEATHERS = ('raindance', 'primordialsea', 'sandstorm', 'hail', 'snowscape')


def _solar_try_move(self, attacker, defender, move):
    if attacker.removeVolatile(move.id):
        return None
    self.add('-prepare', attacker, move.name)
    if attacker.effectiveWeather(None, True) in ('sunnyday', 'desolateland'):
        self.attrLastMove('[still]')
        self.addMove('-anim', attacker, move.name, defender)
        return None
    if not self.runEvent('ChargeMove', attacker, defender, move):
        return None
    attacker.addVolatile('twoturnmove', defender)
    return NULL


def _solar_base_power(self, basePower, pokemon, target):
    if pokemon.effectiveWeather() in _WEAK_WEATHERS:
        self.debug('weakened by weather')
        return self.chainModify(0.5)


class solarbeam:
    onTryMove = _solar_try_move
    onBasePower = _solar_base_power


class solarblade:
    onTryMove = _solar_try_move
    onBasePower = _solar_base_power


class sparklingaria:
    def onAfterMove(self, source, target, move):
        if source.fainted or not move.hitTargets or move.hasSheerForce:
            for pokemon in self.getAllActive():
                pokemon.volatiles.pop('sparklingaria', None)
            return None
        number_targets = len(move.hitTargets)
        for pokemon in move.hitTargets:
            if pokemon is not source and pokemon.isActive and \
                    (pokemon.removeVolatile('sparklingaria') or number_targets > 1) and pokemon.status == 'brn':
                pokemon.cureStatus()


class speedswap:
    def onHit(self, target, source):
        target_spe = target.storedStats['spe']
        target.storedStats['spe'] = source.storedStats['spe']
        source.storedStats['spe'] = target_spe
        self.add('-activate', source, 'move: Speed Swap', f"[of] {target}")


class spikes:
    class condition:
        def onSideStart(self, side):
            self.add('-sidestart', side, 'Spikes')
            self.effectState.layers = 1

        def onSideRestart(self, side):
            if self.effectState.layers >= 3:
                return False
            self.add('-sidestart', side, 'Spikes')
            self.effectState.layers += 1

        def onSwitchIn(self, pokemon):
            if not pokemon.isGrounded() or pokemon.hasItem('heavydutyboots'):
                return None
            damage_amounts = [0, 3, 4, 6]
            self.damage(damage_amounts[self.effectState.layers] * pokemon.maxhp / 24)


class spikyshield:
    onPrepareHit = _protect_prepare_hit
    onHit = _protect_hit

    class condition:
        def onStart(self, target):
            self.add('-singleturn', target, 'move: Protect')

        def onTryHit(self, target, source, move):
            if not _protect_try_hit_common(self, target, source, move):
                return None
            if self.checkMoveMakesContact(move, source, target):
                self.damage(source.baseMaxhp / 8, source, target)
            return self.NOT_FAIL

        def onHit(self, target, source, move):
            if move.isZOrMaxPowered and self.checkMoveMakesContact(move, source, target):
                self.damage(source.baseMaxhp / 8, source, target)


class spiritshackle:
    class secondary:
        def onHit(self, target, source, move):
            if source.isActive:
                target.addVolatile('trapped', source, move, 'trapper')


class spite:
    def onHit(self, target):
        move = target.lastMove
        if not move or move.isZ:
            return False
        if move.isMax and move.baseMove:
            move = self.dex.moves.get(move.baseMove)
        pp_deducted = target.deductPP(move.id, 4)
        if not pp_deducted:
            return False
        self.add('-activate', target, 'move: Spite', move.name, pp_deducted)


class spitup:
    def basePowerCallback(self, pokemon):
        sp = pokemon.volatiles.get('stockpile')
        if not (sp and sp.layers):
            return False
        return sp.layers * 100

    def onTry(self, source):
        return bool(source.volatiles.get('stockpile'))

    def onAfterMove(self, pokemon):
        pokemon.removeVolatile('stockpile')


class stealthrock:
    class condition:
        def onSideStart(self, side):
            self.add('-sidestart', side, 'move: Stealth Rock')

        def onSwitchIn(self, pokemon):
            if pokemon.hasItem('heavydutyboots'):
                return None
            type_mod = self.clampIntRange(pokemon.runEffectiveness(self.dex.getActiveMove('stealthrock')), -6, 6)
            self.damage(pokemon.maxhp * 2 ** type_mod / 8)


class steelbeam:
    def onMoveFail(self, target, source, move):
        if move.multihit:
            return None
        self.damage(js_round(source.maxhp / 2), source, source, self.dex.conditions.get('Steel Beam'))


class steelroller:
    def onTry(self):
        return not self.field.isTerrain('')

    def onHit(self):
        self.field.clearTerrain()

    def onAfterSubDamage(self):
        self.field.clearTerrain()


class stickyweb:
    class condition:
        def onSideStart(self, side):
            self.add('-sidestart', side, 'move: Sticky Web')

        def onSwitchIn(self, pokemon):
            if not pokemon.isGrounded() or pokemon.hasItem('heavydutyboots'):
                return None
            self.add('-activate', pokemon, 'move: Sticky Web')
            self.boost(Obj(spe=-1), pokemon, pokemon.side.foe.active[0], self.dex.getActiveMove('stickyweb'))


class stockpile:
    def onTry(self, source):
        sp = source.volatiles.get('stockpile')
        if sp and sp.layers >= 3:
            return False

    class condition:
        def onStart(self, target):
            self.effectState.layers = 1
            self.effectState['def'] = 0
            self.effectState.spd = 0
            self.add('-start', target, 'stockpile' + str(self.effectState.layers))
            cur_def, cur_spd = target.boosts['def'], target.boosts['spd']
            self.boost(Obj({'def': 1, 'spd': 1}), target, target)
            if cur_def != target.boosts['def']:
                self.effectState['def'] -= 1
            if cur_spd != target.boosts['spd']:
                self.effectState.spd -= 1

        def onRestart(self, target):
            if self.effectState.layers >= 3:
                return False
            self.effectState.layers += 1
            self.add('-start', target, 'stockpile' + str(self.effectState.layers))
            cur_def = target.boosts['def']
            cur_spd = target.boosts['spd']
            self.boost(Obj({'def': 1, 'spd': 1}), target, target)
            if cur_def != target.boosts['def']:
                self.effectState['def'] -= 1
            if cur_spd != target.boosts['spd']:
                self.effectState.spd -= 1

        def onEnd(self, target):
            if self.effectState['def'] or self.effectState.spd:
                boosts = Obj()
                if self.effectState['def']:
                    boosts['def'] = self.effectState['def']
                if self.effectState.spd:
                    boosts['spd'] = self.effectState.spd
                self.boost(boosts, target, target)
            self.add('-end', target, 'Stockpile')
            if self.effectState['def'] != self.effectState.layers * -1 or \
                    self.effectState.spd != self.effectState.layers * -1:
                self.hint('In Gen 7, Stockpile keeps track of how many times it successfully altered each stat individually.')


class stompingtantrum:
    def basePowerCallback(self, pokemon, target, move):
        if pokemon.moveLastTurnResult is False:
            self.debug('doubling Stomping Tantrum BP due to previous move failure')
            return move.basePower * 2
        return move.basePower


class temperflare(stompingtantrum):
    pass


class stoneaxe:
    def onAfterHit(self, target, source, move):
        if not move.hasSheerForce:
            for side in source.side.foeSidesWithConditions():
                side.addSideCondition('stealthrock')

    def onAfterSubDamage(self, damage, target, source, move):
        if not move.hasSheerForce and source.hp:
            for side in source.side.foeSidesWithConditions():
                side.addSideCondition('stealthrock')


class strengthsap:
    def onHit(self, target, source):
        if target.boosts['atk'] == -6:
            return False
        atk = target.getStat('atk', False, True)
        success = self.boost(Obj(atk=-1), target, source, None, False, True)
        return bool(self.heal(atk, source, target) or success)


class struggle:
    def onModifyMove(self, move, pokemon, target):
        move.type = '???'
        self.add('-activate', pokemon, 'move: Struggle')


class stuffcheeks:
    def onTry(self, source):
        return source.getItem().isBerry

    def onHit(self, pokemon):
        if not self.boost(Obj(**{'def': 2})):
            return NULL
        pokemon.eatItem(True)


class substitute:
    def onTryHit(self, source):
        if source.volatiles.get('substitute'):
            self.add('-fail', source, 'move: Substitute')
            return self.NOT_FAIL
        if source.hp <= source.maxhp / 4 or source.maxhp == 1:
            self.add('-fail', source, 'move: Substitute', '[weak]')
            return self.NOT_FAIL

    def onHit(self, target):
        self.directDamage(target.maxhp / 4)

    class condition:
        def onStart(self, target, source, effect):
            if effect and effect.id == 'shedtail':
                self.add('-start', target, 'Substitute', '[from] move: Shed Tail')
            else:
                self.add('-start', target, 'Substitute')
            self.effectState.hp = math.floor(target.maxhp / 4)
            if target.volatiles.get('partiallytrapped'):
                self.add('-end', target, target.volatiles['partiallytrapped'].sourceEffect, '[partiallytrapped]',
                         '[silent]')
                target.volatiles.pop('partiallytrapped', None)

        def onTryPrimaryHit(self, target, source, move):
            if target is source or move.flags.get('bypasssub') or move.infiltrates:
                return None
            damage = self.actions.getDamage(source, target, move)
            if not damage and not (is_number(damage) and damage == 0):
                self.add('-fail', source)
                self.attrLastMove('[still]')
                return NULL
            if damage > target.volatiles['substitute'].hp:
                damage = target.volatiles['substitute'].hp
            target.volatiles['substitute'].hp -= damage
            source.lastDamage = damage
            if target.volatiles['substitute'].hp <= 0:
                if move.ohko:
                    self.add('-ohko')
                target.removeVolatile('substitute')
            else:
                self.add('-activate', target, 'move: Substitute', '[damage]')
            if damage:
                self.actions.applyRecoilDamage(damage, move, source)
            if move.drain:
                self.heal(math.ceil(damage * move.drain[0] / move.drain[1]), source, target, 'drain')
            self.singleEvent('AfterSubDamage', move, None, target, source, move, damage)
            self.runEvent('AfterSubDamage', target, source, move, damage)
            return self.HIT_SUBSTITUTE

        def onEnd(self, target):
            self.add('-end', target, 'Substitute')


class suckerpunch:
    def onTry(self, source, target):
        action = self.queue.willMove(target)
        move = action.move if (action and action.choice == 'move') else None
        if not move or (move.category == 'Status' and move.id != 'mefirst') or target.volatiles.get('mustrecharge'):
            return False


class thunderclap(suckerpunch):
    pass


class supercellslam:
    def onMoveFail(self, target, source, move):
        self.damage(source.baseMaxhp / 2, source, source, self.dex.conditions.get('Supercell Slam'))


class superfang:
    def damageCallback(self, pokemon, target):
        return self.clampIntRange(target.getUndynamaxedHP() / 2, 1)


class swallow:
    def onTry(self, source, target, move):
        if move.sourceEffect == 'snatch':
            return None
        return bool(source.volatiles.get('stockpile'))

    def onHit(self, pokemon):
        sp = pokemon.volatiles.get('stockpile')
        layers = (sp.layers if sp else None) or 1
        heal_amount = [0.25, 0.5, 1]
        success = bool(self.heal(self.modify(pokemon.maxhp, heal_amount[layers - 1])))
        if not success:
            self.add('-fail', pokemon, 'heal')
        pokemon.removeVolatile('stockpile')
        return success or self.NOT_FAIL


def _trick_on_hit(name):
    def onHit(self, target, source, move):
        your_item = target.takeItem(source)
        my_item = source.takeItem()
        if your_item is False or my_item is False or (not your_item and not my_item):
            if your_item:
                target.item = your_item.id
            if my_item:
                source.item = my_item.id
            return False
        if ((my_item and not self.singleEvent('TakeItem', my_item, source.itemState, target, source, move, my_item)) or
                (your_item and not self.singleEvent('TakeItem', your_item, target.itemState, source, target, move,
                                                    your_item))):
            if your_item:
                target.item = your_item.id
            if my_item:
                source.item = my_item.id
            return False
        self.add('-activate', source, 'move: Trick', f"[of] {target}")
        if my_item:
            target.setItem(my_item)
            self.add('-item', target, my_item, f"[from] move: {name}")
        else:
            self.add('-enditem', target, your_item, '[silent]', f"[from] move: {name}")
        if your_item:
            source.setItem(your_item)
            self.add('-item', source, your_item, f"[from] move: {name}")
        else:
            self.add('-enditem', source, my_item, '[silent]', f"[from] move: {name}")
    return onHit


class switcheroo:
    def onTryImmunity(self, target):
        return not target.hasAbility('stickyhold')

    onHit = _trick_on_hit('Switcheroo')


class trick:
    def onTryImmunity(self, target):
        return not target.hasAbility('stickyhold')

    onHit = _trick_on_hit('Trick')


class synthesis:
    onHit = _weather_heal


class syrupbomb:
    class condition:
        def onStart(self, pokemon):
            self.add('-start', pokemon, 'Syrup Bomb')

        def onUpdate(self, pokemon):
            if self.effectState.source and not self.effectState.source.isActive:
                pokemon.removeVolatile('syrupbomb')

        def onResidual(self, pokemon):
            self.boost(Obj(spe=-1), pokemon, self.effectState.source)

        def onEnd(self, pokemon):
            self.add('-end', pokemon, 'Syrup Bomb', '[silent]')


class tailwind:
    class condition:
        def durationCallback(self, target, source, effect):
            if source and source.hasAbility('persistent'):
                self.add('-activate', source, 'ability: Persistent', '[move] Tailwind')
                return 6
            return 4

        def onSideStart(self, side, source):
            if source and source.hasAbility('persistent'):
                self.add('-sidestart', side, 'move: Tailwind', '[persistent]')
            else:
                self.add('-sidestart', side, 'move: Tailwind')

        def onModifySpe(self, spe, pokemon):
            return self.chainModify(2)

        def onSideEnd(self, side):
            self.add('-sideend', side, 'move: Tailwind')


class taunt:
    class condition:
        def onStart(self, target):
            if target.activeTurns and not self.queue.willMove(target):
                self.effectState.duration += 1
            self.add('-start', target, 'move: Taunt')

        def onEnd(self, target):
            self.add('-end', target, 'move: Taunt')

        def onDisableMove(self, pokemon):
            for ms in pokemon.moveSlots:
                move = self.dex.moves.get(ms.id)
                if move.category == 'Status' and move.id != 'mefirst':
                    pokemon.disableMove(ms.id)

        def onBeforeMove(self, attacker, defender, move):
            if not (move.isZ and move.isZOrMaxPowered) and move.category == 'Status' and move.id != 'mefirst':
                self.add('cant', attacker, 'move: Taunt', move)
                return False


class teatime:
    def onHitField(self, target, source, move):
        targets = []
        for pokemon in self.getAllActive():
            if self.runEvent('Invulnerability', pokemon, source, move) is False:
                self.add('-miss', source, pokemon)
            elif self.runEvent('TryHit', pokemon, source, move) and pokemon.getItem().isBerry:
                targets.append(pokemon)
        self.add('-fieldactivate', 'move: Teatime')
        if not targets:
            self.add('-fail', source, 'move: Teatime')
            self.attrLastMove('[still]')
            return self.NOT_FAIL
        for pokemon in targets:
            pokemon.eatItem(True)


class terrainpulse:
    def onModifyType(self, move, pokemon):
        if not pokemon.isGrounded():
            return None
        t = self.field.terrain
        if t == 'electricterrain':
            move.type = 'Electric'
        elif t == 'grassyterrain':
            move.type = 'Grass'
        elif t == 'mistyterrain':
            move.type = 'Fairy'
        elif t == 'psychicterrain':
            move.type = 'Psychic'

    def onModifyMove(self, move, pokemon):
        if self.field.terrain and pokemon.isGrounded():
            move.basePower *= 2
            self.debug('BP doubled in Terrain')


class thief:
    def onAfterHit(self, target, source, move):
        if source.item or source.volatiles.get('gem'):
            return None
        your_item = target.takeItem(source)
        if not your_item:
            return None
        if (not self.singleEvent('TakeItem', your_item, target.itemState, source, target, move, your_item) or
                not source.setItem(your_item)):
            target.item = your_item.id
            return None
        self.add('-enditem', target, your_item, '[silent]', '[from] move: Thief', f"[of] {source}")
        self.add('-item', source, your_item, '[from] move: Thief', f"[of] {target}")


class throatchop:
    class secondary:
        def onHit(self, target):
            target.addVolatile('throatchop')

    class condition:
        def onStart(self, target):
            self.add('-start', target, 'Throat Chop', '[silent]')

        def onDisableMove(self, pokemon):
            for ms in pokemon.moveSlots:
                if self.dex.moves.get(ms.id).flags.get('sound'):
                    pokemon.disableMove(ms.id)

        def onBeforeMove(self, pokemon, target, move):
            if not move.isZOrMaxPowered and move.flags.get('sound'):
                self.add('cant', pokemon, 'move: Throat Chop')
                return False

        def onModifyMove(self, move, pokemon, target):
            if not move.isZOrMaxPowered and move.flags.get('sound'):
                self.add('cant', pokemon, 'move: Throat Chop')
                return False

        def onEnd(self, target):
            self.add('-end', target, 'Throat Chop', '[silent]')


class thunder(hurricane):
    pass


class tidyup:
    def onHit(self, pokemon):
        success = False
        for active in self.getAllActive():
            if active.removeVolatile('substitute'):
                success = True
        sides = [pokemon.side] + pokemon.side.foeSidesWithConditions()
        for side in sides:
            for cond in _HAZARDS:
                if side.removeSideCondition(cond):
                    self.add('-sideend', side, self.dex.conditions.get(cond).name)
                    success = True
        if success:
            self.add('-activate', pokemon, 'move: Tidy Up')
        return bool(self.boost(Obj(atk=1, spe=1), pokemon, pokemon, None, False, True)) or success


class topsyturvy:
    def onHit(self, target):
        success = False
        for i in target.boosts:
            if target.boosts[i] == 0:
                continue
            target.boosts[i] = -target.boosts[i]
            success = True
        if not success:
            return False
        self.add('-invertboost', target, '[from] move: Topsy-Turvy')


class torment:
    class condition:
        def onStart(self, pokemon, source, effect):
            if pokemon.volatiles.get('dynamax'):
                pokemon.volatiles.pop('torment', None)
                return False
            if effect and effect.id == 'gmaxmeltdown':
                self.effectState.duration = 3
            self.add('-start', pokemon, 'Torment')

        def onEnd(self, pokemon):
            self.add('-end', pokemon, 'Torment')

        def onDisableMove(self, pokemon):
            if pokemon.lastMove and pokemon.lastMove.id != 'struggle':
                pokemon.disableMove(pokemon.lastMove.id)


class toxicspikes:
    class condition:
        def onSideStart(self, side):
            self.add('-sidestart', side, 'move: Toxic Spikes')
            self.effectState.layers = 1

        def onSideRestart(self, side):
            if self.effectState.layers >= 2:
                return False
            self.add('-sidestart', side, 'move: Toxic Spikes')
            self.effectState.layers += 1

        def onSwitchIn(self, pokemon):
            if not pokemon.isGrounded():
                return None
            if pokemon.hasType('Poison'):
                self.add('-sideend', pokemon.side, 'move: Toxic Spikes', f"[of] {pokemon}")
                pokemon.side.removeSideCondition('toxicspikes')
            elif pokemon.hasType('Steel') or pokemon.hasItem('heavydutyboots'):
                pass
            elif self.effectState.layers >= 2:
                pokemon.trySetStatus('tox', pokemon.side.foe.active[0])
            else:
                pokemon.trySetStatus('psn', pokemon.side.foe.active[0])


class transform:
    def onHit(self, target, pokemon):
        return pokemon.transformInto(target)


class triattack:
    class secondary:
        def onHit(self, target, source):
            status = self.sample(['brn', 'par', 'frz'])
            target.trySetStatus(status, source)


class trickortreat:
    def onHit(self, target):
        if target.hasType('Ghost'):
            return False
        if not target.addType('Ghost'):
            return False
        self.add('-start', target, 'typeadd', 'Ghost', '[from] move: Trick-or-Treat')
        if len(target.side.active) == 2 and target.position == 1:
            action = self.queue.willMove(target)
            if action and action.move.id == 'curse':
                action.targetLoc = -1


class trickroom:
    class condition:
        durationCallback = _persistent_room_duration('Trick Room')

        def onFieldStart(self, target, source):
            if source and source.hasAbility('persistent'):
                self.add('-fieldstart', 'move: Trick Room', f"[of] {source}", '[persistent]')
            else:
                self.add('-fieldstart', 'move: Trick Room', f"[of] {source}")

        def onFieldRestart(self, target, source):
            self.field.removePseudoWeather('trickroom')

        def onFieldEnd(self):
            self.add('-fieldend', 'move: Trick Room')


class tripleaxel:
    def basePowerCallback(self, pokemon, target, move):
        return 20 * move.hit


class upperhand:
    def onTry(self, source, target):
        action = self.queue.willMove(target)
        move = action.move if (action and action.choice == 'move') else None
        if not move or move.priority <= 0.1 or move.category == 'Status':
            return False


class uproar:
    def onTryHit(self, target):
        active_team = target.side.activeTeam()
        foe_active_team = target.side.foe.activeTeam()
        for i, ally_active in enumerate(active_team):
            if ally_active and ally_active.status == 'slp':
                ally_active.cureStatus()
            foe_active = foe_active_team[i] if i < len(foe_active_team) else None
            if foe_active and foe_active.status == 'slp':
                foe_active.cureStatus()

    class condition:
        def onStart(self, target):
            self.add('-start', target, 'Uproar')

        def onResidual(self, target):
            if target.volatiles.get('throatchop'):
                target.removeVolatile('uproar')
                return None
            if target.lastMove and target.lastMove.id == 'struggle':
                target.volatiles.pop('uproar', None)
            self.add('-start', target, 'Uproar', '[upkeep]')

        def onEnd(self, target):
            self.add('-end', target, 'Uproar')

        def onAnySetStatus(self, status, pokemon):
            if status.id == 'slp':
                if pokemon is self.effectState.target:
                    self.add('-fail', pokemon, 'slp', '[from] Uproar', '[msg]')
                else:
                    self.add('-fail', pokemon, 'slp', '[from] Uproar')
                return NULL


class venoshock(barbbarrage):
    pass


class watershuriken:
    def basePowerCallback(self, pokemon, target, move):
        if pokemon.species.name == 'Greninja-Ash' and pokemon.hasAbility('battlebond') and not pokemon.transformed:
            return move.basePower + 5
        return move.basePower


class waterspout(eruption):
    pass


class weatherball:
    def onModifyType(self, move, pokemon):
        w = pokemon.effectiveWeather()
        if w in ('sunnyday', 'desolateland'):
            move.type = 'Fire'
        elif w in ('raindance', 'primordialsea'):
            move.type = 'Water'
        elif w == 'sandstorm':
            move.type = 'Rock'
        elif w in ('hail', 'snowscape'):
            move.type = 'Ice'

    def onModifyMove(self, move, pokemon):
        w = pokemon.effectiveWeather()
        if w in ('sunnyday', 'desolateland', 'raindance', 'primordialsea', 'sandstorm', 'hail', 'snowscape'):
            move.basePower *= 2
        self.debug(f"BP: {move.basePower}")


class wideguard:
    def onTry(self):
        return bool(self.queue.willAct())

    def onHitSide(self, side, source):
        source.addVolatile('stall')

    class condition:
        def onSideStart(self, target, source):
            self.add('-singleturn', source, 'Wide Guard')

        def onTryHit(self, target, source, move):
            if (move.target if move else None) != 'allAdjacent' and move.target != 'allAdjacentFoes':
                return None
            if self.checkMoveBypassesProtect(move, source, target):
                return None
            self.add('-activate', target, 'move: Wide Guard')
            if source.getVolatile('lockedmove'):
                if source.volatiles['lockedmove'].duration == 2:
                    source.volatiles.pop('lockedmove', None)
            return self.NOT_FAIL


class wish:
    class condition:
        def onStart(self, pokemon, source):
            self.effectState.hp = source.maxhp / 2
            self.effectState.startingTurn = self.getOverflowedTurnCount()
            if self.effectState.startingTurn == 255:
                self.hint(f"In Gen 8+, Wish will never resolve when used on the {self.turn}th turn.")

        def onResidual(self, target):
            if self.getOverflowedTurnCount() <= self.effectState.startingTurn:
                return None
            target.side.removeSlotCondition(self.getAtSlot(self.effectState.sourceSlot), 'wish')

        def onEnd(self, target):
            if target and not target.fainted:
                damage = self.heal(self.effectState.hp, target, target)
                if damage:
                    self.add('-heal', target, target.getHealth, '[from] move: Wish',
                             '[wisher] ' + self.effectState.source.name)


class wonderroom:
    class condition:
        durationCallback = _persistent_room_duration('Wonder Room')

        def onModifyMove(self, move, source, target):
            if not move.overrideOffensiveStat:
                return None
            stat = move.overrideOffensiveStat
            if stat not in ('def', 'spd'):
                return None
            move.overrideOffensiveStat = 'spd' if stat == 'def' else 'def'
            self.hint(f"{move.name} uses {'' if stat == 'def' else 'Sp. '}Def boosts when Wonder Room is active.")

        def onFieldStart(self, field, source):
            if source and source.hasAbility('persistent'):
                self.add('-fieldstart', 'move: Wonder Room', f"[of] {source}", '[persistent]')
            else:
                self.add('-fieldstart', 'move: Wonder Room', f"[of] {source}")

        def onFieldRestart(self, target, source):
            self.field.removePseudoWeather('wonderroom')

        def onFieldEnd(self):
            self.add('-fieldend', 'move: Wonder Room')


class worryseed:
    def onTryImmunity(self, target):
        if target.ability in ('truant', 'insomnia'):
            return False

    def onTryHit(self, target):
        if target.getAbility().flags.get('cantsuppress'):
            return False

    def onHit(self, target, source):
        old = target.setAbility('insomnia')
        if not old:
            return old
        if target.status == 'slp':
            target.cureStatus()


class yawn:
    def onTryHit(self, target):
        if target.status or not target.runStatusImmunity('slp'):
            return False

    class condition:
        def onStart(self, target, source):
            self.add('-start', target, 'move: Yawn', f"[of] {source}")

        def onEnd(self, target):
            self.add('-end', target, 'move: Yawn', '[silent]')
            target.trySetStatus('slp', self.effectState.source)


class healblock:
    """Heal Block itself is not usable in Champions, but Psychic Noise inflicts its volatile."""

    class condition:
        def durationCallback(self, target, source, effect):
            if effect and effect.name == 'Psychic Noise':
                return 2
            if source and source.hasAbility('persistent'):
                self.add('-activate', source, 'ability: Persistent', '[move] Heal Block')
                return 7
            return 5

        def onStart(self, pokemon, source):
            self.add('-start', pokemon, 'move: Heal Block')
            source.moveThisTurnResult = True

        def onDisableMove(self, pokemon):
            for ms in pokemon.moveSlots:
                if self.dex.moves.get(ms.id).flags.get('heal'):
                    pokemon.disableMove(ms.id)

        def onBeforeMove(self, pokemon, target, move):
            if move.flags.get('heal') and not move.isZ and not move.isMax:
                self.add('cant', pokemon, 'move: Heal Block', move)
                return False

        def onModifyMove(self, move, pokemon):
            if move.flags.get('heal') and not move.isZ and not move.isMax:
                self.add('cant', pokemon, 'move: Heal Block', move)
                return False

        def onEnd(self, pokemon):
            self.add('-end', pokemon, 'move: Heal Block')

        def onTryHeal(self, damage, target, source, effect):
            if effect and (effect.id == 'zpower' or effect.isZ):
                return damage
            if source and target is not source and target.hp != target.maxhp and effect.name == 'Pollen Puff':
                self.attrLastMove('[still]')
                self.add('cant', source, 'move: Heal Block', effect)
                return NULL
            return False

        def onRestart(self, target, source, effect):
            if effect and effect.name == 'Psychic Noise':
                return None
            self.add('-fail', target, 'move: Heal Block')
            if not source.moveThisTurnResult:
                source.moveThisTurnResult = False
