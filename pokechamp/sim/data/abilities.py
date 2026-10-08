"""Ability handlers (Showdown ``data/abilities.ts`` + champions overrides), ported to Python.

Each class is an ability id; ``self`` in handlers is the Battle (Showdown's ``this``).
"""
from __future__ import annotations

import math

from ..js import NULL, Obj, clamp_int_range, is_number, js_round, to_id, trunc

_NO_MODIFY_TYPE = ('judgment', 'multiattack', 'naturalgift', 'revelationdance', 'technoblast', 'terrainpulse',
                   'weatherball')


def _type_changer(new_type):
    def onModifyType(self, move, pokemon):
        if (move.type == 'Normal' and (move.id not in _NO_MODIFY_TYPE or (self.activeMove and self.activeMove.isMax))
                and not (move.isZ and move.category != 'Status')
                and not (move.name == 'Tera Blast' and pokemon.terastallized)):
            move.type = new_type
            move.typeChangerBoosted = self.effect

    def onBasePower(self, basePower, pokemon, target, move):
        if move.typeChangerBoosted is self.effect:
            return self.chainModify([4915, 4096])
    return onModifyType, onBasePower


def _pinch(type_):
    def mod(self, atk, attacker, defender, move):
        if move.type == type_ and attacker.hp <= attacker.maxhp / 3:
            self.debug('pinch boost')
            return self.chainModify(1.5)
    return mod


def _clear_body_like(name):
    def onTryBoost(self, boost, target, source, effect):
        if source and target is source:
            return None
        show_msg = False
        for i in list(boost.keys()):
            if boost[i] < 0:
                del boost[i]
                show_msg = True
        if show_msg and not effect.secondaries and effect.id != 'octolock':
            self.add('-fail', target, 'unboost', f"[from] ability: {name}", f"[of] {target}")
    return onTryBoost


def _defiant_like(stat):
    def onAfterEachBoost(self, boost, target, source, effect):
        if not source or target.isAlly(source):
            return None
        stats_lowered = any(boost[i] < 0 for i in boost)
        if stats_lowered:
            self.boost(Obj({stat: 2}), target, target, None, False, True)
    return onAfterEachBoost


def _absorb_heal(type_, name):
    def onTryHit(self, target, source, move):
        if target is not source and move.type == type_:
            if not self.heal(target.baseMaxhp / 4):
                self.add('-immune', target, f"[from] ability: {name}")
            return NULL
    return onTryHit


class adaptability:
    def onModifySTAB(self, stab, source, target, move):
        if move.forceSTAB or source.hasType(move.type):
            if stab == 2:
                return 2.25
            return 2


class aerilate:
    onModifyType, onBasePower = _type_changer('Flying')


class pixilate:
    onModifyType, onBasePower = _type_changer('Fairy')


class refrigerate:
    onModifyType, onBasePower = _type_changer('Ice')


class galvanize:
    onModifyType, onBasePower = _type_changer('Electric')


class dragonize:
    onModifyType, onBasePower = _type_changer('Dragon')


class aftermath:
    def onDamagingHit(self, damage, target, source, move):
        if not target.hp and self.checkMoveMakesContact(move, source, target, True):
            self.damage(source.baseMaxhp / 4, source, target)


class analytic:
    def onBasePower(self, basePower, pokemon):
        boosted = True
        for target in self.getAllActive():
            if target is pokemon:
                continue
            if self.queue.willMove(target):
                boosted = False
                break
        if boosted:
            self.debug('Analytic boost')
            return self.chainModify([5325, 4096])


class angerpoint:
    def onHit(self, target, source, move):
        if not target.hp:
            return None
        if move and move.effectType == 'Move' and target.getMoveHitData(move).crit:
            self.boost(Obj(atk=12), target, target)


class anticipation:
    def onStart(self, pokemon):
        for target in pokemon.foes():
            for ms in target.moveSlots:
                move = self.dex.moves.get(ms.move)
                if move.category == 'Status':
                    continue
                move_type = target.hpType if move.id == 'hiddenpower' else move.type
                if (self.dex.getImmunity(move_type, pokemon) and self.dex.getEffectiveness(move_type, pokemon) > 0) \
                        or move.ohko:
                    self.add('-ability', pokemon, 'Anticipation')
                    return None


class armortail:
    def onFoeTryMove(self, target, source, move):
        if move.target == 'foeSide' or (move.target == 'all' and move.id not in ('perishsong', 'flowershield',
                                                                                 'rototiller')):
            return None
        holder = self.effectState.target
        if (source.isAlly(holder) or move.target == 'all') and move.priority > 0.1:
            self.attrLastMove('[still]')
            self.add('cant', holder, 'ability: Armor Tail', move, f"[of] {target}")
            return False


class aromaveil:
    def onAllyTryAddVolatile(self, status, target, source, effect):
        if status.id in ('attract', 'disable', 'encore', 'healblock', 'taunt', 'torment'):
            if effect.effectType == 'Move':
                holder = self.effectState.target
                self.add('-block', target, 'ability: Aroma Veil', f"[of] {holder}")
            return NULL


class auraguard:
    def onSourceModifyDamage(self, damage, source, target, move):
        if move.flags.get('contact'):
            return self.chainModify(0.5)


class battlebond:
    def onSourceAfterFaint(self, length, target, source, effect):
        if source.bondTriggered:
            return None
        if not effect or effect.effectType != 'Move':
            return None
        if source.species.id == 'greninjabond' and source.hp and not source.transformed and \
                source.side.foePokemonLeft():
            self.boost(Obj(atk=1, spa=1, spe=1), source, source, self.effect)
            self.add('-activate', source, 'ability: Battle Bond')
            source.bondTriggered = True

    def onModifyMove(self, move, attacker):
        if move.id == 'watershuriken' and attacker.species.name == 'Greninja-Ash' and not attacker.transformed:
            move.multihit = 3


_HEALING_BERRIES = ('aguavberry', 'enigmaberry', 'figyberry', 'iapapaberry', 'magoberry', 'sitrusberry', 'wikiberry',
                    'oranberry', 'berryjuice')


class berserk:
    def onDamage(self, damage, target, source, effect):
        self.effectState.checkedBerserk = not (effect.effectType == 'Move' and not effect.multihit)

    def onTryEatItem(self, item):
        if item.id in _HEALING_BERRIES:
            return self.effectState.checkedBerserk
        return True

    def onAfterMoveSecondary(self, target, source, move):
        self.effectState.checkedBerserk = True
        if not source or source is target or not target.hp or not move.totalDamage:
            return None
        last = target.getLastAttackedBy()
        if not last:
            return None
        damage = move.totalDamage if (move.multihit and not move.smartTarget) else last.damage
        if target.hp <= target.maxhp / 2 and target.hp + damage > target.maxhp / 2:
            self.boost(Obj(spa=1), target, target)


class angershell:
    def onDamage(self, damage, target, source, effect):
        self.effectState.checkedAngerShell = not (effect.effectType == 'Move' and not effect.multihit)

    def onTryEatItem(self, item):
        if item.id in _HEALING_BERRIES:
            return self.effectState.checkedAngerShell
        return True

    def onAfterMoveSecondary(self, target, source, move):
        self.effectState.checkedAngerShell = True
        if not source or source is target or not target.hp or not move.totalDamage:
            return None
        last = target.getLastAttackedBy()
        if not last:
            return None
        damage = move.totalDamage if (move.multihit and not move.smartTarget) else last.damage
        if target.hp <= target.maxhp / 2 and target.hp + damage > target.maxhp / 2:
            self.boost(Obj({'atk': 1, 'spa': 1, 'spe': 1, 'def': -1, 'spd': -1}), target, target)


class bigpecks:
    def onTryBoost(self, boost, target, source, effect):
        if source and target is source:
            return None
        if boost.get('def') and boost['def'] < 0:
            del boost['def']
            if not effect.secondaries and effect.id != 'octolock':
                self.add('-fail', target, 'unboost', 'def', '[from] ability: Big Pecks', f"[of] {target}")


class blaze:
    onModifyAtk = _pinch('Fire')
    onModifySpA = _pinch('Fire')


class torrent:
    onModifyAtk = _pinch('Water')
    onModifySpA = _pinch('Water')


class overgrow:
    onModifyAtk = _pinch('Grass')
    onModifySpA = _pinch('Grass')


class swarm:
    onModifyAtk = _pinch('Bug')
    onModifySpA = _pinch('Bug')


class bulletproof:
    def onTryHit(self, pokemon, target, move):
        if move.flags.get('bullet'):
            self.add('-immune', pokemon, '[from] ability: Bulletproof')
            return NULL


class cheekpouch:
    def onEatItem(self, item, pokemon):
        self.heal(pokemon.baseMaxhp / 3)


class chlorophyll:
    def onModifySpe(self, spe, pokemon):
        if pokemon.effectiveWeather() in ('sunnyday', 'desolateland'):
            return self.chainModify(2)


class clearbody:
    onTryBoost = _clear_body_like('Clear Body')


class whitesmoke:
    onTryBoost = _clear_body_like('White Smoke')


class fullmetalbody:
    onTryBoost = _clear_body_like('Full Metal Body')


class cloudnine:
    def onSwitchIn(self, pokemon):
        self.add('-ability', pokemon, 'Cloud Nine')
        self.effect.onStart(self, pokemon)

    def onStart(self, pokemon):
        pokemon.abilityState.ending = False
        self.eachEvent('WeatherChange', self.effect)

    def onEnd(self, pokemon):
        pokemon.abilityState.ending = True
        self.eachEvent('WeatherChange', self.effect)


class airlock(cloudnine):
    def onSwitchIn(self, pokemon):
        self.add('-ability', pokemon, 'Air Lock')
        self.effect.onStart(self, pokemon)


class competitive:
    onAfterEachBoost = _defiant_like('spa')


class defiant:
    onAfterEachBoost = _defiant_like('atk')


class compoundeyes:
    def onSourceModifyAccuracy(self, accuracy):
        if not is_number(accuracy):
            return None
        self.debug('compoundeyes - enhancing accuracy')
        return self.chainModify([5325, 4096])


class contrary:
    def onChangeBoost(self, boost, target, source, effect):
        if effect and effect.id == 'zpower':
            return None
        for i in boost:
            boost[i] *= -1


class cudchew:
    def onEatItem(self, item, pokemon, source, effect):
        if item.isBerry and (not effect or effect.id not in ('bugbite', 'pluck')):
            self.effectState.berry = item
            self.effectState.counter = 2
            if not self.queue.peek():
                self.effectState.counter -= 1

    def onResidual(self, pokemon):
        if not self.effectState.berry or not pokemon.hp:
            return None
        self.effectState.counter -= 1
        if self.effectState.counter <= 0:
            item = self.effectState.berry
            self.add('-activate', pokemon, 'ability: Cud Chew')
            self.add('-enditem', pokemon, item.name, '[eat]')
            if self.singleEvent('Eat', item, None, pokemon, None, None):
                self.runEvent('EatItem', pokemon, None, None, item)
            if item.onEat:
                pokemon.ateBerry = True
            self.effectState.pop('berry', None)
            self.effectState.pop('counter', None)


class curiousmedicine:
    def onStart(self, pokemon):
        for ally in pokemon.adjacentAllies():
            ally.clearBoosts()
            self.add('-clearboost', ally, '[from] ability: Curious Medicine', f"[of] {pokemon}")


class cursedbody:
    def onDamagingHit(self, damage, target, source, move):
        if source.volatiles.get('disable'):
            return None
        if not move.isMax and not move.flags.get('futuremove') and move.id != 'struggle':
            if self.randomChance(3, 10):
                source.addVolatile('disable', self.effectState.target)


class cutecharm:
    def onDamagingHit(self, damage, target, source, move):
        if self.checkMoveMakesContact(move, source, target):
            if self.randomChance(3, 10):
                source.addVolatile('attract', self.effectState.target)


class damp:
    def onAnyTryMove(self, target, source, effect):
        if effect.id in ('explosion', 'mindblown', 'mistyexplosion', 'selfdestruct'):
            self.attrLastMove('[still]')
            self.add('cant', self.effectState.target, 'ability: Damp', effect, f"[of] {target}")
            return False

    def onAnyDamage(self, damage, target, source, effect):
        if effect and effect.name == 'Aftermath':
            return False


def _disguise_blocks(self, target, move):
    hit_sub = target.volatiles.get('substitute') and not move.flags.get('bypasssub') and not move.infiltrates
    if hit_sub:
        return False
    if not target.runImmunity(move):
        return False
    return True


class disguise:
    def onDamage(self, damage, target, source, effect):
        if effect and effect.effectType == 'Move' and target.species.id in ('mimikyu', 'mimikyutotem'):
            self.add('-activate', target, 'ability: Disguise')
            self.effectState.busted = True
            return 0

    def onCriticalHit(self, target, source, move):
        if not target:
            return None
        if target.species.id not in ('mimikyu', 'mimikyutotem'):
            return None
        if not _disguise_blocks(self, target, move):
            return None
        return False

    def onEffectiveness(self, typeMod, target, type_, move):
        if not target or move.category == 'Status':
            return None
        if target.species.id not in ('mimikyu', 'mimikyutotem'):
            return None
        if not _disguise_blocks(self, target, move):
            return None
        return 0

    def onUpdate(self, pokemon):
        if pokemon.species.id in ('mimikyu', 'mimikyutotem') and self.effectState.busted:
            speciesid = 'Mimikyu-Busted-Totem' if pokemon.species.id == 'mimikyutotem' else 'Mimikyu-Busted'
            pokemon.formeChange(speciesid, self.effect, True)
            self.damage(pokemon.baseMaxhp / 8, pokemon, pokemon, self.dex.species.get(speciesid))


class drizzle:
    def onStart(self, source):
        if source.species.id == 'kyogre' and source.item == 'blueorb':
            return None
        self.field.setWeather('raindance')


class drought:
    def onStart(self, source):
        if source.species.id == 'groudon' and source.item == 'redorb':
            return None
        self.field.setWeather('sunnyday')


class dryskin:
    onTryHit = _absorb_heal('Water', 'Dry Skin')

    def onSourceBasePower(self, basePower, attacker, defender, move):
        if move.type == 'Fire':
            return self.chainModify(1.25)

    def onWeather(self, target, source, effect):
        if target.effectiveWeather() != effect.id:
            return None
        if effect.id in ('raindance', 'primordialsea'):
            self.heal(target.baseMaxhp / 8)
        elif effect.id in ('sunnyday', 'desolateland'):
            self.damage(target.baseMaxhp / 8, target, target)


class eartheater:
    onTryHit = _absorb_heal('Ground', 'Earth Eater')


class waterabsorb:
    onTryHit = _absorb_heal('Water', 'Water Absorb')


class voltabsorb:
    onTryHit = _absorb_heal('Electric', 'Volt Absorb')


class eelevate:
    def onSourceAfterFaint(self, length, target, source, effect):
        if effect and effect.effectType == 'Move':
            best = source.getBestStat(True, True)
            self.boost(Obj({best: length}), source)


class effectspore:
    def onDamagingHit(self, damage, target, source, move):
        if self.checkMoveMakesContact(move, source, target) and source.runStatusImmunity('powder'):
            r = self.random(100)
            if r < 11:
                source.trySetStatus('slp', target)
            elif r < 21:
                source.trySetStatus('par', target)
            elif r < 30:
                source.trySetStatus('psn', target)


def _surge(terrain):
    def onStart(self, source):
        self.field.setTerrain(terrain)
    return onStart


class electricsurge:
    onStart = _surge('electricterrain')


class grassysurge:
    onStart = _surge('grassyterrain')


class mistysurge:
    onStart = _surge('mistyterrain')


class psychicsurge:
    onStart = _surge('psychicterrain')


class electromorphosis:
    def onDamagingHit(self, damage, target, source, move):
        target.addVolatile('charge')


def _embody(species_name, stat):
    def onStart(self, pokemon):
        if pokemon.baseSpecies.name == species_name and pokemon.terastallized and not self.effectState.embodied:
            self.effectState.embodied = True
            self.boost(Obj({stat: 1}), pokemon)
    return onStart


class embodyaspectcornerstone:
    onStart = _embody('Ogerpon-Cornerstone-Tera', 'def')


class embodyaspecthearthflame:
    onStart = _embody('Ogerpon-Hearthflame-Tera', 'atk')


class embodyaspectteal:
    onStart = _embody('Ogerpon-Teal-Tera', 'spe')


class embodyaspectwellspring:
    onStart = _embody('Ogerpon-Wellspring-Tera', 'spd')


def _emergency_exit(label):
    def onEmergencyExit(self, originalHp, target):
        # champions override
        if not target.hp or target.hp > target.maxhp / 2 or originalHp <= target.maxhp / 2:
            return None
        if not self.canSwitch(target.side) or target.forceSwitchFlag or target.switchFlag:
            return None
        target.switchFlag = True
        self.add('-activate', target, f"ability: {label}")
    return onEmergencyExit


class emergencyexit:
    onEmergencyExit = _emergency_exit('Emergency Exit')


class wimpout:
    onEmergencyExit = _emergency_exit('Wimp Out')


def _aura(name, type_):
    def onStart(self, pokemon):
        if self.suppressingAbility(pokemon):
            return None
        self.add('-ability', pokemon, name)

    def onAnyBasePower(self, basePower, source, target, move):
        if target is source or move.category == 'Status' or move.type != type_:
            return None
        if not (move.auraBooster and move.auraBooster.hasAbility(name)):
            move.auraBooster = self.effectState.target
        if move.auraBooster is not self.effectState.target:
            return None
        return self.chainModify([3072 if move.hasAuraBreak else 5448, 4096])
    return onStart, onAnyBasePower


class fairyaura:
    onStart, onAnyBasePower = _aura('Fairy Aura', 'Fairy')


class darkaura:
    onStart, onAnyBasePower = _aura('Dark Aura', 'Dark')


class filter:
    def onSourceModifyDamage(self, damage, source, target, move):
        if target.getMoveHitData(move).typeMod > 0:
            self.debug('Filter neutralize')
            return self.chainModify(0.75)


class solidrock(filter):
    pass


class prismarmor(filter):
    pass


class firemane:
    def onModifyAtk(self, atk, attacker, defender, move):
        if move.type == 'Fire':
            self.debug('Fire Mane boost')
            return self.chainModify(1.5)

    def onModifySpA(self, atk, attacker, defender, move):
        if move.type == 'Fire':
            self.debug('Fire Mane boost')
            return self.chainModify(1.5)


def _contact_status(status):
    def onDamagingHit(self, damage, target, source, move):
        if self.checkMoveMakesContact(move, source, target):
            if self.randomChance(3, 10):
                source.trySetStatus(status, target)
    return onDamagingHit


class flamebody:
    onDamagingHit = _contact_status('brn')


class static:
    onDamagingHit = _contact_status('par')


class poisonpoint:
    onDamagingHit = _contact_status('psn')


class flashfire:
    def onTryHit(self, target, source, move):
        if target is not source and move.type == 'Fire':
            move.accuracy = True
            if not target.addVolatile('flashfire'):
                self.add('-immune', target, '[from] ability: Flash Fire')
            return NULL

    def onEnd(self, pokemon):
        pokemon.removeVolatile('flashfire')

    class condition:
        def onStart(self, target):
            self.add('-start', target, 'ability: Flash Fire')

        def onModifyAtk(self, atk, attacker, defender, move):
            if move.type == 'Fire' and attacker.hasAbility('flashfire'):
                self.debug('Flash Fire boost')
                return self.chainModify(1.5)

        def onModifySpA(self, atk, attacker, defender, move):
            if move.type == 'Fire' and attacker.hasAbility('flashfire'):
                self.debug('Flash Fire boost')
                return self.chainModify(1.5)

        def onEnd(self, target):
            self.add('-end', target, 'ability: Flash Fire', '[silent]')


class flowerveil:
    def onAllyTryBoost(self, boost, target, source, effect):
        if (source and target is source) or not target.hasType('Grass'):
            return None
        show_msg = False
        for i in list(boost.keys()):
            if boost[i] < 0:
                del boost[i]
                show_msg = True
        if show_msg and not effect.secondaries:
            self.add('-block', target, 'ability: Flower Veil', f"[of] {self.effectState.target}")

    def onAllySetStatus(self, status, target, source, effect):
        if target.hasType('Grass') and source and target is not source and effect and effect.id != 'yawn':
            self.debug('interrupting setStatus with Flower Veil')
            if effect.name == 'Synchronize' or (effect.effectType == 'Move' and not effect.secondaries):
                self.add('-block', target, 'ability: Flower Veil', f"[of] {self.effectState.target}")
            return NULL

    def onAllyTryAddVolatile(self, status, target):
        if target.hasType('Grass') and status.id == 'yawn':
            self.debug('Flower Veil blocking yawn')
            self.add('-block', target, 'ability: Flower Veil', f"[of] {self.effectState.target}")
            return NULL


class fluffy:
    def onSourceModifyDamage(self, damage, source, target, move):
        mod = 1
        if move.type == 'Fire':
            mod *= 2
        if move.flags.get('contact'):
            mod /= 2
        return self.chainModify(mod)


class forecast:
    def onStart(self, pokemon):
        self.singleEvent('WeatherChange', self.effect, self.effectState, pokemon)

    def onWeatherChange(self, pokemon):
        if pokemon.baseSpecies.baseSpecies != 'Castform' or pokemon.transformed:
            return None
        forme = None
        w = pokemon.effectiveWeather()
        if w in ('sunnyday', 'desolateland'):
            if pokemon.species.id != 'castformsunny':
                forme = 'Castform-Sunny'
        elif w in ('raindance', 'primordialsea'):
            if pokemon.species.id != 'castformrainy':
                forme = 'Castform-Rainy'
        elif w in ('hail', 'snowscape'):
            if pokemon.species.id != 'castformsnowy':
                forme = 'Castform-Snowy'
        else:
            if pokemon.species.id != 'castform':
                forme = 'Castform'
        if pokemon.isActive and forme:
            pokemon.formeChange(forme, self.effect, False, '0', '[msg]')


class forewarn:
    def onStart(self, pokemon):
        warn_moves = []
        warn_bp = 1
        for target in pokemon.foes():
            for ms in target.moveSlots:
                move = self.dex.moves.get(ms.move)
                bp = move.basePower
                if move.ohko:
                    bp = 150
                if move.id in ('counter', 'metalburst', 'mirrorcoat'):
                    bp = 120
                if bp == 1:
                    bp = 80
                if not bp and move.category != 'Status':
                    bp = 80
                if bp > warn_bp:
                    warn_moves = [(move, target)]
                    warn_bp = bp
                elif bp == warn_bp:
                    warn_moves.append((move, target))
        if not warn_moves:
            return None
        warn_move, warn_target = self.sample(warn_moves)
        self.add('-activate', pokemon, 'ability: Forewarn', warn_move, f"[of] {warn_target}")


class friendguard:
    def onAnyModifyDamage(self, damage, source, target, move):
        if target is not self.effectState.target and target.isAlly(self.effectState.target):
            self.debug('Friend Guard weaken')
            return self.chainModify(0.75)


class frisk:
    def onStart(self, pokemon):
        for target in pokemon.foes():
            if target.item:
                self.add('-item', target, target.getItem().name, '[from] ability: Frisk', f"[of] {pokemon}")


class furcoat:
    def onModifyDef(self, def_):
        return self.chainModify(2)


class galewings:
    def onModifyPriority(self, priority, pokemon, target, move):
        if move and move.type == 'Flying' and pokemon.hp == pokemon.maxhp:
            return priority + 1


class gluttony:
    def onStart(self, pokemon):
        pokemon.abilityState.gluttony = True

    def onDamage(self, item, pokemon):
        pokemon.abilityState.gluttony = True


class goodasgold:
    def onTryHit(self, target, source, move):
        if move.category == 'Status' and target is not source:
            self.add('-immune', target, '[from] ability: Good as Gold')
            return NULL


class gooey:
    def onDamagingHit(self, damage, target, source, move):
        if self.checkMoveMakesContact(move, source, target, True):
            self.add('-ability', target, 'Gooey')
            self.boost(Obj(spe=-1), source, target, None, True)


class tanglinghair(gooey):
    def onDamagingHit(self, damage, target, source, move):
        if self.checkMoveMakesContact(move, source, target, True):
            self.add('-ability', target, 'Tangling Hair')
            self.boost(Obj(spe=-1), source, target, None, True)


class grasspelt:
    def onModifyDef(self, pokemon):
        if self.field.isTerrain('grassyterrain'):
            return self.chainModify(1.5)


class guarddog:
    def onDragOut(self, pokemon):
        self.add('-activate', pokemon, 'ability: Guard Dog')
        return NULL

    def onTryBoost(self, boost, target, source, effect):
        if effect.name == 'Intimidate' and boost.get('atk'):
            del boost['atk']
            self.boost(Obj(atk=1), target, target, None, False, True)


class gulpmissile:
    def onDamagingHit(self, damage, target, source, move):
        if not source.hp or not source.isActive or target.isSemiInvulnerable():
            return None
        if target.species.id in ('cramorantgulping', 'cramorantgorging'):
            self.damage(source.baseMaxhp / 4, source, target)
            if target.species.id == 'cramorantgulping':
                self.boost(Obj(**{'def': -1}), source, target, None, True)
            else:
                source.trySetStatus('par', target, move)
            target.formeChange('cramorant', move)

    def onSourceTryPrimaryHit(self, target, source, effect):
        if effect and effect.id == 'surf' and source.hasAbility('gulpmissile') and source.species.name == 'Cramorant':
            forme = 'cramorantgorging' if source.hp <= source.maxhp / 2 else 'cramorantgulping'
            source.formeChange(forme, effect)


class guts:
    def onModifyAtk(self, atk, pokemon):
        if pokemon.status:
            return self.chainModify(1.5)


class harvest:
    def onResidual(self, pokemon):
        if self.field.isWeather(['sunnyday', 'desolateland']) or self.randomChance(1, 2):
            if pokemon.hp and not pokemon.item and self.dex.items.get(pokemon.lastItem).isBerry:
                pokemon.setItem(pokemon.lastItem)
                pokemon.lastItem = ''
                self.add('-item', pokemon, pokemon.getItem(), '[from] ability: Harvest')


class healer:
    def onResidual(self, pokemon):
        # champions override (no position check besides adjacency)
        for ally in pokemon.adjacentAllies():
            if ally.status and self.randomChance(1, 2):
                self.add('-activate', pokemon, 'ability: Healer')
                ally.cureStatus()


class heatproof:
    def onSourceModifyAtk(self, atk, attacker, defender, move):
        if move.type == 'Fire':
            self.debug('Heatproof Atk weaken')
            return self.chainModify(0.5)

    def onSourceModifySpA(self, atk, attacker, defender, move):
        if move.type == 'Fire':
            self.debug('Heatproof SpA weaken')
            return self.chainModify(0.5)

    def onDamage(self, damage, target, source, effect):
        if effect and effect.id == 'brn':
            return damage / 2


class heavymetal:
    def onModifyWeight(self, weighthg):
        return weighthg * 2


class hospitality:
    def onStart(self, pokemon):
        for ally in pokemon.adjacentAllies():
            self.heal(ally.baseMaxhp / 4, ally, pokemon)


class hugepower:
    def onModifyAtk(self, atk):
        return self.chainModify(2)


class purepower(hugepower):
    pass


class hungerswitch:
    def onResidual(self, pokemon):
        if pokemon.species.baseSpecies != 'Morpeko' or pokemon.terastallized:
            return None
        target_forme = 'Morpeko-Hangry' if pokemon.species.name == 'Morpeko' else 'Morpeko'
        pokemon.formeChange(target_forme)


class hustle:
    def onModifyAtk(self, atk):
        return self.modify(atk, 1.5)

    def onSourceModifyAccuracy(self, accuracy, target, source, move):
        if move.category == 'Physical' and is_number(accuracy):
            return self.chainModify([3277, 4096])


class hydration:
    def onResidual(self, pokemon):
        if pokemon.status and pokemon.effectiveWeather() in ('raindance', 'primordialsea'):
            self.debug('hydration')
            self.add('-activate', pokemon, 'ability: Hydration')
            pokemon.cureStatus()


def _stat_guard(stat, name):
    def onTryBoost(self, boost, target, source, effect):
        if source and target is source:
            return None
        if boost.get(stat) and boost[stat] < 0:
            del boost[stat]
            if not effect.secondaries:
                self.add('-fail', target, 'unboost', stat, f"[from] ability: {name}", f"[of] {target}")
    return onTryBoost


class hypercutter:
    onTryBoost = _stat_guard('atk', 'Hyper Cutter')


class icebody:
    def onWeather(self, target, source, effect):
        if effect.id in ('hail', 'snowscape'):
            self.heal(target.baseMaxhp / 16)

    def onImmunity(self, type_, pokemon):
        if type_ == 'hail':
            return False


class iceface:
    def onStart(self, pokemon):
        if self.field.isWeather(['hail', 'snowscape']) and pokemon.species.id == 'eiscuenoice':
            self.add('-activate', pokemon, 'ability: Ice Face')
            self.effectState.busted = False
            pokemon.formeChange('Eiscue', self.effect, True)

    def onDamage(self, damage, target, source, effect):
        if effect and effect.effectType == 'Move' and effect.category == 'Physical' and target.species.id == 'eiscue':
            self.add('-activate', target, 'ability: Ice Face')
            self.effectState.busted = True
            return 0

    def onCriticalHit(self, target, type_, move):
        if not target:
            return None
        if move.category != 'Physical' or target.species.id != 'eiscue':
            return None
        if target.volatiles.get('substitute') and not (move.flags.get('bypasssub') or move.infiltrates):
            return None
        if not target.runImmunity(move):
            return None
        return False

    def onEffectiveness(self, typeMod, target, type_, move):
        if not target:
            return None
        if move.category != 'Physical' or target.species.id != 'eiscue':
            return None
        hit_sub = target.volatiles.get('substitute') and not move.flags.get('bypasssub') and not move.infiltrates
        if hit_sub:
            return None
        if not target.runImmunity(move):
            return None
        return 0

    def onUpdate(self, pokemon):
        if pokemon.species.id == 'eiscue' and self.effectState.busted:
            pokemon.formeChange('Eiscue-Noice', self.effect, True)

    def onWeatherChange(self, pokemon, source, sourceEffect):
        if sourceEffect and sourceEffect.suppressWeather:
            return None
        if not pokemon.hp:
            return None
        if self.field.isWeather(['hail', 'snowscape']) and pokemon.species.id == 'eiscuenoice':
            self.add('-activate', pokemon, 'ability: Ice Face')
            self.effectState.busted = False
            pokemon.formeChange('Eiscue', self.effect, True)


class illuminate:
    onTryBoost = _stat_guard('accuracy', 'Illuminate')

    def onModifyMove(self, move):
        move.ignoreEvasion = True


class keeneye(illuminate):
    onTryBoost = _stat_guard('accuracy', 'Keen Eye')


class mindseye(illuminate):
    onTryBoost = _stat_guard('accuracy', "Mind's Eye")


class illusion:
    def onBeforeSwitchIn(self, pokemon):
        pokemon.illusion = None
        for i in range(len(pokemon.side.pokemon) - 1, pokemon.position, -1):
            possible = pokemon.side.pokemon[i]
            if not possible.fainted:
                if not pokemon.terastallized or possible.species.baseSpecies not in ('Ogerpon', 'Terapagos'):
                    pokemon.illusion = possible
                break

    def onDamagingHit(self, damage, target, source, move):
        if target.illusion:
            self.singleEvent('End', self.dex.abilities.get('Illusion'), target.abilityState, target, source, move)

    def onEnd(self, pokemon):
        if pokemon.illusion and not pokemon.beingCalledBack:
            self.debug('illusion cleared')
            pokemon.illusion = None
            details = pokemon.getUpdatedDetails()
            self.add('replace', pokemon, details)
            self.add('-end', pokemon, 'Illusion')
            if self.ruleTable.has('illusionlevelmod'):
                self.hint("Illusion Level Mod is active, so this Pokémon's true level was hidden.", True)

    def onFaint(self, pokemon):
        pokemon.illusion = None


class immunity:
    def onUpdate(self, pokemon):
        if pokemon.status in ('psn', 'tox'):
            self.add('-activate', pokemon, 'ability: Immunity')
            pokemon.cureStatus()

    def onSetStatus(self, status, target, source, effect):
        if status.id not in ('psn', 'tox'):
            return None
        if effect and effect.status:
            self.add('-immune', target, '[from] ability: Immunity')
        return False


class imposter:
    def onSwitchIn(self, pokemon):
        foe_active = pokemon.side.foe.active
        idx = len(foe_active) - 1 - pokemon.position
        target = foe_active[idx] if 0 <= idx < len(foe_active) else None
        if target:
            pokemon.transformInto(target, self.dex.abilities.get('imposter'))


class infiltrator:
    def onModifyMove(self, move):
        move.infiltrates = True


class innardsout:
    def onDamagingHit(self, damage, target, source, move):
        if not target.hp:
            if not move.smartTarget:
                damage += (move.totalDamage or 0)
            self.damage(target.getUndynamaxedHP(damage), source, target)


class innerfocus:
    def onTryAddVolatile(self, status, pokemon):
        if status.id == 'flinch':
            return NULL

    def onTryBoost(self, boost, target, source, effect):
        if effect.name == 'Intimidate' and boost.get('atk'):
            del boost['atk']
            self.add('-fail', target, 'unboost', 'atk', '[from] ability: Inner Focus', f"[of] {target}")


def _status_immune_ability(status_ids, name, update_status=None):
    class _A:
        def onUpdate(self, pokemon):
            if pokemon.status in (update_status or status_ids):
                self.add('-activate', pokemon, f"ability: {name}")
                pokemon.cureStatus()

        def onSetStatus(self, status, target, source, effect):
            if status.id not in status_ids:
                return None
            if effect and effect.status:
                self.add('-immune', target, f"[from] ability: {name}")
            return False
    return _A


class insomnia(_status_immune_ability(('slp',), 'Insomnia')):
    def onTryAddVolatile(self, status, target):
        if status.id == 'yawn':
            self.add('-immune', target, '[from] ability: Insomnia')
            return NULL


class vitalspirit(_status_immune_ability(('slp',), 'Vital Spirit')):
    def onTryAddVolatile(self, status, target):
        if status.id == 'yawn':
            self.add('-immune', target, '[from] ability: Vital Spirit')
            return NULL


class limber(_status_immune_ability(('par',), 'Limber')):
    pass


class intimidate:
    def onStart(self, pokemon):
        activated = False
        for target in pokemon.adjacentFoes():
            if not activated:
                self.add('-ability', pokemon, 'Intimidate', 'boost')
                activated = True
            if target.volatiles.get('substitute'):
                self.add('-immune', target)
            else:
                self.boost(Obj(atk=-1), target, pokemon, None, True)


class ironfist:
    def onBasePower(self, basePower, attacker, defender, move):
        if move.flags.get('punch'):
            self.debug('Iron Fist boost')
            return self.chainModify([4915, 4096])


class justified:
    def onDamagingHit(self, damage, target, source, move):
        if move.type == 'Dark':
            self.boost(Obj(atk=1))


class klutz:
    def onStart(self, pokemon):
        self.singleEvent('End', pokemon.getItem(), pokemon.itemState, pokemon)


class leafguard:
    def onSetStatus(self, status, target, source, effect):
        if target.effectiveWeather() in ('sunnyday', 'desolateland'):
            if effect and effect.status:
                self.add('-immune', target, '[from] ability: Leaf Guard')
            return False

    def onTryAddVolatile(self, status, target):
        if status.id == 'yawn' and target.effectiveWeather() in ('sunnyday', 'desolateland'):
            self.add('-immune', target, '[from] ability: Leaf Guard')
            return NULL


def _protean(name, key):
    def onPrepareHit(self, source, target, move):
        if self.effectState[key]:
            return None
        if move.hasBounced or move.flags.get('futuremove') or move.sourceEffect == 'snatch' or move.callsMove:
            return None
        type_ = move.type
        if type_ and type_ != '???' and ','.join(source.getTypes()) != type_:
            if not source.setType(type_):
                return None
            self.effectState[key] = True
            self.add('-start', source, 'typechange', type_, f"[from] ability: {name}")
    return onPrepareHit


class libero:
    onPrepareHit = _protean('Libero', 'libero')


class protean:
    onPrepareHit = _protean('Protean', 'protean')


class lightmetal:
    def onModifyWeight(self, weighthg):
        return self.trunc(weighthg / 2)


def _redirect_ability(type_, boost_stat, name):
    class _A:
        def onTryHit(self, target, source, move):
            if target is not source and move.type == type_:
                if not self.boost(Obj({boost_stat: 1})):
                    self.add('-immune', target, f"[from] ability: {name}")
                return NULL

        def onAnyRedirectTarget(self, target, source, source2, move):
            if move.type != type_ or move.flags.get('pledgecombo'):
                return None
            redirect_target = 'normal' if move.target in ('randomNormal', 'adjacentFoe') else move.target
            if self.validTarget(self.effectState.target, source, redirect_target):
                if move.smartTarget:
                    move.smartTarget = False
                if self.effectState.target is not target:
                    self.add('-activate', self.effectState.target, f"ability: {name}")
                return self.effectState.target
    return _A


class lightningrod(_redirect_ability('Electric', 'spa', 'Lightning Rod')):
    pass


class stormdrain(_redirect_ability('Water', 'spa', 'Storm Drain')):
    pass


class liquidooze:
    def onSourceTryHeal(self, damage, target, source, effect):
        self.debug(f"Heal is occurring: {target} <- {source} :: {effect.id}")
        if effect.id in ('drain', 'leechseed', 'strengthsap'):
            self.damage(damage)
            return 0


class liquidvoice:
    def onModifyType(self, move, pokemon):
        if move.flags.get('sound') and not pokemon.volatiles.get('dynamax'):
            move.type = 'Water'


class longreach:
    def onModifyMove(self, move):
        move.flags.pop('contact', None)


class magicbounce:
    def onTryHit(self, target, source, move):
        if target is source or move.hasBounced or not move.flags.get('reflectable') or target.isSemiInvulnerable():
            return None
        new_move = self.dex.getActiveMove(move.id)
        new_move.hasBounced = True
        new_move.pranksterBoosted = False
        self.actions.useMove(new_move, target, Obj(target=source))
        return NULL

    def onAllyTryHitSide(self, target, source, move):
        if target.isAlly(source) or move.hasBounced or not move.flags.get('reflectable') or \
                target.isSemiInvulnerable():
            return None
        new_move = self.dex.getActiveMove(move.id)
        new_move.hasBounced = True
        new_move.pranksterBoosted = False
        self.actions.useMove(new_move, self.effectState.target, Obj(target=source))
        move.hasBounced = True
        return NULL


class magicguard:
    def onDamage(self, damage, target, source, effect):
        if effect.effectType != 'Move':
            if effect.effectType == 'Ability':
                self.add('-activate', source, 'ability: ' + effect.name)
            return False


class magician:
    def onAfterMoveSecondarySelf(self, source, target, move):
        if (not move or source.switchFlag is True or not move.hitTargets or source.item or
                source.volatiles.get('gem') or move.id == 'fling' or move.category == 'Status'):
            return None
        hit_targets = move.hitTargets
        self.speedSort(hit_targets)
        for pokemon in hit_targets:
            if pokemon is not source:
                your_item = pokemon.takeItem(source)
                if not your_item:
                    continue
                if not source.setItem(your_item):
                    pokemon.item = your_item.id
                    continue
                self.add('-item', source, your_item, '[from] ability: Magician', f"[of] {pokemon}")
                return None


class magmaarmor:
    def onUpdate(self, pokemon):
        if pokemon.status == 'frz':
            self.add('-activate', pokemon, 'ability: Magma Armor')
            pokemon.cureStatus()

    def onImmunity(self, type_, pokemon):
        if type_ == 'frz':
            return False


class marvelscale:
    def onModifyDef(self, def_, pokemon):
        if pokemon.status:
            return self.chainModify(1.5)


class megalauncher:
    def onBasePower(self, basePower, attacker, defender, move):
        if move.flags.get('pulse'):
            return self.chainModify(1.5)


class megasol:
    def onWeatherModifyDamage(self, damage, attacker, defender, move):
        self.dex.conditions.getByID('sunnyday').onWeatherModifyDamage(self, damage, attacker, defender, move)
        return damage


class merciless:
    def onModifyCritRatio(self, critRatio, source, target):
        if target and target.status in ('psn', 'tox'):
            return 5


class mimicry:
    def onStart(self, pokemon):
        self.singleEvent('TerrainChange', self.effect, self.effectState, pokemon)

    def onTerrainChange(self, pokemon):
        t = self.field.terrain
        if t == 'electricterrain':
            types = ['Electric']
        elif t == 'grassyterrain':
            types = ['Grass']
        elif t == 'mistyterrain':
            types = ['Fairy']
        elif t == 'psychicterrain':
            types = ['Psychic']
        else:
            types = pokemon.baseSpecies.types
        old_types = pokemon.getTypes()
        if ','.join(old_types) == ','.join(types) or not pokemon.setType(types):
            return None
        if self.field.terrain or pokemon.transformed:
            self.add('-start', pokemon, 'typechange', '/'.join(types), '[from] ability: Mimicry')
            if not self.field.terrain:
                self.hint('Transform Mimicry changes you to your original un-transformed types.')
        else:
            self.add('-activate', pokemon, 'ability: Mimicry')
            self.add('-end', pokemon, 'typechange', '[silent]')


def _plus_minus(self, spa, pokemon):
    for ally in pokemon.allies():
        if ally.hasAbility(['minus', 'plus']):
            return self.chainModify(1.5)


class minus:
    onModifySpA = _plus_minus


class plus:
    onModifySpA = _plus_minus


class mirrorarmor:
    def onTryBoost(self, boost, target, source, effect):
        if not source or target is source or not boost or effect.name == 'Mirror Armor':
            return None
        for b in list(boost.keys()):
            if boost[b] < 0:
                if target.boosts[b] == -6:
                    continue
                negative = Obj({b: boost[b]})
                del boost[b]
                if source.hp:
                    self.add('-ability', target, 'Mirror Armor')
                    self.boost(negative, source, target, None, True)


class moldbreaker:
    def onStart(self, pokemon):
        self.add('-ability', pokemon, 'Mold Breaker')

    def onModifyMove(self, move):
        move.ignoreAbility = True


class teravolt(moldbreaker):
    def onStart(self, pokemon):
        self.add('-ability', pokemon, 'Teravolt')


class turboblaze(moldbreaker):
    def onStart(self, pokemon):
        self.add('-ability', pokemon, 'Turboblaze')


class moody:
    def onResidual(self, pokemon):
        stats = [s for s in pokemon.boosts if s not in ('accuracy', 'evasion') and pokemon.boosts[s] < 6]
        boost = Obj()
        random_stat = self.sample(stats) if stats else None
        if random_stat:
            boost[random_stat] = 2
        stats = [s for s in pokemon.boosts if s not in ('accuracy', 'evasion') and pokemon.boosts[s] > -6 and
                 s != random_stat]
        random_stat = self.sample(stats) if stats else None
        if random_stat:
            boost[random_stat] = -1
        self.boost(boost, pokemon, pokemon)


class motordrive:
    def onTryHit(self, target, source, move):
        if target is not source and move.type == 'Electric':
            if not self.boost(Obj(spe=1)):
                self.add('-immune', target, '[from] ability: Motor Drive')
            return NULL


class moxie:
    def onSourceAfterFaint(self, length, target, source, effect):
        if effect and effect.effectType == 'Move':
            self.boost(Obj(atk=length), source)


class multiscale:
    def onSourceModifyDamage(self, damage, source, target, move):
        if target.hp >= target.maxhp:
            self.debug('Multiscale weaken')
            return self.chainModify(0.5)


class shadowshield(multiscale):
    pass


class mummy:
    def onDamagingHit(self, damage, target, source, move):
        source_ability = source.getAbility()
        if source_ability.flags.get('cantsuppress') or source_ability.id == 'mummy':
            return None
        if self.checkMoveMakesContact(move, source, target, not source.isAlly(target)):
            source.setAbility('mummy', target)


class naturalcure:
    def onSwitchOut(self, pokemon):
        # champions override
        if not pokemon.status or pokemon.status == 'fnt':
            return None
        self.add('-curestatus', pokemon, pokemon.status, '[from] ability: Natural Cure', '[silent]')
        pokemon.clearStatus()


class noguard:
    def onAnyInvulnerability(self, target, source, move):
        if move and (source is self.effectState.target or target is self.effectState.target):
            return 0

    def onAnyAccuracy(self, accuracy, target, source, move):
        if move and (source is self.effectState.target or target is self.effectState.target):
            return True
        return accuracy


class oblivious:
    def onUpdate(self, pokemon):
        if pokemon.volatiles.get('attract'):
            self.add('-activate', pokemon, 'ability: Oblivious')
            pokemon.removeVolatile('attract')
            self.add('-end', pokemon, 'move: Attract', '[from] ability: Oblivious')
        if pokemon.volatiles.get('taunt'):
            self.add('-activate', pokemon, 'ability: Oblivious')
            pokemon.removeVolatile('taunt')

    def onImmunity(self, type_, pokemon):
        if type_ == 'attract':
            return False

    def onTryHit(self, pokemon, target, move):
        if move.id in ('attract', 'captivate', 'taunt'):
            self.add('-immune', pokemon, '[from] ability: Oblivious')
            return NULL

    def onTryBoost(self, boost, target, source, effect):
        if effect.name == 'Intimidate' and boost.get('atk'):
            del boost['atk']
            self.add('-fail', target, 'unboost', 'atk', '[from] ability: Oblivious', f"[of] {target}")


def _opportunist_release(self, *args):
    if not self.effectState.boosts:
        return None
    self.boost(self.effectState.boosts, self.effectState.target)
    self.effectState.pop('boosts', None)


class opportunist:
    def onFoeAfterBoost(self, boost, target, source, effect):
        if effect and effect.name in ('Opportunist', 'Mirror Herb'):
            return None
        if not self.effectState.boosts:
            self.effectState.boosts = Obj()
        plus_ = self.effectState.boosts
        for i in boost:
            if boost[i] > 0:
                plus_[i] = (plus_.get(i) or 0) + boost[i]

    onAnySwitchIn = _opportunist_release
    onAnyAfterMega = _opportunist_release
    onAnyAfterTerastallization = _opportunist_release
    onAnyAfterMove = _opportunist_release
    onResidual = _opportunist_release

    def onEnd(self):
        self.effectState.pop('boosts', None)


class overcoat:
    def onImmunity(self, type_, pokemon):
        if type_ in ('sandstorm', 'hail', 'powder'):
            return False

    def onTryHit(self, target, source, move):
        if move.flags.get('powder') and target is not source and self.dex.getImmunity('powder', target):
            self.add('-immune', target, '[from] ability: Overcoat')
            return NULL


def _intimidate_guard(name):
    def onTryBoost(self, boost, target, source, effect):
        if effect.name == 'Intimidate' and boost.get('atk'):
            del boost['atk']
            self.add('-fail', target, 'unboost', 'atk', f"[from] ability: {name}", f"[of] {target}")
    return onTryBoost


class owntempo:
    def onUpdate(self, pokemon):
        if pokemon.volatiles.get('confusion'):
            self.add('-activate', pokemon, 'ability: Own Tempo')
            pokemon.removeVolatile('confusion')

    def onTryAddVolatile(self, status, pokemon):
        if status.id == 'confusion':
            return NULL

    def onHit(self, target, source, move):
        if move and move.volatileStatus == 'confusion':
            self.add('-immune', target, 'confusion', '[from] ability: Own Tempo')

    onTryBoost = _intimidate_guard('Own Tempo')


class parentalbond:
    def onPrepareHit(self, source, target, move):
        if (move.category == 'Status' or move.multihit or move.flags.get('noparentalbond') or
                move.flags.get('charge') or move.flags.get('futuremove') or move.spreadHit or move.isZ or move.isMax):
            return None
        move.multihit = 2
        move.multihitType = 'parentalbond'

    def onSourceModifySecondaries(self, secondaries, target, source, move):
        if move.multihitType == 'parentalbond' and move.id == 'secretpower' and move.hit < 2:
            return [e for e in secondaries if e.volatileStatus == 'flinch']


class pickpocket:
    def onAfterMoveSecondary(self, target, source, move):
        if source and source is not target and move and move.flags.get('contact'):
            if target.item or target.switchFlag or target.forceSwitchFlag or source.switchFlag is True:
                return None
            your_item = source.takeItem(target)
            if not your_item:
                return None
            if not target.setItem(your_item):
                source.item = your_item.id
                return None
            self.add('-enditem', source, your_item, '[silent]', '[from] ability: Pickpocket', f"[of] {source}")
            self.add('-item', target, your_item, '[from] ability: Pickpocket', f"[of] {source}")


class pickup:
    def onResidual(self, pokemon):
        if pokemon.item:
            return None
        targets = [t for t in self.getAllActive() if t.lastItem and t.usedItemThisTurn and pokemon.isAdjacent(t)]
        if not targets:
            return None
        random_target = self.sample(targets)
        item = random_target.lastItem
        random_target.lastItem = ''
        self.add('-item', pokemon, self.dex.items.get(item), '[from] ability: Pickup')
        pokemon.setItem(item)


def _contact_bypass_protect(self, source, target, move):
    if move.flags.get('contact'):
        target.getMoveHitData(move).bypassProtect = self.effect
        return False


class piercingdrill:
    onHitProtect = _contact_bypass_protect


class unseenfist:
    onHitProtect = _contact_bypass_protect  # champions override


class poisonheal:
    def onDamage(self, damage, target, source, effect):
        if effect.id in ('psn', 'tox'):
            self.heal(target.baseMaxhp / 8)
            return False


class poisontouch:
    def onSourceDamagingHit(self, damage, target, source, move):
        if target.hasAbility('shielddust') or target.hasItem('covertcloak'):
            return None
        if self.checkMoveMakesContact(move, target, source):
            if self.randomChance(3, 10):
                target.trySetStatus('psn', source)


class prankster:
    def onModifyPriority(self, priority, pokemon, target, move):
        if move and move.category == 'Status':
            move.pranksterBoosted = True
            return priority + 1


class pressure:
    def onStart(self, pokemon):
        self.add('-ability', pokemon, 'Pressure')

    def onDeductPP(self, target, source):
        if target.isAlly(source):
            return None
        return 1


class punkrock:
    def onBasePower(self, basePower, attacker, defender, move):
        if move.flags.get('sound'):
            self.debug('Punk Rock boost')
            return self.chainModify([5325, 4096])

    def onSourceModifyDamage(self, damage, source, target, move):
        if move.flags.get('sound'):
            self.debug('Punk Rock weaken')
            return self.chainModify(0.5)


class purifyingsalt:
    def onSetStatus(self, status, target, source, effect):
        if effect and effect.status:
            self.add('-immune', target, '[from] ability: Purifying Salt')
        return False

    def onTryAddVolatile(self, status, target):
        if status.id == 'yawn':
            self.add('-immune', target, '[from] ability: Purifying Salt')
            return NULL

    def onSourceModifyAtk(self, atk, attacker, defender, move):
        if move.type == 'Ghost':
            self.debug('Purifying Salt weaken')
            return self.chainModify(0.5)

    def onSourceModifySpA(self, spa, attacker, defender, move):
        if move.type == 'Ghost':
            self.debug('Purifying Salt weaken')
            return self.chainModify(0.5)


def _dazzling(name):
    def onFoeTryMove(self, target, source, move):
        if move.target == 'foeSide' or (move.target == 'all' and move.id not in ('perishsong', 'flowershield',
                                                                                 'rototiller')):
            return None
        holder = self.effectState.target
        if (source.isAlly(holder) or move.target == 'all') and move.priority > 0.1:
            self.attrLastMove('[still]')
            self.add('cant', holder, f"ability: {name}", move, f"[of] {target}")
            return False
    return onFoeTryMove


class queenlymajesty:
    onFoeTryMove = _dazzling('Queenly Majesty')


class dazzling:
    onFoeTryMove = _dazzling('Dazzling')


class quickdraw:
    def onFractionalPriority(self, priority, pokemon, target, move):
        if move.category != 'Status' and self.randomChance(3, 10):
            self.add('-activate', pokemon, 'ability: Quick Draw')
            return 0.1


class quickfeet:
    def onModifySpe(self, spe, pokemon):
        if pokemon.status:
            return self.chainModify(1.5)


class raindish:
    def onWeather(self, target, source, effect):
        if target.effectiveWeather() != effect.id:
            return None
        if effect.id in ('raindance', 'primordialsea'):
            self.heal(target.baseMaxhp / 16)


class rattled:
    def onDamagingHit(self, damage, target, source, move):
        if move.type in ('Dark', 'Bug', 'Ghost'):
            self.boost(Obj(spe=1))

    def onAfterBoost(self, boost, target, source, effect):
        if effect and effect.name == 'Intimidate' and boost.get('atk'):
            self.boost(Obj(spe=1))


class receiver:
    def onAllyFaint(self, target):
        if not self.effectState.target.hp:
            return None
        ability = target.getAbility()
        if ability.flags.get('noreceiver') or ability.id == 'noability':
            return None
        self.effectState.target.setAbility(ability, target)


class powerofalchemy(receiver):
    pass


class reckless:
    def onBasePower(self, basePower, attacker, defender, move):
        if move.recoil or move.hasCrashDamage:
            self.debug('Reckless boost')
            return self.chainModify([4915, 4096])


class regenerator:
    def onSwitchOut(self, pokemon):
        # champions override
        if pokemon.heal(pokemon.baseMaxhp / 3):
            self.add('-heal', pokemon, pokemon.getHealth, '[from] ability: Regenerator', '[silent]')


_WEAKEN_BERRIES = ('Babiri Berry', 'Charti Berry', 'Chilan Berry', 'Chople Berry', 'Coba Berry', 'Colbur Berry',
                   'Haban Berry', 'Kasib Berry', 'Kebia Berry', 'Occa Berry', 'Passho Berry', 'Payapa Berry',
                   'Rindo Berry', 'Roseli Berry', 'Shuca Berry', 'Tanga Berry', 'Wacan Berry', 'Yache Berry')


class ripen:
    def onTryHeal(self, damage, target, source, effect):
        if not effect:
            return None
        if effect.name in ('Berry Juice', 'Leftovers'):
            self.add('-activate', target, 'ability: Ripen')
        if effect.isBerry:
            return self.chainModify(2)

    def onChangeBoost(self, boost, target, source, effect):
        if effect and effect.isBerry:
            for b in boost:
                boost[b] *= 2

    def onSourceModifyDamage(self, damage, source, target, move):
        if target.abilityState.berryWeaken:
            target.abilityState.berryWeaken = False
            return self.chainModify(0.5)

    def onTryEatItem(self, item, pokemon):
        self.add('-activate', pokemon, 'ability: Ripen')

    def onEatItem(self, item, pokemon):
        pokemon.abilityState.berryWeaken = item.name in _WEAKEN_BERRIES


class rivalry:
    def onBasePower(self, basePower, attacker, defender, move):
        if attacker.gender and defender.gender:
            if attacker.gender == defender.gender:
                self.debug('Rivalry boost')
                return self.chainModify(1.25)
            else:
                self.debug('Rivalry weaken')
                return self.chainModify(0.75)


class rockhead:
    def onDamage(self, damage, target, source, effect):
        if effect.id == 'recoil':
            if not self.activeMove:
                raise ValueError('Battle.activeMove is null')
            if self.activeMove.id != 'struggle':
                return NULL


class roughskin:
    def onDamagingHit(self, damage, target, source, move):
        if self.checkMoveMakesContact(move, source, target, True):
            self.damage(source.baseMaxhp / 8, source, target)


class ironbarbs(roughskin):
    pass


class runaway:
    # champions override: Run Away prevents trapping
    def onTrapPokemon(self, pokemon):
        pokemon.trapped = False

    def onMaybeTrapPokemon(self, pokemon):
        pokemon.maybeTrapped = False


def _sand_immune(self, type_, pokemon):
    if type_ == 'sandstorm':
        return False


class sandforce:
    def onBasePower(self, basePower, attacker, defender, move):
        if self.field.isWeather('sandstorm'):
            if move.type in ('Rock', 'Ground', 'Steel'):
                self.debug('Sand Force boost')
                return self.chainModify([5325, 4096])

    onImmunity = _sand_immune


class sandrush:
    def onModifySpe(self, spe, pokemon):
        if self.field.isWeather('sandstorm'):
            return self.chainModify(2)

    onImmunity = _sand_immune


class sandspit:
    def onDamagingHit(self, damage, target, source, move):
        self.field.setWeather('sandstorm')


class sandstream:
    def onStart(self, source):
        self.field.setWeather('sandstorm')


class sandveil:
    onImmunity = _sand_immune

    def onModifyAccuracy(self, accuracy):
        if not is_number(accuracy):
            return None
        if self.field.isWeather('sandstorm'):
            self.debug('Sand Veil - decreasing accuracy')
            return self.chainModify([3277, 4096])


class sapsipper:
    def onTryHit(self, target, source, move):
        if target is not source and move.type == 'Grass':
            if not self.boost(Obj(atk=1)):
                self.add('-immune', target, '[from] ability: Sap Sipper')
            return NULL

    def onAllyTryHitSide(self, target, source, move):
        if source is self.effectState.target or not target.isAlly(source):
            return None
        if move.type == 'Grass':
            self.boost(Obj(atk=1), self.effectState.target)


class scrappy:
    def onModifyMove(self, move):
        if not move.ignoreImmunity:
            move.ignoreImmunity = Obj()
        if move.ignoreImmunity is not True:
            move.ignoreImmunity['Fighting'] = True
            move.ignoreImmunity['Normal'] = True

    onTryBoost = _intimidate_guard('Scrappy')


class screencleaner:
    def onStart(self, pokemon):
        activated = False
        for cond in ('reflect', 'lightscreen', 'auroraveil'):
            for side in [pokemon.side] + pokemon.side.foeSidesWithConditions():
                if side.getSideCondition(cond):
                    if not activated:
                        self.add('-activate', pokemon, 'ability: Screen Cleaner')
                        activated = True
                    side.removeSideCondition(cond)


class seedsower:
    def onDamagingHit(self, damage, target, source, move):
        self.field.setTerrain('grassyterrain')


class serenegrace:
    def onModifyMove(self, move):
        if move.get('secondaries') is not None:
            self.debug('doubling secondary chance')
            for secondary in move.secondaries:
                if secondary.chance:
                    secondary.chance *= 2
        if move.self and move.self.chance:
            move.self.chance *= 2


class shadowtag:
    def onFoeTrapPokemon(self, pokemon):
        if not pokemon.hasAbility('shadowtag') and pokemon.isAdjacent(self.effectState.target):
            pokemon.tryTrap(True)

    def onFoeMaybeTrapPokemon(self, pokemon, source):
        if not source:
            source = self.effectState.target
        if not source or not pokemon.isAdjacent(source):
            return None
        if not pokemon.hasAbility('shadowtag'):
            pokemon.maybeTrapped = True


class sharpness:
    def onBasePower(self, basePower, attacker, defender, move):
        if move.flags.get('slicing'):
            self.debug('Sharpness boost')
            return self.chainModify(1.5)


class shedskin:
    def onResidual(self, pokemon):
        if pokemon.hp and pokemon.status and self.randomChance(33, 100):
            self.debug('shed skin')
            self.add('-activate', pokemon, 'ability: Shed Skin')
            pokemon.cureStatus()


class sheerforce:
    def onModifyMove(self, move, pokemon):
        if move.get('secondaries') is not None and not move.hasSheerForceBoost:
            move.pop('secondaries', None)
            move.pop('self', None)
            if move.id == 'clangoroussoulblaze':
                move.pop('selfBoost', None)
            move.hasSheerForce = True

    def onBasePower(self, basePower, pokemon, target, move):
        if move.hasSheerForce or move.hasSheerForceBoost:
            return self.chainModify([5325, 4096])


class shielddust:
    def onModifySecondaries(self, secondaries):
        self.debug('Shield Dust prevent secondary')
        return [e for e in secondaries if e.self]


def _minior_check(self, pokemon):
    if pokemon.hp > pokemon.maxhp / 2:
        if pokemon.species.forme != 'Meteor':
            pokemon.formeChange('Minior-Meteor')
    else:
        if pokemon.species.forme == 'Meteor':
            pokemon.formeChange(pokemon.set.get('species'))


class shieldsdown:
    def onStart(self, pokemon):
        if pokemon.baseSpecies.baseSpecies != 'Minior' or pokemon.transformed:
            return None
        _minior_check(self, pokemon)

    def onResidual(self, pokemon):
        if pokemon.baseSpecies.baseSpecies != 'Minior' or pokemon.transformed or not pokemon.hp:
            return None
        _minior_check(self, pokemon)

    def onSetStatus(self, status, target, source, effect):
        if target.species.id != 'miniormeteor' or target.transformed:
            return None
        if effect and effect.status:
            self.add('-immune', target, '[from] ability: Shields Down')
        return False

    def onTryAddVolatile(self, status, target):
        if target.species.id != 'miniormeteor' or target.transformed:
            return None
        if status.id != 'yawn':
            return None
        self.add('-immune', target, '[from] ability: Shields Down')
        return NULL


class simple:
    def onChangeBoost(self, boost, target, source, effect):
        if effect and effect.id == 'zpower':
            return None
        for i in boost:
            boost[i] *= 2


class skilllink:
    def onModifyMove(self, move):
        if move.multihit and isinstance(move.multihit, list) and len(move.multihit):
            move.multihit = move.multihit[1]
        if move.multiaccuracy:
            move.pop('multiaccuracy', None)


class slushrush:
    def onModifySpe(self, spe, pokemon):
        if self.field.isWeather(['hail', 'snowscape']):
            return self.chainModify(2)


class sniper:
    def onModifyDamage(self, damage, source, target, move):
        if target.getMoveHitData(move).crit:
            self.debug('Sniper boost')
            return self.chainModify(1.5)


class snowcloak:
    def onImmunity(self, type_, pokemon):
        if type_ == 'hail':
            return False

    def onModifyAccuracy(self, accuracy):
        if not is_number(accuracy):
            return None
        if self.field.isWeather(['hail', 'snowscape']):
            self.debug('Snow Cloak - decreasing accuracy')
            return self.chainModify([3277, 4096])


class snowwarning:
    def onStart(self, source):
        self.field.setWeather('snowscape')


class solarpower:
    def onModifySpA(self, spa, pokemon):
        if pokemon.effectiveWeather() in ('sunnyday', 'desolateland'):
            return self.chainModify(1.5)

    def onWeather(self, target, source, effect):
        if target.effectiveWeather() != effect.id:
            return None
        if effect.id in ('sunnyday', 'desolateland'):
            self.damage(target.baseMaxhp / 8, target, target)


class soundproof:
    def onTryHit(self, target, source, move):
        if target is not source and move.flags.get('sound'):
            self.add('-immune', target, '[from] ability: Soundproof')
            return NULL

    def onAllyTryHitSide(self, target, source, move):
        if move.flags.get('sound'):
            self.add('-immune', self.effectState.target, '[from] ability: Soundproof')


class speedboost:
    def onResidual(self, pokemon):
        if pokemon.activeTurns:
            self.boost(Obj(spe=1))


class spicyspray:
    def onDamagingHit(self, damage, target, source, move):
        source.trySetStatus('brn', target)


class stakeout:
    def onModifyAtk(self, atk, attacker, defender):
        if not defender.activeTurns:
            self.debug('Stakeout boost')
            return self.chainModify(2)

    def onModifySpA(self, atk, attacker, defender):
        if not defender.activeTurns:
            self.debug('Stakeout boost')
            return self.chainModify(2)


class stalwart:
    def onModifyMove(self, move):
        move.tracksTarget = move.target != 'scripted'


class propellertail(stalwart):
    pass


class stamina:
    def onDamagingHit(self, damage, target, source, effect):
        self.boost(Obj(**{'def': 1}))


class stancechange:
    def onModifyMove(self, move, attacker, defender):
        if attacker.species.baseSpecies != 'Aegislash' or attacker.transformed:
            return None
        if move.category == 'Status' and move.id != 'kingsshield':
            return None
        target_forme = 'Aegislash' if move.id == 'kingsshield' else 'Aegislash-Blade'
        if attacker.species.name != target_forme:
            attacker.formeChange(target_forme)


class steadfast:
    def onFlinch(self, pokemon):
        self.boost(Obj(spe=1))


class steelyspirit:
    def onAllyBasePower(self, basePower, attacker, defender, move):
        if move.type == 'Steel':
            self.debug('Steely Spirit boost')
            return self.chainModify(1.5)


class stench:
    def onModifyMove(self, move):
        if move.category != 'Status':
            self.debug('Adding Stench flinch')
            if move.get('secondaries') is None:
                move.secondaries = []
            for secondary in move.secondaries:
                if secondary.volatileStatus == 'flinch':
                    return None
            move.secondaries.append(Obj(chance=10, volatileStatus='flinch'))


class stickyhold:
    def onTakeItem(self, item, pokemon, source):
        if not self.activeMove:
            raise ValueError('Battle.activeMove is null')
        if not pokemon.hp or pokemon.item == 'stickybarb':
            return None
        if (source and source is not pokemon) or self.activeMove.id == 'knockoff':
            self.add('-activate', pokemon, 'ability: Sticky Hold')
            return False


class strongjaw:
    def onBasePower(self, basePower, attacker, defender, move):
        if move.flags.get('bite'):
            return self.chainModify(1.5)


class sturdy:
    def onTryHit(self, pokemon, target, move):
        if move.ohko:
            self.add('-immune', pokemon, '[from] ability: Sturdy')
            return NULL

    def onDamage(self, damage, target, source, effect):
        if target.hp == target.maxhp and damage >= target.hp and effect and effect.effectType == 'Move':
            self.add('-ability', target, 'Sturdy')
            return target.hp - 1


class suctioncups:
    def onDragOut(self, pokemon):
        self.add('-activate', pokemon, 'ability: Suction Cups')
        return NULL


class superluck:
    def onModifyCritRatio(self, critRatio):
        return critRatio + 1


class supersweetsyrup:
    def onStart(self, pokemon):
        if pokemon.syrupTriggered:
            return None
        pokemon.syrupTriggered = True
        self.add('-ability', pokemon, 'Supersweet Syrup')
        for target in pokemon.adjacentFoes():
            if target.volatiles.get('substitute'):
                self.add('-immune', target)
            else:
                self.boost(Obj(evasion=-1), target, pokemon, None, True)


class supremeoverlord:
    def onStart(self, pokemon):
        if pokemon.side.totalFainted:
            self.add('-activate', pokemon, 'ability: Supreme Overlord')
            fallen = min(pokemon.side.totalFainted, 5)
            self.add('-start', pokemon, f"fallen{fallen}", '[silent]')
            self.effectState.fallen = fallen

    def onEnd(self, pokemon):
        if pokemon.beingCalledBack or not self.effectState.fallen:
            return None
        self.add('-end', pokemon, f"fallen{self.effectState.fallen}", '[silent]')

    def onBasePower(self, basePower, attacker, defender, move):
        if self.effectState.fallen:
            pow_mod = [4096, 4506, 4915, 5325, 5734, 6144]
            self.debug(f"Supreme Overlord boost: {pow_mod[self.effectState.fallen]}/4096")
            return self.chainModify([pow_mod[self.effectState.fallen], 4096])


class surgesurfer:
    def onModifySpe(self, spe):
        if self.field.isTerrain('electricterrain'):
            return self.chainModify(2)


class sweetveil:
    def onAllySetStatus(self, status, target, source, effect):
        if status.id == 'slp':
            self.debug('Sweet Veil interrupts sleep')
            self.add('-block', target, 'ability: Sweet Veil', f"[of] {self.effectState.target}")
            return NULL

    def onAllyTryAddVolatile(self, status, target):
        if status.id == 'yawn':
            self.debug('Sweet Veil blocking yawn')
            self.add('-block', target, 'ability: Sweet Veil', f"[of] {self.effectState.target}")
            return NULL


class swiftswim:
    def onModifySpe(self, spe, pokemon):
        if pokemon.effectiveWeather() in ('raindance', 'primordialsea'):
            return self.chainModify(2)


class symbiosis:
    def onAllyAfterUseItem(self, item, pokemon):
        if pokemon.switchFlag:
            return None
        source = self.effectState.target
        my_item = source.takeItem()
        if not my_item:
            return None
        if (not self.singleEvent('TakeItem', my_item, source.itemState, pokemon, source, self.effect, my_item) or
                not pokemon.setItem(my_item)):
            source.item = my_item.id
            return None
        self.add('-activate', source, 'ability: Symbiosis', my_item, f"[of] {pokemon}")


class synchronize:
    def onAfterSetStatus(self, status, target, source, effect):
        if not source or source is target:
            return None
        if effect and effect.id == 'toxicspikes':
            return None
        if status.id in ('slp', 'frz'):
            return None
        self.add('-activate', target, 'ability: Synchronize')
        source.trySetStatus(status, target, Obj(status=status.id, id='synchronize'))


class tangledfeet:
    def onModifyAccuracy(self, accuracy, target):
        if not is_number(accuracy):
            return None
        if target and target.volatiles.get('confusion'):
            self.debug('Tangled Feet - decreasing accuracy')
            return self.chainModify(0.5)


class technician:
    def onBasePower(self, basePower, attacker, defender, move):
        bp_after = self.modify(basePower, self.event.modifier)
        self.debug(f"Base Power: {bp_after}")
        if bp_after <= 60:
            self.debug('Technician boost')
            return self.chainModify(1.5)


class telepathy:
    def onTryHit(self, target, source, move):
        if target is not source and target.isAlly(source) and move.category != 'Status':
            self.add('-activate', target, 'ability: Telepathy')
            return NULL


class teraformzero:
    def onAfterTerastallization(self, pokemon):
        if pokemon.baseSpecies.name != 'Terapagos-Stellar':
            return None
        if self.field.weather or self.field.terrain:
            self.add('-ability', pokemon, 'Teraform Zero')
            self.field.clearWeather()
            self.field.clearTerrain()


class terashift:
    def onSwitchIn(self, pokemon):
        if pokemon.baseSpecies.baseSpecies != 'Terapagos':
            return None
        if pokemon.species.forme != 'Terastal':
            self.add('-activate', pokemon, 'ability: Tera Shift')
            pokemon.formeChange('Terapagos-Terastal', self.effect, True)


class thermalexchange:
    def onDamagingHit(self, damage, target, source, move):
        if move.type == 'Fire':
            self.boost(Obj(atk=1))

    def onUpdate(self, pokemon):
        if pokemon.status == 'brn':
            self.add('-activate', pokemon, 'ability: Thermal Exchange')
            pokemon.cureStatus()

    def onSetStatus(self, status, target, source, effect):
        if status.id != 'brn':
            return None
        if effect and effect.status:
            self.add('-immune', target, '[from] ability: Thermal Exchange')
        return False


class thickfat:
    def onSourceModifyAtk(self, atk, attacker, defender, move):
        if move.type in ('Ice', 'Fire'):
            self.debug('Thick Fat weaken')
            return self.chainModify(0.5)

    def onSourceModifySpA(self, atk, attacker, defender, move):
        if move.type in ('Ice', 'Fire'):
            self.debug('Thick Fat weaken')
            return self.chainModify(0.5)


class toughclaws:
    def onBasePower(self, basePower, attacker, defender, move):
        if move.flags.get('contact'):
            return self.chainModify([5325, 4096])


class toxicdebris:
    def onDamagingHit(self, damage, target, source, move):
        side = source.side.foe if source.isAlly(target) else source.side
        toxic_spikes = side.sideConditions.get('toxicspikes')
        if move.category == 'Physical' and (not toxic_spikes or toxic_spikes.layers < 2):
            self.add('-activate', target, 'ability: Toxic Debris')
            side.addSideCondition('toxicspikes', target)


class trace:
    def onStart(self, pokemon):
        self.effectState.seek = True
        if any(f.ability == 'noability' for f in pokemon.adjacentFoes()):
            self.effectState.seek = False
        if pokemon.hasItem('Ability Shield'):
            self.add('-block', pokemon, 'item: Ability Shield')
            self.effectState.seek = False
        if self.effectState.seek:
            self.singleEvent('Update', self.effect, self.effectState, pokemon)

    def onUpdate(self, pokemon):
        if not self.effectState.seek:
            return None
        possible = [t for t in pokemon.adjacentFoes() if not t.getAbility().flags.get('notrace') and
                    t.ability != 'noability']
        if not possible:
            return None
        target = self.sample(possible)
        ability = target.getAbility()
        pokemon.setAbility(ability, target)


class unaware:
    def onAnyModifyBoost(self, boosts, pokemon):
        user = self.effectState.target
        if user is pokemon:
            return None
        if user is self.activePokemon and pokemon is self.activeTarget:
            boosts['def'] = 0
            boosts['spd'] = 0
            boosts['evasion'] = 0
        if pokemon is self.activePokemon and user is self.activeTarget:
            boosts['atk'] = 0
            boosts['def'] = 0
            boosts['spa'] = 0
            boosts['accuracy'] = 0


class unburden:
    def onAfterUseItem(self, item, pokemon):
        if pokemon is not self.effectState.target:
            return None
        pokemon.addVolatile('unburden')

    def onTakeItem(self, item, pokemon):
        pokemon.addVolatile('unburden')

    def onEnd(self, pokemon):
        pokemon.removeVolatile('unburden')

    class condition:
        def onModifySpe(self, spe, pokemon):
            if not pokemon.item and not pokemon.ignoringAbility():
                return self.chainModify(2)


class unnerve:
    def onStart(self, pokemon):
        if self.effectState.unnerved:
            return None
        self.add('-ability', pokemon, 'Unnerve')
        self.effectState.unnerved = True

    def onEnd(self):
        self.effectState.unnerved = False

    def onFoeTryEatItem(self):
        return not self.effectState.unnerved


class wanderingspirit:
    def onDamagingHit(self, damage, target, source, move):
        if self.checkMoveMakesContact(move, source, target):
            self.skillSwap(source, target)


class waterbubble:
    def onSourceModifyAtk(self, atk, attacker, defender, move):
        if move.type == 'Fire':
            return self.chainModify(0.5)

    def onSourceModifySpA(self, atk, attacker, defender, move):
        if move.type == 'Fire':
            return self.chainModify(0.5)

    def onModifyAtk(self, atk, attacker, defender, move):
        if move.type == 'Water':
            return self.chainModify(2)

    def onModifySpA(self, atk, attacker, defender, move):
        if move.type == 'Water':
            return self.chainModify(2)

    def onUpdate(self, pokemon):
        if pokemon.status == 'brn':
            self.add('-activate', pokemon, 'ability: Water Bubble')
            pokemon.cureStatus()

    def onSetStatus(self, status, target, source, effect):
        if status.id != 'brn':
            return None
        if effect and effect.status:
            self.add('-immune', target, '[from] ability: Water Bubble')
        return False


class weakarmor:
    def onDamagingHit(self, damage, target, source, move):
        if move.category == 'Physical':
            self.boost(Obj({'def': -1, 'spe': 2}), target, target)


class zerotohero:
    def onSwitchOut(self, pokemon):
        if pokemon.baseSpecies.baseSpecies != 'Palafin':
            return None
        if pokemon.species.forme != 'Hero':
            pokemon.formeChange('Palafin-Hero', self.effect, True)
            pokemon.heroMessageDisplayed = False

    def onSwitchIn(self, pokemon):
        if pokemon.baseSpecies.baseSpecies != 'Palafin':
            return None
        if not pokemon.heroMessageDisplayed and pokemon.species.forme == 'Hero':
            self.add('-activate', pokemon, 'ability: Zero to Hero')
            pokemon.heroMessageDisplayed = True
