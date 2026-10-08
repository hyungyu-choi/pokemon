"""Item handlers (Showdown ``data/items.ts`` + champions ``items.ts``), ported to Python.

Each class is an item id; ``self`` in handlers is the Battle (Showdown's ``this``).
"""
from __future__ import annotations

import json
import math
import os

from ..js import NULL, Obj, is_number, to_id


def _type_boost(type_):
    def onBasePower(self, basePower, user, target, move):
        if move and move.type == type_:
            return self.chainModify([4915, 4096])
    return onBasePower


def _resist_berry(type_):
    class _B:
        def onSourceModifyDamage(self, damage, source, target, move):
            if move.type == type_ and target.getMoveHitData(move).typeMod > 0:
                hit_sub = target.volatiles.get('substitute') and not move.flags.get('bypasssub') and \
                    not move.infiltrates
                if hit_sub:
                    return None
                if target.eatItem():
                    self.debug('-50% reduction')
                    self.add('-enditem', target, self.effect, '[weaken]')
                    return self.chainModify(0.5)

        def onEat(self):
            pass
    return _B


def _status_berry(statuses, volatile=None):
    class _B:
        def onUpdate(self, pokemon):
            if pokemon.status in statuses or (volatile and pokemon.volatiles.get(volatile)):
                pokemon.eatItem()

        def onEat(self, pokemon):
            if pokemon.status in statuses:
                pokemon.cureStatus()
    return _B


def _terrain_seed(terrain):
    class _S:
        def onStart(self, pokemon):
            if not pokemon.ignoringItem() and self.field.isTerrain(terrain):
                pokemon.useItem()

        def onTerrainChange(self, pokemon):
            if self.field.isTerrain(terrain):
                pokemon.useItem()
    return _S


def _ogerpon_mask(prefix):
    class _M:
        def onBasePower(self, basePower, user, target, move):
            if user.baseSpecies.name.startswith(prefix):
                return self.chainModify([4915, 4096])

        def onTakeItem(self, item, source):
            if source.baseSpecies.baseSpecies == 'Ogerpon':
                return False
            return True
    return _M


class airballoon:
    def onStart(self, target):
        if not target.ignoringItem() and not self.field.getPseudoWeather('gravity'):
            self.add('-item', target, 'Air Balloon')

    def onDamagingHit(self, damage, target, source, move):
        self.add('-enditem', target, 'Air Balloon')
        target.item = ''
        self.clearEffectState(target.itemState)
        self.runEvent('AfterUseItem', target, None, None, self.dex.items.get('airballoon'))

    def onAfterSubDamage(self, damage, target, source, effect):
        self.debug('effect: ' + effect.id)
        if effect.effectType == 'Move':
            self.add('-enditem', target, 'Air Balloon')
            target.item = ''
            self.clearEffectState(target.itemState)
            self.runEvent('AfterUseItem', target, None, None, self.dex.items.get('airballoon'))


class aspearberry(_status_berry(('frz',))):
    pass


class cheriberry(_status_berry(('par',))):
    pass


class chestoberry(_status_berry(('slp',))):
    pass


class pechaberry(_status_berry(('psn', 'tox'))):
    pass


class rawstberry(_status_berry(('brn',))):
    pass


class persimberry:
    def onUpdate(self, pokemon):
        if pokemon.volatiles.get('confusion'):
            pokemon.eatItem()

    def onEat(self, pokemon):
        pokemon.removeVolatile('confusion')


class lumberry:
    def onAfterSetStatus(self, status, pokemon):
        pokemon.eatItem()

    def onUpdate(self, pokemon):
        if pokemon.status or pokemon.volatiles.get('confusion'):
            pokemon.eatItem()

    def onEat(self, pokemon):
        pokemon.cureStatus()
        pokemon.removeVolatile('confusion')


babiriberry = _resist_berry('Steel')
chartiberry = _resist_berry('Rock')
chopleberry = _resist_berry('Fighting')
cobaberry = _resist_berry('Flying')
colburberry = _resist_berry('Dark')
habanberry = _resist_berry('Dragon')
kasibberry = _resist_berry('Ghost')
kebiaberry = _resist_berry('Poison')
occaberry = _resist_berry('Fire')
passhoberry = _resist_berry('Water')
payapaberry = _resist_berry('Psychic')
rindoberry = _resist_berry('Grass')
roseliberry = _resist_berry('Fairy')
shucaberry = _resist_berry('Ground')
tangaberry = _resist_berry('Bug')
wacanberry = _resist_berry('Electric')
yacheberry = _resist_berry('Ice')


class chilanberry:
    def onSourceModifyDamage(self, damage, source, target, move):
        if move.type == 'Normal' and (not target.volatiles.get('substitute') or move.flags.get('bypasssub') or
                                      move.infiltrates):
            if target.eatItem():
                self.debug('-50% reduction')
                self.add('-enditem', target, self.effect, '[weaken]')
                return self.chainModify(0.5)

    def onEat(self):
        pass


class bigroot:
    def onTryHeal(self, damage, target, source, effect):
        if effect.id in ('drain', 'leechseed', 'ingrain', 'aquaring', 'strengthsap'):
            return self.chainModify([5324, 4096])


class blackbelt:
    onBasePower = _type_boost('Fighting')


class blackglasses:
    onBasePower = _type_boost('Dark')


class charcoal:
    onBasePower = _type_boost('Fire')


class dragonfang:
    onBasePower = _type_boost('Dragon')


class fairyfeather:
    onBasePower = _type_boost('Fairy')


class hardstone:
    onBasePower = _type_boost('Rock')


class magnet:
    onBasePower = _type_boost('Electric')


class metalcoat:
    onBasePower = _type_boost('Steel')


class miracleseed:
    onBasePower = _type_boost('Grass')


class mysticwater:
    onBasePower = _type_boost('Water')


class nevermeltice:
    onBasePower = _type_boost('Ice')


class poisonbarb:
    onBasePower = _type_boost('Poison')


class sharpbeak:
    onBasePower = _type_boost('Flying')


class silkscarf:
    onBasePower = _type_boost('Normal')


class silverpowder:
    onBasePower = _type_boost('Bug')


class softsand:
    onBasePower = _type_boost('Ground')


class spelltag:
    onBasePower = _type_boost('Ghost')


class twistedspoon:
    onBasePower = _type_boost('Psychic')


class brightpowder:
    def onModifyAccuracy(self, accuracy):
        if not is_number(accuracy):
            return None
        self.debug('brightpowder - decreasing accuracy')
        return self.chainModify([3686, 4096])


class choicescarf:
    def onStart(self, pokemon):
        if pokemon.volatiles.get('choicelock'):
            self.debug('removing choicelock')
        pokemon.removeVolatile('choicelock')

    def onModifyMove(self, move, pokemon):
        pokemon.addVolatile('choicelock')

    def onModifySpe(self, spe, pokemon):
        if pokemon.volatiles.get('dynamax'):
            return None
        return self.chainModify(1.5)


class choiceband(choicescarf):
    def onModifySpe(self, spe, pokemon):
        return None

    def onModifyAtk(self, atk, pokemon):
        if pokemon.volatiles.get('dynamax'):
            return None
        return self.chainModify(1.5)


class choicespecs(choicescarf):
    def onModifySpe(self, spe, pokemon):
        return None

    def onModifySpA(self, spa, pokemon):
        if pokemon.volatiles.get('dynamax'):
            return None
        return self.chainModify(1.5)


cornerstonemask = _ogerpon_mask('Ogerpon-Cornerstone')
hearthflamemask = _ogerpon_mask('Ogerpon-Hearthflame')
wellspringmask = _ogerpon_mask('Ogerpon-Wellspring')


class ejectbutton:
    def onAfterMoveSecondary(self, target, source, move):
        if source and source is not target and target.hp and move and move.category != 'Status' and \
                not move.flags.get('futuremove'):
            if not self.canSwitch(target.side) or target.forceSwitchFlag or target.beingCalledBack or \
                    target.isSkyDropped():
                return None
            if target.volatiles.get('commanding') or target.volatiles.get('commanded'):
                return None
            for pokemon in self.getAllActive():
                if pokemon.switchFlag is True:
                    return None
            target.switchFlag = True
            if not target.useItem():
                target.switchFlag = False


electricseed = _terrain_seed('electricterrain')
grassyseed = _terrain_seed('grassyterrain')
mistyseed = _terrain_seed('mistyterrain')
psychicseed = _terrain_seed('psychicterrain')


class expertbelt:
    def onModifyDamage(self, damage, source, target, move):
        if move and target.getMoveHitData(move).typeMod > 0:
            return self.chainModify([4915, 4096])


class focusband:
    def onDamage(self, damage, target, source, effect):
        if self.randomChance(1, 10) and damage >= target.hp and effect and effect.effectType == 'Move':
            self.add('-activate', target, 'item: Focus Band')
            return target.hp - 1


class focussash:
    def onDamage(self, damage, target, source, effect):
        if target.hp == target.maxhp and damage >= target.hp and effect and effect.effectType == 'Move':
            if target.useItem():
                return target.hp - 1


class ironball:
    def onEffectiveness(self, typeMod, target, type_, move):
        if not target:
            return None
        if target.volatiles.get('ingrain') or target.volatiles.get('smackdown') or \
                self.field.getPseudoWeather('gravity'):
            return None
        if move.type == 'Ground' and target.hasType('Flying'):
            return 0

    def onModifySpe(self, spe):
        return self.chainModify(0.5)


class kingsrock:
    def onModifyMove(self, move):
        if move.category != 'Status':
            if move.get('secondaries') is None:
                move.secondaries = []
            for secondary in move.secondaries:
                if secondary.volatileStatus == 'flinch':
                    return None
            move.secondaries.append(Obj(chance=10, volatileStatus='flinch'))


class leek:
    def onModifyCritRatio(self, critRatio, user):
        if to_id(user.baseSpecies.baseSpecies) in ('farfetchd', 'sirfetchd'):
            return critRatio + 2


class leftovers:
    def onResidual(self, pokemon):
        self.heal(pokemon.baseMaxhp / 16)


class leppaberry:
    def onUpdate(self, pokemon):
        if not pokemon.hp:
            return None
        if any(ms.pp == 0 for ms in pokemon.moveSlots):
            pokemon.eatItem()

    def onEat(self, pokemon):
        move_slot = next((ms for ms in pokemon.moveSlots if ms.pp == 0), None) or \
            next((ms for ms in pokemon.moveSlots if ms.pp < ms.maxpp), None)
        if not move_slot:
            return None
        added = 20 if pokemon.hasAbility('ripen') else 10
        move_slot.pp = min(move_slot.pp + added, move_slot.maxpp)
        self.add('-activate', pokemon, 'item: Leppa Berry', move_slot.move, '[consumed]')


class lifeorb:
    def onModifyDamage(self, damage, source, target, move):
        return self.chainModify([5324, 4096])

    def onAfterMoveSecondarySelf(self, source, target, move):
        if source and source is not target and move and move.category != 'Status' and not source.forceSwitchFlag:
            self.damage(source.baseMaxhp / 10, source, source, self.dex.items.get('lifeorb'))


class lightball:
    def onModifyAtk(self, atk, pokemon):
        if pokemon.baseSpecies.baseSpecies == 'Pikachu':
            return self.chainModify(2)

    def onModifySpA(self, spa, pokemon):
        if pokemon.baseSpecies.baseSpecies == 'Pikachu':
            return self.chainModify(2)


_MENTAL_HERB_CONDITIONS = ('attract', 'taunt', 'encore', 'torment', 'disable', 'healblock')


class mentalherb:
    class fling:
        def effect(self, pokemon):
            for first in _MENTAL_HERB_CONDITIONS:
                if pokemon.volatiles.get(first):
                    for second in _MENTAL_HERB_CONDITIONS:
                        pokemon.removeVolatile(second)
                        if first == 'attract' and second == 'attract':
                            self.add('-end', pokemon, 'move: Attract', '[from] item: Mental Herb')
                    return None

    def onUpdate(self, pokemon):
        for first in _MENTAL_HERB_CONDITIONS:
            if pokemon.volatiles.get(first):
                if not pokemon.useItem():
                    return None
                for second in _MENTAL_HERB_CONDITIONS:
                    pokemon.removeVolatile(second)
                    if first == 'attract' and second == 'attract':
                        self.add('-end', pokemon, 'move: Attract', '[from] item: Mental Herb')
                return None


class metronome:
    def onStart(self, pokemon):
        pokemon.addVolatile('metronome')

    class condition:
        def onStart(self, pokemon):
            self.effectState.lastMove = ''
            self.effectState.numConsecutive = 0

        def onTryMove(self, pokemon, target, move):
            if not pokemon.hasItem('metronome'):
                pokemon.removeVolatile('metronome')
                return None
            if move.callsMove:
                return None
            if self.effectState.lastMove == move.id and pokemon.moveLastTurnResult:
                self.effectState.numConsecutive += 1
            elif pokemon.volatiles.get('twoturnmove'):
                if self.effectState.lastMove != move.id:
                    self.effectState.numConsecutive = 1
                else:
                    self.effectState.numConsecutive += 1
            else:
                self.effectState.numConsecutive = 0
            self.effectState.lastMove = move.id

        def onModifyDamage(self, damage, source, target, move):
            dmg_mod = [4096, 4915, 5734, 6553, 7372, 8192]
            n = min(self.effectState.numConsecutive, 5)
            self.debug(f"Current Metronome boost: {dmg_mod[n]}/4096")
            return self.chainModify([dmg_mod[n], 4096])


class muscleband:
    def onBasePower(self, basePower, user, target, move):
        if move.category == 'Physical':
            return self.chainModify([4505, 4096])


class wiseglasses:
    def onBasePower(self, basePower, user, target, move):
        if move.category == 'Special':
            return self.chainModify([4505, 4096])


class normalgem:
    def onSourceTryPrimaryHit(self, target, source, move):
        if target is source or move.category == 'Status' or move.flags.get('pledgecombo'):
            return None
        if move.type == 'Normal' and source.useItem():
            source.addVolatile('gem')


class oranberry:
    def onUpdate(self, pokemon):
        if pokemon.hp <= pokemon.maxhp / 2:
            pokemon.eatItem()

    def onTryEatItem(self, item, pokemon):
        if not self.runEvent('TryHeal', pokemon, None, self.effect, 10):
            return False

    def onEat(self, pokemon):
        self.heal(10)


class sitrusberry:
    def onUpdate(self, pokemon):
        if pokemon.hp <= pokemon.maxhp / 2:
            pokemon.eatItem()

    def onTryEatItem(self, item, pokemon):
        if not self.runEvent('TryHeal', pokemon, None, self.effect, pokemon.baseMaxhp / 4):
            return False

    def onEat(self, pokemon):
        self.heal(pokemon.baseMaxhp / 4)


class quickclaw:
    def onFractionalPriority(self, priority, pokemon, target, move):
        if move.category == 'Status' and pokemon.hasAbility('myceliummight'):
            return None
        if priority <= 0 and self.randomChance(1, 5):
            self.add('-activate', pokemon, 'item: Quick Claw')
            return 0.1


class redcard:
    def onAfterMoveSecondary(self, target, source, move):
        if source and source is not target and source.hp and target.hp and move and move.category != 'Status':
            if not source.isActive or not self.canSwitch(source.side) or source.forceSwitchFlag or \
                    target.forceSwitchFlag:
                return None
            if target.useItem(source):
                if self.runEvent('DragOut', source, target, move):
                    source.forceSwitchFlag = True


class rockyhelmet:
    def onDamagingHit(self, damage, target, source, move):
        if self.checkMoveMakesContact(move, source, target):
            self.damage(source.baseMaxhp / 6, source, target)


class scopelens:
    def onModifyCritRatio(self, critRatio):
        return critRatio + 1


class shedshell:
    def onTrapPokemon(self, pokemon):
        pokemon.trapped = False

    def onMaybeTrapPokemon(self, pokemon):
        pokemon.maybeTrapped = False


class shellbell:
    def onAfterMoveSecondarySelf(self, pokemon, target, move):
        if move.totalDamage and not pokemon.forceSwitchFlag:
            self.heal(move.totalDamage / 8, pokemon)


class whiteherb:
    class fling:
        def effect(self, pokemon):
            activate = False
            boosts = Obj()
            for i in pokemon.boosts:
                if pokemon.boosts[i] < 0:
                    activate = True
                    boosts[i] = 0
            if activate:
                pokemon.setBoost(boosts)
                self.add('-clearnegativeboost', pokemon, '[silent]')

    def onStart(self, pokemon):
        self.effectState.boosts = Obj()
        ready = False
        for i in pokemon.boosts:
            if pokemon.boosts[i] < 0:
                ready = True
                self.effectState.boosts[i] = 0
        if ready:
            self.effectState.target.useItem()
        self.effectState.pop('boosts', None)

    def onAnySwitchIn(self):
        self.effect.onStart(self, self.effectState.target)

    def onAnyAfterMega(self):
        self.effect.onStart(self, self.effectState.target)

    def onAnyAfterMove(self):
        self.effect.onStart(self, self.effectState.target)

    def onResidual(self, pokemon):
        self.effect.onStart(self, pokemon)

    def onUse(self, pokemon):
        pokemon.setBoost(self.effectState.boosts)
        self.add('-clearnegativeboost', pokemon, '[silent]')


class widelens:
    def onSourceModifyAccuracy(self, accuracy):
        if is_number(accuracy):
            return self.chainModify([4505, 4096])


class zoomlens:
    def onSourceModifyAccuracy(self, accuracy, target):
        if is_number(accuracy) and not self.queue.willMove(target):
            self.debug('Zoom Lens boosting accuracy')
            return self.chainModify([4915, 4096])


# --- Mega Stones: cannot be removed from a Pokemon that can Mega Evolve with them -------------------

def _mega_stone_take_item(self, item, source):
    mega = item.megaStone
    return not (mega and mega.get(source.baseSpecies.baseSpecies))


def _register_mega_stones():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'data', 'items.json')
    try:
        with open(path, encoding='utf-8') as f:
            items = json.load(f)
    except OSError:
        return
    g = globals()
    for item_id, data in items.items():
        if data.get('megaStone') and item_id not in g:
            g[item_id] = type(item_id, (), {'onTakeItem': _mega_stone_take_item})


_register_mega_stones()
