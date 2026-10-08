"""Port of Showdown's ``sim/pokemon.ts`` (with the champions mod overrides)."""
from __future__ import annotations

import math

from .js import NULL, Obj, clamp_int_range, is_number, to_id, trunc

RESTORATIVE_BERRIES = {
    'leppaberry', 'aguavberry', 'enigmaberry', 'figyberry', 'iapapaberry', 'magoberry', 'sitrusberry',
    'wikiberry', 'oranberry',
}

_BOOST_TABLE = (1, 1.5, 2, 2.5, 3, 3.5, 4)
_STAT_TABLE = {'atk': 'Atk', 'def': 'Def', 'spa': 'SpA', 'spd': 'SpD', 'spe': 'Spe'}


def _new_boosts():
    return Obj(atk=0, **{'def': 0}, spa=0, spd=0, spe=0, accuracy=0, evasion=0)


class Pokemon:
    isActive = False  # may be read (as JS `undefined`) before __init__ assigns it

    def __init__(self, set_, side):
        self.side = side
        self.battle = battle = side.battle
        self.m = Obj()
        if isinstance(set_, str):
            set_ = Obj(name=set_)
        self.baseSpecies = battle.dex.species.get(set_.get('species') or set_.get('name'))
        if not self.baseSpecies.exists:
            raise ValueError(f"Unidentified species: {self.baseSpecies.name}")
        self.set = set_
        self.species = self.baseSpecies
        if set_.get('name') == set_.get('species') or not set_.get('name'):
            set_['name'] = self.baseSpecies.baseSpecies
        self.speciesState = battle.initEffectState(Obj(id=self.species.id))

        self.name = set_['name'][:20]
        self.fullname = f"{side.id}: {self.name}"

        set_['level'] = clamp_int_range(set_.get('adjustLevel') or set_.get('level') or 100, 1, 9999)
        self.level = set_['level']
        genders = {'M': 'M', 'F': 'F', 'N': 'N'}
        self.gender = genders.get(set_.get('gender')) or self.species.gender or battle.sample(['M', 'F'])
        if self.gender == 'N':
            self.gender = ''
        happiness = set_.get('happiness')
        self.happiness = clamp_int_range(happiness, 0, 255) if is_number(happiness) else 255
        self.pokeball = to_id(set_.get('pokeball')) or 'pokeball'
        self.dynamaxLevel = 10
        self.gigantamax = set_.get('gigantamax') or False

        self.baseMoveSlots: list[Obj] = []
        self.moveSlots: list[Obj] = []
        self.ppUps = []
        if not set_.get('moves'):
            raise ValueError(f"Set {self.name} has no moves")
        for moveid in set_['moves']:
            move = battle.dex.moves.get(moveid)
            if not move.id:
                continue
            pp_ups = 0 if move.noPPBoosts or move.id == 'trumpcard' else 3
            base_pp = battle.calculatePP(move, pp_ups)
            self.baseMoveSlots.append(Obj(move=move.name, id=move.id, pp=base_pp, maxpp=base_pp,
                                          target=move.target, disabled=False, disabledSource='', used=False))
            self.ppUps.append(pp_ups)

        self.position = 0
        self.details = self.getUpdatedDetails()

        self.status = ''
        self.statusState = battle.initEffectState(Obj())
        self.volatiles: dict[str, Obj] = {}
        self.showCure = None

        if not set_.get('evs'):
            set_['evs'] = Obj(hp=0, atk=0, **{'def': 0}, spa=0, spd=0, spe=0)
        if not set_.get('ivs'):
            set_['ivs'] = Obj(hp=31, atk=31, **{'def': 31}, spa=31, spd=31, spe=31)
        evs = set_['evs']
        ivs = set_['ivs']
        for stat in ('hp', 'atk', 'def', 'spe', 'spa', 'spd'):
            if not evs.get(stat):
                evs[stat] = 0
            if not ivs.get(stat) and ivs.get(stat) != 0:
                ivs[stat] = 31
        for stat in list(evs.keys()):
            evs[stat] = clamp_int_range(evs[stat], 0, 255)
        for stat in list(ivs.keys()):
            ivs[stat] = clamp_int_range(ivs[stat], 0, 31)

        hp_data = battle.dex.getHiddenPower(ivs)
        self.hpType = set_.get('hpType') or hp_data.type
        self.hpPower = hp_data.power
        self.baseHpType = self.hpType
        self.baseHpPower = self.hpPower

        self.baseStoredStats = None
        self.storedStats = Obj(atk=0, **{'def': 0}, spa=0, spd=0, spe=0)
        self.boosts = _new_boosts()

        self.baseAbility = to_id(set_.get('ability'))
        self.ability = self.baseAbility
        self.abilityState = battle.initEffectState(Obj(id=self.ability, target=self))

        self.item = to_id(set_.get('item'))
        self.itemState = battle.initEffectState(Obj(id=self.item, target=self))
        self.lastItem = ''
        self.usedItemThisTurn = False
        self.ateBerry = False
        self.itemKnockedOff = False

        self.trapped = False
        self.maybeTrapped = False
        self.maybeDisabled = False
        self.maybeLocked = False

        self.illusion = None
        self.transformed = False

        self.fainted = False
        self.faintQueued = False
        self.subFainted = None

        self.formeRegression = False

        self.types = self.baseSpecies.types
        self.baseTypes = self.types
        self.addedType = ''
        self.knownType = True
        self.apparentType = '/'.join(self.baseSpecies.types)
        self.teraType = set_.get('teraType') or self.types[0]

        self.switchFlag = False
        self.forceSwitchFlag = False
        self.skipBeforeSwitchOutEventFlag = False
        self.draggedIn = None
        self.newlySwitched = False
        self.beingCalledBack = False

        self.lastMove = None
        self.lastMoveEncore = None
        self.lastMoveUsed = None
        self.lastMoveTargetLoc = None
        self.moveThisTurn = ''
        self.statsRaisedThisTurn = False
        self.statsLoweredThisTurn = False
        self.moveLastTurnResult = None
        self.moveThisTurnResult = None
        self.hurtThisTurn = None
        self.lastDamage = 0
        self.attackedBy: list[Obj] = []
        self.timesAttacked = 0

        self.isActive = False
        self.activeTurns = 0
        self.activeMoveActions = 0
        self.previouslySwitchedIn = 0
        self.truantTurn = False
        self.bondTriggered = False
        self.heroMessageDisplayed = False
        self.swordBoost = False
        self.shieldBoost = False
        self.syrupTriggered = False
        self.stellarBoostedTypes = []
        self.isStarted = False
        self.duringMove = False

        self.weighthg = 1
        self.speed = 0

        self.staleness = None
        self.pendingStaleness = None
        self.volatileStaleness = None
        self.terastallized = None

        self.canMegaEvo = battle.actions.canMegaEvo(self)
        self.canMegaEvoX = None
        self.canMegaEvoY = None
        self.canUltraBurst = None
        self.canGigantamax = None
        self.canTerastallize = None

        self.maxhp = 0
        self.baseMaxhp = 0
        self.hp = 0
        self.clearVolatile()
        self.hp = self.maxhp

    def get(self, name, default=None):
        """Dict-like access (used where Showdown sorts Pokemon with the action comparator)."""
        return getattr(self, name, default)

    # ------------------------------------------------------------------
    @property
    def moves(self):
        return [ms.id for ms in self.moveSlots]

    @property
    def baseMoves(self):
        return [ms.id for ms in self.baseMoveSlots]

    def getSlot(self):
        offset = (self.side.n // 2) * len(self.side.active)
        letter = 'abcdef'[self.position + offset]
        return self.side.id + letter

    def getFieldPositionValue(self):
        return self.side.n + len(self.battle.sides) * self.position

    def __str__(self):
        fullname = self.illusion.fullname if self.illusion else self.fullname
        return self.getSlot() + fullname[2:] if self.isActive else fullname

    def __repr__(self):
        return f"<Pokemon {self}>"

    def getUpdatedDetails(self, level=None):
        name = self.species.name
        if name in ('Greninja-Bond', 'Rockruff-Dusk'):
            name = self.species.baseSpecies
        if not level:
            level = self.level
        return (name + ('' if level == 100 else f", L{level}") +
                ('' if self.gender == '' else f", {self.gender}") + (', shiny' if self.set.get('shiny') else ''))

    def getFullDetails(self):
        health = self.getHealth()
        details = self.details
        if self.illusion:
            details = self.illusion.getUpdatedDetails(self.level)
        if self.terastallized:
            details += f", tera:{self.terastallized}"
        return {'side': health['side'], 'secret': f"{details}|{health['secret']}",
                'shared': f"{details}|{health['shared']}"}

    def updateSpeed(self):
        self.speed = self.getActionSpeed()

    def calculateStat(self, statName, boost, modifier=None, statUser=None):
        statName = to_id(statName)
        if statName == 'hp':
            raise ValueError('Please read `maxhp` directly')
        stat = self.storedStats[statName]
        if 'wonderroom' in self.battle.field.pseudoWeather:
            if statName == 'def':
                stat = self.storedStats['spd']
            elif statName == 'spd':
                stat = self.storedStats['def']
        boosts = Obj({statName: boost})
        boosts = self.battle.runEvent('ModifyBoost', statUser or self, None, None, boosts)
        boost = boosts[statName]
        if boost > 6:
            boost = 6
        if boost < -6:
            boost = -6
        if boost >= 0:
            stat = math.floor(stat * _BOOST_TABLE[boost])
        else:
            stat = math.floor(stat / _BOOST_TABLE[-boost])
        return self.battle.modify(stat, modifier or 1)

    def getStat(self, statName, unboosted=False, unmodified=False):
        statName = to_id(statName)
        if statName == 'hp':
            raise ValueError('Please read `maxhp` directly')
        stat = self.storedStats[statName]
        if unmodified and 'wonderroom' in self.battle.field.pseudoWeather:
            if statName == 'def':
                statName = 'spd'
            elif statName == 'spd':
                statName = 'def'
        if not unboosted:
            boosts = self.boosts
            if not unmodified:
                boosts = self.battle.runEvent('ModifyBoost', self, None, None, Obj(boosts))
            boost = boosts[statName]
            if boost > 6:
                boost = 6
            if boost < -6:
                boost = -6
            if boost >= 0:
                stat = math.floor(stat * _BOOST_TABLE[boost])
            else:
                stat = math.floor(stat / _BOOST_TABLE[-boost])
        if not unmodified:
            stat = self.battle.runEvent('Modify' + _STAT_TABLE[statName], self, None, None, stat)
        if statName == 'spe' and stat > 10000:
            stat = 10000
        return stat

    def getActionSpeed(self):
        # champions: remove Trick Room underflow
        speed = self.getStat('spe', False, False)
        if self.battle.field.getPseudoWeather('trickroom'):
            speed = -speed
        return speed

    def getBestStat(self, unboosted=False, unmodified=False):
        stat_name = 'atk'
        best = 0
        for i in ('atk', 'def', 'spa', 'spd', 'spe'):
            if self.getStat(i, unboosted, unmodified) > best:
                stat_name = i
                best = self.getStat(i, unboosted, unmodified)
        return stat_name

    def getWeight(self):
        weighthg = self.battle.runEvent('ModifyWeight', self, None, None, self.weighthg)
        return max(1, weighthg)

    def getMoveData(self, move):
        move = self.battle.dex.moves.get(move)
        for ms in self.moveSlots:
            if ms.id == move.id:
                return ms
        return None

    def getMoveHitData(self, move):
        if not move.moveHitData:
            move.moveHitData = Obj()
        slot = self.getSlot()
        data = move.moveHitData.get(slot)
        if data is None:
            data = move.moveHitData[slot] = Obj(crit=False, typeMod=0, bypassProtect=False)
        return data

    def alliesAndSelf(self):
        return self.side.allies()

    def allies(self):
        return [a for a in self.side.allies() if a is not self]

    def adjacentAllies(self):
        return [a for a in self.side.allies() if self.isAdjacent(a)]

    def foes(self, all_=False):
        return self.side.foes(all_)

    def adjacentFoes(self):
        if self.battle.activePerHalf <= 2:
            return self.side.foes()
        return [f for f in self.side.foes() if self.isAdjacent(f)]

    def isAlly(self, pokemon):
        return bool(pokemon) and (self.side is pokemon.side or self.side.allySide is pokemon.side)

    def isAdjacent(self, pokemon2):
        if self.fainted or pokemon2.fainted:
            return False
        if self.battle.activePerHalf <= 2:
            return self is not pokemon2
        if self.side is pokemon2.side:
            return abs(self.position - pokemon2.position) == 1
        return abs(self.position + pokemon2.position + 1 - len(self.side.active)) <= 1

    def getUndynamaxedHP(self, amount=None):
        return amount or self.hp

    def getSmartTargets(self, target, move):
        allies = target.adjacentAllies()
        target2 = allies[0] if allies else None
        if not target2 or target2 is self or not target2.hp:
            move.smartTarget = False
            return [target]
        if not target.hp:
            move.smartTarget = False
            return [target2]
        return [target, target2]

    def getAtLoc(self, targetLoc):
        if not targetLoc:
            return None  # JS: side.active[-1] / side.active[NaN] -> undefined
        side = self.battle.sides[self.side.n % 2 if targetLoc < 0 else (self.side.n + 1) % 2]
        targetLoc = abs(targetLoc)
        if targetLoc > len(side.active):
            targetLoc -= len(side.active)
            if side.n + 2 >= len(self.battle.sides):
                return None
            side = self.battle.sides[side.n + 2]
        idx = targetLoc - 1
        return side.active[idx] if 0 <= idx < len(side.active) else None

    def getLocOf(self, target):
        offset = (target.side.n // 2) * len(target.side.active)
        position = target.position + offset + 1
        same_half = (self.side.n % 2) == (target.side.n % 2)
        return -position if same_half else position

    def getMoveTargets(self, move, target):
        battle = self.battle
        targets = []
        mt = move.target
        if mt in ('all', 'foeSide', 'allySide', 'allyTeam'):
            if not mt.startswith('foe'):
                targets += self.alliesAndSelf()
            if not mt.startswith('ally'):
                targets += self.foes(True)
            if targets and target not in targets:
                battle.retargetLastMove(targets[-1])
        elif mt in ('allAdjacent', 'allAdjacentFoes'):
            if mt == 'allAdjacent':
                targets += self.adjacentAllies()
            targets += self.adjacentFoes()
            if targets and target not in targets:
                battle.retargetLastMove(targets[-1])
        elif mt == 'allies':
            targets = self.alliesAndSelf()
        else:
            selected_target = target
            if not target or (target.fainted and not target.isAlly(self)):
                possible = battle.getRandomTarget(self, move)
                if not possible:
                    return [], []
                target = possible
            if battle.activePerHalf > 1 and not move.tracksTarget:
                target = battle.priorityEvent('RedirectTarget', self, self, move, target)
            if move.smartTarget:
                targets = self.getSmartTargets(target, move)
                target = targets[0]
            else:
                targets.append(target)
            if target.fainted and not move.flags.get('futuremove'):
                return [], []
            if selected_target is not target:
                battle.retargetLastMove(target)
        pressure_targets = targets
        if mt == 'foeSide':
            pressure_targets = []
        if move.flags.get('mustpressure'):
            pressure_targets = self.foes()
        return targets, pressure_targets

    def ignoringAbility(self):
        if not self.isActive:
            return True
        ability = self.getAbility()
        if ability.flags.get('notransform') and self.transformed:
            return True
        if ability.flags.get('cantsuppress'):
            return False
        if self.volatiles.get('gastroacid'):
            return True
        if self.hasItem('Ability Shield') or self.ability == 'neutralizinggas':
            return False
        for pokemon in self.battle.getAllActive():
            if (pokemon.ability == 'neutralizinggas' and not pokemon.volatiles.get('gastroacid') and
                    not pokemon.transformed and not pokemon.abilityState.ending and
                    not self.volatiles.get('commanding')):
                return True
        return False

    def ignoringItem(self, isFling=False):
        item = self.getItem()
        if item.isPrimalOrb:
            return False
        if not self.isActive:
            return True
        if self.volatiles.get('embargo') or self.battle.field.pseudoWeather.get('magicroom'):
            return True
        if isFling:
            return self.hasAbility('klutz')
        return not item.ignoreKlutz and self.hasAbility('klutz')

    def deductPP(self, move, amount=None, target=None):
        move = self.battle.dex.moves.get(move)
        pp_data = self.getMoveData(move)
        if not pp_data:
            return 0
        pp_data.used = True
        if not pp_data.pp:
            return 0
        if not amount:
            amount = 1
        pp_data.pp -= amount
        if pp_data.pp < 0:
            amount += pp_data.pp
            pp_data.pp = 0
        return amount

    def moveUsed(self, move, targetLoc=None):
        self.lastMove = move
        self.lastMoveTargetLoc = targetLoc
        self.moveThisTurn = move.id

    def gotAttacked(self, move, damage, source):
        damage_number = damage if is_number(damage) else 0
        move = self.battle.dex.moves.get(move)
        self.attackedBy.append(Obj(source=source, damage=damage_number, move=move.id, thisTurn=True,
                                   slot=source.getSlot(), damageValue=damage))

    def getLastAttackedBy(self):
        if not self.attackedBy:
            return None
        return self.attackedBy[-1]

    def getLastDamagedBy(self, filterOutSameSide=None):
        damaged_by = [a for a in self.attackedBy if is_number(a.damageValue) and
                      (filterOutSameSide is None or not self.isAlly(a.source))]
        if not damaged_by:
            return None
        return damaged_by[-1]

    def getLockedMove(self):
        locked = self.battle.priorityEvent('LockMove', self)
        return None if locked is True else locked

    def getSemiLockedMove(self, restrictData=False):
        if restrictData and self.maybeLocked:
            return None
        locked = self.battle.priorityEvent('SemiLockMove', self)
        return None if locked is True else locked

    def getMoves(self, lockedMove=None, restrictData=False):
        if lockedMove:
            lockedMove = to_id(lockedMove)
            if lockedMove == 'recharge':
                return [{'move': 'Recharge', 'id': 'recharge'}]
            for ms in self.moveSlots:
                if ms.id != lockedMove:
                    continue
                return [{'move': ms.move, 'id': ms.id}]
            return [{'move': self.battle.dex.moves.get(lockedMove).name, 'id': lockedMove}]
        moves = []
        has_valid_move = False
        for ms in self.moveSlots:
            move_name = ms.move
            target = ms.target
            if ms.id == 'curse':
                if not self.hasType('Ghost'):
                    target = 'self'
            elif ms.id == 'pollenpuff':
                if self.volatiles.get('healblock'):
                    target = 'adjacentFoe'
            disabled = ms.disabled
            if ms.pp <= 0:
                disabled = True
            if disabled == 'hidden':
                disabled = not restrictData
            if not disabled:
                has_valid_move = True
            moves.append({'move': move_name, 'id': ms.id, 'pp': ms.pp, 'maxpp': ms.maxpp, 'target': target,
                          'disabled': disabled})
        return moves if has_valid_move else []

    def getMoveRequestData(self):
        locked_move = self.getLockedMove()
        hard_locked = bool(locked_move)
        if locked_move:
            self.trapped = True
        else:
            locked_move = None if self.maybeLocked else self.getSemiLockedMove(True)
        is_last_active = self.isLastActive()
        can_switch_in = self.battle.canSwitch(self.side) > 0
        moves = self.getMoves(locked_move, is_last_active)
        if not moves:
            moves = [{'move': 'Struggle', 'id': 'struggle', 'target': 'randomNormal', 'disabled': False}]
            locked_move = 'struggle'
        data = {'moves': moves}
        if hard_locked or not is_last_active:
            self.maybeDisabled = False
            self.maybeLocked = False
            self.maybeTrapped = False
            if hard_locked or can_switch_in:
                if self.trapped:
                    data['trapped'] = True
        else:
            self.maybeLocked = self.maybeLocked or self.maybeDisabled
            if self.maybeDisabled:
                data['maybeDisabled'] = self.maybeDisabled
            if self.maybeLocked:
                data['maybeLocked'] = self.maybeLocked
            if can_switch_in:
                if self.trapped is True:
                    data['trapped'] = True
                elif self.maybeTrapped:
                    data['maybeTrapped'] = True
        if not locked_move:
            if self.canMegaEvo:
                data['canMegaEvo'] = True
        return data

    def getSwitchRequestData(self, forAlly=False):
        entry = {
            'ident': self.fullname,
            'details': self.details,
            'condition': self.getHealth()['secret'],
            'active': self.position < len(self.side.active),
            'stats': {s: self.baseStoredStats[s] for s in ('atk', 'def', 'spa', 'spd', 'spe')},
            'moves': list(self.baseMoves if forAlly else self.moves),
            'baseAbility': self.baseAbility,
            'item': self.item,
            'pokeball': self.pokeball,
            'ability': self.ability,
            'commanding': bool(self.volatiles.get('commanding')) and not self.fainted,
            'reviving': self.isActive and bool(self.side.slotConditions[self.position].get('revivalblessing')),
        }
        return entry

    def isLastActive(self):
        if not self.isActive:
            return False
        ally_active = self.side.active
        for i in range(self.position + 1, len(ally_active)):
            if ally_active[i] and not ally_active[i].fainted:
                return False
        return True

    def positiveBoosts(self):
        return sum(v for v in self.boosts.values() if v > 0)

    def getCappedBoost(self, boosts):
        capped = Obj()
        for name, boost in boosts.items():
            if not boost:
                continue
            capped[name] = clamp_int_range(self.boosts[name] + boost, -6, 6) - self.boosts[name]
        return capped

    def boostBy(self, boosts):
        boosts = self.getCappedBoost(boosts)
        delta = 0
        for name in boosts:
            delta = boosts[name]
            self.boosts[name] += delta
        return delta

    def clearBoosts(self):
        for name in self.boosts:
            self.boosts[name] = 0

    def setBoost(self, boosts):
        for name in boosts:
            self.boosts[name] = boosts[name]

    def copyVolatileFrom(self, pokemon, switchCause=None):
        battle = self.battle
        self.clearVolatile()
        if switchCause != 'shedtail':
            self.boosts = pokemon.boosts
        for i, state in list(pokemon.volatiles.items()):
            if switchCause == 'shedtail' and i != 'substitute':
                continue
            if battle.dex.conditions.getByID(i).noCopy:
                continue
            new_state = Obj(state)
            new_state['target'] = self
            self.volatiles[i] = battle.initEffectState(new_state)
            if self.volatiles[i].linkedPokemon:
                pokemon.volatiles[i].pop('linkedPokemon', None)
                pokemon.volatiles[i].pop('linkedStatus', None)
                for linked in self.volatiles[i].linkedPokemon:
                    links = linked.volatiles[str(self.volatiles[i].linkedStatus)].linkedPokemon
                    links[links.index(pokemon)] = self
        pokemon.clearVolatile()
        for i in list(self.volatiles.keys()):
            volatile = self.getVolatile(i)
            battle.singleEvent('Copy', volatile, self.volatiles[i], self)

    def transformInto(self, pokemon, effect=None):
        battle = self.battle
        species = pokemon.species
        if (pokemon.fainted or self.illusion or pokemon.illusion or pokemon.volatiles.get('substitute') or
                pokemon.transformed or self.transformed or species.name == 'Eternatus-Eternamax' or
                (species.baseSpecies in ('Ogerpon', 'Terapagos') and (self.terastallized or pokemon.terastallized)) or
                self.terastallized == 'Stellar'):
            return False
        if not self.setSpecies(species, effect, True):
            return False
        self.transformed = True
        self.weighthg = pokemon.weighthg
        types = pokemon.getTypes(True, True)
        self.setType(pokemon.volatiles['roost'].typeWas if pokemon.volatiles.get('roost') else types, True)
        self.addedType = pokemon.addedType
        self.knownType = self.isAlly(pokemon) and pokemon.knownType
        self.apparentType = pokemon.apparentType
        for stat_name in list(self.storedStats.keys()):
            self.storedStats[stat_name] = pokemon.storedStats[stat_name]
        self.moveSlots = []
        self.timesAttacked = pokemon.timesAttacked
        for i, ms in enumerate(pokemon.moveSlots):
            move_name = ms.move
            move = battle.dex.moves.get(ms.id)
            pp = min(5, move.pp)
            self.moveSlots.append(Obj(move=move_name, id=ms.id, pp=pp, maxpp=pp, target=ms.target,
                                      disabled=False, used=False, virtual=True))
        for name in pokemon.boosts:
            self.boosts[name] = pokemon.boosts[name]
        volatiles_to_copy = ['dragoncheer', 'focusenergy', 'gmaxchistrike', 'laserfocus']
        for v in volatiles_to_copy:
            self.removeVolatile(v)
        for v in volatiles_to_copy:
            if pokemon.volatiles.get(v):
                self.addVolatile(v)
                if v == 'gmaxchistrike':
                    self.volatiles[v].layers = pokemon.volatiles[v].layers
                if v == 'dragoncheer':
                    self.volatiles[v].hasDragonType = pokemon.volatiles[v].hasDragonType
        if effect:
            battle.add('-transform', self, pokemon, '[from] ' + effect.fullname)
        else:
            battle.add('-transform', self, pokemon)
        if self.terastallized:
            self.knownType = True
            self.apparentType = self.terastallized
        self.setAbility(pokemon.ability, self, None, True, True)
        if self.species.baseSpecies in ('Ogerpon', 'Terapagos') and self.canTerastallize:
            self.canTerastallize = False
        return True

    def setSpecies(self, rawSpecies, source=NULL, isTransform=False):
        battle = self.battle
        if source is NULL:
            source = battle.effect
        species = battle.runEvent('ModifySpecies', self, None, source, rawSpecies)
        if not species:
            return None
        self.species = species
        self.setType(species.types, True)
        self.apparentType = '/'.join(rawSpecies.types)
        self.addedType = species.addedType or ''
        self.knownType = True
        self.weighthg = species.weighthg
        stats = battle.spreadModify(self.species.baseStats, self.set)
        if self.species.maxHP:
            stats['hp'] = self.species.maxHP
        if not self.maxhp:
            self.baseMaxhp = stats['hp']
            self.maxhp = stats['hp']
            self.hp = stats['hp']
        if not isTransform:
            self.baseStoredStats = stats
        for stat_name in list(self.storedStats.keys()):
            self.storedStats[stat_name] = stats[stat_name]
        self.speed = self.storedStats['spe']
        return species

    def formeChange(self, speciesId, source=None, isPermanent=False, abilitySlot='0', message=None):
        """Champions override: ``source`` has no default (``undefined`` means no source)."""
        battle = self.battle
        raw_species = battle.dex.species.get(speciesId)
        species = self.setSpecies(raw_species, NULL if source is None else source)
        if not species:
            return False
        apparent_species = self.illusion.species.name if self.illusion else species.baseSpecies
        if isPermanent:
            self.baseSpecies = raw_species
            self.details = self.getUpdatedDetails()
            details = (self.illusion or self).details
            if self.terastallized:
                details += f", tera:{self.terastallized}"
            battle.add('detailschange', self, details)
            self.updateMaxHp()
            if not source:
                self.formeRegression = True
            elif source.effectType == 'Item':
                self.canTerastallize = None
                if source.zMove:
                    battle.add('-burst', self, apparent_species, species.requiredItem)
                    self.moveThisTurnResult = True
                elif source.isPrimalOrb:
                    if self.illusion:
                        self.ability = ''
                        battle.add('-primal', self.illusion, species.requiredItem)
                    else:
                        battle.add('-primal', self, species.requiredItem)
                else:
                    battle.add('-mega', self, apparent_species, species.requiredItem)
                    self.moveThisTurnResult = True
                # champions: Mega Evolutions don't revert after fainting
            elif source.effectType == 'Status':
                battle.add('-formechange', self, species.name, message)
        else:
            if source and source.effectType == 'Ability':
                battle.add('-formechange', self, species.name, message, f"[from] ability: {source.name}")
            else:
                battle.add('-formechange', self, self.illusion.species.name if self.illusion else species.name,
                           message)
        if isPermanent and (not source or source.id not in ('disguise', 'iceface')):
            if self.illusion and source:
                self.ability = ''
            ability = species.abilities.get(abilitySlot) or species.abilities['0']
            if source or not self.getAbility().flags.get('cantsuppress'):
                self.setAbility(ability, None, None, True)
            self.baseAbility = to_id(ability)
        if self.terastallized:
            self.knownType = True
            self.apparentType = self.terastallized
        return True

    def updateMaxHp(self):
        new_base = self.battle.statModify(self.species.baseStats, self.set, 'hp')
        if new_base == self.baseMaxhp:
            return
        self.baseMaxhp = new_base
        new_max = self.baseMaxhp
        self.hp = 0 if self.hp <= 0 else max(1, new_max - (self.maxhp - self.hp))
        self.maxhp = new_max
        if self.hp:
            self.battle.add('-heal', self, self.getHealth, '[silent]')

    def clearVolatile(self, includeSwitchFlags=True):
        self.boosts = _new_boosts()
        self.moveSlots = list(self.baseMoveSlots)
        self.transformed = False
        self.ability = self.baseAbility
        self.hpType = self.baseHpType
        self.hpPower = self.baseHpPower
        if self.canTerastallize is False:
            self.canTerastallize = self.teraType
        for i in list(self.volatiles.keys()):
            v = self.volatiles.get(i)
            if v is not None and v.linkedStatus:
                self.removeLinkedVolatiles(v.linkedStatus, v.linkedPokemon)
        self.volatiles = {}
        if includeSwitchFlags:
            self.switchFlag = False
            self.forceSwitchFlag = False
        self.lastMove = None
        self.lastMoveEncore = None
        self.lastMoveUsed = None
        self.moveThisTurn = ''
        self.moveLastTurnResult = None
        self.moveThisTurnResult = None
        self.lastDamage = 0
        self.attackedBy = []
        self.hurtThisTurn = None
        self.newlySwitched = True
        self.beingCalledBack = False
        self.timesAttacked = 0  # champions: Rage Fist counter resets on switch
        self.volatileStaleness = None
        self.abilityState.pop('started', None)
        self.itemState.pop('started', None)
        self.setSpecies(self.baseSpecies)

    def hasType(self, type_):
        this_types = self.getTypes()
        if isinstance(type_, str):
            return type_ in this_types
        for t in type_:
            if t in this_types:
                return True
        return False

    def faint(self, source=None, effect=None):
        if self.fainted or self.faintQueued:
            return 0
        d = self.hp
        self.hp = 0
        self.switchFlag = False
        self.faintQueued = True
        self.battle.faintQueue.append(Obj(target=self, source=source, effect=effect))
        return d

    def damage(self, d, source=None, effect=None):
        if not self.hp or not is_number(d) or d != d or d <= 0:
            return 0
        if 0 < d < 1:
            d = 1
        d = trunc(d)
        self.hp -= d
        if self.hp <= 0:
            d += self.hp
            self.faint(source, effect)
        return d

    def tryTrap(self, isHidden=False):
        if not self.runStatusImmunity('trapped'):
            return False
        if self.trapped and isHidden:
            return True
        self.trapped = 'hidden' if isHidden else True
        return True

    def hasMove(self, moveid):
        moveid = to_id(moveid)
        if moveid[:11] == 'hiddenpower':
            moveid = 'hiddenpower'
        for ms in self.moveSlots:
            if moveid == ms.id:
                return moveid
        return False

    def disableMove(self, moveid, isHidden=False, sourceEffect=None):
        if not sourceEffect and self.battle.event:
            sourceEffect = self.battle.effect
        moveid = to_id(moveid)
        for ms in self.moveSlots:
            if ms.id == moveid and ms.disabled is not True:
                ms.disabled = 'hidden' if isHidden else True
                ms.disabledSource = (sourceEffect.name if sourceEffect else None) or ms.move

    def heal(self, d, source=None, effect=None):
        if not self.hp:
            return False
        d = trunc(d)
        if d <= 0:
            return False
        if self.hp >= self.maxhp:
            return False
        self.hp += d
        if self.hp > self.maxhp:
            d -= self.hp - self.maxhp
            self.hp = self.maxhp
        return d

    def sethp(self, d):
        if not self.hp:
            return 0
        d = trunc(d)
        if d < 1:
            d = 1
        d -= self.hp
        self.hp += d
        if self.hp > self.maxhp:
            d -= self.hp - self.maxhp
            self.hp = self.maxhp
        return d

    def trySetStatus(self, status, source=None, sourceEffect=None):
        return self.setStatus(self.status or status, source, sourceEffect)

    def cureStatus(self, silent=False):
        if not self.hp or not self.status:
            return False
        self.battle.add('-curestatus', self, self.status, '[silent]' if silent else '[msg]')
        if self.status == 'slp' and self.removeVolatile('nightmare'):
            self.battle.add('-end', self, 'Nightmare', '[silent]')
        self.setStatus('')
        return True

    def setStatus(self, status, source=None, sourceEffect=None, ignoreImmunities=False):
        battle = self.battle
        if not self.hp:
            return False
        if not self.isActive and status:
            return False
        status = battle.dex.conditions.get(status)
        if battle.event:
            if not source:
                source = battle.event.source
            if not sourceEffect:
                sourceEffect = battle.effect
        if not source:
            source = self
        if self.status == status.id:
            if sourceEffect and sourceEffect.get('status') == self.status:
                battle.add('-fail', self, self.status)
            elif sourceEffect and sourceEffect.get('status'):
                battle.add('-fail', source)
                battle.attrLastMove('[still]')
            return False
        if (not ignoreImmunities and status.id and
                not (source and source.hasAbility('corrosion') and status.id in ('tox', 'psn'))):
            if not self.runStatusImmunity('psn' if status.id == 'tox' else status.id):
                battle.debug('immune to status')
                if sourceEffect and sourceEffect.get('status'):
                    battle.add('-immune', self)
                return False
        prev_status = self.status
        prev_state = self.statusState
        if status.id:
            result = battle.runEvent('SetStatus', self, source, sourceEffect, status)
            if not result:
                battle.debug('set status [' + status.id + '] interrupted')
                return result
        self.status = status.id
        self.statusState = battle.initEffectState(Obj(id=status.id, target=self))
        if source:
            self.statusState.source = source
        if status.duration:
            self.statusState.duration = status.duration
        if status.get('durationCallback'):
            self.statusState.duration = status.durationCallback(battle, self, source, sourceEffect)
        if status.id and not battle.singleEvent('Start', status, self.statusState, self, source, sourceEffect):
            battle.debug('status start [' + status.id + '] interrupted')
            self.status = prev_status
            self.statusState = prev_state
            return False
        if status.id and not battle.runEvent('AfterSetStatus', self, source, sourceEffect, status):
            return False
        return True

    def clearStatus(self):
        if not self.hp or not self.status:
            return False
        if self.status == 'slp' and self.removeVolatile('nightmare'):
            self.battle.add('-end', self, 'Nightmare', '[silent]')
        self.setStatus('')
        return True

    def getStatus(self):
        return self.battle.dex.conditions.getByID(self.status)

    def eatItem(self, force=False, source=None, sourceEffect=None):
        battle = self.battle
        if not self.item:
            return False
        if (not self.hp and self.item not in ('jabocaberry', 'rowapberry')) or not self.isActive:
            return False
        if not sourceEffect and battle.effect:
            sourceEffect = battle.effect
        if not source and battle.event and battle.event.target:
            source = battle.event.target
        item = self.getItem()
        if sourceEffect and sourceEffect.effectType == 'Item' and self.item != sourceEffect.id and source is self:
            return False
        if battle.runEvent('UseItem', self, None, None, item) and \
                (force or battle.runEvent('TryEatItem', self, None, None, item)):
            battle.add('-enditem', self, item, '[eat]')
            battle.singleEvent('Eat', item, self.itemState, self, source, sourceEffect)
            battle.runEvent('EatItem', self, source, sourceEffect, item)
            if item.id in RESTORATIVE_BERRIES:
                if self.pendingStaleness == 'internal':
                    if self.staleness != 'external':
                        self.staleness = 'internal'
                elif self.pendingStaleness == 'external':
                    self.staleness = 'external'
                self.pendingStaleness = None
            self.lastItem = self.item
            self.item = ''
            battle.clearEffectState(self.itemState)
            self.usedItemThisTurn = True
            self.ateBerry = True
            battle.runEvent('AfterUseItem', self, None, None, item)
            return True
        return False

    def useItem(self, source=None, sourceEffect=None):
        battle = self.battle
        if (not self.hp and not self.getItem().isGem) or not self.isActive:
            return False
        if not self.item:
            return False
        if not sourceEffect and battle.effect:
            sourceEffect = battle.effect
        if not source and battle.event and battle.event.target:
            source = battle.event.target
        item = self.getItem()
        if sourceEffect and sourceEffect.effectType == 'Item' and self.item != sourceEffect.id and source is self:
            return False
        if battle.runEvent('UseItem', self, None, None, item):
            if item.id == 'redcard':
                battle.add('-enditem', self, item, f"[of] {source}")
            elif item.isGem:
                battle.add('-enditem', self, item, '[from] gem', f"[move] {battle.activeMove.name}")
            else:
                battle.add('-enditem', self, item)
            if item.boosts:
                battle.boost(item.boosts, self, source, item)
            battle.singleEvent('Use', item, self.itemState, self, source, sourceEffect)
            self.lastItem = self.item
            self.item = ''
            battle.clearEffectState(self.itemState)
            self.usedItemThisTurn = True
            battle.runEvent('AfterUseItem', self, None, None, item)
            return True
        return False

    def takeItem(self, source=None):
        battle = self.battle
        if not source:
            source = self
        if not self.item:
            return None
        item = self.getItem()
        if battle.runEvent('TakeItem', self, source, None, item):
            self.item = ''
            old_state = self.itemState
            battle.clearEffectState(self.itemState)
            self.pendingStaleness = None
            battle.singleEvent('End', item, old_state, self)
            battle.runEvent('AfterTakeItem', self, None, None, item)
            return item
        return False

    def setItem(self, item, source=None, effect=None):
        battle = self.battle
        if not self.hp or not self.isActive:
            return False
        if isinstance(item, str):
            item = battle.dex.items.get(item)
        effectid = battle.effect.id if battle.effect else ''
        inflicted = effectid in ('trick', 'switcheroo')
        external = inflicted and source and not source.isAlly(self)
        self.pendingStaleness = 'external' if external else 'internal'
        old_item = self.getItem()
        old_state = self.itemState
        self.item = item.id
        self.itemState = battle.initEffectState(Obj(id=item.id, target=self))
        if old_item.exists:
            battle.singleEvent('End', old_item, old_state, self)
        if item.id:
            battle.singleEvent('Start', item, self.itemState, self, source, effect)
        return True

    def getItem(self):
        return self.battle.dex.items.getByID(self.item)

    def hasItem(self, item):
        if isinstance(item, (list, tuple)):
            if self.item not in [to_id(i) for i in item]:
                return False
        else:
            if to_id(item) != self.item:
                return False
        return not self.ignoringItem()

    def clearItem(self):
        return self.setItem('')

    def setAbility(self, ability, source=None, sourceEffect=None, isFromFormeChange=False, isTransform=False):
        battle = self.battle
        if not self.hp:
            return False
        if isinstance(ability, str):
            ability = battle.dex.abilities.get(ability)
        if not sourceEffect and battle.effect:
            sourceEffect = battle.effect
        old_ability = battle.dex.abilities.get(self.ability)
        if not isFromFormeChange:
            if ability.flags.get('cantsuppress') or self.getAbility().flags.get('cantsuppress'):
                return False
        if not isFromFormeChange and not isTransform:
            ev = battle.runEvent('SetAbility', self, source, sourceEffect, ability)
            if not ev:
                return ev
        battle.singleEvent('End', old_ability, self.abilityState, self, source)
        self.ability = ability.id
        self.abilityState = battle.initEffectState(Obj(id=ability.id, target=self))
        if sourceEffect and not isFromFormeChange and not isTransform:
            if sourceEffect.id in ('mummy', 'lingeringaroma'):
                battle.add('-activate', source, sourceEffect.fullname, self, '[ability] ' + old_ability.name)
            elif source:
                battle.add('-ability', self, ability.name, old_ability.name, f"[from] {sourceEffect.fullname}",
                           f"[of] {source}")
            else:
                battle.add('-ability', self, ability.name, old_ability.name, f"[from] {sourceEffect.fullname}")
        if ability.id and (not isTransform or old_ability.id != ability.id):
            battle.singleEvent('Start', ability, self.abilityState, self, source)
        return old_ability.id

    def getAbility(self):
        return self.battle.dex.abilities.getByID(self.ability)

    def hasAbility(self, ability):
        if isinstance(ability, (list, tuple)):
            if self.ability not in [to_id(a) for a in ability]:
                return False
        else:
            if to_id(ability) != self.ability:
                return False
        return not self.ignoringAbility()

    def clearAbility(self):
        return self.setAbility('')

    def getNature(self):
        return self.battle.dex.natures.get(self.set.get('nature'))

    def addVolatile(self, status, source=None, sourceEffect=None, linkedStatus=None):
        battle = self.battle
        status = battle.dex.conditions.get(status)
        if not self.hp and not status.affectsFainted:
            return False
        if linkedStatus and source and not source.hp:
            return False
        if battle.event:
            if not source:
                source = battle.event.source
            if not sourceEffect:
                sourceEffect = battle.effect
        if not source:
            source = self
        if status.id in self.volatiles:
            if not status.get('onRestart'):
                return False
            return battle.singleEvent('Restart', status, self.volatiles[status.id], self, source, sourceEffect)
        if not self.runStatusImmunity(status.id):
            battle.debug('immune to volatile status')
            if sourceEffect and sourceEffect.get('status'):
                battle.add('-immune', self)
            return False
        result = battle.runEvent('TryAddVolatile', self, source, sourceEffect, status)
        if not result:
            battle.debug('add volatile [' + status.id + '] interrupted')
            return result
        state = battle.initEffectState(Obj(id=status.id, name=status.name, target=self))
        self.volatiles[status.id] = state
        if source:
            state.source = source
            state.sourceSlot = source.getSlot()
        if sourceEffect:
            state.sourceEffect = sourceEffect
        if status.duration:
            state.duration = status.duration
        if status.get('durationCallback'):
            state.duration = status.durationCallback(battle, self, source, sourceEffect)
        result = battle.singleEvent('Start', status, state, self, source, sourceEffect)
        if not result:
            self.volatiles.pop(status.id, None)
            return result
        if linkedStatus and source:
            ls = str(linkedStatus)
            if not source.volatiles.get(ls):
                source.addVolatile(linkedStatus, self, sourceEffect)
                source.volatiles[ls].linkedPokemon = [self]
                source.volatiles[ls].linkedStatus = status
            else:
                source.volatiles[ls].linkedPokemon.append(self)
            self.volatiles[str(status)].linkedPokemon = [source]
            self.volatiles[str(status)].linkedStatus = linkedStatus
        return True

    def getVolatile(self, status):
        status = self.battle.dex.conditions.get(status)
        if status.id not in self.volatiles:
            return None
        return status

    def removeVolatile(self, status):
        battle = self.battle
        if not self.hp:
            return False
        status = battle.dex.conditions.get(status)
        if status.id not in self.volatiles:
            return False
        state = self.volatiles[status.id]
        linked_pokemon = state.linkedPokemon
        linked_status = state.linkedStatus
        battle.singleEvent('End', status, state, self)
        self.volatiles.pop(status.id, None)
        if linked_pokemon:
            self.removeLinkedVolatiles(linked_status, linked_pokemon)
        return True

    def removeLinkedVolatiles(self, linkedStatus, linkedPokemon):
        linkedStatus = str(linkedStatus)
        for linked in list(linkedPokemon):
            data = linked.volatiles.get(linkedStatus)
            if not data:
                continue
            lp = data.linkedPokemon
            if self in lp:
                lp.pop(lp.index(self))
            else:
                lp.pop(-1)
            if len(lp) == 0:
                linked.removeVolatile(linkedStatus)

    def getHealth(self):
        if not self.hp:
            return {'side': self.side.id, 'secret': '0 fnt', 'shared': '0 fnt'}
        secret = f"{self.hp}/{self.maxhp}"
        if self.battle.reportExactHP:
            shared = secret
        else:
            percentage = math.floor(100 * self.hp / self.maxhp) or 1
            shared = f"{percentage}/100"
            if percentage == 20:
                shared += 'y' if self.hp * 5 > self.maxhp else 'r'
            elif percentage == 50:
                shared += 'g' if self.hp * 2 > self.maxhp else 'y'
        if self.status:
            secret += f" {self.status}"
            shared += f" {self.status}"
        return {'side': self.side.id, 'secret': secret, 'shared': shared}

    def setType(self, newType, enforce=False):
        if not enforce:
            if (newType == 'Stellar') if isinstance(newType, str) else ('Stellar' in newType):
                return False
            if self.species.num in (493, 773):
                return False
            if self.terastallized:
                return False
        if not newType:
            raise ValueError('Must pass type to setType')
        self.types = [newType] if isinstance(newType, str) else newType
        self.addedType = ''
        self.knownType = True
        self.apparentType = '/'.join(self.types)
        return True

    def addType(self, newType):
        if self.terastallized:
            return False
        self.addedType = newType
        return True

    def getTypes(self, excludeAdded=False, preterastallized=False):
        if not preterastallized and self.terastallized and self.terastallized != 'Stellar':
            return [self.terastallized]
        types = self.battle.runEvent('Type', self, None, None, self.types)
        if not len(types):
            types.append('Normal')
        if not excludeAdded and self.addedType:
            return list(types) + [self.addedType]
        return types

    def isGrounded(self, negateImmunity=False):
        battle = self.battle
        if 'gravity' in battle.field.pseudoWeather:
            return True
        if 'ingrain' in self.volatiles:
            return True
        if 'smackdown' in self.volatiles:
            return True
        item = '' if self.ignoringItem() else self.item
        if item == 'ironball':
            return True
        if not negateImmunity and self.hasType('Flying') and not (self.hasType('???') and 'roost' in self.volatiles):
            return False
        if self.hasAbility(['levitate', 'eelevate']) and not battle.suppressingAbility(self):
            return NULL
        if 'magnetrise' in self.volatiles:
            return False
        if 'telekinesis' in self.volatiles:
            return False
        return item != 'airballoon'

    def isSemiInvulnerable(self):
        v = self.volatiles
        return bool(v.get('fly') or v.get('bounce') or v.get('dive') or v.get('dig') or v.get('phantomforce') or
                    v.get('shadowforce') or self.isSkyDropped())

    def isSkyDropped(self):
        if self.volatiles.get('skydrop'):
            return True
        for foe in self.side.foe.active:
            if foe and foe.volatiles.get('skydrop') and foe.volatiles['skydrop'].source is self:
                return True
        return False

    def isProtected(self):
        v = self.volatiles
        return bool(v.get('protect') or v.get('detect') or v.get('maxguard') or v.get('kingsshield') or
                    v.get('spikyshield') or v.get('banefulbunker') or v.get('obstruct') or v.get('silktrap') or
                    v.get('burningbulwark'))

    def effectiveWeather(self, sourceEffect=None, message=None):
        battle = self.battle
        if not sourceEffect and battle.effect:
            sourceEffect = battle.effect
        weather = battle.field.effectiveWeather()
        if (battle.activePokemon and battle.activePokemon.hasAbility('megasol') and sourceEffect and
                (sourceEffect.id == 'megasol' or sourceEffect.effectType in ('Move', 'Weather'))):
            if weather != 'sunnyday' and message:
                battle.add('-activate', self, 'ability: Mega Sol')
            return 'sunnyday'
        if weather in ('sunnyday', 'raindance', 'desolateland', 'primordialsea'):
            if self.hasItem('utilityumbrella'):
                return ''
        return weather

    def runEffectiveness(self, move):
        battle = self.battle
        total = 0
        for type_ in self.getTypes():
            type_mod = battle.dex.getEffectiveness(move, type_)
            type_mod = battle.singleEvent('Effectiveness', move, None, self, type_, move, type_mod)
            total += battle.runEvent('Effectiveness', self, type_, move, type_mod)
        return total

    def runImmunity(self, source, message=None):
        battle = self.battle
        if not source:
            return True
        type_ = source if isinstance(source, str) else source.type
        if not isinstance(source, str):
            ii = source.ignoreImmunity
            if ii and (ii is True or (isinstance(ii, dict) and ii.get(type_))):
                return True
        if not type_ or type_ == '???':
            return True
        if not battle.dex.types.isName(type_):
            raise ValueError('Use runStatusImmunity for ' + type_)
        negate_immunity = not battle.runEvent('NegateImmunity', self, type_)
        if type_ == 'Ground':
            not_immune = self.isGrounded(negate_immunity)
        else:
            not_immune = negate_immunity or battle.dex.getImmunity(type_, self)
        if not_immune:
            return True
        if not message:
            return False
        if not_immune is NULL:
            if self.hasAbility('levitate'):
                battle.add('-immune', self, '[from] ability: Levitate')
            elif self.hasAbility('eelevate'):
                battle.add('-immune', self, '[from] ability: Eelevate')
            else:
                battle.add('-immune', self)
        else:
            battle.add('-immune', self)
        return False

    def runStatusImmunity(self, type_, message=None):
        battle = self.battle
        if self.fainted:
            return False
        if not type_:
            return True
        if not battle.dex.getImmunity(type_, self):
            battle.debug('natural status immunity')
            if message:
                battle.add('-immune', self)
            return False
        immunity = battle.runEvent('Immunity', self, None, None, type_)
        if not immunity:
            battle.debug('artificial status immunity')
            if message and immunity is not NULL:
                battle.add('-immune', self)
            return False
        return True
