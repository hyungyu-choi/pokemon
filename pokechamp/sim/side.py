"""Port of Showdown's ``sim/side.ts`` (choices, side/slot conditions, requests)."""
from __future__ import annotations

import re

from .js import NULL, Obj, clamp_int_range, to_id

_TARGET_LOC_RE = re.compile(r'\s(?:-|\+)?[1-3]$')


class Side:
    def __init__(self, name: str, battle, sideNum: int, team: list):
        from .pokemon import Pokemon  # noqa: F401
        self.battle = battle
        self.id = ['p1', 'p2', 'p3', 'p4'][sideNum]
        self.n = sideNum
        self.name = name
        self.avatar = ''
        self.foe = None
        self.allySide = None
        self.team = team
        self.pokemon = []
        self.pokemonLeft = 0
        for set_ in self.team:
            self.addPokemon(set_)
        if battle.gameType == 'doubles':
            self.active = [None, None]
        elif battle.gameType in ('triples', 'rotation'):
            self.active = [None, None, None]
        else:
            self.active = [None]
        self.pokemonLeft = len(self.pokemon)
        self.faintedLastTurn = None
        self.faintedThisTurn = None
        self.totalFainted = 0
        self.zMoveUsed = False
        self.dynamaxUsed = True
        self.sideConditions: dict[str, Obj] = {}
        self.slotConditions: list[dict[str, Obj]] = [{} for _ in range(len(self.active))]
        self.activeRequest = None
        self.choice = self._new_choice()
        self.lastMove = None
        self.lastEnemyMove = None

    @staticmethod
    def _new_choice(forced_switches=0, forced_passes=0):
        return Obj(cantUndo=False, error='', actions=[], forcedSwitchesLeft=forced_switches,
                   forcedPassesLeft=forced_passes, switchIns=set(), zMove=False, mega=False, ultra=False,
                   dynamax=False, terastallize=False)

    def __str__(self):
        return f"{self.id}: {self.name}"

    def __repr__(self):
        return f"<Side {self.id}>"

    @property
    def requestState(self):
        req = self.activeRequest
        if not req or req.get('wait'):
            return ''
        if req.get('teamPreview'):
            return 'teampreview'
        if req.get('forceSwitch'):
            return 'switch'
        return 'move'

    def addPokemon(self, set_):
        from .pokemon import Pokemon
        if len(self.pokemon) >= 24:
            return None
        p = Pokemon(set_, self)
        p.position = len(self.pokemon)
        self.pokemon.append(p)
        self.pokemonLeft += 1
        return p

    def canDynamaxNow(self):
        return False

    def getChoice(self):
        actions = self.choice.actions
        if len(actions) > 1 and all(a.choice == 'team' for a in actions):
            return 'team ' + ', '.join(str(a.pokemon.position + 1) for a in actions)
        out = []
        for action in actions:
            c = action.choice
            if c == 'move':
                details = ''
                if action.targetLoc and self.battle.activePerHalf > 1:
                    details += f" {'+' if action.targetLoc > 0 else ''}{action.targetLoc}"
                if action.mega:
                    details += ' mega'
                if action.megax:
                    details += ' megax'
                if action.megay:
                    details += ' megay'
                if action.zmove:
                    details += ' zmove'
                if action.maxMove:
                    details += ' dynamax'
                if action.terastallize:
                    details += ' terastallize'
                out.append(f"move {action.moveid}{details}")
            elif c in ('switch', 'instaswitch', 'revivalblessing'):
                out.append(f"switch {action.target.position + 1}")
            elif c == 'team':
                out.append(f"team {action.pokemon.position + 1}")
            else:
                out.append(c)
        return ', '.join(out)

    def getRequestData(self, forAlly=False):
        return {
            'name': self.name,
            'id': self.id,
            'pokemon': [p.getSwitchRequestData(forAlly) for p in self.pokemon],
        }

    def randomFoe(self):
        actives = self.foes()
        if not actives:
            return None
        return self.battle.sample(actives)

    def foeSidesWithConditions(self):
        return [self.foe]

    def foePokemonLeft(self):
        if self.foe.allySide:
            return self.foe.pokemonLeft + self.foe.allySide.pokemonLeft
        return self.foe.pokemonLeft

    def allies(self, all_=False):
        allies = [a for a in self.activeTeam() if a]
        if not all_:
            allies = [a for a in allies if a.hp]
        return allies

    def foes(self, all_=False):
        return self.foe.allies(all_)

    def activeTeam(self):
        return self.active

    def hasAlly(self, pokemon):
        return pokemon.side is self or pokemon.side is self.allySide

    # --- conditions -----------------------------------------------------------
    def addSideCondition(self, status, source=None, sourceEffect=None):
        battle = self.battle
        if not source and battle.event and battle.event.target:
            source = battle.event.target
        if source == 'debug':
            source = self.active[0]
        if not source:
            raise ValueError('setting sidecond without a source')
        if not hasattr(source, 'getSlot'):
            source = source.active[0]
        status = battle.dex.conditions.get(status)
        if status.id in self.sideConditions:
            if not status.get('onSideRestart'):
                return False
            return battle.singleEvent('SideRestart', status, self.sideConditions[status.id], self, source,
                                      sourceEffect)
        state = battle.initEffectState(Obj(id=status.id, target=self, source=source,
                                           sourceSlot=source.getSlot(), duration=status.duration))
        self.sideConditions[status.id] = state
        if status.get('durationCallback'):
            state.duration = status.durationCallback(battle, self.active[0], source, sourceEffect)
        if not battle.singleEvent('SideStart', status, state, self, source, sourceEffect):
            self.sideConditions.pop(status.id, None)
            return False
        battle.runEvent('SideConditionStart', self, source, status)
        return True

    def getSideCondition(self, status):
        status = self.battle.dex.conditions.get(status)
        if status.id not in self.sideConditions:
            return None
        return status

    def getSideConditionData(self, status):
        status = self.battle.dex.conditions.get(status)
        return self.sideConditions.get(status.id)

    def removeSideCondition(self, status):
        status = self.battle.dex.conditions.get(status)
        if status.id not in self.sideConditions:
            return False
        self.battle.singleEvent('SideEnd', status, self.sideConditions[status.id], self)
        self.sideConditions.pop(status.id, None)
        return True

    def addSlotCondition(self, target, status, source=None, sourceEffect=None):
        from .pokemon import Pokemon
        battle = self.battle
        if source is None:
            source = (battle.event.target if battle.event else None) or None
        if source == 'debug':
            source = self.active[0]
        if isinstance(target, Pokemon):
            target = target.position
        if not source:
            raise ValueError('setting sidecond without a source')
        status = battle.dex.conditions.get(status)
        slot = self.slotConditions[target]
        if status.id in slot:
            if not status.get('onRestart'):
                return False
            return battle.singleEvent('Restart', status, slot[status.id], self, source, sourceEffect)
        state = battle.initEffectState(Obj(id=status.id, target=self, source=source,
                                           sourceSlot=source.getSlot(), isSlotCondition=True,
                                           duration=status.duration))
        slot[status.id] = state
        if status.get('durationCallback'):
            state.duration = status.durationCallback(battle, self.active[0], source, sourceEffect)
        if not battle.singleEvent('Start', status, state, self.active[target], source, sourceEffect):
            slot.pop(status.id, None)
            return False
        return True

    def getSlotCondition(self, target, status):
        from .pokemon import Pokemon
        if isinstance(target, Pokemon):
            target = target.position
        status = self.battle.dex.conditions.get(status)
        if status.id not in self.slotConditions[target]:
            return None
        return status

    def removeSlotCondition(self, target, status):
        from .pokemon import Pokemon
        if isinstance(target, Pokemon):
            target = target.position
        status = self.battle.dex.conditions.get(status)
        slot = self.slotConditions[target]
        if status.id not in slot:
            return False
        self.battle.singleEvent('End', status, slot[status.id], self.active[target])
        slot.pop(status.id, None)
        return True

    # --- protocol helpers -------------------------------------------------
    def send(self, *parts):
        side_update = '|' + '|'.join(str(p(self) if callable(p) else p) for p in parts)
        self.battle.send('sideupdate', f"{self.id}\n{side_update}")

    def emitRequest(self, update=None, updatedRequest=False):
        if update is None:
            update = self.activeRequest
        if updatedRequest:
            self.activeRequest['update'] = True
        self.activeRequest = update

    def emitChoiceError(self, message, update=None):
        self.choice.error = message
        updated = self.updateRequestForPokemon(update['pokemon'], update['update']) if update else None
        type_ = f"[{'Unavailable' if updated else 'Invalid'} choice]"
        self.battle.send('sideupdate', f"{self.id}\n|error|{type_} {message}")
        if self.battle.strictChoices:
            raise ValueError(f"{type_} {message}")
        return False

    def isChoiceDone(self):
        if not self.requestState:
            return True
        if self.choice.forcedSwitchesLeft:
            return False
        if self.requestState == 'teampreview':
            return len(self.choice.actions) >= self.pickedTeamSize()
        self.getChoiceIndex()
        return len(self.choice.actions) >= len(self.active)

    def chooseMove(self, moveText=None, targetLoc=0, event=''):
        battle = self.battle
        if self.requestState != 'move':
            return self.emitChoiceError(f"Can't move: You need a {self.requestState} response")
        index = self.getChoiceIndex()
        if index >= len(self.active):
            return self.emitChoiceError("Can't move: You sent more choices than unfainted Pokémon.")
        auto_choose = not moveText
        pokemon = self.active[index]
        request = pokemon.getMoveRequestData()
        move_slot = None
        moveid = ''
        target_type = ''
        if auto_choose:
            moveText = 1
        if isinstance(moveText, int) or (moveText and re.fullmatch(r'[0-9]+', str(moveText))):
            move_index = int(moveText) - 1
            if move_index < 0 or move_index >= len(request['moves']) or not request['moves'][move_index]:
                return self.emitChoiceError(f"Can't move: Your {pokemon.name} doesn't have a move {move_index + 1}")
            move_slot = move_index
            moveid = request['moves'][move_index]['id']
            target_type = request['moves'][move_index].get('target')
        else:
            moveid = to_id(moveText)
            if moveid.startswith('hiddenpower'):
                moveid = 'hiddenpower'
            for i, m in enumerate(request['moves']):
                if m['id'] != moveid:
                    continue
                move_slot = i
                target_type = m.get('target') or 'normal'
                break
            if not target_type:
                if moveid != 'testfight':
                    return self.emitChoiceError(
                        f"Can't move: Your {pokemon.name} doesn't have a move matching {moveid}")

        moves = pokemon.getMoves()
        if auto_choose:
            for i, m in enumerate(request['moves']):
                if m.get('disabled'):
                    continue
                if i < len(moves) and m['id'] == moves[i]['id'] and moves[i].get('disabled'):
                    continue
                moveid = m['id']
                move_slot = i
                target_type = m.get('target')
                break
        move = battle.dex.moves.get(moveid)

        if auto_choose or moveid == 'testfight':
            targetLoc = 0
        elif battle.actions.targetTypeChoices(target_type):
            if not targetLoc and len(self.active) >= 2:
                return self.emitChoiceError(f"Can't move: {move.name} needs a target")
            if not battle.validTargetLoc(targetLoc, pokemon, target_type):
                return self.emitChoiceError(f"Can't move: Invalid target for {move.name}")
        else:
            if targetLoc:
                return self.emitChoiceError(f"Can't move: You can't choose a target for {move.name}")

        locked_move = pokemon.getLockedMove() or pokemon.getSemiLockedMove()
        if locked_move:
            locked_target_loc = pokemon.lastMoveTargetLoc or 0
            locked_id = to_id(locked_move)
            if pokemon.volatiles.get(locked_id) and pokemon.volatiles[locked_id].targetLoc:
                locked_target_loc = pokemon.volatiles[locked_id].targetLoc
            if pokemon.maybeLocked:
                self.choice.cantUndo = True
            self.choice.actions.append(Obj(choice='move', pokemon=pokemon, targetLoc=locked_target_loc,
                                           moveid=locked_id))
            return True
        elif not moves:
            if pokemon.maybeLocked:
                self.choice.cantUndo = True
            self.choice.actions.append(Obj(choice='move', pokemon=pokemon, moveid='struggle'))
            return True
        elif moveid == 'testfight':
            if not pokemon.maybeLocked:
                return self.emitChoiceError(f"Can't move: {pokemon.name}'s Fight button is known to be safe")
            self.updateRequestForPokemon(pokemon, lambda req: self.updateDisabledRequest(pokemon, req))
            self.emitRequest(self.activeRequest, True)
            self.choice.error = 'Hack to avoid sending error messages to the client :D'
            return False
        else:
            is_enabled = False
            disabled_source = ''
            for m in moves:
                if m['id'] != moveid:
                    continue
                if not m.get('disabled'):
                    is_enabled = True
                    break
                elif m.get('disabledSource'):
                    disabled_source = m['disabledSource']
            if not is_enabled:
                if auto_choose:
                    raise ValueError('autoChoose chose a disabled move')

                def _upd(req, _moveid=moveid, _src=disabled_source):
                    updated = self.updateDisabledRequest(pokemon, req)
                    for m in req['moves']:
                        if m['id'] == _moveid:
                            if not m.get('disabled'):
                                m['disabled'] = True
                                updated = True
                            if m.get('disabledSource') != _src:
                                m['disabledSource'] = _src
                                updated = True
                            break
                    return updated
                return self.emitChoiceError(f"Can't move: {pokemon.name}'s {move.name} is disabled",
                                            {'pokemon': pokemon, 'update': _upd})

        mega = event == 'mega'
        if mega and not pokemon.canMegaEvo:
            return self.emitChoiceError(f"Can't move: {pokemon.name} can't mega evolve")
        if mega and self.choice.mega:
            return self.emitChoiceError("Can't move: You can only mega-evolve once per battle")
        if event in ('megax', 'megay'):
            return self.emitChoiceError(f"Can't move: {pokemon.name} can't mega evolve {event[-1].upper()}")
        if event == 'ultra':
            return self.emitChoiceError(f"Can't move: {pokemon.name} can't ultra burst")
        if event == 'zmove':
            return self.emitChoiceError(f"Can't move: {pokemon.name} can't use {move.name} as a Z-move")
        if event == 'dynamax':
            return self.emitChoiceError("Can't move: Dynamaxing doesn't outside of Gen 8.")
        if event == 'terastallize':
            return self.emitChoiceError(f"Can't move: {pokemon.name} can't Terastallize.")
        if move_slot is None:
            raise ValueError('moveSlot should have been set by this point')

        self.choice.actions.append(Obj(choice='move', pokemon=pokemon, targetLoc=targetLoc, moveid=moveid,
                                       moveSlot=move_slot, mega=mega, megax=False, megay=False,
                                       zmove=None, maxMove=None, terastallize=None))
        if pokemon.maybeDisabled and battle.gameType == 'singles':
            self.choice.cantUndo = True
        if mega:
            self.choice.mega = True
        return True

    def updateDisabledRequest(self, pokemon, req):
        updated = False
        if pokemon.maybeLocked:
            pokemon.maybeLocked = False
            req.pop('maybeLocked', None)
            updated = True
        if pokemon.maybeDisabled and self.battle.gameType != 'singles':
            pokemon.maybeDisabled = False
            req.pop('maybeDisabled', None)
            updated = True
            for m in req['moves']:
                ms = pokemon.getMoveData(m['id'])
                if ms and ms.disabled:
                    m['disabled'] = True
                    updated = True
        if all(m.get('disabled') or m['id'] == 'struggle' for m in req['moves']):
            if req.get('canMegaEvo'):
                req['canMegaEvo'] = False
                updated = True
        return updated

    def updateRequestForPokemon(self, pokemon, update):
        if not self.activeRequest or not self.activeRequest.get('active'):
            raise ValueError("Can't update a request without active Pokemon")
        req = self.activeRequest['active'][pokemon.position]
        if not req:
            raise ValueError("Pokemon not found in request's active field")
        r = update(req)
        return True if r is None else r

    def chooseSwitch(self, slotText=None):
        if self.requestState not in ('move', 'switch'):
            return self.emitChoiceError(f"Can't switch: You need a {self.requestState} response")
        index = self.getChoiceIndex()
        if index >= len(self.active):
            if self.requestState == 'switch':
                return self.emitChoiceError("Can't switch: You sent more switches than Pokémon that need to switch")
            return self.emitChoiceError("Can't switch: You sent more choices than unfainted Pokémon")
        pokemon = self.active[index]
        if not slotText:
            if self.requestState != 'switch':
                return self.emitChoiceError("Can't switch: You need to select a Pokémon to switch in")
            if self.slotConditions[pokemon.position].get('revivalblessing'):
                slot = 0
                while not self.pokemon[slot].fainted:
                    slot += 1
            else:
                if not self.choice.forcedSwitchesLeft:
                    return self.choosePass()
                slot = len(self.active)
                while slot in self.choice.switchIns or self.pokemon[slot].fainted:
                    slot += 1
        else:
            try:
                slot = int(re.match(r'\s*[+-]?\d+', slotText).group(0)) - 1
            except (AttributeError, ValueError):
                slot = -1
                for i, mon in enumerate(self.pokemon):
                    if slotText.lower() == mon.name.lower() or to_id(slotText) == mon.species.id:
                        slot = i
                        break
                if slot < 0:
                    return self.emitChoiceError(
                        f"Can't switch: You do not have a Pokémon named \"{slotText}\" to switch to")
            if slot < 0:
                return self.emitChoiceError(
                    f"Can't switch: You do not have a Pokémon named \"{slotText}\" to switch to")
        if slot >= len(self.pokemon):
            return self.emitChoiceError(f"Can't switch: You do not have a Pokémon in slot {slot + 1} to switch to")
        elif slot < len(self.active) and not self.slotConditions[pokemon.position].get('revivalblessing'):
            return self.emitChoiceError("Can't switch: You can't switch to an active Pokémon")
        elif slot in self.choice.switchIns:
            return self.emitChoiceError(f"Can't switch: The Pokémon in slot {slot + 1} can only switch in once")
        target_pokemon = self.pokemon[slot]

        if self.slotConditions[pokemon.position].get('revivalblessing'):
            if not target_pokemon.fainted:
                return self.emitChoiceError("Can't switch: You have to pass to a fainted Pokémon")
            self.choice.forcedSwitchesLeft = clamp_int_range(self.choice.forcedSwitchesLeft - 1, 0)
            pokemon.switchFlag = False
            self.choice.actions.append(Obj(choice='revivalblessing', pokemon=pokemon, target=target_pokemon))
            return True

        if target_pokemon.fainted:
            return self.emitChoiceError("Can't switch: You can't switch to a fainted Pokémon")

        if self.requestState == 'move':
            if pokemon.trapped:
                def _upd(req):
                    updated = False
                    if req.get('maybeTrapped'):
                        req.pop('maybeTrapped', None)
                        updated = True
                    if not req.get('trapped'):
                        req['trapped'] = True
                        updated = True
                    return updated
                return self.emitChoiceError("Can't switch: The active Pokémon is trapped",
                                            {'pokemon': pokemon, 'update': _upd})
            elif pokemon.maybeTrapped:
                self.choice.cantUndo = True
        elif self.requestState == 'switch':
            if not self.choice.forcedSwitchesLeft:
                raise ValueError('Player somehow switched too many Pokemon')
            self.choice.forcedSwitchesLeft -= 1

        self.choice.switchIns.add(slot)
        self.choice.actions.append(Obj(choice='instaswitch' if self.requestState == 'switch' else 'switch',
                                       pokemon=pokemon, target=target_pokemon))
        return True

    def pickedTeamSize(self):
        picked = self.battle.ruleTable.pickedTeamSize
        return min(len(self.pokemon), picked) if picked else len(self.pokemon)

    def chooseTeam(self, data=None):
        if self.requestState != 'teampreview':
            return self.emitChoiceError("Can't choose for Team Preview: You're not in a Team Preview phase")
        is_bracketed = False
        team_data = data
        if data and data.startswith('[') and data.endswith(']'):
            is_bracketed = True
            team_data = data[1:-1].strip()
        if team_data:
            if is_bracketed or ',' in team_data or len(self.pokemon) >= 10:
                parts = team_data.split(',')
            else:
                parts = list(team_data)
            positions = []
            for datum in parts:
                m = re.match(r'\s*[+-]?\d+', datum)
                positions.append(int(m.group(0)) - 1 if m else None)
        else:
            positions = list(range(len(self.pokemon)))
        picked = self.pickedTeamSize()
        if not is_bracketed:
            del positions[picked:]
        if len(positions) < picked:
            for i in range(picked):
                if i not in positions:
                    positions.append(i)
                if len(positions) >= picked:
                    break
        if len(positions) != picked:
            return self.emitChoiceError(f"Can't choose for Team Preview: You must choose exactly {picked} Pokémon")
        for index, pos in enumerate(positions):
            if pos is None or pos < 0 or pos >= len(self.pokemon):
                return self.emitChoiceError(
                    f"Can't choose for Team Preview: You do not have a Pokémon in slot {(pos if pos is not None else -1) + 1}")
            if positions.index(pos) != index:
                return self.emitChoiceError(
                    f"Can't choose for Team Preview: The Pokémon in slot {pos + 1} can only switch in once")
        for index, pos in enumerate(positions):
            self.choice.switchIns.add(pos)
            self.choice.actions.append(Obj(choice='team', index=index, pokemon=self.pokemon[pos], priority=-index))
        return True

    def chooseShift(self):
        return self.emitChoiceError("Can't shift: You can only shift to the center in triples")

    def clearChoice(self):
        forced_switches = 0
        forced_passes = 0
        if self.battle.requestState == 'switch':
            can_switch_out = len([p for p in self.active if p and p.switchFlag])
            can_switch_in = len([p for p in self.pokemon[len(self.active):] if p and not p.fainted])
            forced_switches = min(can_switch_out, can_switch_in)
            forced_passes = can_switch_out - forced_switches
        self.choice = self._new_choice(forced_switches, forced_passes)

    def commitChoices(self):
        self.battle.queue.addChoice(self.choice.actions)

    def choose(self, input_: str) -> bool:
        if not self.requestState:
            return self.emitChoiceError("Can't do anything: The game is over" if self.battle.ended
                                        else "Can't do anything: It's not your turn")
        if self.choice.cantUndo:
            return self.emitChoiceError("Can't undo: A trapping/disabling effect would cause undo to leak information")
        self.clearChoice()
        choice_strings = [input_] if input_.startswith('team ') else input_.split(',')
        if len(choice_strings) > len(self.active):
            return self.emitChoiceError(
                f"Can't make choices: You sent choices for {len(choice_strings)} Pokémon, but this is a "
                f"{self.battle.gameType} game!")
        for choice_string in choice_strings:
            parts = choice_string.strip().split(' ', 1)
            choice_type = parts[0]
            data = parts[1].strip() if len(parts) > 1 else ''
            if choice_type == 'testfight':
                choice_type = 'move'
                data = 'testfight'
            if choice_type == 'move':
                target_loc = None
                event = ''
                conflict = False
                while True:
                    if _TARGET_LOC_RE.search(data) and to_id(data) != 'conversion2':
                        if target_loc is not None:
                            conflict = True
                            break
                        target_loc = int(data[-2:])
                        data = data[:-2].strip()
                    else:
                        matched = False
                        for suffix, ev in ((' mega', 'mega'), (' megax', 'megax'), (' megay', 'megay'),
                                           (' zmove', 'zmove'), (' ultra', 'ultra'), (' dynamax', 'dynamax'),
                                           (' gigantamax', 'dynamax'), (' max', 'dynamax'),
                                           (' terastal', 'terastallize'), (' terastallize', 'terastallize')):
                            if data.endswith(suffix):
                                if event:
                                    conflict = True
                                    break
                                event = ev
                                data = data[:-len(suffix)]
                                matched = True
                                break
                        if conflict or not matched:
                            break
                if conflict:
                    return self.emitChoiceError(f"Conflicting arguments for \"move\": {parts[1] if len(parts) > 1 else ''}")
                if not self.chooseMove(data, target_loc or 0, event):
                    return False
            elif choice_type == 'switch':
                if not self.chooseSwitch(data):
                    return False
            elif choice_type == 'shift':
                if data:
                    return self.emitChoiceError(f"Unrecognized data after \"shift\": {data}")
                if not self.chooseShift():
                    return False
            elif choice_type == 'team':
                if not self.chooseTeam(data):
                    return False
            elif choice_type in ('pass', 'skip'):
                if data:
                    return self.emitChoiceError(f"Unrecognized data after \"pass\": {data}")
                if not self.choosePass():
                    return False
            elif choice_type in ('auto', 'default'):
                if not self.autoChoose():
                    return False
            else:
                self.emitChoiceError(f"Unrecognized choice: {choice_string}")
        return not self.choice.error

    def getChoiceIndex(self, isPass=False):
        index = len(self.choice.actions)
        if not isPass:
            rs = self.requestState
            if rs == 'move':
                while index < len(self.active) and (self.active[index].fainted or
                                                     self.active[index].volatiles.get('commanding')):
                    self.choosePass()
                    index += 1
            elif rs == 'switch':
                while index < len(self.active) and not self.active[index].switchFlag:
                    self.choosePass()
                    index += 1
        return index

    def choosePass(self):
        index = self.getChoiceIndex(True)
        if index >= len(self.active):
            return False
        pokemon = self.active[index]
        rs = self.requestState
        if rs == 'switch':
            if pokemon.switchFlag:
                if not self.choice.forcedPassesLeft:
                    return self.emitChoiceError(f"Can't pass: You need to switch in a Pokémon to replace {pokemon.name}")
                self.choice.forcedPassesLeft -= 1
        elif rs == 'move':
            if not pokemon.fainted and not pokemon.volatiles.get('commanding'):
                return self.emitChoiceError(f"Can't pass: Your {pokemon.name} must make a move (or switch)")
        else:
            return self.emitChoiceError("Can't pass: Not a move or switch request")
        self.choice.actions.append(Obj(choice='pass'))
        return True

    def autoChoose(self):
        rs = self.requestState
        if rs == 'teampreview':
            if not self.isChoiceDone():
                self.chooseTeam()
        elif rs == 'switch':
            i = 0
            while not self.isChoiceDone():
                if not self.chooseSwitch():
                    raise ValueError(f"autoChoose switch crashed: {self.choice.error}")
                i += 1
                if i > 10:
                    raise ValueError('autoChoose failed: infinite looping')
        elif rs == 'move':
            i = 0
            while not self.isChoiceDone():
                if not self.chooseMove():
                    raise ValueError(f"autoChoose crashed: {self.choice.error}")
                i += 1
                if i > 10:
                    raise ValueError('autoChoose failed: infinite looping')
        return True
