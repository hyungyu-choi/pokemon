"""Port of Showdown's ``sim/battle-queue.ts``."""
from __future__ import annotations

from .js import Obj

_ORDERS = {
    'team': 1,
    'start': 2,
    'instaswitch': 3,
    'beforeTurn': 4,
    'beforeTurnMove': 5,
    'revivalblessing': 6,
    'runSwitch': 101,
    'switch': 103,
    'megaEvo': 104,
    'megaEvoX': 104,
    'megaEvoY': 104,
    'runDynamax': 105,
    'terastallize': 106,
    'priorityChargeMove': 107,
    'shift': 200,
    'residual': 300,
}


class BattleQueue:
    def __init__(self, battle):
        self.battle = battle
        self.list: list[Obj] = []

    def shift(self):
        return self.list.pop(0) if self.list else None

    def peek(self, end=False):
        if not self.list:
            return None
        return self.list[-1] if end else self.list[0]

    def push(self, action):
        self.list.append(action)
        return len(self.list)

    def unshift(self, action):
        self.list.insert(0, action)
        return len(self.list)

    def __iter__(self):
        return iter(self.list)

    def entries(self):
        return enumerate(self.list)

    def resolveAction(self, action: Obj, midTurn=False):
        battle = self.battle
        if not action:
            raise ValueError('Action not passed to resolveAction')
        if action.choice == 'pass':
            return []
        actions = [action]
        if not action.side and action.pokemon:
            action.side = action.pokemon.side
        if not action.move and action.moveid:
            action.move = battle.dex.getActiveMove(action.moveid)
        if not action.order:
            if action.choice in _ORDERS:
                action.order = _ORDERS[action.choice]
            else:
                action.order = 200
                if action.choice not in ('move', 'event'):
                    raise ValueError(f"Unexpected orderless action {action.choice}")
        if not midTurn:
            if action.choice == 'move':
                if not action.maxMove and not action.zmove and action.move.beforeTurnCallback:
                    actions[0:0] = self.resolveAction(Obj(choice='beforeTurnMove', pokemon=action.pokemon,
                                                          move=action.move, targetLoc=action.targetLoc))
                if action.mega and not action.pokemon.isSkyDropped():
                    actions[0:0] = self.resolveAction(Obj(choice='megaEvo', pokemon=action.pokemon))
                if not action.maxMove and not action.zmove and action.move.priorityChargeCallback:
                    actions[0:0] = self.resolveAction(Obj(choice='priorityChargeMove', pokemon=action.pokemon,
                                                          move=action.move))
                action.fractionalPriority = battle.runEvent('FractionalPriority', action.pokemon, None,
                                                            action.move, 0)
            elif action.choice in ('switch', 'instaswitch'):
                if isinstance(action.pokemon.switchFlag, str):
                    action.sourceEffect = battle.dex.moves.get(action.pokemon.switchFlag)
                action.pokemon.switchFlag = False
        if action.move:
            target = None
            action.move = battle.dex.getActiveMove(action.move)
            if action.move.id == 'curse' and not action.pokemon.hasType('Ghost'):
                action.move.target = 'self'
            if not action.targetLoc:
                target = battle.getRandomTarget(action.pokemon, action.move)
                if target:
                    action.targetLoc = action.pokemon.getLocOf(target)
            action.originalTarget = action.pokemon.getAtLoc(action.targetLoc)
        battle.getActionSpeed(action)
        return actions

    def prioritizeAction(self, action, sourceEffect=None):
        for i, cur in enumerate(self.list):
            if cur is action:
                self.list.pop(i)
                break
        action.sourceEffect = sourceEffect
        action.order = 3
        self.list.insert(0, action)

    def changeAction(self, pokemon, action: Obj):
        self.cancelAction(pokemon)
        if not action.pokemon:
            action.pokemon = pokemon
        self.insertChoice(action)

    def addChoice(self, choices):
        if not isinstance(choices, list):
            choices = [choices]
        for choice in choices:
            self.list.extend(self.resolveAction(choice))

    def willAct(self):
        for action in self.list:
            if action.choice in ('move', 'switch', 'instaswitch', 'shift'):
                return action
        return None

    def willMove(self, pokemon):
        if pokemon.fainted:
            return None
        for action in self.list:
            if action.choice == 'move' and action.pokemon is pokemon:
                return action
        return None

    def cancelAction(self, pokemon):
        old_len = len(self.list)
        self.list[:] = [a for a in self.list if a.pokemon is not pokemon]
        return len(self.list) != old_len

    def cancelMove(self, pokemon):
        for i, action in enumerate(self.list):
            if action.choice == 'move' and action.pokemon is pokemon:
                self.list.pop(i)
                return True
        return False

    def willSwitch(self, pokemon):
        for action in self.list:
            if action.choice in ('switch', 'instaswitch') and action.pokemon is pokemon:
                return action
        return None

    def insertChoice(self, choices, midTurn=False):
        if isinstance(choices, list):
            for choice in choices:
                self.insertChoice(choice)
            return
        choice = choices
        if choice.pokemon:
            choice.pokemon.updateSpeed()
        actions = self.resolveAction(choice, midTurn)
        first_index = None
        last_index = None
        for i, cur in enumerate(self.list):
            compared = self.battle.comparePriority(actions[0], cur)
            if compared <= 0 and first_index is None:
                first_index = i
            if compared < 0:
                last_index = i
                break
        if first_index is None:
            self.list.extend(actions)
        else:
            if last_index is None:
                last_index = len(self.list)
            index = first_index if first_index == last_index else self.battle.random(first_index, last_index + 1)
            self.list[index:index] = actions

    def clear(self):
        self.list = []

    def sort(self):
        self.battle.speedSort(self.list)
        return self
