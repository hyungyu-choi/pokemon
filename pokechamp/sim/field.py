"""Port of Showdown's ``sim/field.ts`` (weather, terrain, pseudo-weather)."""
from __future__ import annotations

from .js import NULL, Obj, to_id


class Field:
    def __init__(self, battle):
        self.battle = battle
        self.id = ''
        self.weather = ''
        self.weatherState = battle.initEffectState(Obj(id=''))
        self.terrain = ''
        self.terrainState = battle.initEffectState(Obj(id=''))
        self.pseudoWeather: dict[str, Obj] = {}

    def __repr__(self):
        return '<Field>'

    def setWeather(self, status, source=None, sourceEffect=None):
        battle = self.battle
        status = battle.dex.conditions.get(status)
        if not sourceEffect and battle.effect:
            sourceEffect = battle.effect
        if not source and battle.event and battle.event.target:
            source = battle.event.target
        if source == 'debug':
            source = battle.sides[0].active[0]
        if self.weather == status.id:
            if sourceEffect and sourceEffect.effectType == 'Ability':
                return False
            else:
                return False
        if source:
            result = battle.runEvent('SetWeather', source, source, status)
            if not result:
                if result is False:
                    if sourceEffect and sourceEffect.get('weather'):
                        battle.add('-fail', source, sourceEffect, '[from] ' + self.weather)
                    elif sourceEffect and sourceEffect.effectType == 'Ability':
                        battle.add('-ability', source, sourceEffect, '[from] ' + self.weather, '[fail]')
                return NULL
        prev_weather = self.weather
        prev_state = self.weatherState
        self.weather = status.id
        self.weatherState = battle.initEffectState(Obj(id=status.id))
        if source:
            self.weatherState.source = source
            self.weatherState.sourceSlot = source.getSlot()
        if status.duration:
            self.weatherState.duration = status.duration
        if status.get('durationCallback'):
            if not source:
                raise ValueError('setting weather without a source')
            self.weatherState.duration = status.durationCallback(battle, source, source, sourceEffect)
        if not battle.singleEvent('FieldStart', status, self.weatherState, self, source, sourceEffect):
            self.weather = prev_weather
            self.weatherState = prev_state
            return False
        battle.eachEvent('WeatherChange', sourceEffect)
        return True

    def clearWeather(self):
        if not self.weather:
            return False
        prev = self.getWeather()
        self.battle.singleEvent('FieldEnd', prev, self.weatherState, self)
        self.weather = ''
        self.battle.clearEffectState(self.weatherState)
        self.battle.eachEvent('WeatherChange')
        return True

    def effectiveWeather(self):
        if self.suppressingWeather():
            return ''
        return self.weather

    def suppressingWeather(self):
        for side in self.battle.sides:
            for pokemon in side.active:
                if (pokemon and not pokemon.fainted and not pokemon.ignoringAbility() and
                        pokemon.getAbility().get('suppressWeather') and not pokemon.abilityState.ending):
                    return True
        return False

    def isWeather(self, weather):
        ours = self.effectiveWeather()
        if not isinstance(weather, list):
            return ours == to_id(weather)
        return ours in [to_id(w) for w in weather]

    def getWeather(self):
        return self.battle.dex.conditions.getByID(self.weather)

    def setTerrain(self, status, source=None, sourceEffect=None):
        battle = self.battle
        status = battle.dex.conditions.get(status)
        if not sourceEffect and battle.effect:
            sourceEffect = battle.effect
        if not source and battle.event and battle.event.target:
            source = battle.event.target
        if source == 'debug':
            source = battle.sides[0].active[0]
        if not source:
            raise ValueError('setting terrain without a source')
        if self.terrain == status.id:
            return False
        prev = self.terrain
        prev_state = self.terrainState
        self.terrain = status.id
        self.terrainState = battle.initEffectState(Obj(id=status.id, source=source, sourceSlot=source.getSlot(),
                                                       duration=status.duration))
        if status.get('durationCallback'):
            self.terrainState.duration = status.durationCallback(battle, source, source, sourceEffect)
        if not battle.singleEvent('FieldStart', status, self.terrainState, self, source, sourceEffect):
            self.terrain = prev
            self.terrainState = prev_state
            return False
        battle.eachEvent('TerrainChange', sourceEffect)
        return True

    def clearTerrain(self):
        if not self.terrain:
            return False
        prev = self.getTerrain()
        self.battle.singleEvent('FieldEnd', prev, self.terrainState, self)
        self.terrain = ''
        self.battle.clearEffectState(self.terrainState)
        self.battle.eachEvent('TerrainChange')
        return True

    def effectiveTerrain(self, target=None):
        if self.battle.event and not target:
            target = self.battle.event.target
        return self.terrain if self.battle.runEvent('TryTerrain', target) else ''

    def isTerrain(self, terrain, target=None):
        ours = self.effectiveTerrain(target)
        if not isinstance(terrain, list):
            return ours == to_id(terrain)
        return ours in [to_id(t) for t in terrain]

    def getTerrain(self):
        return self.battle.dex.conditions.getByID(self.terrain)

    def addPseudoWeather(self, status, source=None, sourceEffect=None):
        battle = self.battle
        if not source and battle.event and battle.event.target:
            source = battle.event.target
        if source == 'debug':
            source = battle.sides[0].active[0]
        status = battle.dex.conditions.get(status)
        state = self.pseudoWeather.get(status.id)
        if state is not None:
            if not status.get('onFieldRestart'):
                return False
            return battle.singleEvent('FieldRestart', status, state, self, source, sourceEffect)
        state = battle.initEffectState(Obj(id=status.id, source=source,
                                           sourceSlot=source.getSlot() if source and hasattr(source, 'getSlot') else None,
                                           duration=status.duration))
        self.pseudoWeather[status.id] = state
        if status.get('durationCallback'):
            if not source:
                raise ValueError('setting fieldcond without a source')
            state.duration = status.durationCallback(battle, source, source, sourceEffect)
        if not battle.singleEvent('FieldStart', status, state, self, source, sourceEffect):
            self.pseudoWeather.pop(status.id, None)
            return False
        battle.runEvent('PseudoWeatherChange', source, source, status)
        return True

    def getPseudoWeather(self, status):
        status = self.battle.dex.conditions.get(status)
        return status if status.id in self.pseudoWeather else None

    def removePseudoWeather(self, status):
        status = self.battle.dex.conditions.get(status)
        state = self.pseudoWeather.get(status.id)
        if state is None:
            return False
        self.battle.singleEvent('FieldEnd', status, state, self)
        self.pseudoWeather.pop(status.id, None)
        return True
