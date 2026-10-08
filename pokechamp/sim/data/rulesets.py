"""Battle-time handlers of the rules used by the Champions formats.

Team-validation hooks (``onValidateTeam`` etc.) are implemented separately in
``pokechamp.validator``.
"""
from __future__ import annotations

import re


class teampreview:
    NAME = 'Team Preview'

    def onBegin(self):
        if self.ruleTable.has('teratypepreview'):
            self.add('rule', 'Tera Type Preview: Tera Types are shown at Team Preview')

    def onTeamPreview(self):
        # champions override of the base Team Preview rule
        self.add('clearpoke')
        for pokemon in self.getAllPokemon():
            details = re.sub(r'(Xerneas|Zacian|Zamazenta)(-[a-zA-Z?-]+)?', r'\1-*', pokemon.details)
            self.add('poke', pokemon.side.id, details, '')
        self.makeRequest('teampreview')


class speciesclause:
    NAME = 'Species Clause'

    def onBegin(self):
        self.add('rule', 'Species Clause: Limit one of each Pokémon')


class itemclause:
    NAME = 'Item Clause'

    def onBegin(self):
        self.add('rule', f"Item Clause: Limit {self.ruleTable.valueRules.get('itemclause') or 1} of each item")


class cancelmod:
    NAME = 'Cancel Mod'

    def onBegin(self):
        self.supportCancel = True


class openteamsheets:
    NAME = 'Open Team Sheets'

    def onTeamPreview(self):
        msg = ('uhtml|otsrequest|<button name="send" value="/acceptopenteamsheets" class="button" '
               'style="margin-right: 10px;"><strong>Accept Open Team Sheets</strong></button><button name="send" '
               'value="/rejectopenteamsheets" class="button" style="margin-top: 10px"><strong>Deny Open Team '
               'Sheets</strong></button>')
        for side in self.sides:
            self.addSplit(side.id, [msg])

    def onBattleStart(self):
        for side in self.sides:
            self.addSplit(side.id, ['uhtmlchange|otsrequest|'])


class endlessbattleclause:
    NAME = 'Endless Battle Clause'

    def onBegin(self):
        self.add('rule', 'Endless Battle Clause: Forcing endless battles is banned')


class sleepmovesclause:
    NAME = 'Sleep Moves Clause'

    def onBegin(self):
        self.add('rule', 'Sleep Moves Clause: Sleep-inducing moves are banned')


class ohkoclause:
    NAME = 'OHKO Clause'

    def onBegin(self):
        self.add('rule', 'OHKO Clause: OHKO moves are banned')


class evasionclause:
    NAME = 'Evasion Clause'

    def onBegin(self):
        self.add('rule', 'Evasion Clause: Evasion abilities, items, and moves are banned')


class evasionabilitiesclause:
    NAME = 'Evasion Abilities Clause'

    def onBegin(self):
        self.add('rule', 'Evasion Abilities Clause: Evasion abilities are banned')


class evasionitemsclause:
    NAME = 'Evasion Items Clause'

    def onBegin(self):
        self.add('rule', 'Evasion Items Clause: Evasion items are banned')


class evasionmovesclause:
    NAME = 'Evasion Moves Clause'

    def onBegin(self):
        self.add('rule', 'Evasion Moves Clause: Evasion moves are banned')
