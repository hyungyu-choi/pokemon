# Battle assistant UI — API contract

Server: `python -m pokechamp ui --port 8765` (standard library HTTP server, `pokechamp/ui/server.py`;
logic in `pokechamp/ui/service.py`). The page is `pokechamp/ui/static/index.html`; other static files
are served from `/static/<file>`. All JSON is UTF-8. Errors: `{"error": "..."}` with HTTP 400/500.
Format: singles only, `gen9championsbssregmc` (bring 6, pick 3, level 50).

## GET /api/status
`{"format": "gen9championsbssregmc", "model": "battle_singles.pt" | null, "library": "teams_singles.json" | null, "korean_names": bool}`

## GET /api/data  (~430 KB, cache it in the page)
```
{
  "format": "gen9championsbssregmc", "picked": 3,
  "species": [ {                          // legal species only, sorted by English name
      "name": "Garchomp",                 // canonical Showdown name - use this in every request
      "id": "garchomp", "ko": "한카리아스" | "", "num": 445,
      "types": ["Dragon", "Ground"], "baseStats": {"hp":108,"atk":130,"def":95,"spa":80,"spd":85,"spe":102},
      "abilities": ["Sand Veil", "Rough Skin"],   // legal abilities
      "moves": ["Aerial Ace", ...],               // legal (learnable) moves, English names
      "megas": [ {"stone": "Garchompite", "stone_ko": "", "forme": "Garchomp-Mega", "forme_ko": "",
                  "types": [...], "ability": "Sand Force", "baseStats": {...}} ],
      "weightkg": 95 } ],
  "moves": { "Earthquake": {"ko": "지진" | "", "type": "Ground", "category": "Physical"|"Special"|"Status",
             "basePower": 100, "accuracy": 100 (0 = never misses), "priority": 0, "pp": 10, "target": "allAdjacent"} },
  "items": [ {"name": "Choice Scarf", "ko": "구애스카프" | "", "megaStone": false, "megaFor": []} ],
  "abilities": { "Rough Skin": "까칠한피부" | "" },
  "natures": [ {"name": "Jolly", "ko": "명랑" | "", "plus": "spe", "minus": "spa"} ],   // plus/minus "" = neutral
  "types": { "Fire": "불꽃" | "" },
  "statuses": [ {"id": "", "ko": "없음"}, {"id": "brn", "ko": "화상"}, ... par slp frz psn tox ],
  "weathers": [ {"id": "", "ko": "없음"}, {"id": "sunnyday", ...}, raindance, sandstorm, snowscape ],
  "terrains": [ {"id": "", ...}, electricterrain, grassyterrain, mistyterrain, psychicterrain ],
  "pseudo":   [ {"id": "trickroom", "ko": "트릭룸"}, gravity, magicroom, wonderroom ],
  "side_conditions": [ {"id": "reflect", "ko": "리플렉터", "layers": 1, "timed": true}, lightscreen, auroraveil,
                       tailwind, safeguard (timed), stealthrock, stickyweb (layers 1), spikes (layers 3),
                       toxicspikes (layers 2) ],
  "volatiles": [ {"id": "substitute", "ko": "대타출동"}, confusion, leechseed, taunt, encore, yawn, ... ],
  "boosts": [ {"id": "atk", "ko": "공격"}, def, spa, spd, spe, accuracy, evasion ]
}
```
`ko` is an empty string when no Korean name is known — show the English name then.

## GET /api/recommended_teams
```
{ "teams": [ { "rank": 1, "source": "명예의 전당" | "개체군", "fitness": 0.79 | null,
               "sets": [ <set> x6 ], "text": "<Showdown export>", "species": [...], "species_ko": [...],
               "stats": [ {"hp":..,"atk":..,"def":..,"spa":..,"spd":..,"spe":..} x6 ] } ],
  "top_species": [ {"species": "Primarina", "ko": "", "win_rate": 0.59, "games": 251.7} ],
  "generation": 40 }
```
`<set>` = `{"species","name","item","ability","moves":[...],"nature","evs":{"hp","atk","def","spa","spd","spe"},"level":50}`
(`evs` are the Champions Stat Points: each 0..32, total <= 66).

## POST /api/team/parse
Body `{"text": "<Showdown export with SPs: lines>"}` or `{"sets": [<set>...]}` →
`{"sets": [<set> + {"stats": {...}, "species_ko": ""}], "problems": ["..."], "text": "<normalised export>"}`.
`problems` empty = legal (species/item clause, learnsets, SP limits, 6 Pokemon).

## POST /api/preview  (takes ~10-20 s)
Body `{"my_team": [<set> x6], "foe": ["<species>" x6], "sims": 12}` →
```
{ "options": [ { "order": [0, 2, 5],            // indices into my_team; order[0] = lead
                 "lead": "Garchomp", "names": [...3], "names_ko": [...3],
                 "win_rate": 0.83, "games": 12, "policy": 0.11 | null, "matchup": 7.2, "prior_score": 0.9 } ],
  "considered": 60, "elapsed": 14.2 }
```
Sorted best first.

## POST /api/advise  (takes ~3-15 s depending on samples)
Body `{"state": <state>, "samples": 12, "depth": 2}` →
```
{ "recommendations": [ { "label": "Garchomp: switch to Kingambit", "label_ko": "교체 → 킬가르도",
                         "win_rate": 0.63, "stderr": 0.03, "policy": 0.96, "samples": 12,
                         "option": [33], "meaning": [["switch", "Kingambit"]] | [["move", "Earthquake"]] } ],
  "value": 0.73 | null,                       // value network's win probability for the position
  "foe_replies": [ {"label": "Dragonite: Extreme Speed", "prob": 0.4} ],   // opponent's likely actions
  "beliefs": { "Dragonite": { "mean_stats": {...}, "speed_10_90": [110, 164], "p_choice_scarf": 0.21,
                              "observations": 1, "max_hp_range": [166, 198] } },
  "samples": 12, "elapsed": 4.3, "log": [] }
```
`recommendations` is sorted by `win_rate` (best first). Labels containing "Mega Evolve + " mean
"Mega Evolve and use this move" (`label_ko` starts with "메가진화 + ").

## POST /api/advise_switch  (our active Pokemon fainted: which one to send in)
Body `{"state": <state with the fainted Pokemon marked "fainted": true>, "samples": 8}` →
`{"options": [ {"switch_to": "Rotom-Wash", "switch_to_ko": "", "win_rate": 0.62, "best_next_action": "...", "value": 0.6} ], "elapsed": 10.2}`

## <state> — the battle situation (same format as `examples/advisor_state.json`)
Use canonical English species/move/item/ability names from `/api/data` everywhere.
```
{
  "format": "gen9championsbssregmc",
  "turn": 3,
  "me": {
    "team": [<set> x6] | "<Showdown export text>",
    "brought": ["Garchomp", "Rotom-Wash", "Kingambit"],      // the 3 we picked (lead first is fine)
    "active": ["Garchomp"],                                  // exactly one, must not be fainted
    "pokemon": {                                             // per brought Pokemon (omit = full HP, healthy)
      "Garchomp": {
        "hp": "120/185" (exact) | "64%",
        "status": "" | "brn" | "par" | "slp" | "frz" | "psn" | "tox",
        "boosts": {"atk": -1, "spe": 1},                     // active only; -6..6
        "item": "" (consumed / knocked off)  -- omit or null = still holding its set item,
        "mega": true,                                        // has Mega Evolved
        "fainted": true,
        "volatiles": ["substitute", "confusion", "leechseed", "taunt", "encore", "yawn", ...],
        "substitute_hp": 0.25,                               // fraction of max HP, if substitute
        "locked_move": "Earthquake",                         // Choice-locked into this move
        "sleep_turns": 1, "toxic_turns": 2,                  // turns already asleep / badly poisoned
        "fresh": true                                        // switched in this turn (Fake Out works)
      }
    },
    "side": {"reflect": 3, "lightscreen": 2, "auroraveil": 4, "tailwind": 2, "safeguard": 3,   // turns left
             "stealthrock": 1, "stickyweb": 1, "spikes": 2, "toxicspikes": 1},                 // layers
    "mega_used": false                                       // our side already Mega Evolved this battle
  },
  "foe": {
    "team": ["Dragonite", "Gholdengo", "Hippowdon", "Primarina", "Tyranitar", "Corviknight"],  // team preview
    "brought": ["Dragonite", "Tyranitar"],                   // opponent Pokemon seen so far
    "active": ["Dragonite"],
    "pokemon": {                                             // only what has been seen
      "Dragonite": {
        "hp": "54%", "status": "", "boosts": {"atk": 1},
        "item": "Choice Band" | "" (known: none / consumed) -- omit or null = unknown,
        "ability": "Multiscale" -- omit or null = unknown,
        "moves": ["Extreme Speed", "Dragon Dance"],          // revealed moves
        "mega": false, "fainted": false, "volatiles": [],
        "faster_than": ["Rotom-Wash"],   // it moved before these Pokemon of ours (same priority)
        "slower_than": ["Garchomp"]      // these Pokemon of ours moved before it
      }
    },
    "side": {...}, "mega_used": false
  },
  "field": {"weather": "" | "sunnyday" | "raindance" | "sandstorm" | "snowscape", "weather_turns": 3,
            "terrain": "" | "electricterrain" | ..., "terrain_turns": 4,
            "trickroom": 2, "gravity": 0, "magicroom": 0, "wonderroom": 0}       // turns left, 0 = off
}
```
