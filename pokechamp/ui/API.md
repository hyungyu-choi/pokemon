# Battle assistant UI — API contract

Server: `python -m pokechamp ui --port 8765` (standard library HTTP server, `pokechamp/ui/server.py`;
logic in `pokechamp/ui/service.py`; start-up checks and console messages in `pokechamp/ui/console.py`; launchers
`run_ui.bat` / `run_ui.sh` at the project root). The page is `pokechamp/ui/static/index.html`; other static files
are served from `/static/<file>` (GET and HEAD; only plain file names directly inside `static/`, with fixed
Content-Types: `.html` text/html, `.js` text/javascript, `.css` text/css, `.json` application/json, all `charset=utf-8`).
All JSON is UTF-8. POST bodies must be JSON objects (at most 2 MB).
Errors: `{"error": "..."}` with HTTP 400 (malformed request, unknown name, impossible situation - the message says
which), 413 (body too large) or 500. Format: singles only, `gen9championsbssregmc` (bring 6, pick 3, level 50).

UI files (plain HTML/CSS/JS, no build step, no external resources): `static/index.html`, `static/app.js`
(screens, state, <state> builder), `static/combobox.js` (searchable dropdown: Korean / English / initial-consonant
search), `static/style.css`. The page keeps its state in `localStorage` (key `pokechamp-ui-v1`; the calculation
preset is `preset`: `fast` | `normal` | `precise` - older saves with `samples` 6 / 12 / 24 are mapped to these).
Field effects of recorded moves / abilities (Trick Room, weather, terrain, screens, Tailwind, hazards, Defog...) are
applied to the <state> by the page itself; the server only receives the resulting `field` / `side` counters.

AI requests (`/api/preview`, `/api/advise`, `/api/advise_switch`) run one at a time: each new AI request (or
`POST /api/cancel`) makes the running one stop after its current sample / simulated game; the stopped request still
answers, with what it has so far and `"cancelled": true`. The UI's "취소" button aborts its fetch and sends
`POST /api/cancel`, so the server stops computing right away; the next request does not wait for the cancelled one.

## GET /api/status
`{"format": "gen9championsbssregmc", "model": "battle_singles.pt" | null, "model_error": null | "<why the model could not be loaded>", "library": "teams_singles.json" | null, "korean_names": bool}`

`model` is null when the server runs without the neural network (`--no-model`, no `models/battle_singles.pt`, or the
checkpoint could not be loaded - then `model_error` says why); the AI then uses the heuristic agent.

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
Body `{"my_team": [<set> x6], "foe": ["<species>" x6, all different], "sims": 12 (1..48)}` →
```
{ "options": [ { "order": [0, 2, 5],            // indices into my_team; order[0] = lead
                 "lead": "Garchomp", "names": [...3], "names_ko": [...3],
                 "win_rate": 0.83, "games": 12, "policy": 0.11 | null, "matchup": 7.2, "prior_score": 0.9 } ],
  "considered": 60, "elapsed": 14.2, "cancelled": false }
```
Sorted best first. When cancelled, `options` holds the combinations simulated so far (`games` may be < `sims`).

## POST /api/advise  (12 samples, 6 legal actions: ~5 s at depth 1, ~8 s at depth 2 on a typical PC)
Body `{"state": <state>, "samples": 12 (1..64), "depth": 2 (1..3)}` →
`samples` = determinizations (the opponent's hidden items / abilities / moves / stats sampled each time);
`depth` = turns simulated after each candidate action before the value network scores the position. Time grows
with samples × legal actions (and with depth). The UI's presets send 빠름 `{"samples": 6, "depth": 1}` (~2-5 s),
보통 `{"samples": 12, "depth": 1}` (~4-10 s) and 정밀 `{"samples": 24, "depth": 2}` (~10-30 s).
```
{ "recommendations": [ { "label": "Garchomp: switch to Kingambit", "label_ko": "교체 → 대도각참",
                         "win_rate": 0.63, "stderr": 0.03, "policy": 0.96, "samples": 12,
                         "option": [33], "meaning": [["switch", "Kingambit"]] | [["move", "Earthquake"]] } ],
  "value": 0.73 | null,                       // value network's instant estimate for the position (no simulation;
                                              //  rough - the UI shows it below the simulated win rates)
  "foe_replies": [ {"label": "Dragonite: Extreme Speed", "prob": 0.4} ],   // opponent's likely actions
  "beliefs": { "Dragonite": { "mean_stats": {...}, "speed_10_90": [110, 164], "p_choice_scarf": 0.21,
                              "observations": 1, "max_hp_range": [166, 198] } },
  "samples": 12, "elapsed": 4.3, "cancelled": false, "log": [] }
```
`recommendations` is sorted by `win_rate` (best first). Labels containing "Mega Evolve + " mean
"Mega Evolve and use this move" (`label_ko` starts with "메가진화 + "). `cancelled: true` = stopped early by
`/api/cancel` or a newer request (fewer samples than asked). An empty `recommendations` list with lines in `log`
means the situation could not be rebuilt in the simulator.

The state is checked first (400 with the reason): every move / item / ability name must exist, our
`locked_move` must be one of that Pokemon's moves, and the opponent's revealed `moves` must be learnable by its
species in this format.

## POST /api/advise_switch  (our active Pokemon fainted: which one to send in)
Body `{"state": <state with the fainted Pokemon marked "fainted": true>, "samples": 8 (1..64), "depth": 2 (1..3)}` →
(the UI sends `{"samples": 6, "depth": 1}`: one advise run per remaining Pokemon, ~3-10 s for two)
`{"options": [ {"switch_to": "Rotom-Wash", "switch_to_ko": "", "win_rate": 0.62, "best_next_action": "...", "value": 0.6} ], "elapsed": 10.2, "cancelled": false}`
(one advise run per remaining Pokemon, all in one AI request; when cancelled only fully evaluated candidates are listed).

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
        "item": "Life Orb" | "" (known: none / consumed) -- omit or null = unknown,
        "ability": "Multiscale" -- omit or null = unknown,
        "moves": ["Extreme Speed", "Dragon Dance"],          // revealed moves
        "mega": false,                   // Mega Evolved; give its Mega Stone as "item" to say which Mega
        "fainted": false, "volatiles": [],
        "faster_than": ["Rotom-Wash"],   // it moved before these Pokemon of ours (same priority; compared with
        "slower_than": ["Garchomp"]      //  plain Speed stats, base formes, our Choice Scarf x1.5)
      }
    },
    "side": {...}, "mega_used": false
  },
  "field": {"weather": "" | "sunnyday" | "raindance" | "sandstorm" | "snowscape", "weather_turns": 3,
            "terrain": "" | "electricterrain" | ..., "terrain_turns": 4,
            "trickroom": 2, "gravity": 0, "magicroom": 0, "wonderroom": 0}       // turns left, 0 = off
}
```

## POST /api/cancel
Body `{}` → `{"cancelled_generation": n}`. The running `/api/advise`, `/api/advise_switch` or `/api/preview` stops
after its current sample / game and returns its partial result with `"cancelled": true`. Starting a new AI request
also supersedes a running one.
