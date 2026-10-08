# pokechamp — 포켓몬 챔피언스 배틀 시뮬레이터 + 자기학습 AI

포켓몬 챔피언스(Pokémon Champions)의 배틀 규칙을 **그대로** 재현하는 파이썬 시뮬레이터와,
그 위에서 스스로 학습하는 AI입니다.

* **배틀 엔진** (`pokechamp/sim`): Pokémon Showdown의 `champions` 모드(실제 게임 규칙을 구현한
  레퍼런스 시뮬레이터)를 파이썬으로 1:1 이식했습니다. 포켓몬·기술·특성·도구·타입 상성·부가효과·
  메가진화·스탯 포인트(SP) 등 모든 데이터와 규칙이 포함되어 있습니다. 화면/모션 없이 결과와
  배틀 로그만 기록합니다.
* **팀 빌딩 AI** (`pokechamp/teambuilder`): 포켓몬·도구·기술·특성·성격·SP가 완전히 랜덤인 팀에서
  시작해 서로 대전시키며 진화(유전 알고리즘 + 성공한 유전자 학습)하여 강한 팀을 스스로 구성합니다.
* **배틀 AI** (`pokechamp/ai`): 휴리스틱 선생님에게서 행동 복제로 시작한 뒤 PPO 자기대전(self-play)
  리그로 강화학습하는 Transformer 정책/가치 신경망.
* **실전 어드바이저** (`python -m pokechamp advise`): 실제 게임에서 배틀 중인 상황(내 팀, 상대가
  보여준 정보, HP·상태이상·랭크·필드)을 입력하면, 상대의 숨겨진 정보를 샘플링해 시뮬레이터에서
  각 선택지를 미리 플레이해 보고 **승률이 높은 순서로 기술/교체를 추천**합니다.

## 정확성: Showdown과 로그 단위로 동일

엔진은 Showdown과 **같은 시드·같은 선택이면 배틀 로그 전체가 한 줄도 다르지 않도록** 검증했습니다.
랜덤 합법 팀 + 랜덤 선택으로 만든 싱글(BSS Reg M-C)·더블(VGC 2026 Reg M-C) 배틀을 실제 Showdown
(Node.js)과 이 엔진에서 각각 돌려 비교합니다.

* 3,000판 이상 비교, 최종 버전 기준 **1,000/1,000판 완전 일치**
  (데미지 난수, 급소, 부가효과 확률, 스피드 동률 처리, 메가진화, 교체, 날씨/필드 등 전부 포함)
* 테스트 스위트에 Showdown 레퍼런스 배틀 40판이 들어 있어 Node.js 없이도 `pytest`로 재검증됩니다.
* 챔피언스 고유 규칙: 레벨 50, IV 31 고정, SP(스탯당 최대 32 / 합계 66, HP = 종족값 + SP + 75,
  나머지 = (종족값 + SP + 20) × 성격 보정), PP 상한, 메가진화만 사용(테라스탈 없음), 마비 1/8로
  행동불능, 수면 2~3턴, 얼음 3턴 내 해동 등 Showdown champions 모드의 변경점이 모두 반영되어 있습니다.

데이터 출처: Showdown `champions` 모드 (커밋은 `pokechamp/data/meta.json` 참고). op.gg 등은 이
환경에서 접근할 수 없어, 같은 데이터를 담고 있는 Showdown 소스에서 내보냈습니다. 합법성(어떤
포켓몬이 어떤 기술/특성/도구를 쓸 수 있는지)도 Showdown의 TeamValidator로 계산했습니다
(`pokechamp/data/legality.json`).

## 설치

```bash
pip install numpy torch pytest      # torch는 AI 학습/추론에만 필요 (CPU로 충분)
python -m pytest -q                 # 엔진/AI 테스트 (약 20초)
```

## 빠른 시작

```bash
# 랜덤 합법 팀 출력 / 팀 검증 (Showdown 텍스트 형식, "SPs:" 줄 사용)
python -m pokechamp randomteam
python -m pokechamp validate examples/team_singles.txt

# AI끼리 대전 (random | maxdamage | heuristic | 모델 파일.pt)
python -m pokechamp battle --p1 heuristic --p2 maxdamage -n 200
python -m pokechamp battle --p1 models/battle_singles.pt --p2 heuristic -n 200
python -m pokechamp battle --p1 heuristic --p2 random -n 5 --log-dir logs/   # 배틀 로그 저장

# 더블 배틀
python -m pokechamp battle --format gen9championsvgc2026regmc --p1 heuristic --p2 maxdamage
```

포맷: `gen9championsbssregmc` (싱글, 6마리 중 3마리 선출), `gen9championsvgc2026regmc` (더블,
4마리 선출, 오픈 팀시트), `gen9championsou`.

## 실전 어드바이저 (실제 게임 중 최선의 선택 추천)

배틀 상황을 JSON으로 적어서 넘기면 됩니다. 예시: [`examples/advisor_state.json`](examples/advisor_state.json)

```bash
python -m pokechamp advise examples/advisor_state.json
```

```
 #    win%     ±  policy  action
 1    64.5   5.9   ...    Garchomp: switch to Kingambit
 2    60.6   5.3   ...    Garchomp: switch to Rotom-Wash
 3    51.9   2.7   ...    Garchomp: Earthquake
```

입력 항목 (모르는 정보는 비워두면 됩니다):

| 위치 | 내용 |
|---|---|
| `format`, `turn` | 포맷, 현재 턴 |
| `me.team` | 내 팀 6마리 전체 (Showdown 텍스트 형식 또는 세트 목록) |
| `me.brought`, `me.active` | 선출한 포켓몬, 현재 필드에 나와 있는 포켓몬 |
| `me.pokemon.<이름>` | `hp`("72%" 또는 실수치), `status`, `boosts`, `item`(소모 시 `""`), `mega`, `fainted`, `pp`, `locked_move`(구애 고정), `volatiles`(대타출동 등), `sleep_turns`, `toxic_turns` |
| `foe.team` | 팀 프리뷰에서 본 상대 6마리 |
| `foe.brought`, `foe.active` | 지금까지 나온 상대 포켓몬, 현재 상대 필드 |
| `foe.pokemon.<이름>` | 보인 정보만: `hp`(%), `status`, `boosts`, `item`, `ability`, `moves`, `mega` |
| `me.side` / `foe.side` | `reflect`, `lightscreen`, `tailwind`(남은 턴), `stealthrock`, `spikes`(층수) 등 |
| `field` | `weather`(sun/rain/sand/snow) + `weather_turns`, `terrain` + `terrain_turns`, `trickroom` 등 |

동작 방식:
1. 입력으로 플레이어 시점의 상태를 만들고 정책망이 각 선택지의 사전확률과 승률을 냅니다.
2. 상대의 숨겨진 정보(안 보인 기술·도구·특성·SP, 아직 안 나온 선출 포켓몬)를 사전분포에서
   샘플링합니다. 팀 진화 결과(`models/teams_singles.json`)가 있으면 거기 나온 세트를 우선 사용합니다.
3. 샘플마다 시뮬레이터에 실제 배틀을 재구성하고, 내 선택지 각각에 대해 상대의 예상 대응과 함께
   몇 턴을 미리 플레이한 뒤 가치망으로 승률을 평가합니다.
4. 샘플 평균 승률 순으로 추천합니다. `--samples`(샘플 수)와 `--depth`(미리 보는 턴 수)를
   늘리면 더 정확하지만 느려집니다.

## 상대의 성격·SP·기술을 모르는 상황에서의 AI

실제 대전에서는 상대의 성격, 스탯 포인트(SP) 배분, 아직 쓰지 않은 기술, 도구를 모릅니다.
이 AI는 처음부터 **그 조건 그대로** 만들어져 있습니다.

1. **공정한 정보만 사용**: 배틀 AI(휴리스틱·신경망·탐색)는 플레이어 시점(`BattleView`)만 봅니다.
   상대 기술은 사용해서 드러난 것만, HP는 퍼센트만, 도구/특성은 발동해서 보인 것만 압니다.
2. **숨겨진 스탯 추론** (`pokechamp/ai/inference.py`): 상대 포켓몬마다 약 1만 개의 가설
   (SP 배분 × 성격 × 구애스카프 여부)에 대한 확률분포를 유지하고, 배틀 중 관측으로 베이즈 갱신합니다.
   * 같은 우선도에서 누가 먼저 움직였는지 → 스피드 범위 (스카프 여부 포함)
   * 내 공격으로 상대 HP가 몇 % 줄었는지 → 상대 HP·방어/특방
   * 상대 공격으로 내 HP가 정확히 얼마 줄었는지 → 상대 공격/특공 (구애 도구 가능성 포함)
   * 모르는 특성/도구 때문에 어긋나는 관측도 분포가 붕괴하지 않도록 작은 하한 확률을 둡니다.
   추론 결과는 데미지 계산, "상대가 나보다 빠를 확률", 신경망 입력(확신도, 스카프 확률, 스피드
   범위)에 바로 쓰입니다. 랜덤 팀 배틀에서 평균 6~7번의 관측 후 상대 실능치 추정 오차가
   15~25% 줄었습니다.
3. **안 보인 기술 추론**: 상대가 지금까지 고른 행동도 정보입니다. 더 강한 공격이 있었다면
   보통 그걸 썼을 것이므로, 그런 기술을 가진 후보 세트의 확률을 낮춥니다.
4. **결정화 탐색** (어드바이저/탐색 AI): 위 사후분포와 메타 사전분포(진화한 팀들의 세트)에서
   상대의 숨은 정보를 여러 번 샘플링하고, 각 경우를 시뮬레이터로 끝까지(또는 몇 턴) 플레이해
   **모든 가능성에 대한 평균 승률**이 가장 높은 행동을 고릅니다. 특정 추측에 과신하지 않습니다.
5. **다양한 상대로 학습**: 학습 상대 팀은 성격·SP·기술·도구가 완전히 랜덤인 팀과 진화한 팀이
   섞여 있어서, 정책이 특정 배분을 가정하지 않고 관측에 반응하도록 학습됩니다.

어드바이저 입력에 관측 힌트를 넣을 수도 있습니다: 상대 포켓몬 항목에
`"faster_than": ["Rotom-Wash"]`(같은 우선도에서 내 로토무보다 먼저 행동) /
`"slower_than": [...]`를 적으면 스피드 분포가 갱신되어 샘플링에 반영됩니다.

## 학습

```bash
# 1) 배틀 AI: 휴리스틱 행동 복제 -> PPO 자기대전 (CPU 코어 수에 맞춰 --workers)
python -m pokechamp train --out runs/singles --bc-games 3000 --iters 300

# 2) 팀 빌딩 AI: 완전 랜덤 팀에서 시작해 진화 (적합도 = 같은 AI로 싸운 승률)
python -m pokechamp evolve --out runs/teams --generations 100 --agent runs/singles/best.pt

# 3) 공진화: 진화한 팀들로 배틀 AI를 다시 학습 (랜덤 팀과 섞어서)
python -m pokechamp train --out runs/singles2 --init runs/singles/best.pt \
       --team-pool runs/teams/population.json --iters 300

# 1~3을 자동으로 반복 (팀 진화 <-> 배틀 AI 학습)
python -m pokechamp coevolve --out runs/coevolve --rounds 5 --generations 30 --iters 100
```

* 행동 복제 → PPO: 상대는 현재 정책(자기대전), 과거 스냅샷, 휴리스틱 AI를 섞은 리그입니다.
  보상은 승패(+1/−1)에 HP 차이 기반 포텐셜 셰이핑을 더한 것(최적 정책은 바뀌지 않음)입니다.
  `runs/<이름>/log.jsonl`에 반복마다 손실·상대별 승률이, 주기적으로 휴리스틱 상대 승률 평가가 기록됩니다.
* 팀 진화: 개체군 팀들이 서로, 그리고 명예의 전당(역대 최고 팀)·새 랜덤 팀과 싸웁니다. 상위 팀은
  보존되고, 나머지는 돌연변이(기술·도구·특성·성격·SP 변경, 포켓몬 교체)와 교차로 채워집니다.
  이긴 팀에 있던 기술/도구/특성/포켓몬은 이후 더 자주 제안됩니다. 결과물:
  `population.json`(팀+적합도), `best_team.txt`(Showdown 형식), `usage.json`(학습된 사용률·승률).
* 모든 생성 팀은 팀 검증기(종 중복 금지, 도구 중복 금지, SP 제한, 습득 가능 기술 등)를 통과합니다.

## 구조

```
pokechamp/
  sim/            배틀 엔진 (Showdown champions 모드 이식: battle, side, pokemon, field, queue,
                  battle_actions, dex, prng, teams; data/ 에 기술·특성·도구·상태 핸들러)
  data/           Showdown에서 내보낸 데이터 (species, moves, abilities, items, typechart, learnsets,
                  legality, formats ...)
  env/            플레이어 시점 상태 추적(BattleView/LogTracker), 행동 공간, 배틀 실행기
  teambuilder/    랜덤 팀 생성, 팀 검증기, 팀 진화
  ai/             데미지 추정, 휴리스틱 AI, 특징 인코딩, 신경망, 학습(PPO), 평가, 어드바이저, 탐색 AI
  tools/          Showdown과의 차등 테스트(difftest), 리플레이
tools/            Node.js 스크립트: Showdown 데이터 내보내기, 레퍼런스 배틀 실행
tests/            pytest (Showdown 레퍼런스 배틀 재생 포함)
examples/         어드바이저 입력 예시, 예시 팀
models/           학습된 모델 / 진화한 팀 (있을 경우)
```

### 데이터 갱신 / 엔진 재검증 (선택)

Showdown이 업데이트되면 데이터를 다시 내보내고 엔진을 비교할 수 있습니다 (Node.js 필요).

```bash
git clone https://github.com/smogon/pokemon-showdown ps && (cd ps && npm install && node build)
node tools/export_showdown_data.js ps pokechamp/data
python -m pokechamp difftest --showdown ps -n 300 --seed 1
```

## 한계

* "실제 게임과 동일"의 기준은 Showdown champions 모드입니다. 닌텐도/포켓몬의 공식 게임에서
  Showdown에 아직 반영되지 않은 차이가 있다면 그 부분은 다를 수 있습니다.
* 일루전(조로아크)처럼 플레이어가 원래 알 수 없는 정보는 AI도 모릅니다(공정한 정보 조건).
* 어드바이저의 정확도는 상대 세트 사전분포와 정책/가치망 품질에 좌우됩니다. 학습과 팀 진화를
  오래 돌릴수록 좋아집니다(이 저장소의 모델은 CPU 4코어에서 짧게 학습한 출발점입니다).
