# 스포츠 데이터 분석 I — 월드컵 축구 데이터로 배우는 분석 방법 (집필 중)

축구 경기의 공개 데이터로 분석 방법을 가르치는 교재의 저장소다. 데이터에 기본적인 분석 방법을 하나씩 적용하고, 방법마다 수식과 가정, 추정 절차, 결과를 해석하는 법을 자세히 설명한다. I권은 2022년 카타르 월드컵의 64경기를 다룬다. 분석은 모두 파이썬으로 한다.

소개 대시보드: <https://pilsunchoi.github.io/sports-data-analysis/>

자매 교재: [재활 데이터 분석 I](https://github.com/pilsunchoi/rehab-data-analysis)

## 장

| 묶음 | 장 | 본문 |
|---|---|---|
| 데이터 | 1 | [데이터와 좌표 — 영상은 어떻게 표가 되는가](chapters/01-data-and-coordinates.md) |
| 회귀 | 1 | [로지스틱 회귀 — 어디서 찬 슛이 골이 되는가](chapters/02-shot-logistic.md) |
| 회귀 | 2 | [포아송 회귀 — 득점은 포아송 분포를 따르는가](chapters/03-goal-counts.md) |
| 다층 모형 | 1 | [축소 추정 — 패스 성공률이 높은 선수는 정말 패스를 잘하는가](chapters/04-player-shrinkage.md) |
| 생존 분석 | 1 | [사건까지의 시간 — 첫 골은 언제 나오는가](chapters/05-time-to-goal.md) |
| 연쇄 | 1 | [마르코프 연쇄 — 공이 이 구역에 있으면 골이 날 확률은 얼마인가](chapters/06-possession-markov.md) |
| 연쇄 | 2 | [패스 네트워크 — 공은 누구를 거쳐 가는가](chapters/07-pass-network.md) |
| 공간 | 1 | [커널 밀도와 2차원 평활 — 슛과 압박은 경기장의 어디에 몰리는가](chapters/08-spatial-smoothing.md) |
| 공간 | 2 | [보로노이 분할 — 패스를 받는 선수는 얼마나 비어 있는가](chapters/09-space-control.md) |
| 차원 축소 | 1 | [주성분 분석과 군집 — 선수의 역할은 몇 가지인가](chapters/10-pca-and-clustering.md) |
| 운동학 | 1 | [위치의 미분과 평활 — 위치 기록에서 속도를 어떻게 얻는가](chapters/11-speed-from-tracking.md) |
| 운동학 | 2 | [반복 측정의 선형 혼합모형 — 선수는 후반에 덜 뛰는가](chapters/12-within-match-fatigue.md) |

## 구조

```
sports-data-analysis/
├── README.md
├── chapters/
│   ├── _quarto.yml                        렌더 설정(렌더 뒤 MD 손질)
│   ├── 01-data-and-coordinates.qmd        데이터 1장 원본(파이썬 코드 포함)
│   ├── 01-data-and-coordinates.md         데이터 1장 렌더 결과(GitHub에서 읽는 본문)
│   ├── 01-data-and-coordinates_files/     데이터 1장 그림
│   ├── 02-shot-logistic.qmd               회귀 1장 원본
│   ├── 02-shot-logistic.md                회귀 1장 렌더 결과
│   ├── 02-shot-logistic_files/            회귀 1장 그림
│   ├── 03-goal-counts.qmd                 회귀 2장 원본
│   ├── 03-goal-counts.md                  회귀 2장 렌더 결과
│   ├── 03-goal-counts_files/              회귀 2장 그림
│   ├── 04-player-shrinkage.qmd            다층 모형 1장 원본
│   ├── 04-player-shrinkage.md             다층 모형 1장 렌더 결과
│   ├── 04-player-shrinkage_files/         다층 모형 1장 그림
│   ├── 05-time-to-goal.qmd                생존 분석 1장 원본
│   ├── 05-time-to-goal.md                 생존 분석 1장 렌더 결과
│   ├── 05-time-to-goal_files/             생존 분석 1장 그림
│   ├── 06-possession-markov.qmd           연쇄 1장 원본
│   ├── 06-possession-markov.md            연쇄 1장 렌더 결과
│   ├── 06-possession-markov_files/        연쇄 1장 그림
│   ├── 07-pass-network.qmd                연쇄 2장 원본
│   ├── 07-pass-network.md                 연쇄 2장 렌더 결과
│   ├── 07-pass-network_files/             연쇄 2장 그림
│   ├── 08-spatial-smoothing.qmd           공간 1장 원본
│   ├── 08-spatial-smoothing.md            공간 1장 렌더 결과
│   ├── 08-spatial-smoothing_files/        공간 1장 그림
│   ├── 09-space-control.qmd               공간 2장 원본
│   ├── 09-space-control.md                공간 2장 렌더 결과
│   ├── 09-space-control_files/            공간 2장 그림
│   ├── 10-pca-and-clustering.qmd          차원 축소 1장 원본
│   ├── 10-pca-and-clustering.md           차원 축소 1장 렌더 결과
│   ├── 10-pca-and-clustering_files/       차원 축소 1장 그림
│   ├── 11-speed-from-tracking.qmd         운동학 1장 원본
│   ├── 11-speed-from-tracking.md          운동학 1장 렌더 결과
│   ├── 11-speed-from-tracking_files/      운동학 1장 그림
│   ├── 12-within-match-fatigue.qmd        운동학 2장 원본
│   ├── 12-within-match-fatigue.md         운동학 2장 렌더 결과
│   └── 12-within-match-fatigue_files/     운동학 2장 그림
├── docs/
│   ├── index.html                         소개 대시보드(GitHub Pages)
│   └── img/                               대시보드의 그림(장의 그림을 복사)
├── sportsdata/                            모든 장이 불러 쓰는 파이썬 모듈
│   ├── statsbomb.py                       JSON 읽기, 경기·사건·명단·프리즈 프레임의 표
│   ├── features.py                        슛·패스·팀·경기·골 시각·선수별 변수의 표, 골대까지의 거리와 각도
│   ├── cluster.py                         주성분 분석, k-평균, 실루엣, 간격 통계량, 조정 랜드 지수
│   ├── idsse.py                           분데스리가 추적 데이터(XML) 읽기, 경기·명단·위치의 배열
│   ├── kinematics.py                      차분과 Savitzky-Golay 가중치, 속도와 가속도, 끊긴 궤적의 처리
│   ├── multilevel.py                      다층 모형의 추정(베타-이항, 감마-포아송, 무작위 절편 로지스틱, 선형 혼합모형)
│   ├── workload.py                        추적 데이터에서 선수의 구간별 이동 거리 표
│   ├── markov.py                          경기장의 구역, 점유의 연쇄, 흡수 연쇄의 계산
│   ├── network.py                         패스 네트워크의 인접 행렬, 중심성, 요약
│   ├── smooth.py                          2차원 커널 밀도 추정, 벌점 스플라인 로지스틱 회귀
│   ├── space.py                           보로노이 칸, 보이지 않는 곳까지의 거리, 받는 순간의 공간
│   ├── metrics.py                         예측 확률의 평가(AUC, 브라이어 점수)
│   ├── pitch.py                           경기장 그림과 그림의 공통 양식
│   └── report.py                          본문과 표에 쓰는 수의 표기, 숫자 뒤의 조사
├── scripts/
│   ├── fetch_statsbomb.py                 2022 월드컵의 사건과 프리즈 프레임을 받는다
│   ├── fetch_idsse.py                     분데스리가 7경기의 추적·사건 데이터를 받는다(운동학 묶음)
│   ├── data_status.py                     받은 데이터의 경기와 사건 수를 센다
│   └── postrender_md.py                   렌더된 MD를 GitHub용으로 손질(물결 이스케이프)
└── data/                                  공개하지 않는다
    ├── raw/statsbomb/                     받은 파일(경로와 이름을 바꾸지 않는다)
    ├── raw/idsse/                         분데스리가 7경기의 XML(이름을 바꾸지 않는다)
    └── cache/                             대회 전체를 쌓은 표(처음 읽을 때 만든다)
```

## 데이터 받기

```bash
python scripts/fetch_statsbomb.py
python scripts/data_status.py
```

첫 명령이 StatsBomb 공개 데이터 저장소에서 2022 남자 월드컵의 파일을 받아 `data/raw/statsbomb/`에 둔다. 이미 받은 파일은 다시 받지 않는다. 둘째 명령이 경기, 사건, 슛, 프리즈 프레임의 수를 낸다.

운동학 묶음의 두 장은 다른 데이터를 쓴다. 독일 분데스리가 2022/23 시즌 7경기의 추적·사건 데이터(Bassek et al., 2025, CC BY 4.0)이고 다음 명령으로 받는다. figshare에서 XML 21개(2.6GB)를 받아 `data/raw/idsse/`에 두고 MD5를 대조한다. 다른 묶음의 장을 렌더하는 데는 필요 없다. 위치 파일은 처음 읽을 때 배열로 바꾸어 `data/cache/`에 저장하며 1분쯤 걸린다.

```bash
python scripts/fetch_idsse.py
```

## 렌더하기

장의 `.md`는 `.qmd` 원본을 렌더해 만든 것이므로 손으로 고치지 않는다. `chapters/`에서 렌더한다. Quarto가 쓸 파이썬은 `QUARTO_PYTHON`으로 알려 준다. 처음 렌더할 때는 대회 전체의 표를 만들어 `data/cache/`에 저장하므로 몇 분이 걸리고, 다음부터는 1분 안에 끝난다.

```bash
cd chapters
quarto render 01-data-and-coordinates.qmd
```

필요한 패키지는 다음과 같다.

```bash
pip install pandas numpy scipy matplotlib statsmodels lifelines tabulate jupyter
```

## 주의

- 데이터는 StatsBomb의 것이다. 이용 약관(`data/raw/statsbomb/LICENSE.pdf`, 2023-09-08 판)에 따라 데이터를 다시 배포하지 않으므로 `data/`는 git에 올리지 않는다. 데이터와 그것으로 한 분석의 상업적 이용도 약관이 막는다.
- 이 데이터로 한 분석을 공개할 때는 출처를 StatsBomb으로 밝히고 로고를 붙인다. 장의 첫머리에 출처를 적었고, 로고는 아직 붙이지 않았다.
- 좌표는 가로 120, 세로 80이고 사건을 한 팀이 언제나 왼쪽에서 오른쪽으로 공격하도록 적혀 있다. 좌표의 1단위는 1야드로 본다(데이터 1장 III장).
- 대시보드의 수치와 그림은 손으로 옮긴 것이다. 장을 다시 렌더해 수치가 달라지면 `docs/index.html`의 해당 문장을 고치고 그림을 `docs/img/`에 다시 복사한다.
- 대시보드 주소와 장의 링크는 GitHub 저장소 `pilsunchoi/sports-data-analysis`의 `main` 가지와 Pages(`/docs`)를 전제로 적었다. 저장소를 만들기 전에는 열리지 않는다.
- `sportsdata/statsbomb.py`를 고쳐 표의 열이 달라지면 `data/cache/`를 지우고 다시 렌더한다.
