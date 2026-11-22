# 스포츠 데이터 분석 I (집필 중)

축구 경기의 공개 데이터로 분석 방법을 가르치는 교재의 저장소다. 데이터에 기본적인 분석 방법을 하나씩 적용하고, 방법마다 수식과 가정, 추정 절차, 결과를 읽는 법을 자세히 설명한다. I권은 2022년 카타르 월드컵의 64경기를 다룬다. 분석은 모두 파이썬으로 한다.

자매 교재: [재활 데이터 분석 I](https://github.com/pilsunchoi/rehab-data-analysis)

## 장

| 묶음 | 장 | 본문 |
|---|---|---|
| 데이터 | 1 | [데이터와 좌표 — 영상은 어떻게 표가 되는가](chapters/01-data-and-coordinates.md) |
| 회귀 | 1 | [로지스틱 회귀 — 어디서 찬 슛이 골이 되는가](chapters/02-shot-logistic.md) |
| 회귀 | 2 | [포아송 회귀 — 득점은 포아송 분포를 따르는가](chapters/03-goal-counts.md) |

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
│   └── 03-goal-counts_files/              회귀 2장 그림
├── sportsdata/                            모든 장이 불러 쓰는 파이썬 모듈
│   ├── statsbomb.py                       JSON 읽기, 경기·사건·프리즈 프레임의 표
│   ├── features.py                        슛의 표, 팀·경기의 표, 골대까지의 거리와 각도
│   ├── metrics.py                         예측 확률의 평가(AUC, 브라이어 점수)
│   ├── pitch.py                           경기장 그림과 그림의 공통 양식
│   └── report.py                          본문과 표에 쓰는 수의 표기, 숫자 뒤의 조사
├── scripts/
│   ├── fetch_statsbomb.py                 2022 월드컵의 사건과 프리즈 프레임을 받는다
│   ├── data_status.py                     받은 데이터의 경기와 사건 수를 센다
│   └── postrender_md.py                   렌더된 MD를 GitHub용으로 손질(물결 이스케이프)
└── data/                                  공개하지 않는다
    ├── raw/statsbomb/                     받은 파일(경로와 이름을 바꾸지 않는다)
    └── cache/                             대회 전체를 쌓은 표(처음 읽을 때 만든다)
```

## 데이터 받기

```bash
python scripts/fetch_statsbomb.py
python scripts/data_status.py
```

첫 명령이 StatsBomb 공개 데이터 저장소에서 2022 남자 월드컵의 파일을 받아 `data/raw/statsbomb/`에 둔다. 이미 받은 파일은 다시 받지 않는다. 둘째 명령이 경기, 사건, 슛, 프리즈 프레임의 수를 낸다.

## 렌더하기

장의 `.md`는 `.qmd` 원본을 렌더해 만든 것이므로 손으로 고치지 않는다. `chapters/`에서 렌더한다. Quarto가 쓸 파이썬은 `QUARTO_PYTHON`으로 알려 준다. 처음 렌더할 때는 대회 전체의 표를 만들어 `data/cache/`에 저장하므로 몇 분이 걸리고, 다음부터는 1분 안에 끝난다.

```bash
cd chapters
quarto render 01-data-and-coordinates.qmd
```

필요한 패키지는 다음과 같다.

```bash
pip install pandas numpy scipy matplotlib statsmodels tabulate jupyter
```

## 주의

- 데이터는 StatsBomb의 것이다. 이용 약관(`data/raw/statsbomb/LICENSE.pdf`, 2023-09-08 판)에 따라 데이터를 다시 배포하지 않으므로 `data/`는 git에 올리지 않는다. 데이터와 그것으로 한 분석의 상업적 이용도 약관이 막는다.
- 이 데이터로 한 분석을 공개할 때는 출처를 StatsBomb으로 밝히고 로고를 붙인다. 장의 첫머리에 출처를 적었고, 로고는 아직 붙이지 않았다.
- 좌표는 가로 120, 세로 80이고 사건을 한 팀이 언제나 왼쪽에서 오른쪽으로 공격하도록 적혀 있다. 좌표의 1단위는 1야드로 읽는다(데이터 1장 III장).
- `sportsdata/statsbomb.py`를 고쳐 표의 열이 달라지면 `data/cache/`를 지우고 다시 렌더한다.
