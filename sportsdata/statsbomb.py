"""StatsBomb 공개 데이터 읽기. 모든 장이 이 모듈로 데이터를 읽는다.

파일은 scripts/fetch_statsbomb.py가 data/raw/statsbomb/에 받아 둔 것이다. 대회 전체를
쌓은 표는 처음 만들 때 data/cache/에 저장하고 다음부터 그것을 읽는다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "statsbomb" / "data"
CACHE = ROOT / "data" / "cache"

PITCH_X, PITCH_Y = 120.0, 80.0   # 좌표의 범위
YARD = 0.9144                    # 좌표 1단위를 1야드로 읽을 때의 미터


def _read(rel: str):
    path = RAW / rel
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없다. 먼저 python scripts/fetch_statsbomb.py를 돌린다.")
    return json.loads(path.read_text(encoding="utf-8"))


def _cached(name: str, build):
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    out = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


def raw_events(match_id: int) -> list:
    """한 경기의 사건을 JSON 그대로 읽는다."""
    return _read(f"events/{match_id}.json")


def matches(competition: int = 43, season: int = 106) -> pd.DataFrame:
    """경기 하나에 한 행. 날짜순."""
    d = pd.json_normalize(_read(f"matches/{competition}/{season}.json"), sep="_")
    return d.sort_values(["match_date", "kick_off"]).reset_index(drop=True)


def events(match_id: int) -> pd.DataFrame:
    """한 경기의 사건. 사건 하나에 한 행이고 겹친 항목은 pass_end_location처럼 밑줄로 편다.

    location은 [x, y]를 그대로 두고 x, y 열을 따로 더한다. 좌표는 가로 120, 세로 80이고
    사건을 한 팀이 언제나 왼쪽에서 오른쪽으로 공격한다.
    """
    d = pd.json_normalize(raw_events(match_id), sep="_")
    d.insert(0, "match_id", match_id)
    if "location" in d:
        d["x"] = [p[0] if isinstance(p, list) else np.nan for p in d["location"]]
        d["y"] = [p[1] if isinstance(p, list) else np.nan for p in d["location"]]
    return d


def frames(match_id: int) -> pd.DataFrame:
    """한 경기의 프리즈 프레임. 사건 순간 화면에 보이는 선수 한 명에 한 행.

    event_uuid가 events()의 id와 맞는다. 선수가 누구인지는 없고 같은 팀인지, 사건의
    당사자인지, 골키퍼인지만 있다.
    """
    rows = [
        {"match_id": match_id, "event_uuid": f["event_uuid"], "teammate": p["teammate"],
         "actor": p["actor"], "keeper": p["keeper"], "x": p["location"][0], "y": p["location"][1]}
        for f in _read(f"three-sixty/{match_id}.json") for p in f["freeze_frame"]
    ]
    return pd.DataFrame(rows)


def lineups(match_id: int) -> pd.DataFrame:
    """한 경기의 명단. 선수 한 명에 한 행. name은 흔히 부르는 이름이고 없으면 등록된 이름이다."""
    rows = [{"match_id": match_id, "team_name": t["team_name"], "player_id": p["player_id"],
             "name": p.get("player_nickname") or p["player_name"], "jersey": p["jersey_number"]}
            for t in _read(f"lineups/{match_id}.json") for p in t["lineup"]]
    return pd.DataFrame(rows)


def polygon_area(flat) -> float:
    """[x1, y1, x2, y2, ...]로 적힌 다각형의 넓이(신발끈 공식)."""
    p = np.asarray(flat, dtype=float).reshape(-1, 2)
    if len(p) < 3:
        return np.nan
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def visible_area(match_id: int) -> pd.DataFrame:
    """한 경기의 프레임마다 카메라에 보인 영역. 프레임 하나에 한 행."""
    rows = [{"match_id": match_id, "event_uuid": f["event_uuid"], "polygon": f["visible_area"],
             "area": polygon_area(f["visible_area"])} for f in _read(f"three-sixty/{match_id}.json")]
    return pd.DataFrame(rows)


def _stack(one, competition, season):
    ids = matches(competition, season)["match_id"]
    return pd.concat([one(i) for i in ids], ignore_index=True)


def all_events(competition: int = 43, season: int = 106) -> pd.DataFrame:
    """대회 전체의 사건."""
    return _cached(f"events-{competition}-{season}", lambda: _stack(events, competition, season))


def all_frames(competition: int = 43, season: int = 106) -> pd.DataFrame:
    """대회 전체의 프리즈 프레임."""
    return _cached(f"frames-{competition}-{season}", lambda: _stack(frames, competition, season))


def all_visible_areas(competition: int = 43, season: int = 106) -> pd.DataFrame:
    """대회 전체의 보인 영역."""
    return _cached(f"visible-{competition}-{season}", lambda: _stack(visible_area, competition, season))
