"""IDSSE 추적 데이터 읽기. 운동학 묶음의 장이 이 모듈로 데이터를 읽는다.

Bassek et al. (2025)의 독일 분데스리가 7경기다. 파일은 scripts/fetch_idsse.py가
data/raw/idsse/에 받아 둔 XML이다. 위치 파일은 경기마다 350~420MB이므로 처음 읽을 때
배열로 바꾸어 data/cache/에 저장하고 다음부터 그것을 읽는다.

좌표는 경기장 중앙이 원점인 미터다. x는 긴 변, y는 짧은 변의 방향이다. 초당 25프레임이다.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "idsse"
CACHE = ROOT / "data" / "cache"

HZ = 25                      # 초당 프레임 수
DT = 1.0 / HZ
SECTIONS = ("firstHalf", "secondHalf")

_SET = re.compile(rb'<FrameSet GameSection="(\w+)" MatchId="[^"]+" TeamId="([^"]+)" PersonId="([^"]+)">')
_FRAME = re.compile(rb'<Frame N="(\d+)" T="[^"]+" X="([-\d.]+)" Y="([-\d.]+)"(?: Z="[-\d.]+")? D="([-\d.]+)" S="([-\d.]+)" '
                    rb'A="([-\d.]+)" M="\d+"(?: BallPossession="(\d)" BallStatus="(\d)")?/>')


def _file(kind: str, match_id: str) -> Path:
    found = sorted(RAW.glob(f"DFL_*_{kind}_*_{match_id}.xml"))
    if not found:
        raise FileNotFoundError(f"{RAW}에 {match_id}의 {kind} 파일이 없다. 먼저 python scripts/fetch_idsse.py를 돌린다.")
    return found[0]


def match_ids() -> list:
    """받아 둔 경기의 식별자."""
    return sorted(p.stem.split("_")[-1] for p in RAW.glob("DFL_*_matchinformation_*.xml"))


def matches() -> pd.DataFrame:
    """경기 하나에 한 행."""
    rows = []
    for mid in match_ids():
        info = ET.parse(_file("matchinformation", mid)).getroot().find("MatchInformation")
        g, e = info.find("General").attrib, info.find("Environment").attrib
        rows.append({"match_id": mid, "competition": g["CompetitionName"], "date": g["KickoffTime"][:10],
                     "home": g["HomeTeamName"], "guest": g["GuestTeamName"], "result": g["Result"],
                     "home_id": g["HomeTeamId"], "guest_id": g["GuestTeamId"],
                     "pitch_x": float(e["PitchX"]), "pitch_y": float(e["PitchY"])})
    return pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)


def players(match_id: str) -> pd.DataFrame:
    """한 경기의 명단. 선수 하나에 한 행."""
    info = ET.parse(_file("matchinformation", match_id)).getroot().find("MatchInformation")
    rows = []
    for team in info.find("Teams"):
        for p in team.find("Players"):
            a = p.attrib
            rows.append({"match_id": match_id, "person_id": a["PersonId"], "team_id": team.attrib["TeamId"],
                         "team": team.attrib["TeamName"], "side": team.attrib["Role"], "shirt": int(a["ShirtNumber"]),
                         "name": a.get("Shortname", a["LastName"]), "position": a.get("PlayingPosition", ""),
                         "starting": a["Starting"] == "true"})
    return pd.DataFrame(rows)


def _parse(match_id: str) -> dict:
    """위치 파일을 읽어 전반과 후반의 배열로 만든다."""
    data = _file("positions_raw_observed", match_id).read_bytes()
    heads = list(_SET.finditer(data))
    sets = {s: [] for s in SECTIONS}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(data)
        rows = np.array(_FRAME.findall(data, h.end(), end))
        sets[h.group(1).decode()].append((h.group(2).decode(), h.group(3).decode(), rows))
    del data
    out = {}
    for sec in SECTIONS:
        ball = [r for t, _, r in sets[sec] if t == "BALL"][0]
        n0, n = int(ball[0, 0]), len(ball)
        assert np.array_equal(ball[:, 0].astype(np.int64), np.arange(n0, n0 + n)), "공의 프레임 번호가 이어지지 않는다"
        people = [(t, p, r) for t, p, r in sets[sec] if t != "BALL"]
        arr = {k: np.full((len(people), n), np.nan, dtype=np.float32) for k in ("x", "y", "d", "s", "a")}
        for j, (_, _, r) in enumerate(people):
            idx = r[:, 0].astype(np.int64) - n0
            assert np.all(np.diff(idx) == 1), "궤적 안에 빠진 프레임이 있다"
            for c, k in enumerate(("x", "y", "d", "s", "a"), start=1):
                arr[k][j, idx] = r[:, c].astype(np.float32)
        out[sec] = dict(n0=n0, ids=np.array([p for _, p, _ in people]), team=np.array([t for t, _, _ in people]),
                        ball_x=ball[:, 1].astype(np.float32), ball_y=ball[:, 2].astype(np.float32),
                        possession=ball[:, 6].astype(np.int8), alive=ball[:, 7].astype(np.int8) == 1, **arr)
    return out


def tracking(match_id: str) -> dict:
    """한 경기의 추적 데이터. {"firstHalf": {...}, "secondHalf": {...}}.

    구간마다 ids와 team은 길이가 사람 수인 배열이고(심판의 team은 "referee"), x, y, d, s, a는
    (사람 수, 프레임 수)의 배열이다. 그 사람이 경기장에 없던 프레임은 NaN이다. x와 y는 미터,
    s는 제공자가 계산한 속도(km/h), a는 제공자의 가속도(m/s^2), d는 제공자의 프레임 간
    이동 거리(cm)다. ball_x와 ball_y는 공의 위치, alive는 경기가 진행 중인지, possession은
    공을 가진 팀(1은 홈, 2는 원정)이다.
    """
    path = CACHE / f"idsse-{match_id}.npz"
    if not path.exists():
        parsed = _parse(match_id)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"{sec}__{k}": v for sec, d in parsed.items() for k, v in d.items()})
    out = {s: {} for s in SECTIONS}
    with np.load(path) as z:
        for key in z.files:
            sec, k = key.split("__")
            out[sec][k] = z[key] if z[key].ndim else z[key].item()
    return out


def outfield(match_id: str) -> pd.DataFrame:
    """추적 배열의 행을 명단과 잇는다. 구간과 사람마다 한 행. 심판은 뺀다."""
    pl = players(match_id).set_index("person_id")
    rows = []
    for sec, d in tracking(match_id).items():
        for j, (pid, team) in enumerate(zip(d["ids"], d["team"])):
            if team == "referee":
                continue
            seen = np.isfinite(d["x"][j])
            rows.append({"match_id": match_id, "section": sec, "row": j, "person_id": pid, "team": pl.at[pid, "team"],
                         "name": pl.at[pid, "name"], "position": pl.at[pid, "position"], "frames": int(seen.sum()),
                         "first": int(np.argmax(seen)), "full": bool(seen.all())})
    return pd.DataFrame(rows)


def goals(match_id: str) -> pd.DataFrame:
    """한 경기의 골. 골 하나에 한 행. seconds는 그 구간의 킥오프부터 흐른 시간이다."""
    root = ET.parse(_file("events_raw", match_id)).getroot()
    kick, rows = {}, []
    for ev in root.iter("Event"):
        ko = ev.find("KickOff")
        if ko is not None and "GameSection" in ko.attrib:
            kick[ko.attrib["GameSection"]] = pd.Timestamp(ev.attrib["EventTime"])
    for ev in root.iter("Event"):
        for shot in ev.iter("ShotAtGoal"):
            if shot.find("SuccessfulShot") is not None:
                t = pd.Timestamp(ev.attrib["EventTime"])
                sec = "secondHalf" if t >= kick["secondHalf"] else "firstHalf"
                rows.append({"match_id": match_id, "section": sec, "seconds": (t - kick[sec]).total_seconds(),
                             "team_id": shot.attrib["Team"], "result": shot.find("SuccessfulShot").attrib["CurrentResult"]})
        for og in ev.iter("OwnGoal"):
            raise NotImplementedError("자책골의 처리가 없다")
    return pd.DataFrame(rows, columns=["match_id", "section", "seconds", "team_id", "result"]).sort_values(["section", "seconds"]).reset_index(drop=True)
