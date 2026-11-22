"""사건 표에서 분석용 변수를 만든다. 정의는 데이터 1장 III.4절에 있다."""

import numpy as np
import pandas as pd

from .statsbomb import YARD

GOAL_X, GOAL_Y, HALF_GOAL = 120.0, 40.0, 4.0


def goal_distance(x, y):
    """골대의 중심까지의 거리(미터). 좌표의 1단위를 1야드로 읽는다."""
    return np.hypot(GOAL_X - x, y - GOAL_Y) * YARD


def goal_angle(x, y):
    """두 골대 기둥을 바라보는 방향 사이의 각(도)."""
    a, b = GOAL_X - x, y - GOAL_Y
    return np.degrees(np.arctan2(2 * HALF_GOAL * a, a**2 + b**2 - HALF_GOAL**2))


def team_match_table(matches: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """한 팀의 한 경기에 한 행. 득점은 자책골로 얻은 골과 페널티킥을 포함하고 승부차기는 뺀다.

    goals90과 shots90은 전반과 후반(period 1, 2)의 것이고, goals와 minutes는 연장전까지의 것이다.
    """
    e = events[events["period"] <= 4]
    scored = pd.concat([e[(e["type_name"] == "Shot") & (e["shot_outcome_name"] == "Goal")],
                        e[e["type_name"] == "Own Goal For"]])
    shots = e[(e["type_name"] == "Shot") & (e["period"] <= 2)]
    extra = set(e.loc[e["period"] >= 3, "match_id"])
    rows = []
    for m in matches.itertuples():
        for team, opp in ((m.home_team_home_team_name, m.away_team_away_team_name),
                          (m.away_team_away_team_name, m.home_team_home_team_name)):
            g = scored[(scored["match_id"] == m.match_id) & (scored["team_name"] == team)]
            rows.append({"match_id": m.match_id, "team": team, "opp": opp, "stage": m.competition_stage_name,
                         "goals": len(g), "goals90": int((g["period"] <= 2).sum()),
                         "shots90": int(((shots["match_id"] == m.match_id) & (shots["team_name"] == team)).sum()),
                         "minutes": 120 if m.match_id in extra else 90})
    t = pd.DataFrame(rows)
    t["knockout"] = (t["stage"] != "Group Stage").astype(int)
    return t


def goal_table(events: pd.DataFrame) -> pd.DataFrame:
    """전반과 후반의 골. 골 하나에 한 행이고 자책골은 골을 얻은 팀의 것으로 센다.

    t는 경기가 시작된 뒤 실제로 흐른 경기 시간(분)이다. 전반은 timestamp 그대로, 후반은 그
    경기의 전반 길이에 timestamp를 더한다. 하프타임의 휴식은 세지 않는다.
    """
    e = events[events["period"] <= 2].copy()
    e["sec"] = pd.to_timedelta(e["timestamp"]).dt.total_seconds()
    half = e.groupby(["match_id", "period"])["sec"].max().unstack()
    g = e[((e["type_name"] == "Shot") & (e["shot_outcome_name"] == "Goal")) | (e["type_name"] == "Own Goal For")].copy()
    g["t"] = (g["sec"] + np.where(g["period"] == 2, g["match_id"].map(half[1]), 0)) / 60
    out = g[["match_id", "team_name", "period", "t"]].rename(columns={"team_name": "team"})
    out.attrs["half1"] = (half[1] / 60).to_dict()                 # 경기마다 전반의 길이(분)
    out.attrs["end"] = ((half[1] + half[2]) / 60).to_dict()       # 경기마다 후반이 끝난 시각(분)
    return out.sort_values(["match_id", "t"]).reset_index(drop=True)


def scoring_episodes(matches: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """한 팀의 한 경기를 점수가 바뀌는 시각과 하프타임에서 자른 구간. 구간 하나에 한 행.

    구간 (start, stop] 동안 그 팀의 상황은 바뀌지 않는다. event는 구간의 끝에서 그 팀이 골을
    넣었으면 1이다. diff는 구간 동안의 점수 차(그 팀의 골에서 상대의 골을 뺀 값), own과 opp는
    그때까지 그 팀과 상대가 넣은 골의 수다.
    """
    goals = goal_table(events)
    half1, end = goals.attrs["half1"], goals.attrs["end"]
    rows = []
    for m in matches.itertuples():
        gm = goals[goals["match_id"] == m.match_id]
        cuts = sorted(set(gm["t"]) | {half1[m.match_id], end[m.match_id]})
        for team, opp in ((m.home_team_home_team_name, m.away_team_away_team_name),
                          (m.away_team_away_team_name, m.home_team_home_team_name)):
            start, own, against = 0.0, 0, 0
            for stop in cuts:
                scored = gm[(gm["t"] == stop)]
                ev_own = int((scored["team"] == team).sum())
                ev_opp = int((scored["team"] == opp).sum())
                rows.append({"match_id": m.match_id, "team": team, "opp": opp, "start": start, "stop": stop,
                             "event": int(ev_own > 0), "own": own, "against": against, "diff": own - against,
                             "second_half": int(start >= half1[m.match_id]),
                             "knockout": int(m.competition_stage_name != "Group Stage")})
                own += ev_own
                against += ev_opp
                start = stop
    d = pd.DataFrame(rows)
    d["tm"] = d["match_id"].astype(str) + ":" + d["team"]
    return d


def _position_group(name) -> str:
    if not isinstance(name, str):
        return "미상"
    if name == "Goalkeeper":
        return "골키퍼"
    if "Back" in name:
        return "수비수"
    if "Midfield" in name:
        return "미드필더"
    return "공격수"


def pass_table(events: pd.DataFrame) -> pd.DataFrame:
    """경기 중의 패스. 패스 하나에 한 행.

    결과가 Unknown이거나 부상 때문에 일부러 내보낸 패스(Injury Clearance)는 뺀다. complete는
    같은 팀 선수가 받았으면 1이다. group은 그 선수가 패스를 가장 많이 한 자리의 묶음이다.
    """
    p = events[(events["type_name"] == "Pass") & (events["period"] <= 4)
               & ~events["pass_outcome_name"].isin(["Unknown", "Injury Clearance"])].copy()
    p["complete"] = p["pass_outcome_name"].isna().astype(int)
    p["length"] = p["pass_length"] * YARD
    p["pressure"] = (p["under_pressure"] == True).astype(int)          # noqa: E712
    p["high"] = (p["pass_height_name"] == "High Pass").astype(int)
    p["group"] = p["position_name"].map(_position_group)
    main = p.groupby("player_id")["group"].agg(lambda s: s.mode().iloc[0])
    p["group"] = p["player_id"].map(main)
    cols = ["match_id", "team_name", "player_id", "player_name", "group", "complete", "length", "pressure", "high"]
    return p[cols].reset_index(drop=True)


def shot_table(events: pd.DataFrame) -> pd.DataFrame:
    """경기 중의 슛. 승부차기(period 5)와 페널티킥을 뺀다. 슛 하나에 한 행."""
    s = events[(events["type_name"] == "Shot") & (events["period"] <= 4)
               & (events["shot_type_name"] != "Penalty")].copy()
    s["dist"] = goal_distance(s["x"], s["y"])
    s["angle"] = goal_angle(s["x"], s["y"])
    s["goal"] = (s["shot_outcome_name"] == "Goal").astype(int)
    s["head"] = (s["shot_body_part_name"] == "Head").astype(int)
    return s.reset_index(drop=True)
