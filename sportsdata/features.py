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


def shot_table(events: pd.DataFrame) -> pd.DataFrame:
    """경기 중의 슛. 승부차기(period 5)와 페널티킥을 뺀다. 슛 하나에 한 행."""
    s = events[(events["type_name"] == "Shot") & (events["period"] <= 4)
               & (events["shot_type_name"] != "Penalty")].copy()
    s["dist"] = goal_distance(s["x"], s["y"])
    s["angle"] = goal_angle(s["x"], s["y"])
    s["goal"] = (s["shot_outcome_name"] == "Goal").astype(int)
    s["head"] = (s["shot_body_part_name"] == "Head").astype(int)
    return s.reset_index(drop=True)
