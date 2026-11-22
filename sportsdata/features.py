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


def shot_table(events: pd.DataFrame) -> pd.DataFrame:
    """경기 중의 슛. 승부차기(period 5)와 페널티킥을 뺀다. 슛 하나에 한 행."""
    s = events[(events["type_name"] == "Shot") & (events["period"] <= 4)
               & (events["shot_type_name"] != "Penalty")].copy()
    s["dist"] = goal_distance(s["x"], s["y"])
    s["angle"] = goal_angle(s["x"], s["y"])
    s["goal"] = (s["shot_outcome_name"] == "Goal").astype(int)
    s["head"] = (s["shot_body_part_name"] == "Head").astype(int)
    return s.reset_index(drop=True)
