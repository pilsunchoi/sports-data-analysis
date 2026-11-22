"""점유를 경기장 구역의 마르코프 연쇄로 다룬다. 방법은 연쇄 1장에 있다."""

import numpy as np
import pandas as pd

GOAL, END = "골", "점유 끝"


class Grid:
    """경기장을 가로 nx칸, 세로 ny칸으로 나눈 구역. 번호는 0부터, x가 작은 쪽에서 큰 쪽으로 매긴다."""

    def __init__(self, nx: int = 8, ny: int = 4):
        self.nx, self.ny, self.n = nx, ny, nx * ny

    def zone(self, x, y):
        ix = np.clip((np.asarray(x, float) / (120 / self.nx)).astype(int), 0, self.nx - 1)
        iy = np.clip((np.asarray(y, float) / (80 / self.ny)).astype(int), 0, self.ny - 1)
        return ix * self.ny + iy

    def to_matrix(self, values):
        """구역별 값을 세로 ny행, 가로 nx열의 행렬로(그림에 쓴다)."""
        return np.asarray(values).reshape(self.nx, self.ny).T

    def label(self, z: int) -> str:
        ix, iy = divmod(int(z), self.ny)
        w, h = 120 / self.nx, 80 / self.ny
        return f"x {ix * w:.0f}~{(ix + 1) * w:.0f}, y {iy * h:.0f}~{(iy + 1) * h:.0f}"


def possession_chains(events: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    """점유마다 공이 거친 구역의 열. 행 하나가 전이 하나다.

    점유한 팀의 패스, 공을 가지고 이동, 슛을 순서대로 놓고, 각 동작의 위치가 속한 구역을 상태로
    둔다. state는 그 동작의 구역, next는 다음 동작의 구역이고, 점유의 마지막 동작에서는 그 점유가
    골로 끝났으면 GOAL, 아니면 END다. 승부차기와 페널티킥은 뺀다.
    """
    e = events[(events["period"] <= 4) & (events["team_id"] == events["possession_team_id"])]
    own_goal = e[e["type_name"] == "Own Goal For"][["match_id", "possession"]].drop_duplicates().assign(og=True)
    a = e[e["type_name"].isin(["Pass", "Carry", "Shot"]) & (e["shot_type_name"] != "Penalty") & e["x"].notna()].copy()
    a = a.sort_values(["match_id", "possession", "index"])
    a["state"] = grid.zone(a["x"], a["y"])
    a["scored"] = (a["type_name"] == "Shot") & (a["shot_outcome_name"] == "Goal")
    key = ["match_id", "possession"]
    a = a.merge(own_goal, on=key, how="left")
    a["goal_poss"] = a.groupby(key)["scored"].transform("max") | a["og"].fillna(False).astype(bool)
    a["step"] = a.groupby(key).cumcount()
    nxt = a.groupby(key)["state"].shift(-1)
    last = nxt.isna()
    # 골이 들어간 뒤의 동작은 없다고 본다: 골이 된 슛에서 연쇄를 끝낸다
    after_goal = a.groupby(key)["scored"].cumsum().sub(a["scored"].astype(int)) > 0
    a = a[~after_goal].copy()
    nxt, last = nxt[a.index], last[a.index] | a["scored"]
    a["next"] = np.where(last, np.where(a["goal_poss"], GOAL, END), nxt.fillna(-1).astype(int).astype(str))
    a["prev"] = a.groupby(key)["state"].shift(1)
    cols = ["match_id", "possession", "step", "team_name", "player_id", "player_name", "type_name", "state", "next", "prev", "goal_poss"]
    return a[cols].reset_index(drop=True)


def transition_counts(chains: pd.DataFrame, n: int) -> np.ndarray:
    """전이의 수. 행은 구역 n개, 열은 구역 n개와 GOAL, END."""
    col = chains["next"].map(lambda s: n if s == GOAL else n + 1 if s == END else int(s)).to_numpy()
    C = np.zeros((n, n + 2))
    np.add.at(C, (chains["state"].to_numpy(), col), 1)
    return C


def absorbing(C: np.ndarray):
    """전이의 수에서 전이 확률 P = [Q R], 기본 행렬 N = (I - Q)^-1, 흡수 확률 B = N R을 구한다."""
    n = C.shape[0]
    P = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    Q, R = P[:, :n], P[:, n:]
    N = np.linalg.inv(np.eye(n) - Q)
    return P, N, N @ R
