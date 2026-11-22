"""패스 네트워크. 선수를 노드로, 성공한 패스를 방향과 가중치가 있는 변으로 둔다. 방법은 연쇄 2장에 있다.

인접 행렬 A의 (i, j) 원소는 선수 i가 선수 j에게 성공시킨 패스의 수다. 매개 중심성은 시뮬레이션 그래프
200개와 이 대회의 네트워크 128개에서 networkx의 값과 일치하는 것을 확인했다(2026-11-22). 비교할 때
networkx에는 거리를 정수로 바꾸어 주었다. 1/a_ij를 실수로 주면 networkx는 길이가 같은 경로를
반올림 오차 때문에 다르다고 보는 경우가 있다. 이 모듈은 허용 오차를 두고 비교한다.
"""

import numpy as np
import pandas as pd


def pass_list(events: pd.DataFrame) -> pd.DataFrame:
    """성공한 패스. 패스 하나에 한 행이고 준 선수(player_id)와 받은 선수(to_id)가 있다. 승부차기는 없다."""
    p = events[(events["type_name"] == "Pass") & (events["period"] <= 4) & events["pass_outcome_name"].isna()]
    p = p[["match_id", "team_name", "possession", "minute", "player_id", "pass_recipient_id", "x", "y"]].copy()
    p = p.rename(columns={"pass_recipient_id": "to_id"})
    p["player_id"], p["to_id"] = p["player_id"].astype(int), p["to_id"].astype(int)
    return p.reset_index(drop=True)


def team_network(passes: pd.DataFrame, match_id: int, team_name: str):
    """한 경기 한 팀의 패스를 노드 번호로 바꾼다. 선수의 id(번호순), 준 선수와 받은 선수의 번호, 패스의 표를 돌려준다."""
    d = passes[(passes["match_id"] == match_id) & (passes["team_name"] == team_name)]
    ids = np.sort(np.unique(np.r_[d["player_id"].to_numpy(), d["to_id"].to_numpy()]))
    code = pd.Series(np.arange(len(ids)), index=ids)
    return ids, code[d["player_id"]].to_numpy(), code[d["to_id"]].to_numpy(), d


def adjacency(src, dst, n: int, weights=None) -> np.ndarray:
    """노드 번호(0부터)의 쌍에서 n행 n열의 인접 행렬을 만든다. weights는 쌍마다의 가중치다."""
    A = np.zeros((n, n))
    np.add.at(A, (np.asarray(src), np.asarray(dst)), 1.0 if weights is None else weights)
    return A


def strength(A: np.ndarray):
    """나가는 강도(준 패스)와 들어오는 강도(받은 패스)."""
    return A.sum(axis=1), A.sum(axis=0)


def shortest_paths(A: np.ndarray):
    """변의 길이를 1/a_ij로 둔 최단 거리 D와 최단 경로의 수 S. 갈 수 없는 쌍의 거리는 inf다."""
    n = len(A)
    with np.errstate(divide="ignore"):
        W = np.where(A > 0, 1.0 / A, np.inf)
    np.fill_diagonal(W, np.inf)
    D = W.copy()
    np.fill_diagonal(D, 0.0)
    for k in range(n):                                    # 플로이드-워셜
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    S = np.zeros((n, n))
    for s in range(n):                                    # 가까운 노드부터 최단 경로의 수를 센다
        S[s, s] = 1.0
        for t in np.argsort(D[s]):
            if t == s or not np.isfinite(D[s, t]):
                continue
            on_path = np.isclose(D[s] + W[:, t], D[s, t], rtol=1e-9, atol=1e-12)
            S[s, t] = S[s, on_path].sum()
    return D, S


def betweenness(A: np.ndarray) -> np.ndarray:
    """매개 중심성. 다른 두 선수 사이의 최단 경로 가운데 이 선수를 거치는 것의 비율을 모든 쌍에 대해
    평균한 값이다. 서로 닿을 수 없는 쌍은 0으로 센다."""
    n = len(A)
    D, S = shortest_paths(A)
    b = np.zeros(n)
    for v in range(n):
        through = np.isclose(D[:, [v]] + D[[v], :], D, rtol=1e-9, atol=1e-12) & np.isfinite(D)
        frac = np.where(through & (S > 0), S[:, [v]] * S[[v], :] / np.where(S > 0, S, 1.0), 0.0)
        frac[v, :], frac[:, v] = 0.0, 0.0
        np.fill_diagonal(frac, 0.0)
        b[v] = frac.sum()
    return b / ((n - 1) * (n - 2))


def transition_matrix(A: np.ndarray) -> np.ndarray:
    """행의 합이 1인 전이 행렬. 패스를 한 번도 주지 않은 선수의 행은 모든 선수에게 고르게 나눈다."""
    out = A.sum(axis=1, keepdims=True)
    return np.where(out > 0, A / np.where(out > 0, out, 1.0), 1.0 / len(A))


def pagerank(A: np.ndarray, damping: float = 0.85) -> np.ndarray:
    """페이지랭크. 확률 damping으로 패스를 따라가고 나머지 확률로 아무 선수에게나 옮겨 가는 연쇄의 정상 분포."""
    n = len(A)
    G = damping * transition_matrix(A) + (1 - damping) / n
    pi = np.full(n, 1.0 / n)
    for _ in range(1000):
        new = pi @ G
        if np.abs(new - pi).sum() < 1e-13:
            break
        pi = new
    return new / new.sum()


def density(A: np.ndarray) -> float:
    """방향이 있는 선수 쌍 가운데 패스가 한 번이라도 있었던 쌍의 비율."""
    n = len(A)
    off = ~np.eye(n, dtype=bool)
    return float((A[off] > 0).mean())


def shares(A: np.ndarray) -> np.ndarray:
    """선수마다 패스에 관여한 몫. 준 패스와 받은 패스를 더해 전체의 합이 1이 되게 나눈다."""
    s = A.sum(axis=1) + A.sum(axis=0)
    return s / s.sum()


def effective_players(A: np.ndarray) -> float:
    """유효 선수 수. 관여한 몫의 제곱의 합의 역수다. 모든 선수가 같은 몫이면 선수의 수와 같다."""
    q = shares(A)
    return float(1.0 / (q ** 2).sum())
