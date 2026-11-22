"""선수의 배치에서 공간을 계산한다. 보로노이 칸, 보인 영역의 경계까지의 거리, 슛의 길이다. 방법은 공간 2장에 있다.

다각형은 꼭짓점의 (k, 2) 배열이고 볼록하다고 가정한다. 이 대회의 보인 영역은 모두 볼록 다각형이다.
"""

import numpy as np

PITCH_POLY = np.array([[0.0, 0.0], [120.0, 0.0], [120.0, 80.0], [0.0, 80.0]])


def as_polygon(flat) -> np.ndarray:
    """[x1, y1, x2, y2, ...]를 꼭짓점의 배열로. 끝점이 첫 점과 같으면 뺀다."""
    p = np.asarray(flat, float).reshape(-1, 2)
    return p[:-1] if len(p) > 1 and np.allclose(p[0], p[-1]) else p


def area(poly: np.ndarray) -> float:
    """다각형의 넓이(신발끈 공식)."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_halfplane(poly: np.ndarray, normal, c: float) -> np.ndarray:
    """볼록 다각형에서 normal . x <= c인 쪽만 남긴다(서덜랜드-호지먼)."""
    if len(poly) == 0:
        return poly
    s = poly @ np.asarray(normal, float) - c
    out = []
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        if s[i] <= 0:
            out.append(poly[i])
        if (s[i] < 0 < s[j]) or (s[j] < 0 < s[i]):
            t = s[i] / (s[i] - s[j])
            out.append(poly[i] + t * (poly[j] - poly[i]))
    return np.array(out) if out else np.empty((0, 2))


def voronoi_cell(p, others, boundary: np.ndarray) -> np.ndarray:
    """경계 다각형 안에서 점 p의 보로노이 칸. 다른 점마다 수직이등분선의 p 쪽만 남긴다."""
    p = np.asarray(p, float)
    cell = boundary
    for q in np.asarray(others, float).reshape(-1, 2):
        cell = clip_halfplane(cell, 2 * (q - p), q @ q - p @ p)
        if len(cell) == 0:
            break
    return cell


def disc(center, radius: float, k: int = 72) -> np.ndarray:
    """원을 k각형으로 근사한 다각형."""
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.asarray(center, float) + radius * np.column_stack([np.cos(a), np.sin(a)])


def clip_convex(poly: np.ndarray, window: np.ndarray) -> np.ndarray:
    """볼록 다각형 poly를 볼록 다각형 window로 자른다."""
    w = window if _signed(window) > 0 else window[::-1]
    for i in range(len(w)):
        a, b = w[i], w[(i + 1) % len(w)]
        n = np.array([b[1] - a[1], a[0] - b[0]])          # 반시계 방향 다각형의 바깥쪽 법선
        poly = clip_halfplane(poly, n, n @ a)
        if len(poly) == 0:
            break
    return poly


def _signed(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def unseen_distance(point, visible: np.ndarray, tol: float = 0.01) -> float:
    """점에서 보이지 않는 곳까지의 거리. 보인 영역의 변 가운데 경기장의 경계선 위에 있는 변은 세지 않는다.
    그 변 너머는 경기장 밖이어서 선수가 없기 때문이다. 그런 변뿐이면 inf다."""
    p = np.asarray(point, float)
    best = np.inf
    for i in range(len(visible)):
        a, b = visible[i], visible[(i + 1) % len(visible)]
        on_line = any(abs(a[k] - v) < tol and abs(b[k] - v) < tol for k, v in ((0, 0.0), (0, 120.0), (1, 0.0), (1, 80.0)))
        if on_line:
            continue
        ab = b - a
        t = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
        best = min(best, float(np.hypot(*(p - (a + t * ab)))))
    return best


def in_triangle(points, a, b, c) -> np.ndarray:
    """점들이 삼각형 abc 안에 있는지."""
    pts = np.asarray(points, float).reshape(-1, 2)
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    cross = lambda u, v, w: (v[0] - u[0]) * (w[:, 1] - u[1]) - (v[1] - u[1]) * (w[:, 0] - u[0])
    d1, d2, d3 = cross(a, b, pts), cross(b, c, pts), cross(c, a, pts)
    return ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))


def inside(poly: np.ndarray, points, tol: float = 1e-9) -> np.ndarray:
    """점들이 볼록 다각형 안(경계 포함)에 있는지."""
    pts = np.asarray(points, float).reshape(-1, 2)
    w = poly if _signed(poly) > 0 else poly[::-1]
    ok = np.ones(len(pts), bool)
    for i in range(len(w)):
        a, b = w[i], w[(i + 1) % len(w)]
        ok &= (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0]) >= -tol
    return ok


def cell_is_determined(p, cell: np.ndarray, visible: np.ndarray) -> bool:
    """보인 영역 안에서 구한 칸이 보이지 않는 선수가 어디에 있든 달라지지 않는지.
    칸의 꼭짓점마다 p까지의 거리가 보이지 않는 곳까지의 거리 이하이면 그렇다."""
    p = np.asarray(p, float)
    return len(cell) > 0 and all(np.hypot(*(v - p)) <= unseen_distance(v, visible) + 1e-6 for v in cell)


def receipt_table(events, frames, areas):
    """공을 받은 순간마다 받은 선수의 공간. 성공한 받음(Ball Receipt*) 가운데 프리즈 프레임이 있는 것이다.

    d1은 프레임에 보인 가장 가까운 상대까지의 거리(없으면 inf), du는 보이지 않는 곳까지의 거리,
    cell은 보인 영역 안에서 구한 보로노이 칸의 넓이, det는 그 칸이 확정되는지다. 단위는 좌표의 단위(야드)다.
    """
    import pandas as pd
    e = events[(events["period"] <= 4) & (events["type_name"] == "Ball Receipt*") & events["ball_receipt_outcome_name"].isna()]
    e = e[e["id"].isin(areas["event_uuid"])]
    poly = areas.set_index("event_uuid")["polygon"]
    groups = {k: g for k, g in frames[frames["event_uuid"].isin(e["id"])].groupby("event_uuid")}
    rows = []
    for r in e.itertuples():
        g = groups[r.id]
        xy, mate, actor = g[["x", "y"]].to_numpy(), g["teammate"].to_numpy(), g["actor"].to_numpy()
        if actor.sum() != 1:
            continue
        p, vis = xy[actor][0], as_polygon(poly[r.id])
        opp = xy[~mate]
        d1 = float(np.min(np.hypot(*(opp - p).T))) if len(opp) else np.inf
        cell = voronoi_cell(p, xy[~actor], vis)
        rows.append({"id": r.id, "match_id": r.match_id, "x": r.x, "y": r.y, "under_pressure": bool(r.under_pressure == True),  # noqa: E712  (빈 값은 False)
                     "players": len(g), "visible": area(vis), "d1": d1, "du": unseen_distance(p, vis),
                     "cell": area(cell), "det": cell_is_determined(p, cell, vis)})
    return pd.DataFrame(rows)
