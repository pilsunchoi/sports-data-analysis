"""주성분 분석과 군집. 방법은 차원 축소 1장에 있다.

k-평균, 실루엣, 조정 랜드 지수는 scikit-learn의 값과 일치하는 것을 확인했다(2026-11-22).
"""

import numpy as np


def standardize(X) -> np.ndarray:
    """열마다 평균을 빼고 표준편차로 나눈다."""
    X = np.asarray(X, float)
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)


def pca(Z) -> dict:
    """표준화한 자료 Z의 주성분 분석. 고윳값(분산), 적재(열이 주성분), 점수, 분산의 비율을 돌려준다."""
    Z = np.asarray(Z, float)
    U, s, Vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    values = s ** 2 / (len(Z) - 1)
    sign = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])      # 가장 큰 적재가 양수가 되게 부호를 정한다
    V = (Vt * sign[:, None]).T
    return {"values": values, "loadings": V, "scores": Z @ V, "ratio": values / values.sum()}


def parallel_analysis(Z, n_sim: int = 500, rng=None, q: float = 95) -> np.ndarray:
    """열마다 값을 따로 섞어 변수 사이의 상관을 없앤 자료에서 고윳값의 q% 분위수를 구한다."""
    rng = rng or np.random.default_rng(0)
    Z = np.asarray(Z, float)
    sims = np.empty((n_sim, Z.shape[1]))
    for b in range(n_sim):
        P = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        sims[b] = np.linalg.svd(P - P.mean(axis=0), compute_uv=False) ** 2 / (len(Z) - 1)
    return np.percentile(sims, q, axis=0)


def sqdist(X, C) -> np.ndarray:
    """X의 점마다 C의 점마다의 거리의 제곱."""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def kmeans(X, k: int, n_init: int = 20, rng=None, max_iter: int = 300):
    """k-평균(로이드 알고리즘). 시작점을 n_init번 바꾸어 군집 내 제곱합이 가장 작은 해를 고른다.
    군집의 번호, 중심, 군집 내 제곱합을 돌려준다. 시작점은 k-means++ 방식으로 뽑는다."""
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, float)
    best = None
    for _ in range(n_init):
        C = X[[rng.integers(len(X))]]
        for _ in range(1, k):                             # 기존 중심에서 먼 점을 더 높은 확률로 고른다
            d = sqdist(X, C).min(axis=1)
            C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())]])
        for _ in range(max_iter):
            lab = sqdist(X, C).argmin(axis=1)
            new = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(new, C):
                break
            C = new
        lab = sqdist(X, C).argmin(axis=1)
        wss = float(((X - C[lab]) ** 2).sum())
        if best is None or wss < best[2]:
            best = (lab, C, wss)
    return best


def silhouette(X, labels) -> np.ndarray:
    """점마다의 실루엣 값. a는 자기 군집의 다른 점까지의 평균 거리, b는 가장 가까운 다른 군집까지의 평균 거리다."""
    X, labels = np.asarray(X, float), np.asarray(labels)
    D = np.sqrt(sqdist(X, X))
    ks = np.unique(labels)
    out = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() == 1:
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == j].mean() for j in ks if j != labels[i])
        out[i] = (b - a) / max(a, b)
    return out


def adjusted_rand(a, b) -> float:
    """두 분할의 조정 랜드 지수. 같으면 1이고 우연히 일치하는 정도이면 0이다."""
    a, b = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    T = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(T, (a, b), 1)
    c2 = lambda v: v * (v - 1) / 2
    s, sa, sb, n = c2(T).sum(), c2(T.sum(axis=1)).sum(), c2(T.sum(axis=0)).sum(), c2(len(a))
    expected = sa * sb / n
    return float((s - expected) / (0.5 * (sa + sb) - expected))


def gap_statistic(X, ks, n_ref: int = 20, rng=None, n_init: int = 10):
    """간격 통계량. 자료의 범위 안에 고르게 뿌린 기준 자료의 log(군집 내 제곱합)의 평균에서 자료의 것을 뺀다.
    k마다의 간격과 그 표준오차를 돌려준다."""
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    gap, se = [], []
    for k in ks:
        w = np.log(kmeans(X, k, n_init, rng)[2])
        ref = np.array([np.log(kmeans(rng.uniform(lo, hi, X.shape), k, 3, rng)[2]) for _ in range(n_ref)])
        gap.append(ref.mean() - w)
        se.append(ref.std() * np.sqrt(1 + 1 / n_ref))
    return np.array(gap), np.array(se)
