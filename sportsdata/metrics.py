"""예측 확률을 평가하는 척도. 정의는 회귀 1장 VI장에 있다."""

import numpy as np
from scipy.stats import rankdata


def auc(y, p) -> float:
    """ROC 곡선 아래의 넓이. 결과가 1인 관측의 예측 확률이 0인 관측보다 클 확률(동률은 절반)."""
    y = np.asarray(y).astype(bool)
    r = rankdata(np.asarray(p, dtype=float))
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p) -> float:
    """예측 확률과 결과의 차이의 제곱 평균."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))
