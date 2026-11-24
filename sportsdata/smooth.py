"""경기장 위의 평활. 2차원 커널 밀도 추정과 벌점 스플라인 로지스틱 회귀다. 방법은 공간 1장에 있다.

좌표는 StatsBomb의 것(가로 0~120, 세로 0~80)이고 띠너비의 단위도 같다(1단위 = 1야드).
"""

import numpy as np
from scipy.interpolate import BSpline

PITCH = ((0.0, 120.0), (0.0, 80.0))


def _kernel(at, x, h: float, bounds=None) -> np.ndarray:
    """1차원 정규 커널의 값. 행은 평가하는 점, 열은 데이터의 점이다. bounds를 주면 양쪽 경계에서 반사한다."""
    at, x = np.asarray(at, float)[:, None], np.asarray(x, float)[None, :]
    phi = lambda u: np.exp(-0.5 * (u / h) ** 2) / (h * np.sqrt(2 * np.pi))
    k = phi(at - x)
    if bounds is not None:
        lo, hi = bounds
        k = k + phi(at - (2 * lo - x)) + phi(at - (2 * hi - x))
    return k


def kde_grid(x, y, gx, gy, h: float, reflect: bool = True) -> np.ndarray:
    """격자 위의 밀도. 행은 gy, 열은 gx다. 두 좌표에 같은 띠너비 h의 정규 커널을 곱해 사용한다."""
    bx, by = PITCH if reflect else (None, None)
    return _kernel(gy, y, h, by) @ _kernel(gx, x, h, bx).T / len(x)


def kde_at(x, y, ax, ay, h: float, reflect: bool = True, chunk: int = 2000) -> np.ndarray:
    """점 (ax, ay)들에서의 밀도. 데이터는 (x, y)다."""
    bx, by = PITCH if reflect else (None, None)
    ax, ay = np.asarray(ax, float), np.asarray(ay, float)
    out = np.empty(len(ax))
    for s in range(0, len(ax), chunk):
        e = slice(s, s + chunk)
        out[e] = (_kernel(ax[e], x, h, bx) * _kernel(ay[e], y, h, by)).sum(axis=1) / len(x)
    return out


def bin_counts(x, y, cell: float = 0.5):
    """경기장을 한 변이 cell인 칸으로 나누어 점을 센다. 개수의 행렬(행은 y, 열은 x)과 칸의 중심 좌표를 돌려준다."""
    (x0, x1), (y0, y1) = PITCH
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    ix = np.clip(((np.asarray(x, float) - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((np.asarray(y, float) - y0) / cell).astype(int), 0, ny - 1)
    C = np.zeros((ny, nx))
    np.add.at(C, (iy, ix), 1.0)
    return C, x0 + cell * (np.arange(nx) + 0.5), y0 + cell * (np.arange(ny) + 0.5)


def kde_binned(C: np.ndarray, gx, gy, h: float, reflect: bool = True) -> np.ndarray:
    """칸에 센 개수 C에서 칸의 중심마다의 밀도. 점을 칸의 중심으로 옮긴 뒤의 kde_grid와 같다.
    점이 많을 때 사용한다. 띠너비가 칸의 크기보다 충분히 크면 옮긴 것의 영향은 작다."""
    bx, by = PITCH if reflect else (None, None)
    return _kernel(gy, gy, h, by) @ C @ _kernel(gx, gx, h, bx).T / C.sum()


def scott_bandwidth(x, y) -> float:
    """Scott의 기준 규칙(2차원). 두 좌표의 표준편차의 기하평균에 n^(-1/6)을 곱한다."""
    return float(np.sqrt(np.std(x, ddof=1) * np.std(y, ddof=1)) * len(x) ** (-1 / 6))


def bspline_basis(x, lo: float, hi: float, nseg: int, degree: int = 3) -> np.ndarray:
    """구간 [lo, hi]를 nseg등분한 매듭 위의 B-스플라인 기저. 열은 nseg + degree개다."""
    step = (hi - lo) / nseg
    knots = lo + step * np.arange(-degree, nseg + degree + 1)
    x = np.clip(np.asarray(x, float), lo, hi - 1e-9)
    return BSpline.design_matrix(x, knots, degree).toarray()


class PSplineLogit2D:
    """두 좌표의 텐서곱 B-스플라인에 차분 벌점을 준 로지스틱 회귀(Eilers and Marx, 1996).

    logit p(x, y) = sum_jk beta_jk B_j(x) B_k(y)이고, 벌점은 lam * (x 방향과 y 방향의 d차 차분의 제곱합)이다.
    fit 뒤에 beta, edf(유효 자유도), deviance가 있다.
    """

    def __init__(self, xlim, ylim, nseg=(8, 8), degree: int = 3, diff: int = 2):
        self.xlim, self.ylim, self.nseg, self.degree, self.diff = xlim, ylim, nseg, degree, diff
        kx, ky = nseg[0] + degree, nseg[1] + degree
        Dx, Dy = np.diff(np.eye(kx), diff, axis=0), np.diff(np.eye(ky), diff, axis=0)
        self.penalty = np.kron(Dx.T @ Dx, np.eye(ky)) + np.kron(np.eye(kx), Dy.T @ Dy)

    def basis(self, x, y) -> np.ndarray:
        Bx = bspline_basis(x, *self.xlim, self.nseg[0], self.degree)
        By = bspline_basis(y, *self.ylim, self.nseg[1], self.degree)
        return (Bx[:, :, None] * By[:, None, :]).reshape(len(Bx), -1)

    def fit(self, x, y, z, lam: float, max_iter: int = 100):
        B, z = self.basis(x, y), np.asarray(z, float)
        P = lam * self.penalty + 1e-8 * np.eye(B.shape[1])
        beta = np.zeros(B.shape[1])
        beta_old = beta + 1.0
        for _ in range(max_iter):                         # 벌점이 있는 반복 재가중 최소제곱
            eta = B @ beta
            p = 1 / (1 + np.exp(-eta))
            w = np.maximum(p * (1 - p), 1e-10)
            BtWB = B.T @ (B * w[:, None])
            beta = np.linalg.solve(BtWB + P, B.T @ (w * eta + z - p))
            if np.max(np.abs(beta - beta_old)) < 1e-8:
                break
            beta_old = beta
        p = np.clip(1 / (1 + np.exp(-(B @ beta))), 1e-12, 1 - 1e-12)
        w = p * (1 - p)
        BtWB = B.T @ (B * w[:, None])
        self.beta, self.lam = beta, lam
        self.edf = float(np.trace(np.linalg.solve(BtWB + P, BtWB)))
        self.deviance = float(-2 * np.sum(z * np.log(p) + (1 - z) * np.log(1 - p)))
        return self

    def predict(self, x, y) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.basis(x, y) @ self.beta)))
