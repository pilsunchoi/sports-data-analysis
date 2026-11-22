"""위치의 시계열에서 속도와 가속도를 얻는 계산.

모든 함수는 간격이 일정한 시계열을 받는다. 마지막 축이 시간이다. 창이 닿지 않는 양 끝과
NaN이 낀 곳의 값은 NaN이다.
"""

from math import factorial

import numpy as np


def savgol_weights(half: int, order: int, deriv: int = 0, dt: float = 1.0) -> np.ndarray:
    """Savitzky-Golay 가중치. 길이 2*half+1의 창에 order차 다항식을 최소제곱으로 맞추고
    창 가운데에서의 deriv계 미분을 주는 가중치다. 가중치와 창 안의 값의 내적이 추정값이다."""
    if order > 2 * half:
        raise ValueError("다항식의 차수가 창의 점의 수보다 작아야 한다")
    i = np.arange(-half, half + 1)
    A = np.vander(i * dt, order + 1, increasing=True)      # 열 j는 (i*dt)^j
    return factorial(deriv) * np.linalg.pinv(A)[deriv]


def difference_weights(half: int, dt: float = 1.0) -> np.ndarray:
    """두 점 중심 차분의 가중치. 앞뒤로 half칸 떨어진 두 값의 차를 그 사이의 시간으로 나눈다."""
    w = np.zeros(2 * half + 1)
    w[0], w[-1] = -1.0, 1.0
    return w / (2 * half * dt)


def mean_weights(half: int) -> np.ndarray:
    """이동 평균의 가중치."""
    return np.full(2 * half + 1, 1.0 / (2 * half + 1))


def apply(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """가중치 w를 시계열의 모든 시점에 적용한다. 결과의 k번째 값은 x[k-half..k+half]와 w의 내적이다."""
    x = np.asarray(x, dtype=float)
    half = (len(w) - 1) // 2
    out = np.full(x.shape, np.nan)
    if x.shape[-1] < len(w):
        return out
    flat, res = x.reshape(-1, x.shape[-1]), out.reshape(-1, x.shape[-1])
    for r in range(flat.shape[0]):
        res[r, half: flat.shape[1] - half] = np.correlate(flat[r], w, mode="valid")
    return out


def gain(w: np.ndarray, freq: np.ndarray, dt: float) -> np.ndarray:
    """가중치 w가 주파수 freq(Hz)의 사인파의 진폭을 몇 배로 만드는지."""
    half = (len(w) - 1) // 2
    i = np.arange(-half, half + 1)
    return np.abs(np.exp(2j * np.pi * np.outer(freq, i) * dt) @ w)


def forward_speed(x: np.ndarray, y: np.ndarray, dt: float) -> np.ndarray:
    """이웃한 두 프레임 사이의 이동 거리를 시간으로 나눈 속력. 첫 값은 NaN이다."""
    v = np.full(np.shape(x), np.nan)
    v[..., 1:] = np.hypot(np.diff(x, axis=-1), np.diff(y, axis=-1)) / dt
    return v


def motion(x: np.ndarray, y: np.ndarray, w1: np.ndarray, w2: np.ndarray = None) -> dict:
    """속도와 가속도. w1은 1계 미분의 가중치, w2는 2계 미분의 가중치다.

    speed는 속력, a_tan은 속력의 변화율(진행 방향의 가속도), a_tot은 가속도 벡터의 크기다.
    속력이 0에 가까우면 진행 방향이 정해지지 않으므로 a_tan은 0으로 둔다.
    """
    vx, vy = apply(x, w1), apply(y, w1)
    out = {"vx": vx, "vy": vy, "speed": np.hypot(vx, vy)}
    if w2 is not None:
        ax, ay = apply(x, w2), apply(y, w2)
        with np.errstate(invalid="ignore", divide="ignore"):
            tan = np.where(out["speed"] > 1e-6, (vx * ax + vy * ay) / out["speed"], 0.0)
        tan[~np.isfinite(ax)] = np.nan
        out.update(ax=ax, ay=ay, a_tot=np.hypot(ax, ay), a_tan=tan)
    return out


def bouts(mask: np.ndarray, min_len: int = 1) -> int:
    """mask가 참인 채로 min_len개 이상 이어진 구간의 수."""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edge = np.flatnonzero(m[1:] != m[:-1])
    return int(np.sum(edge[1::2] - edge[0::2] >= min_len))


def jumps(x: np.ndarray, y: np.ndarray, limit: float) -> np.ndarray:
    """앞 프레임에서 limit보다 멀리 옮겨진 프레임을 참으로 표시한다."""
    j = np.zeros(np.shape(x), dtype=bool)
    j[..., 1:] = np.hypot(np.diff(x, axis=-1), np.diff(y, axis=-1)) > limit
    return j


def mask_jumps(values: np.ndarray, jump: np.ndarray, half: int) -> np.ndarray:
    """창이 jump를 걸치는 시점의 값을 NaN으로 바꾼다. 창은 앞뒤로 half칸이다.

    jump[k]가 참이면 k-1과 k 사이가 끊긴 것이므로, 두 프레임을 모두 포함하는 창,
    곧 중심이 k-half부터 k+half-1까지인 창의 값을 버린다. half가 0이면 k만 버린다.
    """
    out = np.array(values, dtype=float)
    n = out.shape[-1]
    for r, k in zip(*np.nonzero(np.atleast_2d(jump))):
        np.atleast_2d(out)[r, max(0, k - half): min(n, k + max(half, 1))] = np.nan
    return out


def gaussian_lowpass(x: np.ndarray, f0: float, dt: float, deriv: int = 0) -> np.ndarray:
    """주파수 f의 성분에 exp(-(f/f0)^2/2)를 곱해 높은 주파수를 줄인 시계열, 또는 그것의 deriv계 미분.

    미분은 주파수 영역에서 (2*pi*i*f)^deriv를 곱해 구하므로 차분의 오차가 없다. 양 끝이
    이어지도록 시계열을 거울처럼 뒤집어 붙여 변환한다. 양 끝 근처의 미분은 쓰지 않는다.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    f = np.fft.rfftfreq(2 * n, dt)
    F = np.fft.rfft(np.concatenate([x, x[..., ::-1]], axis=-1), axis=-1)
    return np.fft.irfft(F * np.exp(-0.5 * (f / f0) ** 2) * (2j * np.pi * f) ** deriv, 2 * n, axis=-1)[..., :n]
