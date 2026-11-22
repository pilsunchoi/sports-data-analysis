"""두 수준 모형의 추정. 방법은 다층 모형 1장에 있다.

- beta_binomial_fit: 집단마다의 성공 확률이 베타 분포를 따르는 모형(베타-이항)
- gamma_poisson_fit: 집단마다의 비율이 감마 분포를 따르는 모형(감마-포아송)
- RandomInterceptLogit: 무작위 절편이 있는 로지스틱 회귀(라플라스 근사)
- LinearMixed: 분산 성분이 여럿인 선형 혼합모형(REML 또는 최대가능도). 방법은 운동학 2장에 있다
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit, gammaln
from statsmodels.tools.numdiff import approx_hess


def beta_binomial_fit(y, n):
    """y_j ~ Bin(n_j, theta_j), theta_j ~ Beta(a, b)에서 a, b의 최대가능도 추정치."""
    y, n = np.asarray(y, float), np.asarray(n, float)

    def nll(par):
        a, b = np.exp(par)
        return -np.sum(betaln(a + y, b + n - y) - betaln(a, b))

    m = y.sum() / n.sum()
    res = minimize(nll, np.log([m * 10, (1 - m) * 10]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    return tuple(np.exp(res.x))


def gamma_poisson_fit(y, t):
    """y_j ~ Poisson(t_j lambda_j), lambda_j ~ Gamma(모양 k, 비율 r)에서 k, r의 최대가능도 추정치."""
    y, t = np.asarray(y, float), np.asarray(t, float)

    def nll(par):
        k, r = np.exp(par)
        return -np.sum(gammaln(k + y) - gammaln(k) + k * np.log(r) + y * np.log(t) - (k + y) * np.log(r + t))

    m = y.sum() / t.sum()
    res = minimize(nll, np.log([5.0, 5.0 / m]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    return tuple(np.exp(res.x))


class RandomInterceptLogit:
    """logit P(y_ij = 1) = x_ij' beta + u_j,  u_j ~ N(0, tau^2).

    주변 가능도의 적분을 집단마다 라플라스 근사로 계산한다. u_j의 조건부 최빈값을 뉴턴 방법으로
    찾고, 그 점에서의 2차 미분으로 적분을 근사한다.
    """

    def __init__(self, y, X, group):
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.levels, self.g = np.unique(np.asarray(group), return_inverse=True)
        self.J = len(self.levels)
        self.sum_y = np.bincount(self.g, self.y, self.J)

    def _modes(self, eta0, tau, iters=100):
        """집단마다 u_j의 조건부 최빈값과 그 점에서의 -g''. 0에서 출발하는 뉴턴 방법."""
        u = np.zeros(self.J)
        for _ in range(iters):
            p = expit(eta0 + u[self.g])
            grad = self.sum_y - np.bincount(self.g, p, self.J) - u / tau**2
            hess = np.bincount(self.g, p * (1 - p), self.J) + 1 / tau**2      # -g''
            step = np.clip(grad / hess, -2, 2)
            u += step
            if np.max(np.abs(step)) < 1e-10:
                break
        p = expit(eta0 + u[self.g])
        return u, np.bincount(self.g, p * (1 - p), self.J) + 1 / tau**2

    def _nll(self, par):
        beta, tau = par[:-1], np.exp(par[-1])
        eta0 = self.X @ beta
        u, hess = self._modes(eta0, tau)
        eta = eta0 + u[self.g]
        ll = np.sum(self.y * eta - np.logaddexp(0, eta))
        return -(ll - np.sum(u**2) / (2 * tau**2) - 0.5 * np.sum(np.log(tau**2 * hess)))

    def fit(self):
        k = self.X.shape[1]
        start = np.zeros(k + 1)
        start[-1] = np.log(0.5)
        res = minimize(self._nll, start, method="BFGS", options={"gtol": 1e-5})
        self.par = res.x
        H = approx_hess(res.x, self._nll)
        self.cov = np.linalg.inv(H)
        self.beta, self.tau = res.x[:-1], float(np.exp(res.x[-1]))
        self.se_beta = np.sqrt(np.diag(self.cov))[:-1]
        self.se_log_tau = float(np.sqrt(self.cov[-1, -1]))
        self.loglik = -res.fun
        self.u, hess = self._modes(self.X @ self.beta, self.tau)
        self.u_sd = 1 / np.sqrt(hess)          # u_j의 조건부 표준편차(고정 효과와 tau를 추정치로 고정)
        return self


class LinearMixed:
    """선형 혼합모형 y = X beta + sum_k Z_k u_k + e. u_k ~ N(0, s_k^2 I), e ~ N(0, s^2 I)이고 서로 독립이다.

    Z_k는 집단의 표지 행렬이거나(무작위 절편) 그것에 변수를 곱한 것이다(무작위 기울기). 성분끼리
    겹치거나 교차해도 된다. V = I + sum_k g_k Z_k Z_k'을 통째로 계산하므로 관측이 수천 개를
    넘으면 느리다. g_k = s_k^2 / s^2이고, beta와 s^2은 g가 주어지면 식으로 구해지므로 g만 찾는다.
    """

    def __init__(self, reml: bool = True):
        self.reml = reml

    def _profile(self, logg):
        n, p = self.X.shape
        V = np.eye(n)
        for g, ZZ in zip(np.exp(logg), self._ZZ):
            V += g * ZZ
        L = np.linalg.cholesky(V)
        Xs, ys = np.linalg.solve(L, self.X), np.linalg.solve(L, self.y)
        XtX = Xs.T @ Xs
        beta = np.linalg.solve(XtX, Xs.T @ ys)
        r = ys - Xs @ beta
        df = n - p if self.reml else n
        s2 = float(r @ r) / df
        ll = -0.5 * (df * np.log(s2) + 2 * np.log(np.diag(L)).sum() + df * (1 + np.log(2 * np.pi)))
        if self.reml:
            ll -= 0.5 * np.linalg.slogdet(XtX)[1]
        return ll, beta, s2, XtX, L

    def fit(self, y, X, Z):
        """Z는 행렬의 목록이다. 빈 목록이면 보통의 최소제곱이다."""
        self.y, self.X = np.asarray(y, float), np.asarray(X, float)
        self.Z = [np.asarray(z, float) for z in Z]
        self._ZZ = [z @ z.T for z in self.Z]
        best = None
        for start in (0.0, -2.0, 2.0) if self.Z else (0.0,):
            res = minimize(lambda t: -self._profile(t)[0], np.full(len(self.Z), start), method="L-BFGS-B",
                           bounds=[(-12, 8)] * len(self.Z)) if self.Z else None
            x = res.x if res is not None else np.zeros(0)
            val = self._profile(x)[0]
            if best is None or val > best[0]:
                best = (val, x)
        self.loglik, logg = best
        _, self.beta, self.s2, XtX, L = self._profile(logg)
        self.cov = self.s2 * np.linalg.inv(XtX)
        self.se = np.sqrt(np.diag(self.cov))
        self.var = np.exp(logg) * self.s2                     # 성분마다의 분산
        r = self.y - self.X @ self.beta
        Vinv_r = np.linalg.solve(L.T, np.linalg.solve(L, r))
        self.u = [g * z.T @ Vinv_r for g, z in zip(np.exp(logg), self.Z)]   # 무작위 효과의 예측값
        return self


def indicator(labels) -> np.ndarray:
    """집단의 표지 행렬. 행은 관측, 열은 집단이다."""
    codes, _ = pd.factorize(pd.Series(list(labels)))
    Z = np.zeros((len(codes), codes.max() + 1))
    Z[np.arange(len(codes)), codes] = 1.0
    return Z
