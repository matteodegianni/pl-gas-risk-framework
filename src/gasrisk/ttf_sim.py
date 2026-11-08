"""Shared TTF GARCH fit and path simulation (Stages 10, 12 and 13).

The Stage 4 specification for TTF (mean equation and Student-t innovations) is refitted on the last
``var.garch_window`` roll-adjusted returns up to the as-of date, as in the Stage 5 VaR engine.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gasrisk.config import Config
from gasrisk.volatility import MeanSpec, garch_model, garch_variance_path


@dataclass
class TTFGarch:
    omega: float
    alpha: float
    beta: float
    nu: float
    sigma2_next: float          # one-step-ahead conditional variance (decimal returns squared)
    std_resid: np.ndarray       # standardised residuals of the fit, demeaned and rescaled to unit variance (FHS)
    mean_name: str
    params: pd.Series
    spec: MeanSpec
    resid_last: float
    sigma2_last: float

    def variance_path(self, h: int) -> np.ndarray:
        """Conditional variance forecasts t+1..t+h in percent squared (as in Stage 4)."""
        return garch_variance_path(self.params, self.resid_last, self.sigma2_last, h)


def fit_ttf_garch(cfg: Config, rets: pd.Series, as_of: pd.Timestamp) -> TTFGarch:
    selected = pd.read_csv(cfg.path("outputs_dir") / "stage4_selected_spec.csv").set_index("series").loc["ttf"]
    cand = {d["name"]: MeanSpec.from_cfg(d) for d in cfg["volatility"]["mean_candidates"]}
    spec = cand[selected["mean"]]
    y = rets.loc[:as_of].dropna().iloc[-int(cfg["var"]["garch_window"]):] * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = garch_model(y, selected["dist"], spec).fit(disp="off", options={"maxiter": 2000})
    p = res.params
    rl = float(np.asarray(res.resid)[-1])
    s2l = float(np.asarray(res.conditional_volatility)[-1]) ** 2
    s2n = float(garch_variance_path(p, rl, s2l, 1)[0]) / 1e4
    z = np.asarray(res.std_resid)
    z = z[~np.isnan(z)]
    z = (z - z.mean()) / z.std(ddof=0)  # demeaned and unit variance so bootstrapped shocks keep the price driftless
    return TTFGarch(float(p["omega"]) / 1e4, float(p["alpha[1]"]), float(p["beta[1]"]), float(p.get("nu", np.inf)),
                    s2n, z, selected["mean"], p, spec, rl, s2l)


def simulate_returns(m: TTFGarch, n_paths: int, n_days: int, rng: np.random.Generator, innovations: str = "fhs",
                     martingale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1) daily log returns (n_paths x n_days) and the conditional variance of each day.

    ``innovations``: "fhs" bootstraps the standardised residuals; "t" draws unit-variance Student-t shocks.
    With ``martingale`` the log return is shifted by -sigma2/2 so that the futures price has no drift (a futures
    price is a martingale under the pricing measure); the AR mean of Stage 4 is not used for multi-day paths.
    """
    r = np.empty((n_paths, n_days))
    s2 = np.empty((n_paths, n_days))
    cur = np.full(n_paths, m.sigma2_next)
    for d in range(n_days):
        if innovations == "fhs":
            z = rng.choice(m.std_resid, size=n_paths, replace=True)
        else:
            z = rng.standard_t(m.nu, size=n_paths) / np.sqrt(m.nu / (m.nu - 2))
        s2[:, d] = cur
        eps = np.sqrt(cur) * z
        r[:, d] = eps - 0.5 * cur if martingale else eps
        cur = m.omega + m.alpha * eps**2 + m.beta * cur
    return r, s2
