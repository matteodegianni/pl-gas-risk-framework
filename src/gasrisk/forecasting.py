"""Stage 3: walk-forward price forecasting with a random-walk benchmark.

Usage::

    python -m gasrisk.forecasting

Series: TTF (roll-adjusted index of the second month held to roll), EUR/PLN
(ECB), TGEgasDA (trade date) and the location plus tenor basis (TGE day-ahead
spot in EUR minus TTF second month). Horizons: 1, 5 and 21 business days.

Design:

* Target: the level at t+h. Price series are modelled in logs, so the primary
  loss is the squared log-price error (scale-free across the 2021-2022 crisis);
  the basis can be negative and is modelled in EUR/MWh. Errors in price units
  are reported too.
* Walk-forward: rolling ``train_window`` observations; parameters re-estimated
  every ``refit_every`` trading days; between re-estimations the model state is
  updated daily with fixed parameters. No observation after the forecast
  origin is ever used (tested).
* Models: random walk (benchmark), ARIMA (ARMA on log returns, or on the basis
  level, order by AIC), ETS (specification by AIC), gradient boosting (direct
  h-step model on lag, rolling and cross-asset features) and, for TGE and the
  basis, a linear direct regression (for TGE including an error-correction term
  that pulls TGE back toward TTF x EUR/PLN), with a step-by-step ablation.
* Day-of-week effects: TGE trades on Friday are for weekend delivery, so TGE
  and basis models get day-of-week terms.
* Evaluation: RMSE, MAE, directional accuracy (binomial test against 50%) and
  the Diebold-Mariano test against the random walk with the
  Harvey-Leybourne-Newbold correction for overlapping h-step errors.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from gasrisk.config import Config, load_config

# --------------------------------------------------------------------------- data


@dataclass
class SeriesData:
    """One forecast target with everything the models need."""

    name: str
    y: pd.Series                  # modelled variable: log price, or basis level in EUR/MWh
    is_log: bool                  # True: level = exp(y)
    arima_on_level: bool          # True: ARMA on y itself (stationary basis); False: ARMA on diff(y)
    seasonal: bool                # day-of-week terms (TGE and basis)
    features: pd.DataFrame        # predictors known at each date (for ML and ECM)
    dow: pd.DataFrame             # day-of-week dummies (deterministic, known in advance)
    ecm_features: pd.DataFrame | None = None
    unit: str = ""


def _dow_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.get_dummies(pd.Series(index.day_of_week, index=index), prefix="dow").astype(float)
    return d.drop(columns=[c for c in d.columns if c == "dow_0"])  # Monday is the base


def build_series(panel: pd.DataFrame, cross_lag: int = 0) -> list[SeriesData]:
    """Build the four forecast targets and their feature sets from the Stage 1 panel.

    Timing of every input at row t (forecast made after the close of day t):
    TGEgasDA trade date t (volume weighted price of trades on t), TTF settlement
    of day t (around 17:00 CET), ECB EUR/PLN reference rate of day t (early
    afternoon CET). Targets are values at t+h, h >= 1. ``tests/test_forecasting.py``
    checks mechanically that no feature at t changes when data after t change.

    Args:
        cross_lag: Placebo switch. Extra days by which every TTF- and FX-derived
            input of the TGE and basis models is lagged (0 = normal). The target
            itself is unchanged; for the basis, its own level and changes still
            contain TTF, so only the explicit TTF and FX inputs are lagged.
    """
    idx = panel.index
    ttf_ret = panel["ttf_risk_ret"].fillna(0.0)  # two excluded roll days contribute zero, as in Stage 2
    y_ttf = np.log(panel["ttf_c2"].iloc[0]) + ttf_ret.cumsum()
    y = {
        "ttf": y_ttf,
        "eurpln": np.log(panel["eurpln"]),
        "tge": np.log(panel["tge_pln"]),
        "basis": panel["basis_tge_ttfc2_eur"],
    }
    chg = {k: v.diff() for k, v in y.items()}
    # log ratio of TGE (EUR) to TTF: the error-correction term (positive, unlike the EUR basis)
    z = np.log(panel["tge_eur"] / panel["ttf_c2"])
    dow = _dow_dummies(idx)
    L = int(cross_lag)

    def x(series: pd.Series, own: str) -> pd.Series:
        """Apply the placebo lag to a TTF/FX-derived input of the TGE or basis models."""
        return series.shift(L) if (L and own in ("tge", "basis")) else series

    def feats(own: str) -> pd.DataFrame:
        f = pd.DataFrame(index=idx)
        r = chg[own]
        for k in (0, 1, 2, 4, 9, 20):
            f[f"own_chg_lag{k}"] = r.shift(k)
        f["own_sum5"] = r.rolling(5).sum()
        f["own_sum21"] = r.rolling(21).sum()
        for w in (5, 21, 63):
            f[f"own_sd{w}"] = r.rolling(w).std()
        for other in ("ttf", "tge", "eurpln"):
            if other != own:
                lag = (lambda v: x(v, own)) if other in ("ttf", "eurpln") else (lambda v: v)
                f[f"{other}_chg"] = lag(chg[other])
                f[f"{other}_sum5"] = lag(chg[other].rolling(5).sum())
        f["z_tge_ttf"] = x(z, own)
        f["z_chg5"] = x(z.diff(5), own)
        f["dow"] = idx.day_of_week.astype(float)
        if own == "basis":
            f["level"] = y["basis"]
        return f

    # Linear direct models. Column order follows the ablation groups (see ABLATION_GROUPS).
    ecm = pd.concat([dow, chg["tge"].rename("tge_chg"), chg["tge"].shift(1).rename("tge_chg_lag1"),
                     x(chg["ttf"], "tge").rename("ttf_chg"), x(chg["eurpln"], "tge").rename("eurpln_chg"),
                     x(z, "tge").rename("z_tge_ttf")], axis=1)
    basis_lin = pd.concat([y["basis"].rename("basis_level"), dow, chg["basis"].rename("basis_chg"),
                           chg["basis"].shift(1).rename("basis_chg_lag1"), x(chg["ttf"], "basis").rename("ttf_chg"),
                           x(chg["eurpln"], "basis").rename("eurpln_chg")], axis=1)
    return [
        SeriesData("ttf", y["ttf"], True, False, False, feats("ttf"), dow, unit="EUR/MWh"),
        SeriesData("eurpln", y["eurpln"], True, False, False, feats("eurpln"), dow, unit="PLN per EUR"),
        SeriesData("tge", y["tge"], True, False, True, feats("tge"), dow, ecm_features=ecm, unit="PLN/MWh"),
        SeriesData("basis", y["basis"], False, True, True, feats("basis"), dow, ecm_features=basis_lin, unit="EUR/MWh"),
    ]


def training_rows(fit_pos: int, h: int, window: int) -> range:
    """Rows usable to fit a direct h-step model at ``fit_pos``.

    A row i is usable only if its target y[i+h] is observed by ``fit_pos``,
    i.e. i + h <= fit_pos. This is the guard against look-ahead.
    """
    return range(max(0, fit_pos - window + 1), fit_pos - h + 1)


# Cumulative feature groups for the linear-model ablation: each step adds one component.
ABLATION_GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "tge": [
        ("day_of_week", ["dow_1", "dow_2", "dow_3", "dow_4"]),
        ("+ own reversal", ["tge_chg", "tge_chg_lag1"]),
        ("+ same-day TTF and FX moves", ["ttf_chg", "eurpln_chg"]),
        ("+ error-correction term", ["z_tge_ttf"]),
    ],
    "basis": [
        ("mean reversion (level)", ["basis_level"]),
        ("+ day_of_week", ["dow_1", "dow_2", "dow_3", "dow_4"]),
        ("+ own reversal", ["basis_chg", "basis_chg_lag1"]),
        ("+ same-day TTF and FX moves", ["ttf_chg", "eurpln_chg"]),
    ],
}


# --------------------------------------------------------------------------- models


class RandomWalk:
    name = "random_walk"

    def fit(self, sd: SeriesData, pos: int, cfg: dict) -> None:
        pass

    def predict(self, sd: SeriesData, pos: int, horizons: list[int]) -> dict[int, float]:
        return {h: float(sd.y.iloc[pos]) for h in horizons}


class ArimaModel:
    """ARMA(p, q) with constant on log returns (or on the basis level), optional day-of-week exog."""

    name = "arima"

    def __init__(self) -> None:
        self.res = None
        self.last = -1
        self.order: tuple[int, int] | None = None

    def _endog(self, sd: SeriesData) -> pd.Series:
        return sd.y if sd.arima_on_level else sd.y.diff()

    def fit(self, sd: SeriesData, pos: int, cfg: dict) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        e = self._endog(sd)
        lo = max(1, pos - cfg["train_window"] + 1)
        endog = e.iloc[lo:pos + 1].to_numpy()
        exog = sd.dow.iloc[lo:pos + 1].to_numpy() if sd.seasonal else None
        best = None
        for p, q in cfg["arima_orders"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = ARIMA(endog, exog=exog, order=(p, 0, q), trend="c").fit()
                except Exception:  # noqa: BLE001 - a candidate that fails to converge is skipped
                    continue
            if best is None or r.aic < best.aic:
                best, self.order = r, (p, q)
        self.res, self.last = best, pos

    def predict(self, sd: SeriesData, pos: int, horizons: list[int]) -> dict[int, float]:
        e = self._endog(sd)
        if pos > self.last:  # daily state update with fixed parameters
            new = e.iloc[self.last + 1:pos + 1].to_numpy()
            exog = sd.dow.iloc[self.last + 1:pos + 1].to_numpy() if sd.seasonal else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.res = self.res.append(new, exog=exog)
            self.last = pos
        H = max(horizons)
        fexog = sd.dow.iloc[pos + 1:pos + 1 + H].to_numpy() if sd.seasonal else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = np.asarray(self.res.forecast(H, exog=fexog))
        if sd.arima_on_level:
            return {h: float(f[h - 1]) for h in horizons}
        cum = np.cumsum(f)
        return {h: float(sd.y.iloc[pos] + cum[h - 1]) for h in horizons}


class EtsModel:
    """Exponential smoothing on y; candidates ANN and AAdN, plus ANA (weekly) for seasonal series."""

    name = "ets"

    def __init__(self) -> None:
        self.spec: dict | None = None
        self.params = None
        self.lo = 0

    @staticmethod
    def _specs(seasonal: bool) -> list[dict]:
        s = [dict(error="add", trend=None), dict(error="add", trend="add", damped_trend=True)]
        if seasonal:
            s.append(dict(error="add", trend=None, seasonal="add", seasonal_periods=5))
        return s

    def fit(self, sd: SeriesData, pos: int, cfg: dict) -> None:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        self.lo = max(0, pos - cfg["train_window"] + 1)
        endog = sd.y.iloc[self.lo:pos + 1].to_numpy()
        best = None
        for spec in self._specs(sd.seasonal):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = ETSModel(endog, **spec).fit(disp=False)
                except Exception:  # noqa: BLE001
                    continue
            if best is None or r.aic < best[0].aic:
                best = (r, spec)
        self.params, self.spec = best[0].params, best[1]

    def predict(self, sd: SeriesData, pos: int, horizons: list[int]) -> dict[int, float]:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        endog = sd.y.iloc[self.lo:pos + 1].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = np.asarray(ETSModel(endog, **self.spec).smooth(self.params).forecast(max(horizons)))
        return {h: float(f[h - 1]) for h in horizons}


class DirectRegression:
    """Direct h-step regression of y[t+h] - y[t] on features at t (one model per horizon)."""

    def __init__(self, kind: str, gbm_params: dict | None = None, seed: int = 0) -> None:
        self.kind = kind
        self.name = {"gbm": "gbm", "ecm": "linear"}[kind]
        self.gbm_params = gbm_params or {}
        self.seed = seed
        self.models: dict[int, object] = {}
        self.columns: list[str] = []

    def _X(self, sd: SeriesData) -> pd.DataFrame:
        return sd.features if self.kind == "gbm" else sd.ecm_features

    def _new_model(self):
        if self.kind == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(random_state=self.seed, **self.gbm_params)
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    def fit(self, sd: SeriesData, pos: int, cfg: dict, horizons: list[int] | None = None) -> None:
        X = self._X(sd)
        self.columns = list(X.columns)
        Xv, yv = X.to_numpy(), sd.y.to_numpy()
        for h in horizons or cfg["horizons"]:
            rows = np.array(training_rows(pos, h, cfg["train_window"]))
            target = yv[rows + h] - yv[rows]
            ok = ~np.isnan(Xv[rows]).any(axis=1) & ~np.isnan(target)
            m = self._new_model()
            m.fit(Xv[rows][ok], target[ok])
            self.models[h] = m

    def predict(self, sd: SeriesData, pos: int, horizons: list[int]) -> dict[int, float]:
        x = self._X(sd).iloc[[pos]].to_numpy()
        if np.isnan(x).any():
            return {h: float("nan") for h in horizons}
        return {h: float(sd.y.iloc[pos] + self.models[h].predict(x)[0]) for h in horizons}


def models_for(sd: SeriesData, cfg: dict, seed: int) -> list:
    m = [RandomWalk(), ArimaModel(), EtsModel(), DirectRegression("gbm", cfg["gbm"], seed)]
    if sd.ecm_features is not None:
        m.append(DirectRegression("ecm"))
    return m


# --------------------------------------------------------------------------- walk-forward


def walk_forward(sd: SeriesData, cfg: dict, seed: int, models: list | None = None,
                 horizons: list[int] | None = None) -> pd.DataFrame:
    """Run models (default: all for this series) on one series. One row per origin, horizon and model."""
    horizons = list(horizons or cfg["horizons"])
    start, n = int(cfg["train_window"]), len(sd.y)
    last_origin = n - 1 - min(horizons)
    idx, y = sd.y.index, sd.y.to_numpy()
    rows = []
    for model in models or models_for(sd, cfg, seed):
        for pos in range(start, last_origin + 1):
            if (pos - start) % int(cfg["refit_every"]) == 0:
                if isinstance(model, DirectRegression):
                    model.fit(sd, pos, cfg, horizons)
                else:
                    model.fit(sd, pos, cfg)
            hs = [h for h in horizons if pos + h <= n - 1]
            preds = model.predict(sd, pos, hs)
            for h in hs:
                rows.append((sd.name, model.name, h, idx[pos], idx[pos + h], y[pos], y[pos + h], preds[h],
                             getattr(model, "order", None) and str(model.order)))
    return pd.DataFrame(rows, columns=["series", "model", "h", "origin", "target_date", "y_origin", "y_true", "y_pred", "detail"])


# --------------------------------------------------------------------------- evaluation


def hln_factor(n: int, h: int) -> float:
    """Harvey-Leybourne-Newbold (1997) small-sample correction for the DM statistic."""
    return float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))


def diebold_mariano_full(e_model: np.ndarray, e_bench: np.ndarray, h: int, kernel: str = "rectangular") -> dict:
    """Diebold-Mariano test on squared-error loss, HAC variance with h-1 lags, HLN correction.

    d_t = e_model^2 - e_bench^2. h-step errors from daily origins overlap, so d_t
    is MA(h-1) under the null and its long-run variance is estimated from
    autocovariances at lags 0..h-1:

    * ``kernel="rectangular"``: equal weights, as in Diebold-Mariano (1995) and
      the HLN (1997) correction (default, used for the verdicts). If this
      estimate is not positive, the lag-0 variance is used and ``fallback`` is True.
    * ``kernel="bartlett"``: Newey-West weights 1 - k/h, always non-negative
      (sensitivity check).

    The statistic is multiplied by the HLN factor and compared with t(n-1).
    """
    d = np.asarray(e_model) ** 2 - np.asarray(e_bench) ** 2
    n = len(d)
    dc = d - d.mean()
    gamma = np.array([np.dot(dc[k:], dc[:n - k]) / n for k in range(h)])
    if kernel == "rectangular":
        w = np.ones(h)
    elif kernel == "bartlett":
        w = 1 - np.arange(h) / h
    else:
        raise ValueError("kernel must be 'rectangular' or 'bartlett'")
    lrv = gamma[0] + 2 * np.sum(w[1:] * gamma[1:])
    fallback = bool(lrv <= 0)
    if fallback:
        lrv = gamma[0]
    stat = d.mean() / np.sqrt(lrv / n) * hln_factor(n, h)
    return {"stat": float(stat), "p_two": float(2 * stats.t.sf(abs(stat), df=n - 1)),
            "p_better": float(stats.t.cdf(stat, df=n - 1)), "fallback": fallback, "hac_lags": h - 1}


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> tuple[float, float, float]:
    """(statistic, two-sided p, one-sided p for "model more accurate") from :func:`diebold_mariano_full`."""
    r = diebold_mariano_full(e_model, e_bench, h)
    return r["stat"], r["p_two"], r["p_better"]


def directional_accuracy(y_origin: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int, float]:
    """Hit rate of predicted direction; days with zero actual or predicted change are excluded.

    Returns (hit rate, n, one-sided binomial p-value against 50%).
    """
    a, p = np.sign(y_true - y_origin), np.sign(y_pred - y_origin)
    ok = (a != 0) & (p != 0) & ~np.isnan(p)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0, float("nan")
    hits = int((a[ok] == p[ok]).sum())
    return hits / n, n, float(stats.binomtest(hits, n, 0.5, alternative="greater").pvalue)


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values (family-wise error rate control)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def add_holm(metrics: pd.DataFrame, prespecified: tuple[str, ...], alpha: float) -> pd.DataFrame:
    """Holm-adjust the one-sided DM p-values.

    * ``holm_p_all``: family = every model, horizon and series (primary).
    * ``holm_p_prespecified``: family = TGE and basis only, the series Stage 2
      identified as candidates before Stage 3 was run (secondary).
    """
    out = metrics.copy()
    tested = out["model"] != "random_walk"
    out["holm_p_all"] = np.nan
    out.loc[tested, "holm_p_all"] = holm(out.loc[tested, "dm_p_better"].to_numpy())
    pre = tested & out["series"].isin(prespecified)
    out["holm_p_prespecified"] = np.nan
    out.loc[pre, "holm_p_prespecified"] = holm(out.loc[pre, "dm_p_better"].to_numpy())
    out["beats_rw_after_holm_all"] = np.where(tested, out["holm_p_all"] < alpha, np.nan)
    return out


def evaluate(fc: pd.DataFrame, series: dict[str, SeriesData], alpha: float) -> pd.DataFrame:
    """Metrics per series, horizon and model, with DM tests against the random walk."""
    rows = []
    for (s, h), g in fc.groupby(["series", "h"]):
        rw = g[g["model"] == "random_walk"].set_index("origin")
        for m, gm in g.groupby("model"):
            gm = gm.set_index("origin").dropna(subset=["y_pred"])
            common = gm.index.intersection(rw.index)
            gm, r = gm.loc[common], rw.loc[common]
            e = (gm["y_pred"] - gm["y_true"]).to_numpy()
            e_rw = (r["y_pred"] - r["y_true"]).to_numpy()
            if series[s].is_log:
                lvl_e = np.exp(gm["y_pred"]) - np.exp(gm["y_true"])
            else:
                lvl_e = gm["y_pred"] - gm["y_true"]
            row = {"series": s, "h": int(h), "model": m, "n": len(e),
                   "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))),
                   "rmse_level": float(np.sqrt(np.mean(lvl_e**2))), "mae_level": float(np.mean(np.abs(lvl_e))),
                   "rmse_ratio_vs_rw": float(np.sqrt(np.mean(e**2)) / np.sqrt(np.mean(e_rw**2)))}
            if m == "random_walk":
                row.update(dir_acc=np.nan, dir_n=0, dir_p=np.nan, dm_stat=np.nan, dm_p_two=np.nan, dm_p_better=np.nan,
                           verdict="benchmark")
            else:
                acc, nd, pd_ = directional_accuracy(gm["y_origin"].to_numpy(), gm["y_true"].to_numpy(), gm["y_pred"].to_numpy())
                dm = diebold_mariano_full(e, e_rw, int(h))
                dm_nw = diebold_mariano_full(e, e_rw, int(h), kernel="bartlett")
                stat, p2, pb = dm["stat"], dm["p_two"], dm["p_better"]
                verdict = ("beats random walk" if pb < alpha else
                           "worse than random walk" if (1 - pb) < alpha else "no significant difference")
                row.update(dir_acc=acc, dir_n=nd, dir_p=pd_, dm_stat=stat, dm_p_two=p2, dm_p_better=pb, verdict=verdict,
                           dm_hac_lags=dm["hac_lags"], dm_lrv_fallback=dm["fallback"], dm_p_better_newey_west=dm_nw["p_better"])
            rows.append(row)
    out = pd.DataFrame(rows)
    out = add_holm(out, prespecified=("tge", "basis"), alpha=alpha)
    order = {"random_walk": 0, "arima": 1, "ets": 2, "gbm": 3, "linear": 4}
    return out.sort_values(["series", "h", "model"], key=lambda c: c.map(order) if c.name == "model" else c).reset_index(drop=True)


def evaluate_by_subperiod(fc: pd.DataFrame, subperiods: dict[str, list]) -> pd.DataFrame:
    """RMSE ratio vs random walk by sub-period of the target date."""
    rows = []
    for name, (a, b) in subperiods.items():
        sel = fc[(fc["target_date"] >= pd.Timestamp(a)) & ((fc["target_date"] <= pd.Timestamp(b)) if b else True)]
        for (s, h), g in sel.groupby(["series", "h"]):
            rw = g[g["model"] == "random_walk"].set_index("origin")
            rmse_rw = np.sqrt(np.mean((rw["y_pred"] - rw["y_true"]) ** 2))
            for m, gm in g.groupby("model"):
                gm = gm.set_index("origin").dropna(subset=["y_pred"])
                if gm.empty:
                    continue
                rows.append({"subperiod": name, "series": s, "h": int(h), "model": m, "n": len(gm),
                             "rmse_ratio_vs_rw": float(np.sqrt(np.mean((gm["y_pred"] - gm["y_true"]) ** 2)) / rmse_rw)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- interpretation helpers


def ecm_insample(sd: SeriesData) -> pd.DataFrame:
    """Full-sample OLS of the 1-day TGE change on the error-correction term, with HAC errors (descriptive)."""
    import statsmodels.api as sm

    X = sd.ecm_features.copy()
    dy = sd.y.shift(-1) - sd.y
    df = pd.concat([dy.rename("dy"), X], axis=1).dropna()
    res = sm.OLS(df["dy"], sm.add_constant(df.drop(columns="dy"))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    out = pd.DataFrame({"coef": res.params, "hac_se": res.bse, "t": res.tvalues, "p": res.pvalues}).reset_index(names="term")
    b = res.params["z_tge_ttf"]
    half_life = np.log(0.5) / np.log(1 + b) if -1 < b < 0 else np.nan
    extra = pd.DataFrame([{"term": "implied_half_life_days", "coef": half_life, "hac_se": np.nan, "t": np.nan, "p": np.nan},
                          {"term": "r_squared", "coef": res.rsquared, "hac_se": np.nan, "t": np.nan, "p": np.nan},
                          {"term": "n", "coef": res.nobs, "hac_se": np.nan, "t": np.nan, "p": np.nan}])
    return pd.concat([out, extra], ignore_index=True)


def basis_ar1(sd: SeriesData) -> pd.DataFrame:
    """Full-sample AR(1) of the basis level with HAC errors and implied half-life (descriptive)."""
    import statsmodels.api as sm

    y = sd.y.dropna()
    df = pd.concat([y.rename("y"), y.shift(1).rename("y_lag1")], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df["y_lag1"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    phi = res.params["y_lag1"]
    return pd.DataFrame([{"metric": "phi", "value": phi}, {"metric": "phi_hac_se", "value": res.bse["y_lag1"]},
                         {"metric": "implied_half_life_days", "value": np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan},
                         {"metric": "long_run_mean_eur_mwh", "value": res.params["const"] / (1 - phi)},
                         {"metric": "n", "value": res.nobs}])


def gbm_importance(sd: SeriesData, cfg: dict, seed: int, h: int) -> pd.DataFrame:
    """Permutation importance of the gradient boosting model fitted on the last window (h-step)."""
    from sklearn.inspection import permutation_importance

    m = DirectRegression("gbm", cfg["gbm"], seed)
    pos = len(sd.y) - 1
    m.fit(sd, pos, cfg, [h])
    rows = np.array(training_rows(pos, h, cfg["train_window"]))
    X = sd.features.to_numpy()[rows]
    y = sd.y.to_numpy()[rows + h] - sd.y.to_numpy()[rows]
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    imp = permutation_importance(m.models[h], X[ok], y[ok], n_repeats=10, random_state=seed)
    return (pd.DataFrame({"feature": m.columns, "importance_mean": imp.importances_mean, "importance_sd": imp.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def _placebo_one(sd: SeriesData, cfg: dict, seed: int, horizons: list[int]) -> pd.DataFrame:
    models = [RandomWalk(), DirectRegression("ecm"), DirectRegression("gbm", cfg["gbm"], seed)]
    return walk_forward(sd, cfg, seed, models, horizons)


def placebo(panel: pd.DataFrame, fc: pd.DataFrame, cfg: dict, seed: int, horizons: list[int]) -> pd.DataFrame:
    """Placebo: lag every TTF and FX input of the TGE and basis models by one more day.

    If the 1-day gain came from information that is not really available at the
    forecast time, it would vanish here only partly; if it comes from the
    genuine same-day TTF move (settled at 17:00 CET, before the forecast), most
    of it should disappear. ``gain_surviving`` = (1 - placebo ratio) / (1 - original ratio).
    """
    from joblib import Parallel, delayed

    plac = {s.name: s for s in build_series(panel, cross_lag=1)}
    names = ["tge", "basis"]
    parts = Parallel(n_jobs=2)(delayed(_placebo_one)(plac[n], cfg, seed, horizons) for n in names)
    rows = []
    for name, pf in zip(names, parts):
        for h in horizons:
            rw = pf[(pf["model"] == "random_walk") & (pf["h"] == h)].set_index("origin")
            for m in ("linear", "gbm"):
                a = pf[(pf["model"] == m) & (pf["h"] == h)].set_index("origin").dropna(subset=["y_pred"])
                o = fc[(fc["series"] == name) & (fc["model"] == m) & (fc["h"] == h)].set_index("origin").dropna(subset=["y_pred"])
                common = a.index.intersection(o.index).intersection(rw.index)
                a, o, r = a.loc[common], o.loc[common], rw.loc[common]
                e_p, e_o, e_rw = (a["y_pred"] - a["y_true"]).to_numpy(), (o["y_pred"] - o["y_true"]).to_numpy(), (r["y_pred"] - r["y_true"]).to_numpy()
                rmse = lambda e: float(np.sqrt(np.mean(e**2)))  # noqa: E731
                ratio_o, ratio_p = rmse(e_o) / rmse(e_rw), rmse(e_p) / rmse(e_rw)
                rows.append({"series": name, "h": h, "model": m, "n": len(common),
                             "rmse_ratio_original": ratio_o, "rmse_ratio_placebo": ratio_p,
                             "dm_p_better_original": diebold_mariano_full(e_o, e_rw, h)["p_better"],
                             "dm_p_better_placebo": diebold_mariano_full(e_p, e_rw, h)["p_better"],
                             "gain_surviving": (1 - ratio_p) / (1 - ratio_o) if ratio_o < 1 else np.nan})
    return pd.DataFrame(rows)


def ablation(sd: SeriesData, cfg: dict, seed: int, horizons: list[int]) -> pd.DataFrame:
    """Out-of-sample contribution of each feature group in the linear direct model.

    Each step re-runs the full walk-forward with one more feature group and is
    compared with the random walk (RMSE ratio, DM test) and with the previous step.
    """
    from dataclasses import replace

    rows, used = [], []
    rw = walk_forward(sd, cfg, seed, [RandomWalk()], horizons)
    prev = None
    for label, cols in ABLATION_GROUPS[sd.name]:
        used = used + cols
        sub = replace(sd, ecm_features=sd.ecm_features[used])
        fc = walk_forward(sub, cfg, seed, [DirectRegression("ecm")], horizons)
        for h in horizons:
            a = fc[fc["h"] == h].set_index("origin").dropna(subset=["y_pred"])
            b = rw[rw["h"] == h].set_index("origin").loc[a.index]
            e, e_rw = (a["y_pred"] - a["y_true"]).to_numpy(), (b["y_pred"] - b["y_true"]).to_numpy()
            stat, _, pb = diebold_mariano(e, e_rw, h)
            row = {"series": sd.name, "h": h, "step": label, "features": len(used), "n": len(e),
                   "rmse_ratio_vs_rw": float(np.sqrt(np.mean(e**2)) / np.sqrt(np.mean(e_rw**2))),
                   "dm_p_better_vs_rw": pb,
                   "dir_acc": directional_accuracy(a["y_origin"].to_numpy(), a["y_true"].to_numpy(), a["y_pred"].to_numpy())[0]}
            if prev is not None:
                pa = prev[prev["h"] == h].set_index("origin").loc[a.index]
                e_prev = (pa["y_pred"] - pa["y_true"]).to_numpy()
                row["dm_p_better_vs_previous_step"] = diebold_mariano(e, e_prev, h)[2]
            else:
                row["dm_p_better_vs_previous_step"] = np.nan
            rows.append(row)
        prev = fc
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- run


def _run_one(sd: SeriesData, cfg: dict, seed: int) -> pd.DataFrame:
    return walk_forward(sd, cfg, seed)


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    from joblib import Parallel, delayed

    fcfg = dict(cfg["forecasting"])
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    series = build_series(panel)
    parts = Parallel(n_jobs=int(fcfg["n_jobs"]))(delayed(_run_one)(sd, fcfg, cfg.seed) for sd in series)
    fc = pd.concat(parts, ignore_index=True)
    fc.to_csv(cfg.path("clean_dir") / "forecasts.csv", index=False)  # contains price levels: gitignored

    by_name = {s.name: s for s in series}
    alpha = float(fcfg["alpha"])
    tables = {
        "metrics": evaluate(fc, by_name, alpha),
        "metrics_by_subperiod": evaluate_by_subperiod(fc, cfg["sample"]["subperiods"]),
        "placebo": placebo(panel, fc, fcfg, cfg.seed, [1, 5]),
        "ablation": pd.concat([ablation(by_name[k], fcfg, cfg.seed, [1, 5]) for k in ("tge", "basis")], ignore_index=True),
        "tge_ecm_insample": ecm_insample(by_name["tge"]),
        "basis_ar1_insample": basis_ar1(by_name["basis"]),
        "tge_gbm_importance_h1": gbm_importance(by_name["tge"], fcfg, cfg.seed, 1),
        "basis_gbm_importance_h1": gbm_importance(by_name["basis"], fcfg, cfg.seed, 1),
        "arima_orders_chosen": (fc[fc["model"] == "arima"].dropna(subset=["detail"]).groupby(["series", "detail"])
                                ["origin"].nunique().rename("origins").reset_index()),
        "test_window": (fc.groupby(["series", "h"]).agg(first_origin=("origin", "min"), last_origin=("origin", "max"),
                                                        first_target=("target_date", "min"), last_target=("target_date", "max"))
                        .reset_index()),
    }
    out = cfg.path("outputs_dir")
    for name, t in tables.items():
        t.to_csv(out / f"stage3_{name}.csv", index=False)
    from gasrisk.forecasting_report import rmse_ratio_figure, write_stage3_report

    rmse_ratio_figure(tables["metrics"], out / "figures" / "stage3_rmse_ratio.png")
    write_stage3_report(cfg, tables, panel)
    return tables


def main() -> None:
    cfg = load_config()
    t = run(cfg)["metrics"]
    print(t[["series", "h", "model", "rmse_ratio_vs_rw", "dm_p_better", "verdict"]].to_string(index=False))
    print(f"Report: {cfg.root / cfg['paths']['outputs_dir'] / 'stage3_report.md'}")


if __name__ == "__main__":
    main()
