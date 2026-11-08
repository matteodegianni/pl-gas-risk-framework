"""Stage 8: stress tests and sensitivities of the SYNTHETIC book (today's positions, full revaluation).

Usage::

    python -m gasrisk.stress

* Sensitivities: P&L per +1 PLN/MWh TGE (EUR/PLN fixed), per +1 EUR/MWh TTF and per +0.01 EUR/PLN
  (TGE price in EUR fixed, so the PLN cost of TGE rises), plus a grid of larger moves.
* Historical scenarios: the worst k-day joint moves of TGE (EUR), TTF and EUR/PLN for today's book inside the
  2021-2022 crisis window, and named episodes; cumulative log returns applied to today's levels.
* Hypothetical shocks: TTF +50% alone, gas complex +/-50%, EUR/PLN +5%, combined, TGE-TTF basis widening.
* Reverse stress: the TGE rise (TTF and EUR/PLN unchanged) and the gas-complex rise that would lose exactly
  the 99% VaR limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from gasrisk.config import Config, load_config
from gasrisk.var import build_factors, pnl_legs, positions


def _legs_row(name: str, lv: dict, rets: np.ndarray, pos: dict, extra: dict | None = None) -> dict:
    legs = pnl_legs(lv, np.asarray(rets, dtype=float).reshape(1, 3), pos)
    return {"scenario": name, **(extra or {}), "pnl_tge_leg": legs["tge"][0], "pnl_ttf_leg": legs["ttf"][0],
            "pnl_fx_leg": legs["fx"][0], "pnl_total": legs["total"][0]}


def level_shock_returns(lv: dict, d_tge_pln: float = 0.0, d_ttf_eur: float = 0.0, d_fx: float = 0.0) -> np.ndarray:
    """Log returns for absolute moves: TGE in PLN/MWh (EUR/PLN fixed), TTF in EUR/MWh, EUR/PLN in PLN."""
    g = lv["tge_eur"]
    return np.array([np.log((g + d_tge_pln / lv["fx"]) / g), np.log((lv["ttf"] + d_ttf_eur) / lv["ttf"]),
                     np.log((lv["fx"] + d_fx) / lv["fx"])])


def sensitivities(lv: dict, pos: dict) -> pd.DataFrame:
    rows = [
        _legs_row("+1 PLN/MWh TGE", lv, level_shock_returns(lv, d_tge_pln=1.0), pos),
        _legs_row("+1 EUR/MWh TTF", lv, level_shock_returns(lv, d_ttf_eur=1.0), pos),
        _legs_row("+0.01 EUR/PLN", lv, level_shock_returns(lv, d_fx=0.01), pos),
    ]
    return pd.DataFrame(rows)


def sensitivity_grid(lv: dict, pos: dict, grid: dict) -> pd.DataFrame:
    rows = []
    for d in grid["tge_pln"]:
        rows.append(_legs_row(f"TGE {d:+g} PLN/MWh", lv, level_shock_returns(lv, d_tge_pln=d), pos))
    for d in grid["ttf_eur"]:
        rows.append(_legs_row(f"TTF {d:+g} EUR/MWh", lv, level_shock_returns(lv, d_ttf_eur=d), pos))
    for d in grid["fx"]:
        rows.append(_legs_row(f"EUR/PLN {d:+g}", lv, level_shock_returns(lv, d_fx=d), pos))
    return pd.DataFrame(rows)


def worst_windows(rets: pd.DataFrame, lv: dict, pos: dict, window: list, ks: list[int]) -> pd.DataFrame:
    """Worst k-day cumulative joint move for today's book inside ``window``."""
    r = rets.loc[window[0]:window[1]].dropna()
    rows = []
    for k in ks:
        cum = r.rolling(k).sum().dropna()
        pnl = pnl_legs(lv, cum.to_numpy(), pos)["total"]
        i = int(np.argmin(pnl))
        end = cum.index[i]
        start = r.index[r.index.get_loc(end) - k + 1]
        move = cum.iloc[i].to_numpy()
        rows.append(_legs_row(f"worst {k}-day crisis move", lv, move, pos,
                              {"from": start.date(), "to": end.date(), "move_tge_eur": np.expm1(move[0]),
                               "move_ttf": np.expm1(move[1]), "move_fx": np.expm1(move[2])}))
    return pd.DataFrame(rows)


def named_episodes(rets: pd.DataFrame, lv: dict, pos: dict, episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for e in episodes:
        r = rets.loc[e["start"]:e["end"]].iloc[1:].fillna(0.0)  # close of start to close of end
        move = r.sum().to_numpy()
        rows.append(_legs_row(e["name"], lv, move, pos, {"from": r.index[0].date(), "to": r.index[-1].date(),
                                                          "move_tge_eur": np.expm1(move[0]), "move_ttf": np.expm1(move[1]),
                                                          "move_fx": np.expm1(move[2])}))
    return pd.DataFrame(rows)


def hypothetical(lv: dict, pos: dict, shocks: list[dict], basis: list[float]) -> pd.DataFrame:
    rows = [_legs_row(s["name"], lv, np.log1p([s["tge"], s["ttf"], s["fx"]]), pos) for s in shocks]
    for b in basis:
        rows.append(_legs_row(f"basis widening: TGE +{b:g} EUR/MWh, TTF unchanged", lv,
                              [np.log((lv["tge_eur"] + b) / lv["tge_eur"]), 0.0, 0.0], pos))
    return pd.DataFrame(rows)


def reverse_stress(lv: dict, pos: dict, loss: float) -> pd.DataFrame:
    """Moves that lose exactly ``loss`` PLN."""
    f_tge = lambda x: pnl_legs(lv, np.array([[np.log1p(x), 0, 0]]), pos)["total"][0] + loss  # noqa: E731
    f_gas = lambda x: pnl_legs(lv, np.array([[np.log1p(x), np.log1p(x), 0]]), pos)["total"][0] + loss  # noqa: E731
    out = [{"scenario": "TGE rise alone (TTF, EUR/PLN unchanged)", "required_move": optimize.brentq(f_tge, 0, 50)}]
    try:
        out.append({"scenario": "gas complex rise (TGE and TTF equally)", "required_move": optimize.brentq(f_gas, 0, 50)})
    except ValueError:
        out.append({"scenario": "gas complex rise (TGE and TTF equally)", "required_move": np.nan})
    return pd.DataFrame(out).assign(loss_pln=loss)


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    s = cfg["stress"]
    panel = pd.read_csv(cfg.path("clean_dir") / "panel.csv", index_col="date", parse_dates=["date"])
    levels, rets = build_factors(panel)
    as_of = levels.index[-1]
    lv = {f: float(levels[f].iloc[-1]) for f in levels.columns}
    pos = positions(cfg["book"], lv["tge_eur"])
    out = cfg.path("outputs_dir")
    official = pd.read_csv(out / "stage7_official_method.csv")["method"].iloc[0]
    today = pd.read_csv(out / "stage5_var_today.csv").set_index("method").loc[official]
    hist = pd.concat([worst_windows(rets, lv, pos, s["crisis_window"], list(s["windows_days"])),
                      named_episodes(rets, lv, pos, s["named_episodes"])], ignore_index=True)
    hyp = hypothetical(lv, pos, s["hypothetical"], list(s["basis_widening_eur_mwh"]))
    for t in (hist, hyp):
        t["loss_over_var99"] = -t["pnl_total"] / today["var_99"]
        t["loss_over_es97_5"] = -t["pnl_total"] / today["es_97.5"]
    tables = {
        "sensitivities": sensitivities(lv, pos),
        "sensitivity_grid": sensitivity_grid(lv, pos, s["sensitivity_grid"]),
        "historical": hist,
        "hypothetical": hyp,
        "reverse_stress": reverse_stress(lv, pos, float(cfg["limits"]["var99_limit_pln"])),
        "reference": pd.DataFrame([{"as_of": as_of.date(), "official_method": official, "var_99": today["var_99"],
                                    "es_97.5": today["es_97.5"], "var99_limit_pln": cfg["limits"]["var99_limit_pln"]}]),
    }
    for k, t in tables.items():
        t.to_csv(out / f"stage8_{k}.csv", index=False)
    from gasrisk.stress_report import write_stage8_report

    write_stage8_report(cfg, tables)
    return tables


def main() -> None:
    t = run(load_config())
    for k in ("sensitivities", "historical", "hypothetical", "reverse_stress"):
        print(t[k].to_string(index=False))


if __name__ == "__main__":
    main()
