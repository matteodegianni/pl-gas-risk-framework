"""Stage 16: daily risk pipeline in one command.

Usage::

    python -m gasrisk.pipeline                 # full daily run: refresh, validate, risk, export, validate, report
    python -m gasrisk.pipeline --offline       # rebuild from stored raw files (no API calls)
    python -m gasrisk.pipeline --step var      # one step (used by the Airflow DAG)
    python -m gasrisk.pipeline --monthly       # also re-run Stages 2 and 4 (diagnostics, GARCH re-selection)

Daily steps, in order (each step reads only what earlier steps wrote):

refresh -> validate_inputs -> var -> backtest -> limits -> stress -> attribution -> pricing -> hedging -> margin ->
credit -> export -> validate_outputs -> report

The pipeline stops at the first failing step; a failed input or output validation stops publication of the report.
Stage 3 forecasting is research only and is not part of the daily run.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

import pandas as pd

from gasrisk.config import Config, load_config

DAILY_STEPS = ["refresh", "validate_inputs", "var", "backtest", "limits", "stress", "attribution", "pricing",
               "hedging", "margin", "credit", "export", "validate_outputs", "report"]
MONTHLY_STEPS = ["diagnostics", "volatility"]   # inserted after validate_inputs with --monthly


class StepFailed(RuntimeError):
    pass


def _module_run(name: str, cfg: Config, **kw):
    return importlib.import_module(f"gasrisk.{name}").run(cfg, **kw)


def run_step(step: str, cfg: Config, offline: bool) -> str:
    from gasrisk import validation

    if step == "refresh":
        _module_run("data.build", cfg, offline=offline)
        return "panel rebuilt"
    if step == "validate_inputs":
        df = validation.run(cfg, inputs=True, outputs=False)
        bad = df[(~df["passed"]) & (df["severity"] == "error")]
        if len(bad):
            raise StepFailed(f"input validation failed: {', '.join(bad['id'])}")
        return f"{len(df)} input checks passed"
    if step == "validate_outputs":
        df = validation.run(cfg, inputs=True, outputs=True)
        bad = df[(~df["passed"]) & (df["severity"] == "error")]
        if len(bad):
            raise StepFailed(f"output validation failed: {', '.join(bad['id'])}")
        return f"{len(df)} checks passed"
    if step == "report":
        from gasrisk.daily_report import write_daily_report

        return write_daily_report(cfg)
    if step in ("pricing", "hedging"):
        _module_run(step, cfg, offline=offline)
    else:
        _module_run(step, cfg)
    return "ok"


def run(cfg: Config, offline: bool = False, steps: list[str] | None = None, monthly: bool = False) -> pd.DataFrame:
    if steps is None:
        steps = list(DAILY_STEPS)
        if monthly:
            steps[2:2] = MONTHLY_STEPS
    log = []
    for s in steps:
        t0 = time.time()
        try:
            msg = run_step(s, cfg, offline)
            log.append({"step": s, "status": "ok", "seconds": round(time.time() - t0, 1), "message": msg})
            print(f"[ok]   {s:<17} {time.time() - t0:6.1f}s  {msg}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and stop
            log.append({"step": s, "status": "failed", "seconds": round(time.time() - t0, 1), "message": str(exc)})
            print(f"[FAIL] {s:<17} {exc}", flush=True)
            break
    df = pd.DataFrame(log)
    df.to_csv(cfg.path("outputs_dir") / "pipeline_log.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily risk pipeline.")
    ap.add_argument("--offline", action="store_true", help="rebuild from stored raw files, no API calls")
    ap.add_argument("--step", choices=DAILY_STEPS + MONTHLY_STEPS, help="run a single step")
    ap.add_argument("--monthly", action="store_true", help="include diagnostics and GARCH re-selection")
    a = ap.parse_args()
    cfg = load_config()
    if a.step:
        # a single step (Airflow task) must not overwrite the full-run log
        try:
            print(run_step(a.step, cfg, a.offline))
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {a.step}: {exc}")
            sys.exit(1)
        return
    df = run(cfg, a.offline, monthly=a.monthly)
    sys.exit(0 if (df["status"] == "ok").all() and len(df) else 1)


if __name__ == "__main__":
    main()
