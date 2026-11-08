"""Airflow DAG for the daily gas risk pipeline (Stage 16).

Why Airflow rather than cron: the run has 14 dependent steps, three of which call external APIs (Instrat, ECB, NBP)
that fail intermittently; Airflow gives per-step retries, a visible dependency graph, backfills for missed days and an
audit trail of each run, and stops downstream steps when validation fails. A single cron line calling
``python -m gasrisk.pipeline`` is a valid simpler alternative for one desk (the pipeline already stops at the first
failure and writes outputs/pipeline_log.csv); Airflow is the choice once the run must be monitored by a team.

Each task runs one pipeline step in the Docker image (``docker build -t gasrisk .``); with a local virtualenv replace
the command prefix by ``python -m gasrisk.pipeline``.

Schedule: weekdays 19:30 Europe/Warsaw, after the ICE Endex TTF settlement (about 17:30 London) and the ECB reference
rate (about 16:00 CET); TGE day-ahead for the next delivery day is published by then.
"""

from __future__ import annotations

from datetime import timedelta

STEPS = ["refresh", "validate_inputs", "var", "backtest", "limits", "stress", "attribution", "pricing", "hedging",
         "margin", "credit", "export", "validate_outputs", "report"]
# Steps that only need the validated panel can run in parallel after validate_inputs; the rest follow their inputs.
UPSTREAM = {
    "refresh": [],
    "validate_inputs": ["refresh"],
    "var": ["validate_inputs"],
    "attribution": ["validate_inputs"],
    "pricing": ["validate_inputs"],
    "backtest": ["var"],
    "limits": ["backtest"],
    "stress": ["limits"],
    "hedging": ["pricing"],
    "margin": ["stress"],
    "credit": ["pricing"],
    "export": ["limits", "stress", "attribution", "hedging", "margin", "credit"],
    "validate_outputs": ["export"],
    "report": ["validate_outputs"],
}
CMD = ("docker run --rm -v {{ var.value.get('gasrisk_repo', '/opt/gasrisk') }}/data:/app/data "
       "-v {{ var.value.get('gasrisk_repo', '/opt/gasrisk') }}/outputs:/app/outputs "
       "-v {{ var.value.get('gasrisk_repo', '/opt/gasrisk') }}/exports:/app/exports gasrisk --step ")

try:  # the DAG file stays importable (and testable) without Airflow installed
    import pendulum
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:  # pragma: no cover
    DAG = None

if DAG is not None:
    with DAG(
        dag_id="gasrisk_daily",
        description="Daily market, margin and credit risk run for the Polish gas supply book",
        schedule="30 19 * * 1-5",
        start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Warsaw"),
        catchup=False,
        max_active_runs=1,
        default_args={"retries": 2, "retry_delay": timedelta(minutes=10), "owner": "risk"},
        tags=["risk", "gas"],
    ) as dag:
        tasks = {s: BashOperator(task_id=s, bash_command=CMD + s, retries=3 if s == "refresh" else 2,
                                 execution_timeout=timedelta(minutes=30)) for s in STEPS}
        for s, ups in UPSTREAM.items():
            for u in ups:
                tasks[u] >> tasks[s]
