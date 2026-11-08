"""Tests for the Stage 18 number check on the risk commentary."""

from gasrisk.commentary import check_numbers, template

FACTS = {"var": {"value": 18323059.7, "printed": "18,323,060 PLN"}, "util": {"value": 0.7329, "printed": "73.3%"},
         "x": {"value": 37.64, "printed": "37.6 times"}, "as_of": {"value": "2026-11-06", "printed": "2026-11-06"}}


def test_exact_figures_pass_and_altered_figures_fail() -> None:
    assert check_numbers("VaR is 18,323,060 PLN, 73.3% of the limit, on 2026-11-06, a 10-day view.", FACTS) == []
    assert check_numbers("VaR is 18,323,000 PLN, about 73% of the limit and 38 times VaR.", FACTS) == ["18,323,000", "73%", "38"]


def test_partial_matches_are_not_accepted() -> None:
    assert check_numbers("Utilisation 3.3% and 323,060.", FACTS) == ["3.3%", "323,060"]


def test_template_uses_only_fact_sheet_numbers() -> None:
    f = {k: {"value": 0, "printed": p} for k, p in {
        "as_of": "2026-11-06", "official_var99_pln": "1,000 PLN", "official_es975_pln": "1,200 PLN", "var_limit_utilisation": "40.0%",
        "backtest_zone": "green", "open_gas_limit_utilisation": "101.5%", "fx_open_limit_utilisation": "115.6%",
        "worst_stress_scenario": "worst 21-day crisis move", "worst_stress_loss_pln": "9,999 PLN", "worst_stress_x_var": "9.9 times",
        "initial_margin_eur": "500 EUR", "liquidity_buffer_eur": "700 EUR", "buffer_share_of_credit_line": "28.3%",
        "worst_margin_call_eur": "1,700 EUR", "worst_margin_call_share_of_line": "71.5%", "counterparties_in_breach": "1",
        "worst_counterparty": "CP04", "worst_counterparty_utilisation": "14.35 times its limit"}.items()}
    f["counterparties_in_breach"]["value"] = 1
    assert check_numbers(template(f), f) == []
