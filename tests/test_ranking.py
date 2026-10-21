"""Configurable ranking: equivalence with the original formula, weights, missing values, filters."""

from __future__ import annotations

import pandas as pd
import pytest

from psma_pipeline.ranking import criteria_table, rank_templates, template_ranking

MODE = {"criteria": [{"column": "Resolution (Å)", "direction": "lower", "weight": 1},
                     {"column": "Clashscore", "direction": "lower", "weight": 1},
                     {"column": "Best UniProt coverage", "direction": "higher", "weight": 1}],
        "missing": "ignore"}


def crit():
    return pd.DataFrame({"Method": ["X-RAY", "X-RAY", "EM"], "Resolution (Å)": [1.5, 3.0, 2.0],
                         "Clashscore": [2.0, 15.0, None], "Best UniProt coverage": [0.94, 0.94, 0.99]},
                        index=pd.Index(["A", "B", "C"], name="PDB ID"))


def original_formula(c: pd.DataFrame) -> pd.Series:
    """The scoring code of the original notebook, verbatim in spirit."""
    ranks = []
    for col, higher in (("Resolution (Å)", False), ("Clashscore", False), ("Best UniProt coverage", True)):
        r = pd.to_numeric(c[col], errors="coerce").rank(ascending=not higher)
        n = r.notna().sum()
        ranks.append((r - 1) / (n - 1) if n > 1 else r * 0)
    return (1 - pd.concat(ranks, axis=1).mean(axis=1)).round(3)


def test_equal_weights_reproduce_original_scores():
    out = rank_templates(crit(), MODE).set_index("PDB ID")
    expected = original_formula(crit())
    for pid in expected.index:
        assert out.loc[pid, "Score (0-1, higher = better)"] == pytest.approx(expected[pid])
    assert list(out.sort_values("Rank").index) == list(expected.sort_values(ascending=False).index)


def test_missing_value_policies():
    ignore = rank_templates(crit(), MODE).set_index("PDB ID")
    worst = rank_templates(crit(), {**MODE, "missing": "worst"}).set_index("PDB ID")
    assert "Clashscore: missing (ignore)" in ignore.loc["C", "Score explanation"]
    assert worst.loc["C", "component: Clashscore"] == 1.0
    assert worst.loc["C", "Score (0-1, higher = better)"] < ignore.loc["C", "Score (0-1, higher = better)"]


def test_weights_change_the_order():
    heavy = {**MODE, "criteria": [{**c, "weight": 10 if c["column"] == "Best UniProt coverage" else 1}
                                  for c in MODE["criteria"]]}
    assert rank_templates(crit(), heavy).iloc[0]["PDB ID"] == "C"


def test_filters_and_audit_columns():
    out = rank_templates(crit(), {**MODE, "include": {"Method": "X-RAY"},
                                  "exclude": {"Resolution (Å)": {"min": 2.5}}}, "xray")
    assert list(out["PDB ID"]) == ["A"]
    assert out.loc[0, "Ranking mode"] == "xray"
    for col in ("component: Resolution (Å)", "Criteria used", "Score explanation"):
        assert col in out
    with pytest.raises(KeyError):
        rank_templates(crit(), {**MODE, "include": {"No such column": 1}})


def test_all_missing_gives_no_score():
    c = crit().assign(**{"Resolution (Å)": None, "Clashscore": None, "Best UniProt coverage": None})
    out = rank_templates(c, MODE)
    assert out["Score (0-1, higher = better)"].isna().all()


def test_template_ranking_from_tables(offline_survey):
    t = offline_survey.tables
    out = template_ranking(t["Summary"], t["Chains"], t["ActiveSite"])
    assert out.iloc[0]["PDB ID"] == "1AAA"
    assert out.columns[:2].tolist() == ["PDB ID", "Rank"]          # original column order
    assert set(criteria_table(t["Summary"], t["Chains"], t["ActiveSite"]).index) == set(t["Summary"]["PDB ID"])
