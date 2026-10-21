"""Opt-in integration test on real PDB entries (needs internet and gemmi).

Run with:  PSMA_NETWORK_TESTS=1 pytest -q -m network
"""

from __future__ import annotations

import pytest

from psma_pipeline import load_config
from psma_pipeline.sanity import critical_failures, run_sanity_checks


@pytest.mark.network
def test_calibration_structures_end_to_end(tmp_path):
    pytest.importorskip("gemmi")
    from psma_pipeline.survey import run_survey
    cfg = load_config(overrides={"paths": {"outputs": str(tmp_path)}})
    res = run_survey(cfg, ids=["3BHX", "2XEG"])
    s = res.tables["Summary"].set_index("PDB ID")
    assert set(s.index) == {"3BHX", "2XEG"}
    assert (s["Monomer / dimer"] == "dimer").all()          # PSMA is a homodimer
    sanity = run_sanity_checks(res.tables, cfg)
    assert critical_failures(sanity).empty, critical_failures(sanity).to_string()
