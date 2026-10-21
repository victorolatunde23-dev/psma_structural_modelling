"""Shared pytest fixtures.

Run all offline tests:        pytest -q
Include network tests:        PSMA_NETWORK_TESTS=1 pytest -q -m network
Tests that need gemmi are skipped automatically when gemmi is not installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures as F  # noqa: E402
from psma_pipeline import load_config  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs internet access (opt-in: PSMA_NETWORK_TESTS=1)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PSMA_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test; set PSMA_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def cfg(tmp_path):
    """Default configuration with cache/outputs redirected to a temporary folder."""
    return load_config(overrides={"paths": {"cache": str(tmp_path / "cache"),
                                            "outputs": str(tmp_path / "outputs")},
                                  "run": {"use_coordinates": False}})


@pytest.fixture
def offline_survey(cfg, monkeypatch):
    """Run the survey on synthetic metadata with every network call stubbed."""
    from psma_pipeline import survey
    meta = F.metadata()
    monkeypatch.setattr(survey, "rcsb_search", lambda s, a, url: sorted(meta))
    monkeypatch.setattr(survey, "rcsb_metadata", lambda s, ids, c, url, **k: {i: meta[i] for i in ids})
    monkeypatch.setattr(survey, "uniprot_entry", lambda s, a, c, url, **k: F.uniprot_entry())
    monkeypatch.setattr(survey, "fetch_best_structures", lambda s, a, c, url, **k: F.best_structures())
    monkeypatch.setattr(survey, "rcsb_validation",
                        lambda s, ids, c, url, **k: ({p: v for p, v in F.rcsb_validation().items() if p in ids},
                                                     "split"))
    monkeypatch.setattr(survey, "fetch_sifts", lambda s, p, c, url, **k: F.sifts(p))
    monkeypatch.setattr(survey, "pdbe_validation", lambda s, p, c, u1, u2, **k: F.pdbe_validation(p))
    return survey.run_survey(cfg)
