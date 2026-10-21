"""Configuration loading and validation of configured residues against the sequence."""

from __future__ import annotations

import pytest
import yaml

from psma_pipeline.config import ConfigError, load_config, parse_residue_code, validate_residue_codes
from psma_pipeline.ligands import classify_component

import fixtures as F


def test_default_config_loads_target_files():
    cfg = load_config()
    assert cfg.accession == "Q04609"
    assert cfg.lid["range"] == (541, 548)
    assert cfg.lid["known_states"] == {"3BHX": "closed", "2OOT": "open"}
    assert "H377" in cfg.active_site["literature_sites"]
    assert "ZN" in cfg.ligand_classes["ions_of_interest"]
    assert cfg.get("ranking.modes.global.criteria")[0]["column"] == "Resolution (Å)"


def test_overrides_are_deep_merged():
    cfg = load_config(overrides={"contacts": {"ligand": 4.5}})
    assert cfg.get("contacts.ligand") == 4.5 and cfg.get("contacts.metal") == 2.8


def test_every_configured_residue_matches_canonical_sequence():
    cfg = load_config()
    codes = list(cfg.active_site["literature_sites"]) + cfg.lid["arene_site"]
    valid, dropped = validate_residue_codes(codes, F.USEQ)
    assert dropped == [] and len(valid) == len(codes)


def test_mismatched_residue_is_reported_not_silently_used():
    valid, dropped = validate_residue_codes(["H377", "W377", "K9999"], F.USEQ)
    assert valid == ["H377"]
    assert "not W377" in dropped[0]["Reason"] and "outside sequence" in dropped[1]["Reason"]


def test_bad_residue_code():
    assert parse_residue_code("W541") == ("W", 541)
    with pytest.raises(ConfigError):
        parse_residue_code("Trp541")


def test_accession_mismatch_between_files_is_an_error(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    base = load_config()
    for name in ("psma_active_site.yaml", "psma_regions.yaml", "ligand_classes.yaml"):
        (cfg_dir / name).write_text((base.config_dir / name).read_text())
    main = yaml.safe_load((base.config_dir / "default_config.yaml").read_text())
    main["target"]["accession"] = "P00533"
    (cfg_dir / "default_config.yaml").write_text(yaml.safe_dump(main))
    with pytest.raises(ConfigError, match="is for Q04609"):
        load_config(cfg_dir / "default_config.yaml")


def test_unquoted_numeric_pdb_id_is_rejected():
    with pytest.raises(ConfigError, match="quote PDB IDs"):
        load_config(overrides={"validation_set": {"extra_ids": [1e10]}})


def test_ligand_classes_from_config():
    lc = load_config().ligand_classes
    assert classify_component("zn", lc) == "ion of interest"
    assert classify_component("NAG", lc) == "glycan"
    assert classify_component("GOL", lc) == "additive / buffer"
    assert classify_component("XYZ", lc) == "ligand of interest"


def test_offline_active_site_definition(offline_survey):
    d = offline_survey.tables["ActiveSiteDefinition"]
    assert (d["Status"] == "OK").all()
    assert {377, 424}.issubset(set(offline_survey.sites))
    assert "UniProt binding site: Zn(2+)" in offline_survey.sites[377]
