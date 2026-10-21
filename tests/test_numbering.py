"""SIFTS label_seq ↔ UniProt mapping, author/label chain mapping, construct differences."""

from __future__ import annotations

import pytest

from psma_pipeline.numbering import (construct_differences, label_to_unp, parse_ranges, to_ranges,
                                     unp_to_label)
from psma_pipeline.pdbe import sifts_segments


def seg(s, e, us, ue, chain="A", asym="A", ent="1"):
    return {"entity_id": ent, "chain_id": chain, "seq_start": s, "seq_end": e, "unp_start": us, "unp_end": ue}


def test_ranges_round_trip():
    nums = [1, 2, 3, 7, 9, 10]
    assert to_ranges(nums) == "1-3, 7, 9-10"
    assert parse_ranges(to_ranges(nums)) == nums
    assert to_ranges([]) == ""
    assert to_ranges([5, 5, 4]) == "4-5"


def test_label_to_unp_offset_for_tagged_construct():
    # 8-residue tag, then UniProt 44-750 at label_seq 9-715
    m = label_to_unp([seg(9, 715, 44, 750)])
    assert m[9] == 44 and m[715] == 750
    assert 1 not in m and 8 not in m            # tag residues have no UniProt number
    assert m[9 + (424 - 44)] == 424             # catalytic E424
    assert unp_to_label([seg(9, 715, 44, 750)])[424] == 389


def test_inconsistent_segment_is_not_numbered():
    m = label_to_unp([seg(1, 10, 44, 60), seg(20, 29, 100, 109)])
    assert 1 not in m                           # 10 label residues vs 17 UniProt: skipped
    assert m[20] == 100 and m[29] == 109


def test_construct_differences_tag_mutation_and_gap():
    useq = "M" * 43 + "ACDEFGHIKL" + "MNPQRSTVWY"     # UniProt 44-63
    # tag GS, UniProt 44-50 with F48 -> A, deletion of 51-55, insertion 'GG', then 56-63
    seq = "GS" + "ACDEAGH" + "GG" + useq[55:63]           # useq[55:63] = UniProt 56-63
    segs = [seg(3, 9, 44, 50), seg(12, 19, 56, 63)]
    d = construct_differences(seq, segs, useq)
    assert d["n_ext"] == "GS" and d["c_ext"] == ""
    assert d["mutations"] == "F48A"
    assert "UniProt 51-55 absent from construct" in d["gaps"]
    assert "non-UniProt insertion after 50: GG" in d["gaps"]
    assert d["unp_range"] == "44-50; 56-63"
    assert d["coverage"] == pytest.approx(15 / 63, abs=1e-3)


def test_construct_without_sifts():
    d = construct_differences("ACD", [], "ACD")
    assert d["unp_range"] == "no SIFTS mapping" and d["coverage"] is None


def test_sifts_segments_uses_label_asym_and_keeps_author_chain():
    sifts = {"1abc": {"UniProt": {"Q04609": {"mappings": [
        {"entity_id": 1, "chain_id": "H", "struct_asym_id": "B",
         "start": {"residue_number": 30}, "end": {"residue_number": 40}, "unp_start": 130, "unp_end": 140},
        {"entity_id": 1, "chain_id": "H", "struct_asym_id": "B",
         "start": {"residue_number": 1}, "end": {"residue_number": 20}, "unp_start": 44, "unp_end": 63},
    ]}}}}
    s = sifts_segments(sifts, "1ABC", "Q04609")
    assert list(s) == ["B"]                                  # keyed by label_asym_id
    assert [x["seq_start"] for x in s["B"]] == [1, 30]       # sorted by label_seq
    assert all(x["chain_id"] == "H" for x in s["B"])         # author chain preserved
    assert sifts_segments(sifts, "1ABC", "P00000") == {}


def test_offline_survey_numbering(offline_survey):
    ch = offline_survey.tables["Chains"].set_index(["PDB ID", "Chain (auth)"])
    assert ch.loc[("2BBB", "A"), "N-terminal extension (non-UniProt)"] == "GSHHHHHH"
    assert ch.loc[("2BBB", "A"), "Substitutions vs UniProt"] == "E424A"
    assert ch.loc[("1AAA", "B"), "UniProt range (SIFTS)"] == "44-750"
    assert ch.loc[("3CCC", "A"), "UniProt range (SIFTS)"] == "no SIFTS mapping"
    assert ch.loc[("3CCC", "A"), "Calculation status"] == "WARNING"
