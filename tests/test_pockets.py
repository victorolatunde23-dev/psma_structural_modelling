"""Binding-site contacts, metal coordination, glycosylation and active-site B-factors
on a tiny hand-built structure (requires gemmi)."""

from __future__ import annotations

import pytest

from psma_pipeline.config import load_config
from psma_pipeline.pockets import coordination_distances

import fixtures as F

MMCIF_SITE = """data_SITE
_cell.length_a 100
_cell.length_b 100
_cell.length_c 100
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_symmetry.space_group_name_H-M 'P 1'
loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
3 non-polymer
4 water
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 HIS
1 2 ASP
1 3 ASN
1 4 GLY
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
C 3
D 4
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N NE2 . HIS A 1 1 ? 52.05 50.00 50.00 1.00 20.0 377 A 1
ATOM   2 C CA  . HIS A 1 1 ? 55.00 50.00 50.00 1.00 20.0 377 A 1
ATOM   3 O OD1 . ASP A 1 2 ? 50.00 52.10 50.00 1.00 30.0 387 A 1
ATOM   4 C CA  . ASP A 1 2 ? 50.00 55.00 50.00 0.50 30.0 387 A 1
ATOM   5 N ND2 . ASN A 1 3 ? 60.00 60.00 60.00 1.00 40.0 476 A 1
ATOM   6 C CA  . ASN A 1 3 ? 62.00 62.00 62.00 1.00 40.0 476 A 1
ATOM   7 C CA  . GLY A 1 4 ? 70.00 70.00 70.00 1.00 50.0 477 A 1
HETATM 8 ZN ZN  . ZN  B 2 . ? 50.00 50.00 50.00 1.00 25.0 801 A 1
HETATM 9 C C1  . NAG C 3 . ? 61.45 60.00 60.00 1.00 60.0 901 A 1
HETATM 10 O O  . HOH D 4 . ? 50.00 50.00 52.00 1.00 35.0 1001 A 1
"""

SEGS = {"A": [{"entity_id": "1", "chain_id": "A", "seq_start": s, "seq_end": s, "unp_start": u, "unp_end": u}
              for s, u in ((1, 377), (2, 387), (3, 476), (4, 477))]}


@pytest.fixture
def ctx():
    gemmi = pytest.importorskip("gemmi")
    from psma_pipeline.assemblies import coordinate_analysis
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(MMCIF_SITE).sole_block())
    st.setup_entities()
    return {"coords": coordinate_analysis(st, {"1"}), "segs": SEGS, "method": "X-RAY DIFFRACTION",
            "resolution": 2.0}


def test_zinc_coordination_and_contacts(ctx):
    from psma_pipeline.pockets import binding_site_analysis
    cfg = load_config()
    rows = binding_site_analysis("TEST", ctx, {("A", 377, ""): {"clash"}}, cfg.ligand_classes,
                                 cfg.require("contacts"), F.USEQ)
    assert [r["Component"] for r in rows] == ["ZN"]          # glycan and water are not binding-site rows
    zn = rows[0]
    assert zn["Protein residues (UniProt/chain)"] == "H377/A, D387/A"
    dist = dict(coordination_distances(zn))
    assert dist["H377/A NE2"] == pytest.approx(2.05, abs=0.01)
    assert dist["D387/A OD1"] == pytest.approx(2.10, abs=0.01)
    assert zn["Coordination number"] == 3                    # His, Asp and one water
    assert zn["Waters in contact"] == 1 and zn["Ligand atom–water pairs"] == 1
    assert zn["Symmetry-mate contact"] is False
    assert zn["Cutoff (Å)"] == 2.8
    assert "H377/A: clash" in zn["Validation outliers in site"]
    assert zn["Residue/UniProt mismatches"] == ""
    assert zn["Chain (label)"] == "B" and zn["Residue no."] == "801" and zn["Model"] == 1


def test_glycosylation_site_in_uniprot_and_author_numbering(ctx):
    from psma_pipeline.glycans import find_glycosylation_sites
    cfg = load_config()
    rows = find_glycosylation_sites("TEST", ctx, cfg.ligand_classes, cfg.require("contacts"))
    assert len(rows) == 1
    r = rows[0]
    assert r["Asn (UniProt)"] == 476 and r["Asn (author)"] == "476"
    assert r["C1-ND2 (Å)"] == pytest.approx(1.45, abs=0.01)
    assert r["Root sugar"] == "NAG901"


def test_active_site_b_factors_and_occupancy(ctx):
    from psma_pipeline.active_site import active_site_analysis
    rows = active_site_analysis("TEST", ctx, {377: "Zn1 ligand", 387: "bridging", 424: "catalytic"},
                                F.USEQ, {}, {"N", "CA", "C", "O", "OXT"})
    by = {r["UniProt residue"]: r for r in rows}
    assert by["E424"]["Modelled"] is False
    assert by["H377"]["Matches UniProt"] is True and by["H377"]["Author residue number"] == "377"
    assert by["D387"]["Min occupancy"] == 0.5
    # chain B values 20,20,30,30,40,40,50: mean 32.86, population SD 10.30
    assert by["H377"]["Normalised B (z)"] == pytest.approx(-1.25, abs=0.01)
    assert by["D387"]["Normalised B (z)"] == pytest.approx(-0.28, abs=0.01)
    assert by["H377"]["Chain (label)"] == "A" and by["H377"]["Entity"] == "1"


MMCIF_EDGE = """data_EDGE
_cell.length_a 20
_cell.length_b 40
_cell.length_c 40
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_symmetry.space_group_name_H-M 'P 1'
loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 HIS
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N NE2 . HIS A 1 1 ? 18.60 20.00 20.00 1.00 20.0 377 A 1
HETATM 2 ZN ZN  . ZN  B 2 . ? 0.50 20.00 20.00 1.00 25.0 801 A 1
"""


def test_contact_across_cell_edge_uses_nearest_image():
    """Zn at x=0.5 and His NE2 at x=18.6 in a 20 Å cell are 1.9 Å apart through
    the cell edge (18.1 Å directly): the contact must be found, with the
    correct distance, and marked as a lattice image."""
    gemmi = pytest.importorskip("gemmi")
    from psma_pipeline.assemblies import coordinate_analysis
    from psma_pipeline.pockets import binding_site_analysis
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(MMCIF_EDGE).sole_block())
    st.setup_entities()
    ctx = {"coords": coordinate_analysis(st, {"1"}), "segs": {"A": [SEGS["A"][0]]},
           "method": "X-RAY DIFFRACTION", "resolution": 2.0}
    cfg = load_config()
    zn = binding_site_analysis("EDGE", ctx, {}, cfg.ligand_classes, cfg.require("contacts"), F.USEQ)[0]
    assert dict(coordination_distances(zn))["H377/A* NE2"] == pytest.approx(1.90, abs=0.01)
    assert zn["Symmetry-mate contact"] is True
