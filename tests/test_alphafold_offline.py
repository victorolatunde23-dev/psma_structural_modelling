"""Milestone 4 and membrane context: offline tests (synthetic models, no network, no gemmi).

Covers the exit criteria: PAE indexing and chain boundaries, and the confident-dimer rule,
plus AlphaFold2 run inside Colab (ColabFold, faked) through to membrane-ready inputs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import af_fixtures as A
import fixtures as F
from psma_pipeline import alphafold as AF
from psma_pipeline.numbering import align_to_uniprot
from psma_pipeline.status import IssueLog, Status

tmfold = pytest.importorskip("tmfold", reason="membrane geometry needs tmfold (pip install -e '.[membrane]')")


@pytest.fixture
def acfg(cfg):
    return cfg.with_overrides({"membrane": {"deeptmhmm": {"enabled": False}}})


# ---------------------------------------------------------------------------
# numbering
# ---------------------------------------------------------------------------
def test_align_to_uniprot_construct_offset_and_tag():
    # the fixture sequence is mostly Ala (alignment would be ambiguous), so use a realistic random one
    rng = np.random.default_rng(1)
    useq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 750))
    m = align_to_uniprot(useq[43:], useq)                      # ectodomain construct 44-750
    assert m[0] == 44 and m[len(useq) - 44] == 750
    mutated = useq[55:200]
    mutated = mutated[:5] + ("A" if mutated[5] != "A" else "G") + mutated[6:]
    tagged = "MGSSHHHHHH" + mutated + useq[210:]                # tag, point mutation, 10-residue gap
    m2 = align_to_uniprot(tagged, useq)
    assert all(i not in m2 for i in range(10)) and m2[10] == 56   # tag unmapped, first native residue = 56
    assert m2[15] == 61                                         # mutated residue keeps its number
    assert m2[10 + 145] == 211                                  # residues after the gap


def test_number_model_flags_target_and_offset(tmp_path):
    path, _ = A.write_model(tmp_path, "ecto", {"A": A.protomer_ca()[43:]}, start=44)
    atoms = AF.read_model_atoms(path)
    atoms, info = AF.number_model(atoms, F.USEQ)
    assert info[0]["Target"] and info[0]["UniProt range"] == "44-750"
    ca = atoms[atoms["atom"] == "CA"]
    assert int(ca["unp"].min()) == 44 and (ca["unp"] == ca["resnum"]).all()


# ---------------------------------------------------------------------------
# PAE indexing and chain boundaries (exit criterion)
# ---------------------------------------------------------------------------
def _dimer_atoms(tmp_path):
    a = A.protomer_ca()
    path, _ = A.write_model(tmp_path, "dimer", {"A": a, "B": A.c2_image(a)})
    atoms, info = AF.number_model(AF.read_model_atoms(path), F.USEQ)
    return atoms


def test_residue_pae_matrix_chain_boundaries(tmp_path):
    atoms = _dimer_atoms(tmp_path)
    pae = A.pae_matrix(750, [750, 750])
    sub, idx = AF.pae_from_residue_matrix(pae, atoms)
    assert sub.shape == (1500, 1500)
    assert idx[0] == ("A", 1) and idx[749] == ("A", 750) and idx[750] == ("B", 1)
    assert sub[0, 760] == 4.0 and sub[0, 100] == 25.0        # inter-chain vs TM->ectodomain block
    with pytest.raises(ValueError, match="polymer residues"):
        AF.pae_from_residue_matrix(pae[:-1, :-1], atoms)


def test_pae_index_check_detects_problems(tmp_path):
    atoms = _dimer_atoms(tmp_path)
    m = AF.AFModel("T", "test", "multimer", atoms)
    m.pae, m.pae_index = AF.pae_from_residue_matrix(A.pae_matrix(750, [750, 750]), atoms)
    assert AF.pae_index_check(m)["Status"] == Status.PASS.value
    m.pae_index = m.pae_index[750:] + m.pae_index[:750][::-1]
    assert "UniProt order" in AF.pae_index_check(m)["Detail"]
    m.pae_index = [("A", 1), ("B", 1), ("A", 2)] + m.pae_index[3:]
    assert "contiguous" in AF.pae_index_check(m)["Detail"]
    m2 = AF.AFModel("T2", "test", "monomer", atoms)
    m2.pae, m2.pae_index = np.full((3, 3), 9.0), [("A", 1), ("A", 2), ("A", 3)]
    r = AF.pae_index_check(m2)
    assert r["Status"] == Status.FAIL.value and "diagonal" in r["Detail"]


# ---------------------------------------------------------------------------
# confident-dimer rule (exit criterion)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("iptm,pae,jac,expected", [
    (0.90, 3.0, 0.8, "confident"),
    (0.90, 3.0, 0.3, "uncertain"),          # confident but not the canonical interface
    (0.90, 3.0, None, "uncertain"),         # canonical criterion NOT RUN -> never confident
    (0.70, 3.0, 0.8, "uncertain"),
    (0.50, 3.0, 0.8, "low confidence"),
    (0.90, 12.0, 0.8, "low confidence"),
    (None, 3.0, 0.8, "uncertain"),
])
def test_confident_dimer_rule(cfg, iptm, pae, jac, expected):
    call, why = AF.confident_dimer_call(iptm, pae, jac, cfg.get("alphafold.dimer"))
    assert call == expected, why


# ---------------------------------------------------------------------------
# AlphaFold DB loader: current and retired field names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fields", ["current", "retired"])
def test_load_afdb_field_names(acfg, tmp_path, monkeypatch, fields):
    path, _ = A.write_model(tmp_path, "afdb", {"A": A.protomer_ca()})
    pae = A.pae_matrix(750, [750])
    e = {"latestVersion": 6, "modelCreatedDate": "2025-10-21", "pdbUrl": "https://afdb/AF-Q04609-F1-model_v6.pdb",
         "paeDocUrl": "https://afdb/AF-Q04609-F1-predicted_aligned_error_v6.json"}
    if fields == "current":
        e.update(modelEntityId="AF-Q04609-F1", sequence=F.USEQ, sequenceStart=1, sequenceEnd=750)
    else:
        e.update(entryId="AF-Q04609-F1", uniprotSequence=F.USEQ, uniprotStart=1, uniprotEnd=750)

    def fake_json(sess, url, cache_file, **k):
        return [e] if "prediction" in url else [{"predicted_aligned_error": pae.tolist()}]

    def fake_dl(sess, url, dest, **k):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(path.read_text())
        return dest
    monkeypatch.setattr(AF, "cached_json", fake_json)
    monkeypatch.setattr(AF, "cached_download", fake_dl)
    issues = IssueLog()
    models = AF.load_afdb(None, "Q04609", F.USEQ, acfg, issues)
    assert len(models) == 1
    m = models[0]
    assert m.model_id == "AFDB:AF-Q04609-F1-v6" and m.meta["Version"] == "v6" and m.meta["Canonical sequence"]
    assert m.pae.shape == (750, 750) and AF.pae_index_check(m)["Status"] == "PASS"
    assert issues.count(Status.FAIL) == 0


def test_colabfold_result_folder(acfg, tmp_path):
    a = A.protomer_ca()
    cf = tmp_path / "cf"
    cf.mkdir()
    for r in (1, 2):
        (cf / f"psma_unrelaxed_rank_00{r}_alphafold2_multimer_v3_model_{r}_seed_000.pdb").write_text(
            A.pdb_text({"A": a, "B": A.c2_image(a)}, F.USEQ, A.plddt_profile()))
        (cf / f"psma_scores_rank_00{r}_alphafold2_multimer_v3_model_{r}_seed_000.json").write_text(json.dumps(
            {"ptm": 0.8, "iptm": 0.85 - 0.1 * r, "pae": A.pae_matrix(750, [750, 750]).round(1).tolist()}))
    issues = IssueLog()
    models = AF.load_colabfold(cf, F.USEQ, acfg, issues)
    assert [m.model_id for m in models] == ["CF:cf:rank_001", "CF:cf:rank_002"]
    assert models[0].kind == "multimer" and models[0].scores["iptm"] == pytest.approx(0.75)
    assert models[0].pae.shape == (1500, 1500)
    assert (tmp_path / "cf").exists()                       # original untouched
    assert issues.count(Status.FAIL) == 0


# ---------------------------------------------------------------------------
# membrane placement
# ---------------------------------------------------------------------------
def _model(tmp_path, acfg, dimer=False):
    a = A.protomer_ca()
    chains = {"A": a, "B": A.c2_image(a)} if dimer else {"A": a}
    lens = [750, 750] if dimer else [750]
    path, pj = A.write_model(tmp_path, "dimer" if dimer else "mono", chains, pae=A.pae_matrix(750, lens))
    return AF.load_user_model(path, F.USEQ, acfg, IssueLog(), "dimer" if dimer else "mono", True, pj)[0]


def test_topology_from_uniprot(acfg):
    from psma_pipeline.membrane import topology_table
    topo, comp = topology_table(acfg, A.uniprot_entry_with_topology(), F.USEQ, IssueLog())
    lab = "".join(topo["Membrane label"])
    assert lab[:19] == "I" * 19 and lab[19:43] == "M" * 24 and set(lab[43:]) == {"O"}
    assert comp.iloc[0]["TM ranges"] == "20-43" and comp.iloc[0]["N-terminus"] == "cytoplasmic"


@pytest.mark.parametrize("dimer", [False, True])
def test_membrane_placement(acfg, tmp_path, dimer):
    from psma_pipeline.alphafold import region_sets
    from psma_pipeline.membrane import place_membrane, placement_metrics, topology_table

    m = _model(tmp_path, acfg, dimer)
    topo, _ = topology_table(acfg, A.uniprot_entry_with_topology(), F.USEQ, IssueLog())
    labels = "".join(topo["Membrane label"])
    issues = IssueLog()
    pl = place_membrane(m, labels, acfg, issues)
    assert np.dot(pl.membrane.normal, [0, 0, 1]) > 0.97        # cytoplasm -> outside along +z
    row = placement_metrics(m, pl, acfg, F.USEQ, region_sets(acfg, F.USEQ, [377, 387, 425, 453, 553]))
    assert row["TM in slab"] >= 0.9 and row["Ectodomain atoms in slab"] == 0
    assert row["Membrane thickness (Å)"] <= 36                 # the large ectodomain must not inflate the slab
    assert row["Ectodomain lowest atom above outer plane (Å)"] > 0
    assert row["TM-ectodomain PAE (Å)"] == pytest.approx(25.0)
    assert row["Orientation reliability"] == "weak"            # high TM-ectodomain PAE: orientation not determined
    if dimer:
        assert pl.method.startswith("C2-axis constrained")
        assert row["C2 axis to membrane normal (°)"] < 3 and abs(row["C2 rotation angle (°)"] - 180) < 1


def test_md_model_selection_requires_confident_dimer():
    from types import SimpleNamespace

    from psma_pipeline.membrane import _select_md_model
    top = pd.DataFrame({"Ranking mode": ["membrane_dimer", "membrane_anchor", "membrane_anchor"],
                        "Model ID": ["D", "D", "M"], "Rank": [1, 1, 2]})
    for conf, expected in (("confident", "D"), ("uncertain", "M"), ("low confidence", "M")):
        summ = pd.DataFrame({"Model ID": ["D", "M"], "Model kind": ["multimer", "monomer"],
                             "Dimer confidence": [conf, None]})
        mid, why = _select_md_model(SimpleNamespace(tables={"AFModels": summ}), top)
        assert mid == expected, (conf, why)
        assert ("not used" in why) == (expected == "M")
    # only a non-confident dimer available: it is used, and the reason says it is not confirmed
    summ = pd.DataFrame({"Model ID": ["D"], "Model kind": ["multimer"], "Dimer confidence": ["uncertain"]})
    mid, why = _select_md_model(SimpleNamespace(tables={"AFModels": summ}), top[top["Model ID"] == "D"])
    assert mid == "D" and "no monomer available" in why and "not used as confirmed" in why


def test_c2_misalignment_warns(acfg, tmp_path):
    """Tilt chain B's ectodomain so the dimer axis is 30° off the TM normal."""
    from psma_pipeline.membrane import place_membrane, topology_table

    a = A.protomer_ca()
    th = np.radians(30)
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    piv = np.array([11.0, 0, 0])
    b = A.c2_image(a)
    a2, b2 = a.copy(), b.copy()
    a2[56:] = (a[56:] - piv) @ R.T + piv                      # tilt both ectodomains: C2 axis tilts 30°
    b2[56:] = (b[56:] - piv) @ R.T + piv
    path, pj = A.write_model(tmp_path, "tilted", {"A": a2, "B": b2}, pae=A.pae_matrix(750, [750, 750]))
    m = AF.load_user_model(path, F.USEQ, acfg, IssueLog(), "tilted", True, pj)[0]
    topo, _ = topology_table(acfg, A.uniprot_entry_with_topology(), F.USEQ, IssueLog())
    issues = IssueLog()
    pl = place_membrane(m, "".join(topo["Membrane label"]), acfg, issues)
    assert pl.c2_angle_to_normal > 20
    assert any("C2 axis" in i.message for i in issues.issues)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def _survey_tables():
    return {
        "TemplateRanking": pd.DataFrame({"PDB ID": ["REF1"]}),
        "Chains": pd.DataFrame({"PDB ID": ["REF1", "REF2"], "Chain (auth)": ["A", "A"], "Entity": ["1", "1"],
                                "UniProt range (SIFTS)": ["56-750", "44-750"]}),
        "ActiveSiteDefinition": pd.DataFrame({"Status": ["OK"] * 5, "UniProt number": [377, 387, 425, 453, 553]}),
        "Ligands": pd.DataFrame({"Component": ["ZN", "CL", "CA"]}),
    }


def test_run_alphafold_end_to_end(acfg, tmp_path):
    from psma_pipeline.alphafold import run_alphafold, save_alphafold
    from psma_pipeline.excel_output import data_dictionary

    a = A.protomer_ca()
    mono, pj1 = A.write_model(tmp_path, "mono", {"A": a}, pae=A.pae_matrix(750, [750]))
    dim, pj2 = A.write_model(tmp_path, "dimer", {"A": a, "B": A.c2_image(a)}, pae=A.pae_matrix(750, [750, 750]))
    uploads = [{"kind": "model", "path": mono, "label": "mono", "plddt_in_b": True, "pae_json": pj1},
               {"kind": "model", "path": dim, "label": "dimer", "plddt_in_b": True, "pae_json": pj2}]
    # canonical interface = residues the synthetic dimer actually uses (so the Jaccard test is meaningful)
    assessment = {"CanonicalInterface": pd.DataFrame({"Item": ["canonical residues"], "Value": ["600, 601, 602"]})}
    res = run_alphafold(acfg, _survey_tables(), assessment, uploads=uploads, use_afdb=False,
                        refs={("REF1", "A"): A.reference_atoms()}, uniprot=A.uniprot_entry_with_topology())
    t = res.tables
    assert set(t["AFModels"]["Model ID"]) == {"MODEL:mono", "MODEL:dimer"}
    mono_row = t["AFModels"].set_index("Model ID").loc["MODEL:mono"]
    assert mono_row["Core CA RMSD to best template (Å)"] < 1.0       # reference = same ectodomain + 0.4 Å noise
    assert mono_row["Mean pLDDT TM"] == pytest.approx(82.0)
    assert mono_row["Orientation reliability"] == "weak"
    assert (t["AlphaFoldChecks"].query("Critical")["Status"] == "PASS").all()
    assert len(t["AFInterfaces"]) == 1
    iface = t["AFInterfaces"].iloc[0]
    assert iface["Dimer confidence"] in ("uncertain", "low confidence")   # synthetic interface is not canonical
    assert iface["Interface PAE (Å)"] is not None
    rd = t["MembraneMDReadiness"].set_index("Item")
    assert rd.loc["OVERALL", "Status"] == "FAIL" and rd.loc["zinc ions", "Status"] == "FAIL"
    assert "CA" in rd.loc["other experimental ions", "Detail"] and "CL" in rd.loc["other experimental ions", "Detail"]
    assert set(t["AFRankings"]["Ranking mode"]) >= {"ectodomain_template", "membrane_anchor"}
    ctx = t["ConstructContext"]
    assert "prediction-only" in ctx.iloc[-1]["Note"]
    # membrane-ready inputs: oriented model with TM at z ~ 0 and the cytoplasm at z < 0
    folder = res.membrane.md_files["folder"]
    from psma_pipeline.alphafold import _atoms_from_pdb_text
    oriented = _atoms_from_pdb_text((folder / "model_membrane_oriented.pdb").read_text())
    ca = oriented[(oriented["atom"] == "CA") & (oriented["chain"] == "A")].set_index("resnum")
    assert abs(ca.loc[20:43, "z"].mean()) < 4 and ca.loc[1:19, "z"].max() < 0 and ca.loc[57:, "z"].min() > 0
    assert "--preoriented" in (folder / "packmol_memgen_command.sh").read_text()
    # variant hook
    vm = pd.DataFrame({"Position": [377, 20, 9999]})
    filled = AF.variant_alphafold_columns(vm, t["AFResidueConfidence"])
    assert filled["AlphaFold pLDDT"].iloc[0] == pytest.approx(92.0)
    assert filled["AlphaFold status"].iloc[2].startswith("NOT APPLICABLE")
    # workbook and data dictionary
    paths = save_alphafold(acfg, res, tag="test")
    assert paths["workbook"].exists()
    sheets = {**t, "AlphaFoldIssues": res.issues.to_frame()}
    dd = data_dictionary(sheets)
    und = dd[dd["Description"] == "UNDOCUMENTED"]
    assert und.empty, und[["Sheet", "Column"]].to_string()


# ---------------------------------------------------------------------------
# network (opt-in: PSMA_NETWORK_TESTS=1): the real AlphaFold DB entry for Q04609
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_afdb_q04609_network(acfg):
    pytest.importorskip("gemmi")
    from psma_pipeline.cache import session_from_config
    from psma_pipeline.uniprot import uniprot_entry, uniprot_info
    sess = session_from_config(acfg)
    useq = uniprot_info(uniprot_entry(sess, "Q04609", acfg.cache_dir, acfg.endpoint("uniprot_json")))[0]
    issues = IssueLog()
    models = AF.load_afdb(sess, "Q04609", useq, acfg, issues)
    assert models and models[0].pae is not None
    assert AF.pae_index_check(models[0])["Status"] == "PASS"
    assert models[0].chain_info[0]["UniProt range"] == f"1-{len(useq)}"


def test_upload_detection(tmp_path):
    import zipfile
    z2 = tmp_path / "cf.zip"
    with zipfile.ZipFile(z2, "w") as z:
        z.writestr("x_scores_rank_001_alphafold2_ptm_model_1_seed_000.json", "{}")
    assert AF.detect_upload_kind(z2) == "colabfold"
    assert AF.detect_upload_kind(tmp_path / "model.pdb") == "model"


# ---------------------------------------------------------------------------
# AlphaFold2 inside Colab (ColabFold): fake colabfold_batch with ColabFold's real output names
# ---------------------------------------------------------------------------
FAKE_COLABFOLD = r'''#!{python}
import json, sys
from pathlib import Path
sys.path[:0] = [{tests!r}, {src!r}]
import af_fixtures as A, fixtures as F
args = sys.argv[1:]
fasta, out = Path(args[-2]), Path(args[-1])
n = int(args[args.index("--num-models") + 1])
job = fasta.stem
seq = fasta.read_text().split("\n")[1]
chains = seq.split(":")
out.mkdir(parents=True, exist_ok=True)
a = A.protomer_ca()
xyz = {{"A": a, "B": A.c2_image(a)}} if len(chains) == 2 else {{"A": a}}
lens = [750] * len(chains)
for r in range(1, n + 1):
    tag = f"rank_{{r:03d}}_alphafold2_multimer_v3_model_{{r}}_seed_000"
    (out / f"{{job}}_unrelaxed_{{tag}}.pdb").write_text(A.pdb_text(xyz, F.USEQ, A.plddt_profile()))
    (out / f"{{job}}_scores_{{tag}}.json").write_text(json.dumps(
        {{"ptm": 0.8, "iptm": 0.9 - 0.05 * r, "pae": A.pae_matrix(750, lens).round(1).tolist()}}))
(out / f"{{job}}.done.txt").write_text("")
(Path(str(out)) / "calls.txt").open("a").write(" ".join(args) + "\n")
'''


@pytest.fixture
def fake_colabfold(tmp_path, monkeypatch):
    import os
    import stat
    import sys
    from pathlib import Path as P
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "colabfold_batch"
    here = P(__file__).resolve().parent
    exe.write_text(FAKE_COLABFOLD.format(python=sys.executable, tests=str(here), src=str(here.parent / "src")))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return exe


def test_colabfold_query_and_command(acfg):
    from psma_pipeline.colabfold_run import colabfold_command, job_sequence
    s = job_sequence(F.USEQ, copies=2)
    assert s.count(":") == 1 and len(s) == 2 * 750 + 1
    assert len(job_sequence(F.USEQ, 2, start=44)) == 2 * 707 + 1
    cmd = colabfold_command(__import__("pathlib").Path("d.fasta"), __import__("pathlib").Path("o"),
                            acfg.get("alphafold.colabfold"))
    assert cmd[0] == "colabfold_batch" and cmd[-2:] == ["d.fasta", "o"]
    assert cmd[cmd.index("--num-models") + 1] == "3" and "--amber" not in cmd


def test_colabfold_run_resume_and_import(acfg, fake_colabfold):
    from psma_pipeline.colabfold_run import run_colabfold_jobs
    issues = IssueLog()
    jobs = {"dimer": {"copies": 2, "num_models": 2}}
    ups = run_colabfold_jobs(acfg, F.USEQ, issues, jobs=jobs)
    assert len(ups) == 1 and ups[0]["stage"] is False and issues.count(Status.FAIL) == 0
    calls = acfg.cache_dir / "alphafold" / "colabfold" / "dimer" / "calls.txt"
    assert len(calls.read_text().splitlines()) == 1
    run_colabfold_jobs(acfg, F.USEQ, issues, jobs=jobs)                     # resumed run: job skipped
    assert len(calls.read_text().splitlines()) == 1
    models = AF.load_upload(ups[0], F.USEQ, acfg, issues)
    assert [m.model_id for m in models] == ["CF:dimer:rank_001", "CF:dimer:rank_002"]
    m = models[0]
    assert m.source == "ColabFold (AlphaFold2)" and m.kind == "multimer" and m.meta["Model"].startswith("alphafold2_multimer_v3")
    assert m.scores["iptm"] == pytest.approx(0.85) and AF.pae_index_check(m)["Status"] == "PASS"


def test_colabfold_missing_is_not_run(acfg, monkeypatch):
    from psma_pipeline import colabfold_run as CR
    monkeypatch.setattr(CR, "colabfold_available", lambda: False)
    issues = IssueLog()
    assert CR.run_colabfold_jobs(acfg, F.USEQ, issues, jobs={"x": {"copies": 1}}) == []
    assert issues.count(Status.NOT_RUN) == 1


def test_end_to_end_with_alphafold2_dimer(acfg, fake_colabfold):
    """AlphaFold2-multimer run in 'Colab' -> import -> membrane (C2-constrained) -> MD inputs."""
    from psma_pipeline.alphafold import run_alphafold
    from psma_pipeline.colabfold_run import run_colabfold_jobs
    ups = run_colabfold_jobs(acfg, F.USEQ, IssueLog(), jobs={"dimer": {"copies": 2, "num_models": 1}})
    assessment = {"CanonicalInterface": pd.DataFrame({"Item": ["canonical residues"], "Value": ["600, 601"]})}
    res = run_alphafold(acfg, _survey_tables(), assessment, uploads=ups, use_afdb=False,
                        refs={("REF1", "A"): A.reference_atoms()}, uniprot=A.uniprot_entry_with_topology())
    pl = res.tables["MembranePlacement"].set_index("Model ID").loc["CF:dimer:rank_001"]
    assert pl["Placement method"].startswith("C2-axis constrained") and pl["Membrane thickness (Å)"] <= 36
    assert res.tables["AFInterfaces"].iloc[0]["ipTM"] == pytest.approx(0.85)
    assert (res.tables["AlphaFoldChecks"].query("Critical")["Status"] == "PASS").all()
