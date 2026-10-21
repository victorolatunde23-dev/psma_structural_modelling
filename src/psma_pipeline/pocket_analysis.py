"""Pockets, conserved waters and variants (notebook 02 part 2, Milestone 3).

Input: survey tables (notebook 01) and, optionally, assessment tables
(notebook 02 part 1 – used for ranking columns such as ligand local support).
Output: ``<ACC>_pockets_waters_variants.xlsx`` with

* **SubpocketDefinition** – validated sub-pocket residues (dropped ones listed);
* **PocketFingerprints** – one row per ligand instance in the biological assembly;
* **LigandInteractions** – one row per ligand atom – protein interaction;
* **SubpocketOccupancy** – instance × sub-pocket;
* **PocketSimilarity** – pairwise similarities of representative instances;
* **LigandClusters** / **ClusterDiagnostics** – hierarchical (several thresholds) and DBSCAN;
* **LigandChemistry** – CCD definitions + RDKit descriptors (NOT RUN without RDKit);
* **ConservedWaters**, **WaterMembers**, **WaterSites**;
* **Variants**, **VariantStructureMapping**, **VariantSummary**;
* **LigandConditionedRanking** – only when a reference ligand is given.

A separate workbook keeps notebook 02's two parts independently re-runnable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from .cache import session_from_config
from .config import Config, parse_residue_code, validate_residue_codes
from .fingerprints import cluster_ligands, ligand_fingerprint, similarity_tables, tanimoto, zinc_similarity
from .ligand_chem import chemical_similarity, fetch_chem_comps, ligand_table, rdkit_available
from .ligands import LIGAND, classify_component
from .pdbe import fetch_sifts, sifts_segments
from .ranking import criteria_table, rank_templates
from .rcsb import load_structure
from .status import IssueLog, Status
from .structure_atoms import assembly_generators, asu_as_assembly, atoms_from_gemmi, build_assembly, target_protomers
from .survey import http_kwargs
from .uniprot import uniprot_entry
from .variants import (ca_rmsf, disulfides, fetch_ebi_variants, map_variants, residue_environment, uniprot_variants,
                       validate_variants, variant_summary)
from .waters import conserved_presence, conserved_waters, site_from_atoms

log = logging.getLogger("psma_pipeline.pocket_analysis")


@dataclass
class PocketResults:
    """Tables plus the data needed to re-run ligand-conditioned rankings cheaply."""

    tables: dict[str, pd.DataFrame]
    fingerprints: dict[str, dict]
    instance_meta: pd.DataFrame
    similarity: np.ndarray
    similarity_ids: list[str]
    chem_defs: dict[str, dict]
    entry_extra: pd.DataFrame
    criteria: pd.DataFrame
    subpockets: dict[str, set[int]]
    issues: IssueLog
    ids: list[str]
    input_files: list = field(default_factory=list)


def subpocket_sets(cfg: Config, seq: str) -> tuple[dict[str, set[int]], pd.DataFrame]:
    """Validated ``{sub-pocket: UniProt numbers}`` and a definition table."""
    out, rows = {}, []
    for name, codes in (cfg.active_site.get("subpockets") or {}).items():
        valid, dropped = validate_residue_codes(codes, seq)
        out[name] = {parse_residue_code(c)[1] for c in valid}
        rows += [{"Sub-pocket": name, "Residue": c, "UniProt number": parse_residue_code(c)[1],
                  "Status": Status.OK.value, "Reason": ""} for c in valid]
        rows += [{"Sub-pocket": name, "Residue": d["Residue"], "UniProt number": None,
                  "Status": Status.FAIL.value, "Reason": d["Reason"] + " – dropped"} for d in dropped]
    return out, pd.DataFrame(rows)


def _site_numbers(tables: dict) -> list[int]:
    sd = tables.get("ActiveSiteDefinition", pd.DataFrame())
    if not len(sd):
        return []
    return sorted(int(n) for n in sd[sd["Status"] == "OK"]["UniProt number"].dropna().unique())


def _reference_entry(tables: dict, assessment: dict | None, ids: list[str]) -> str | None:
    """Variant-mapping reference: top md_start entry, else top global entry, else first ID."""
    order: list[str] = []
    if assessment and len(assessment.get("TemplateRankings", pd.DataFrame())):
        tr = assessment["TemplateRankings"]
        order += tr[tr["RankingMode"] == "md_start"].sort_values("Rank")["PDB ID"].tolist()
    if len(tables.get("TemplateRanking", pd.DataFrame())):
        order += tables["TemplateRanking"].sort_values("Rank")["PDB ID"].tolist()
    order += ids
    return next((p for p in order if p in ids), None)


def run_pocket_analysis(cfg: Config, tables: dict[str, pd.DataFrame], assessment: dict[str, pd.DataFrame] | None = None,
                        ids: list[str] | None = None, sess: requests.Session | None = None,
                        issues: IssueLog | None = None) -> PocketResults:
    """Ligand fingerprints, conserved waters and variant mapping for survey entries."""
    import gemmi  # noqa: F401

    issues = issues if issues is not None else IssueLog()
    sess = sess or session_from_config(cfg)
    json_kw, dl_kw = http_kwargs(cfg)
    pcfg, wcfg, vcfg = cfg.require("pockets"), cfg.require("waters"), cfg.require("variants")
    icfg = cfg.require("interfaces")
    acc = cfg.accession
    lc = cfg.ligand_classes
    classify = lambda c: classify_component(c, lc)  # noqa: E731
    metals = {m.upper() for m in lc["metals"]}
    summary = tables["Summary"]
    sidx = summary.set_index("PDB ID")
    ids = [i.upper() for i in (summary["PDB ID"].tolist() if ids is None else ids)]
    entry = uniprot_entry(sess, acc, cfg.cache_dir, cfg.endpoint("uniprot_json"), **json_kw)
    seq = entry["sequence"]["value"]
    subpockets, sp_def = subpocket_sets(cfg, seq)
    for _, r in sp_def[sp_def["Status"] == Status.FAIL.value].iterrows():
        issues.add("pockets", Status.WARNING, f"sub-pocket residue {r['Residue']} dropped: {r['Reason']}")
    broad_sites = _site_numbers(tables)          # survey active-site definition (incl. UniProt Ca/Cl sites)
    lit_valid, _ = validate_residue_codes(list((cfg.active_site.get("literature_sites") or {}).keys()), seq)
    zn_lig = {parse_residue_code(c)[1] for lst in (cfg.active_site.get("metal_site", {}).get("zn_ligands") or {}).values()
              for c in validate_residue_codes(lst, seq)[0]}
    core_sites = {parse_residue_code(c)[1] for c in lit_valid} | zn_lig
    pocket_union = set().union(*subpockets.values()) if subpockets else set()
    # catalytic pocket = literature residues + Zn ligands + sub-pockets: region for waters and
    # "distance to catalytic pocket"; superposition uses the broader survey set as well.
    sites = sorted(core_sites | pocket_union) or broad_sites
    super_sites = sorted(set(sites) | set(broad_sites))
    if not sites:
        issues.add("pockets", Status.WARNING, "no validated catalytic-pocket residues")
    ref_pdb = _reference_entry(tables, assessment, ids)

    fps: dict[str, dict] = {}
    meta, inter_tabs, occ_rows, water_sites = [], [], [], []
    ca_all: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    modelled_count: dict[int, int] = {}
    n_chains = 0
    missing_near: dict[str, int] = {}
    ref_env = pd.DataFrame()
    ref_label = ""
    files = []
    for k, pdb in enumerate(ids, 1):
        if pdb not in sidx.index:
            issues.add("pockets", Status.FAIL, "entry not in survey Summary", pdb_id=pdb)
            continue
        s = sidx.loc[pdb]
        ch = tables["Chains"][tables["Chains"]["PDB ID"] == pdb]
        tents = {str(e) for e in ch["Entity"].dropna()}
        asm_id = str(s.get("Assembly used") or "none").split()[0]
        try:
            st = load_structure(sess, pdb, cfg.cache_dir / "mmcif", cfg.endpoint("rcsb_cif"), **dl_kw)
            files.append(cfg.cache_dir / "mmcif" / f"{pdb}.cif.gz")
            segs = sifts_segments(fetch_sifts(sess, pdb, cfg.cache_dir, cfg.endpoint("pdbe_sifts"), **json_kw), pdb, acc)
            atoms = atoms_from_gemmi(st, 0, segs, tents, icfg.get("altloc", "first"))
        except Exception as ex:  # noqa: BLE001
            issues.add("pockets", Status.FAIL, f"structure could not be loaded: {ex}", pdb_id=pdb)
            continue
        gens = assembly_generators(st, asm_id) if asm_id != "none" else None
        if gens is None:
            issues.add("pockets", Status.WARNING, f"assembly {asm_id} not found; ASU used", pdb_id=pdb)
            asm = asu_as_assembly(atoms)
        else:
            asm = build_assembly(atoms, gens)
        # ---- per-chain CA, modelled pocket residues ---------------------------------------
        tgt = atoms[atoms["target"] & atoms["polymer"] & atoms["unp"].notna()]
        miss = []
        for chain, g in tgt.groupby("chain"):
            ca = g[g["atom"] == "CA"]
            d = {int(r.unp): np.array([r.x, r.y, r.z]) for r in ca.itertuples()}
            ca_all[(pdb, chain)] = d
            n_chains += 1
            for n in d:
                modelled_count[n] = modelled_count.get(n, 0) + 1
            miss.append(len(pocket_union - set(d)))
        missing_near[pdb] = min(miss) if miss else None
        # ---- ligand instances -----------------------------------------------------------------
        het = asm[~asm["polymer"] & ~asm["water"]]
        lig = het[het["resname"].map(classify) == LIGAND]
        tprot = asm[asm["target"] & asm["polymer"]]
        ptree = cKDTree(tprot[["x", "y", "z"]].to_numpy()) if len(tprot) else None
        atree = cKDTree(asm[["x", "y", "z"]].to_numpy())
        for key, la in lig.groupby(["protomer", "label_asym", "resnum", "icode", "resname"], sort=True):
            if ptree is None or not any(ptree.query_ball_point(la[["x", "y", "z"]].to_numpy(), float(pcfg["contact_max"]))):
                continue                     # not touching the target protein
            prot_, sub, resnum, icode, comp = key
            iid = f"{pdb}:{comp}:{la['chain'].iloc[0]}{resnum}{icode or ''}|{la['copy'].iloc[0]}"
            lxyz = la[["x", "y", "z"]].to_numpy()
            near = atree.query_ball_point(lxyz, 12.0)
            idx = sorted({i for h in near for i in h} - set(np.flatnonzero(asm.index.isin(la.index))))
            env = asm.iloc[idx]
            try:
                tab, fp = ligand_fingerprint(pdb, la, env, pcfg, subpockets, metals)
            except Exception as ex:  # noqa: BLE001
                issues.add("pockets", Status.FAIL, f"fingerprint {iid}: {ex}", pdb_id=pdb)
                continue
            fps[iid] = fp
            chains_contacted = sorted(set(tab["Chain"])) if len(tab) else []
            meta.append({"Instance": iid, "PDB ID": pdb, "Component": comp, "Chain": la["chain"].iloc[0],
                         "Residue no. (author)": f"{resnum}{icode or ''}", "Copy": la["copy"].iloc[0],
                         "Heavy atoms": len(la), "Residues contacted": len(fp["residues"]),
                         "Contacted residues (UniProt)": ", ".join(map(str, sorted(fp["residues"]))),
                         "Interaction bits": ", ".join(sorted(fp["typed"])),
                         **{f"N {t}": fp["counts"].get(t, 0) for t in
                            ("hydrogen bond (possible)", "ionic (possible salt bridge)", "hydrophobic",
                             "aromatic (π-stacking, possible)", "cation–π (possible)", "metal coordination",
                             "water-mediated")},
                         "Metal coordination": ", ".join(fp["metal_atoms"]),
                         "Zn-coordinating atoms": sum("ZN" in m for m in fp["metal_atoms"]),
                         "Occupied sub-pockets": ", ".join(n for n, v in fp["subpockets"].items() if v),
                         "Contacts partner protomer": fp["partner"],
                         "Chains contacted": ", ".join(map(str, chains_contacted)),
                         "Entrance lid state": s.get("Entrance lid state"),
                         "Status": Status.OK.value if fp["residues"] else Status.WARNING.value})
            if len(tab):
                inter_tabs.append(tab.assign(Instance=iid))
            chain_mod = set(ca_all.get((pdb, la["chain"].iloc[0]), {}))
            for name, sset in subpockets.items():
                hit = sorted(fp["residues"] & sset)
                occ_rows.append({"Instance": iid, "PDB ID": pdb, "Component": comp, "Sub-pocket": name,
                                 "Occupied": bool(hit), "Contacted residues": ", ".join(map(str, hit)),
                                 "Defined residues": len(sset), "Modelled in ligand chain": len(sset & chain_mod)})
        # ---- water sites -----------------------------------------------------------------------
        res = pd.to_numeric(pd.Series([s.get("Resolution (Å)")]), errors="coerce").iloc[0]
        clash = pd.to_numeric(pd.Series([s.get("Clashscore")]), errors="coerce").iloc[0]
        xray = "X-RAY" in str(s.get("Method", "")).upper()
        if xray and pd.notna(res) and res <= float(wcfg["max_resolution"]) and (
                pd.isna(clash) or clash <= float(wcfg["max_clashscore"])):
            if pd.isna(clash):
                issues.add("waters", Status.WARNING, "clashscore missing; entry kept as water site", pdb_id=pdb)
            asu_lig = atoms[~atoms["polymer"] & ~atoms["water"] & (atoms["resname"].map(classify) == LIGAND)]
            for chain in sorted(tgt["chain"].unique()):
                lx = asu_lig[asu_lig["chain"] == chain][["x", "y", "z"]].to_numpy() if len(asu_lig) else None
                ws = site_from_atoms(pdb, chain, atoms, sites, float(wcfg["region_radius"]), lx)
                if ws is not None:
                    water_sites.append(ws)
        # ---- variant reference environment ---------------------------------------------------
        if pdb == ref_pdb:
            prots = target_protomers(asm)
            if prots:
                try:
                    ref_env = residue_environment(asm, prots[0], sites, lc, classify, icfg.get("sasa", {}),
                                                  float(pcfg["contact_max"]))
                    ref_label = f"{pdb} assembly {asm_id}, protomer {prots[0]}"
                except Exception as ex:  # noqa: BLE001
                    issues.add("variants", Status.FAIL, f"reference environment failed: {ex}", pdb_id=pdb)
        if k % 10 == 0 or k == len(ids):
            log.info("pockets %d/%d", k, len(ids))

    meta_df = pd.DataFrame(meta)
    # ---- representatives, similarity, clustering ------------------------------------------------
    rep_ids: list[str] = []
    if len(meta_df):
        meta_df = meta_df.sort_values(["PDB ID", "Component", "Residues contacted", "Instance"],
                                      ascending=[True, True, False, True])
        rep_ids = meta_df.drop_duplicates(["PDB ID", "Component"])["Instance"].tolist()
        meta_df["Representative"] = meta_df["Instance"].isin(rep_ids)
        rep_ids = [i for i in rep_ids if fps[i]["residues"]]
    sim_tab, T, T_ids = similarity_tables({i: fps[i] for i in rep_ids})
    clusters, diag = cluster_ligands(T, T_ids, pcfg.get("clustering", {}))
    # ---- ligand chemistry --------------------------------------------------------------------------
    comps = sorted(set(meta_df["Component"])) if len(meta_df) else []
    defs: dict[str, dict] = {}
    if comps:
        try:
            defs = fetch_chem_comps(sess, comps, cfg.cache_dir, cfg.endpoint("rcsb_chem_comp"), **json_kw)
        except Exception as ex:  # noqa: BLE001
            issues.add("ligand chemistry", Status.NOT_RUN, f"chemical component definitions unavailable: {ex}")
    chem = ligand_table(defs, comps) if comps else pd.DataFrame()
    if comps and not rdkit_available():
        issues.add("ligand chemistry", Status.NOT_RUN, "RDKit not installed: descriptors and chemical similarity skipped")
    if len(clusters):
        clusters = clusters.merge(meta_df[["Instance", "PDB ID", "Component", "Occupied sub-pockets",
                                           "Zn-coordinating atoms", "Entrance lid state"]],
                                  left_on="Ligand instance", right_on="Instance", how="left").drop(columns="Instance")
        if len(chem):
            clusters = clusters.merge(chem[["Component", "Ligand name", "Formula weight"]], on="Component", how="left")
    # ---- conserved waters -----------------------------------------------------------------------------
    ref_site = None
    if ref_pdb and any(w.pdb == ref_pdb for w in water_sites):
        ref_site = next((w.pdb, w.chain) for w in water_sites if w.pdb == ref_pdb)
    w_clusters, w_members, w_sites = conserved_waters(water_sites, wcfg, super_sites, ref_site)
    if not water_sites:
        issues.add("waters", Status.NOT_APPLICABLE, "no eligible high-resolution X-ray sites in this run")
    presence = conserved_presence(w_members, w_clusters)
    # ---- variants --------------------------------------------------------------------------------------
    vrows = uniprot_variants(entry)
    if "ebi_proteins" in (vcfg.get("sources") or []):
        try:
            vrows += fetch_ebi_variants(sess, acc, cfg.cache_dir, cfg.endpoint("ebi_variation"), **json_kw)
        except Exception as ex:  # noqa: BLE001
            issues.add("variants", Status.NOT_RUN, f"EBI Proteins API unavailable: {ex}")
    variants = validate_variants(vrows, seq)
    failed = variants[variants["Status"] == Status.FAIL.value] if len(variants) else pd.DataFrame()
    for _, r in failed.iterrows():
        issues.add("variants", Status.WARNING, f"{r['Substitution']}: {r['Status reason']}")
    rmsf, _ = ca_rmsf(ca_all, None)
    ctx = {"site_numbers": core_sites, "zn_ligands": zn_lig, "subpockets": subpockets, "rmsf": rmsf,
           "regions": cfg.flexible_regions, "disulfide_cys": {c for pair in disulfides(entry) for c in pair}}
    frac = {n: round(c / n_chains, 3) for n, c in modelled_count.items()} if n_chains else {}
    if ref_env.empty:
        issues.add("variants", Status.WARNING, "no reference structure environment; structural columns missing")
    mapping = map_variants(variants, ref_env, seq, ctx, vcfg, ref_label, frac) if len(variants) else pd.DataFrame()
    vsum = variant_summary(mapping, variants)
    # ---- entry-level extras for ranking ------------------------------------------------------------------
    extra = pd.DataFrame(index=pd.Index(ids, name="PDB ID"))
    extra["Missing residues near pocket"] = pd.Series(missing_near)
    extra["Conserved waters present"] = pd.Series(presence)
    asum = (assessment or {}).get("AssessmentSummary")
    base = asum if asum is not None and len(asum) else summary
    crit = criteria_table(base, tables["Chains"], tables.get("ActiveSite", pd.DataFrame()))
    for c in base.columns:
        if c not in crit and c != "PDB ID":
            crit[c] = base.set_index("PDB ID")[c]
    crit = crit.join(extra, how="left")

    out = {"SubpocketDefinition": sp_def, "PocketFingerprints": meta_df,
           "LigandInteractions": pd.concat(inter_tabs, ignore_index=True) if inter_tabs else pd.DataFrame(),
           "SubpocketOccupancy": pd.DataFrame(occ_rows), "PocketSimilarity": sim_tab,
           "LigandClusters": clusters, "ClusterDiagnostics": diag, "LigandChemistry": chem,
           "ConservedWaters": w_clusters, "WaterMembers": w_members.drop(columns=["x_ref", "y_ref", "z_ref"],
                                                                         errors="ignore"),
           "WaterSites": w_sites, "Variants": variants, "VariantStructureMapping": mapping, "VariantSummary": vsum}
    res = PocketResults(out, fps, meta_df, T, T_ids, defs, extra, crit, subpockets, issues, ids, files)
    ref_comp = pcfg.get("reference_ligand")
    if ref_comp or pcfg.get("reference_smiles"):
        res.tables["LigandConditionedRanking"] = rank_for_ligand(
            res, cfg, comp=ref_comp, smiles=pcfg.get("reference_smiles"),
            required_subpockets=pcfg.get("required_subpockets"), lid_state=pcfg.get("desired_lid_state"))
    return res


def rank_for_ligand(res: PocketResults, cfg: Config, comp: str | None = None, smiles: str | None = None,
                    required_subpockets: list[str] | None = None, lid_state: str | None = None) -> pd.DataFrame:
    """Ligand-conditioned template ranking.

    Parameters
    ----------
    comp:
        Reference ligand component ID present in the analysed structures
        (its best-resolved instance provides the reference fingerprint and,
        via the CCD, the SMILES).
    smiles:
        Reference SMILES (overrides the CCD SMILES; needed for a ligand not
        in the PDB). Without ``comp`` the pocket similarity uses only the
        required sub-pockets.
    required_subpockets:
        Sub-pockets the ligand is expected to occupy.
    lid_state:
        ``open`` / ``closed`` – desired entrance-lid state.

    Every entry with at least one ligand instance is scored with its
    best-matching instance; apo entries are listed with missing similarity
    (ranked last by ``missing: worst``).
    """
    pcfg = cfg.require("pockets")
    meta = res.instance_meta
    comp = comp.upper() if comp else None
    ref_fp, ref_iid = None, None
    if comp is not None:
        cand = meta[meta["Component"] == comp] if len(meta) else meta
        if not len(cand):
            raise ValueError(f"reference ligand {comp} not found in the analysed structures")
        resol = res.criteria["Resolution (Å)"] if "Resolution (Å)" in res.criteria else pd.Series(dtype=float)
        cand = cand.assign(_r=cand["PDB ID"].map(resol)).sort_values(["_r", "Residues contacted"],
                                                                      ascending=[True, False])
        ref_iid = cand["Instance"].iloc[0]
        ref_fp = res.fingerprints[ref_iid]
        smiles = smiles or (res.chem_defs.get(comp, {}).get("smiles_stereo") or res.chem_defs.get(comp, {}).get("smiles"))
    req = set(required_subpockets or [])
    unknown = req - set(res.subpockets)
    if unknown:
        raise ValueError(f"unknown sub-pockets: {sorted(unknown)}; known: {sorted(res.subpockets)}")
    rows = []
    for iid, fp in res.fingerprints.items():
        m = meta[meta["Instance"] == iid].iloc[0]
        c = m["Component"]
        csmi = res.chem_defs.get(c, {}).get("smiles_stereo") or res.chem_defs.get(c, {}).get("smiles")
        rows.append({"PDB ID": m["PDB ID"], "Best instance": iid, "Component": c,
                     "Pocket-contact similarity": tanimoto(fp["residues"], ref_fp["residues"]) if ref_fp else None,
                     "Interaction-fingerprint similarity": tanimoto(fp["typed"], ref_fp["typed"]) if ref_fp else None,
                     "Required sub-pockets occupied": (sum(fp["subpockets"].get(n, False) for n in req) / len(req))
                     if req else None,
                     "Ligand chemical similarity": chemical_similarity(smiles, csmi),
                     "Zinc-coordination similarity": zinc_similarity(fp["metal_atoms"], ref_fp["metal_atoms"])
                     if ref_fp else None})
    per = pd.DataFrame(rows)
    if len(per):
        key = per["Pocket-contact similarity"].fillna(per["Required sub-pockets occupied"]).fillna(-1)
        per = (per.assign(_k=key, _c=per["Ligand chemical similarity"].fillna(-1))
               .sort_values(["PDB ID", "_k", "_c", "Best instance"], ascending=[True, False, False, True])
               .drop_duplicates("PDB ID").drop(columns=["_k", "_c"]).set_index("PDB ID"))
    crit = res.criteria.copy()
    for col in ("Best instance", "Component", "Pocket-contact similarity", "Interaction-fingerprint similarity",
                "Required sub-pockets occupied", "Ligand chemical similarity", "Zinc-coordination similarity"):
        crit[col] = per[col].reindex(crit.index) if len(per) else None
    if lid_state:
        st = crit.get("Entrance lid state", pd.Series(index=crit.index, dtype=object)).astype(str)
        crit["Entrance-lid state match"] = np.where(st.str.lower() == lid_state.lower(), 1.0,
                                                    np.where(st.str.lower().isin(["open", "closed"]), 0.0, np.nan))
    mode = {"criteria": list(pcfg["ligand_conditioned"]["criteria"]), "missing": pcfg["ligand_conditioned"].get("missing", "worst")}
    if req:
        mode["criteria"].append({"column": "Required sub-pockets occupied", "direction": "higher", "weight": 2})
    if not rdkit_available() or not smiles:
        mode["criteria"] = [c for c in mode["criteria"] if c["column"] != "Ligand chemical similarity"]
    if not lid_state:
        mode["criteria"] = [c for c in mode["criteria"] if c["column"] != "Entrance-lid state match"]
    used = {c["column"] for c in mode["criteria"]}
    keep = [c for c in crit.columns if c in used or c in ("Method", "Best instance", "Component",
                                                          "Interaction-fingerprint similarity", "Entrance lid state")]
    out = rank_templates(crit[keep], mode, "ligand_conditioned")
    out.insert(1, "Reference ligand", comp or "")
    out.insert(2, "Reference instance", ref_iid or "")
    out["Reference SMILES"] = smiles or ""
    out["Required sub-pockets"] = ", ".join(sorted(req))
    out["Desired lid state"] = lid_state or ""
    return out


def save_pocket_analysis(cfg: Config, res: PocketResults, tag: str = "") -> dict[str, Path]:
    """Write the pockets/waters/variants workbook, tables and manifest."""
    from .excel_output import write_tables, write_workbook
    from .manifest import build_manifest, detect_software, manifest_table, write_manifest

    software = detect_software()
    sheets = dict(res.tables)
    sheets["PocketIssues"] = res.issues.to_frame()
    sheets["Software"] = software
    manifest = build_manifest(cfg, structure_ids=res.ids, input_files=res.input_files, software=software,
                              extra={"stage": "pockets, waters, variants (notebook 02 part 2)",
                                     "subpockets": {k: sorted(v) for k, v in res.subpockets.items()},
                                     "rdkit": rdkit_available(),
                                     "issue_counts": {s.value: res.issues.count(s) for s in Status}})
    sheets["Reproducibility"] = manifest_table(manifest)
    suffix = f"_{tag}" if tag else ""
    wb = cfg.output_file("pocket_workbook")
    wb = wb.with_name(wb.stem + suffix + ".xlsx")
    write_workbook(wb, sheets)
    mpath = write_manifest(manifest, cfg.outputs_dir / f"{cfg.accession}_pocket_manifest{suffix}.json")
    tdir = cfg.outputs_dir / f"pocket_tables{suffix}"
    write_tables(tdir, sheets, csv=bool(cfg.get("outputs.write_csv", True)),
                 parquet=bool(cfg.get("outputs.write_parquet", True)))
    if len(res.similarity_ids):
        np.savetxt(tdir / "ligand_interaction_tanimoto.csv", res.similarity, delimiter=",",
                   header=",".join(res.similarity_ids), comments="")
    return {"workbook": wb, "manifest": mpath, "tables_dir": tdir}


__all__: list[Any] = ["run_pocket_analysis", "rank_for_ligand", "save_pocket_analysis", "subpocket_sets",
                      "PocketResults"]
