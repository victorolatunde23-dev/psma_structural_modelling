"""Structure survey: per-entry processing, cross-check, summary and the full run.

This is the refactored core of the original ``PDB_structure_pipeline``
notebook. :func:`run_survey` reproduces the original workbook (all 14 sheets)
and adds identifier, status and provenance columns plus a few new sheets.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .active_site import active_site_analysis, active_site_residues
from .assemblies import assembly_note, choose_assembly, coordinate_analysis, global_symmetry, oligomer_label
from .cache import FetchError, session_from_config
from .config import Config
from .conformations import (ca_coordinates, construct_ranges, lid_conformation, lid_entry_summary,
                            lid_ligand_association, structural_variability)
from .glycans import find_glycosylation_sites
from .ligands import classify_component
from .numbering import construct_differences, label_to_unp, to_ranges
from .pdbe import fetch_best_structures, fetch_sifts, sifts_segments
from .pockets import binding_site_analysis
from .ranking import criteria_table, rank_templates
from .rcsb import entity_is_target, load_structure, rcsb_metadata, rcsb_search
from .status import IssueLog, Status
from .uniprot import uniprot_entry, uniprot_info
from .validation import pdbe_validation, rcsb_validation

log = logging.getLogger("psma_pipeline.survey")

ORIGINAL_SHEETS = ["Summary", "TemplateRanking", "Chains", "Ligands", "BindingSites", "Glycosylation",
                   "ActiveSite", "Validation", "Variability", "EntranceLid", "LidVsLigand",
                   "Assemblies", "CrossCheck", "Notes"]


def http_kwargs(cfg: Config) -> tuple[dict, dict]:
    """Keyword arguments for :func:`cached_json` and :func:`cached_download`."""
    j = {"timeout": cfg.get("http.timeout_s", 120), "polite_delay": cfg.get("http.polite_delay_s", 0.15),
         "refresh_404_after_days": cfg.get("http.refresh_404_after_days", 30)}
    d = {"timeout": max(300, cfg.get("http.timeout_s", 120)), "polite_delay": cfg.get("http.polite_delay_s", 0.15)}
    return j, d


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------
def process_entry(pdb: str, e: dict, sifts: dict, cfg: Config, useq: str,
                  sess: requests.Session, use_coords: bool) -> dict:
    """Summary, chain, ligand and assembly rows for one PDB entry.

    Returns
    -------
    dict with ``summary`` (dict), ``chains``/``ligands``/``assemblies`` (lists
    of dicts) and ``_ctx`` (context for the coordinate analyses; not written
    to Excel).
    """
    acc = cfg.accession
    classes = cfg.ligand_classes
    ions = classes["ions_of_interest"]
    glycan_set = set(classes["glycans"])
    _, dl_kw = http_kwargs(cfg)

    rows: dict[str, Any] = {"summary": None, "chains": [], "ligands": [], "assemblies": []}
    info = e.get("rcsb_entry_info") or {}
    res = (info.get("resolution_combined") or [None])[0]
    method = "; ".join(x["method"] for x in e.get("exptl") or [])
    rfree = next((r.get("ls_R_factor_R_free") for r in e.get("refine") or []
                  if r.get("ls_R_factor_R_free") is not None), None)
    sg = (e.get("symmetry") or {}).get("space_group_name_H_M")

    targets = [p for p in e.get("polymer_entities") or [] if entity_is_target(p, acc)]
    others = [p for p in e.get("polymer_entities") or [] if not entity_is_target(p, acc)]
    target_ids = {p["rcsb_polymer_entity_container_identifiers"]["entity_id"] for p in targets}
    segs_by_asym = sifts_segments(sifts, pdb, acc)
    warnings: list[str] = []
    if not segs_by_asym:
        warnings.append("no SIFTS mapping to the target accession")

    # --- coordinates ----------------------------------------------------------
    coords, coord_err = None, ""
    if use_coords:
        try:
            st = load_structure(sess, pdb, cfg.cache_dir / "mmcif", cfg.endpoint("rcsb_cif"), **dl_kw)
            coords = coordinate_analysis(st, target_ids, model_index=0)
        except Exception as ex:  # noqa: BLE001 - recorded in Warnings, entry continues
            coord_err = f"coordinate analysis failed: {ex}"
    model_no = coords["model_number"] if coords else None

    # --- chains / construct / missing residues ----------------------------------
    missing_summary, internal_summary, construct_notes = [], [], []
    for p in targets:
        ids = p["rcsb_polymer_entity_container_identifiers"]
        seq = (p["entity_poly"].get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
        asyms = ids.get("asym_ids") or []
        auths = ids.get("auth_asym_ids") or []
        ent_segs = next((segs_by_asym[a] for a in asyms if a in segs_by_asym), [])
        diff = construct_differences(seq, ent_segs, useq)
        construct_notes.append(
            f"ent {ids['entity_id']}: UniProt {diff['unp_range']}"
            + (f"; N-ext {len(diff['n_ext'])} aa" if diff["n_ext"] else "")
            + (f"; C-ext {len(diff['c_ext'])} aa" if diff["c_ext"] else "")
            + (f"; mutations {diff['mutations']}" if diff["mutations"] else ""))

        for asym, auth in zip(asyms, auths):
            segs = segs_by_asym.get(asym, ent_segs)
            l2u = label_to_unp(segs)
            miss_label, miss_unp, miss_tag, internal = [], [], 0, []
            n_obs = None
            if coords:
                obs = coords["observed"].get(asym, set())
                n_obs = len(obs)
                full = set(range(1, len(seq) + 1))
                miss_label = sorted(full - obs)
                miss_unp = sorted(l2u[i] for i in miss_label if i in l2u)
                miss_tag = sum(1 for i in miss_label if i not in l2u)
                if obs:
                    lo, hi = min(obs), max(obs)
                    internal = sorted(l2u[i] for i in miss_label if lo < i < hi and i in l2u)
            mstr = to_ranges(miss_unp)
            istr = to_ranges(internal)
            if mstr:
                missing_summary.append(f"{auth}: {mstr}")
            if istr:
                internal_summary.append(f"{auth}: {istr}")
            chain_warn = []
            if asym not in segs_by_asym:
                chain_warn.append("chain has no SIFTS segment; entity mapping used" if ent_segs
                                  else "chain has no SIFTS mapping")
            if "length mismatch" in diff["gaps"]:
                chain_warn.append("inconsistent SIFTS segment (excluded from numbering)")
            if coords is None:
                chain_warn.append("coordinates not analysed")
            rows["chains"].append({
                "PDB ID": pdb, "Entity": ids["entity_id"],
                "Chain (auth)": auth, "Chain (label)": asym,
                "Description": p["rcsb_polymer_entity"].get("pdbx_description"),
                "Entity length": len(seq),
                "Observed residues": n_obs,
                "UniProt range (SIFTS)": diff["unp_range"],
                "UniProt coverage": diff["coverage"],
                "N-terminal extension (non-UniProt)": diff["n_ext"],
                "C-terminal extension (non-UniProt)": diff["c_ext"],
                "Substitutions vs UniProt": diff["mutations"],
                "Mutation (author annotation)": p["rcsb_polymer_entity"].get("pdbx_mutation"),
                "Internal gaps / insertions": diff["gaps"],
                "Missing residues (UniProt numbering)": mstr,
                "N missing (UniProt)": len(miss_unp) if coords else None,
                "Missing internal (loop) residues": istr,
                "N missing internal": len(internal) if coords else None,
                "Missing non-UniProt (tag) residues": miss_tag if coords else None,
                "Missing residues (entity label_seq)": to_ranges(miss_label),
                "Entity sequence": seq,
                "Model": model_no,
                "Calculation status": (Status.WARNING if chain_warn else Status.OK).value,
                "Warning": "; ".join(chain_warn),
            })

    # --- assemblies -------------------------------------------------------------
    chosen, reason = choose_assembly(e.get("assemblies") or [],
                                     cfg.get("structure.assembly_preference", ["author", "software", "first"]))
    asu_copies = sum(len(p["rcsb_polymer_entity_container_identifiers"].get("asym_ids") or [])
                     for p in targets)
    for a in e.get("assemblies") or []:
        psa = a.get("pdbx_struct_assembly") or {}
        sym = global_symmetry(a)
        copies, _ = coords["expand"](psa.get("id")) if coords else (None, None)
        rows["assemblies"].append({
            "PDB ID": pdb, "Assembly": psa.get("id"),
            "Used in summary": bool(chosen and a["rcsb_id"] == chosen["rcsb_id"]),
            "Defined by": psa.get("details"), "Method details": psa.get("method_details"),
            "Oligomeric details": psa.get("oligomeric_details"),
            "Total polymer chains": psa.get("oligomeric_count"),
            "Target copies (gemmi)": copies,
            "Target state": oligomer_label(copies) if copies is not None else "",
            "Symmetry": sym.get("symbol"), "Stoichiometry": ", ".join(sym.get("stoichiometry") or []),
            "RCSB oligomeric state": sym.get("oligomeric_state"),
            "Selection reason": reason if chosen and a["rcsb_id"] == chosen["rcsb_id"] else "",
        })

    asm_id = (chosen or {}).get("pdbx_struct_assembly", {}).get("id")
    asm_copies, asm_het = (coords["expand"](asm_id) if coords and asm_id else (None, None))
    if coords and asm_id and asm_copies is None:
        warnings.append(f"assembly {asm_id} not found in mmCIF (gemmi assemblies: "
                        f"{coords.get('assembly_names') or 'none'}); copies taken from RCSB if possible")
    copies_source = "gemmi generators" if asm_copies is not None else ""
    if asm_copies is None and chosen and len(e.get("polymer_entities") or []) == 1:
        # Fallback without coordinates: homomeric entry -> chain count = copies
        asm_copies = (chosen.get("pdbx_struct_assembly") or {}).get("oligomeric_count")
        copies_source = "RCSB oligomeric_count (homomer, no coordinates)" if asm_copies is not None else ""
    note = assembly_note(asu_copies, asm_copies)

    # --- ligands ------------------------------------------------------------------
    lig_by_class: dict[str, list[str]] = defaultdict(list)
    for n in e.get("nonpolymer_entities") or []:
        comp = n["pdbx_entity_nonpoly"]["comp_id"]
        cls = classify_component(comp, classes)
        n_asu = len((n.get("rcsb_nonpolymer_entity_container_identifiers") or {}).get("asym_ids") or [])
        n_asm = asm_het.get(comp) if asm_het is not None else None
        rows["ligands"].append({
            "PDB ID": pdb, "Component": comp, "Name": n["pdbx_entity_nonpoly"]["name"],
            "Class": cls, "Count in ASU": n_asu,
            "Count in assembly": n_asm, "Assembly": asm_id,
            "Chains (auth)": ", ".join((n.get("rcsb_nonpolymer_entity_container_identifiers") or {})
                                       .get("auth_asym_ids") or []),
            "Chains (label)": ", ".join((n.get("rcsb_nonpolymer_entity_container_identifiers") or {})
                                        .get("asym_ids") or []),
        })
        lig_by_class[cls].append(f"{comp}({n_asu})")
    glycan_chains_asu, glycan_chains_asm = 0, (0 if coords and asm_id else None)
    for b in e.get("branched_entities") or []:
        b_asyms = (b.get("rcsb_branched_entity_container_identifiers") or {}).get("asym_ids") or []
        n_asu = len(b_asyms)
        n_asm = coords["count_subs"](asm_id, b_asyms) if coords and asm_id else None
        glycan_chains_asu += n_asu
        if glycan_chains_asm is not None and n_asm is not None:
            glycan_chains_asm += n_asm
        desc = (b.get("rcsb_branched_entity") or {}).get("pdbx_description")
        rows["ligands"].append({
            "PDB ID": pdb, "Component": "branched (oligosaccharide)", "Name": desc,
            "Class": "glycan", "Count in ASU": n_asu, "Count in assembly": n_asm,
            "Assembly": asm_id,
            "Chains (auth)": ", ".join((b.get("rcsb_branched_entity_container_identifiers") or {})
                                       .get("auth_asym_ids") or []),
            "Chains (label)": ", ".join(b_asyms),
        })
        lig_by_class["glycan"].append(f"oligosaccharide({n_asu})")

    def ion_count(where: Counter | None, ion: str) -> int | None:
        return None if where is None else where.get(ion, 0)

    asu_ions = {ion: sum(len((n.get("rcsb_nonpolymer_entity_container_identifiers") or {})
                             .get("asym_ids") or [])
                         for n in e.get("nonpolymer_entities") or []
                         if n["pdbx_entity_nonpoly"]["comp_id"] == ion)
                for ion in ions}

    sym = global_symmetry(chosen) if chosen else {}
    summary: dict[str, Any] = {
        "PDB ID": pdb,
        "Title": (e.get("struct") or {}).get("title"),
        "Release date": (e.get("rcsb_accession_info") or {}).get("initial_release_date", "")[:10],
        "Method": method,
        "Resolution (Å)": res,
        "R-free": rfree,
        "Space group": sg,
        "Target chains (auth)": ", ".join(c["Chain (auth)"] for c in rows["chains"]),
        "Construct (UniProt range, extensions, mutations)": " | ".join(construct_notes),
        "Missing residues (UniProt numbering)": " | ".join(missing_summary),
        "Missing internal residues": " | ".join(internal_summary),
    }
    for ion in ions:   # default: Zn, Ca, Cl (same column names as the original)
        label = ion.capitalize()
        summary[f"{label} ASU"] = asu_ions[ion]
        summary[f"{label} assembly"] = ion_count(asm_het, ion)
    summary.update({
        "Ligands of interest": ", ".join(lig_by_class["ligand of interest"]),
        "Glycans": ", ".join(lig_by_class["glycan"]),
        "Glycan chains ASU": glycan_chains_asu,
        "Glycan chains assembly": glycan_chains_asm,
        "Glycan residues assembly (NAG+MAN+...)": (sum(n for k, n in asm_het.items() if k in glycan_set)
                                                    if asm_het is not None else None),
        "Additives / buffer": ", ".join(lig_by_class["additive / buffer"]),
        "Other polymer entities": "; ".join(
            (o["rcsb_polymer_entity"] or {}).get("pdbx_description") or "" for o in others),
        "Assembly used": f"{asm_id} ({(chosen or {}).get('pdbx_struct_assembly', {}).get('details')})"
                         if chosen else "none",
        "RCSB oligomeric state": sym.get("oligomeric_state"),
        "Target copies in ASU": asu_copies,
        "Target copies in assembly": asm_copies,
        "Monomer / dimer": oligomer_label(asm_copies) if asm_copies is not None else "unknown",
        "Assembly note": note,
        "Warnings": "; ".join([w for w in [coord_err] + warnings if w]),
        "Assembly choice reason": reason,
        "Assembly copies source": copies_source,
        "RCSB assembly chain count": (chosen or {}).get("pdbx_struct_assembly", {}).get("oligomeric_count"),
        "Polymer entities": len(e.get("polymer_entities") or []),
        "Model analysed": model_no,
    })
    rows["summary"] = summary
    rows["_ctx"] = {"coords": coords, "asm_id": asm_id, "segs": segs_by_asym,
                    "method": method, "resolution": res}
    return rows


def cross_check(rcsb_ids: set[str], meta: dict, pdbe_best: list[dict],
                uniprot_xrefs: dict[str, dict]) -> pd.DataFrame:
    """RCSB vs PDBe vs UniProt entry lists, methods and resolutions (one row per ID)."""
    pdbe_ids: dict[str, list] = defaultdict(list)
    for b in pdbe_best:
        pdbe_ids[b["pdb_id"].upper()].append(b)
    all_ids = sorted(set(rcsb_ids) | set(pdbe_ids) | set(uniprot_xrefs))
    rows = []
    for pid in all_ids:
        e = meta.get(pid, {})
        r_res = ((e.get("rcsb_entry_info") or {}).get("resolution_combined") or [None])[0]
        b = pdbe_ids.get(pid, [])
        u = uniprot_xrefs.get(pid, {})
        u_res = None
        if u.get("Resolution") and u["Resolution"] != "-":
            try:
                u_res = float(u["Resolution"].split()[0])
            except ValueError:
                pass
        p_res = b[0].get("resolution") if b else None
        flags = []
        if pid not in rcsb_ids:
            flags.append("not returned by RCSB search")
        if pid not in pdbe_ids:
            flags.append("absent from PDBe best_structures")
        if pid not in uniprot_xrefs:
            flags.append("absent from UniProt xrefs")
        for label, other in (("PDBe", p_res), ("UniProt", u_res)):
            if r_res is not None and other is not None and abs(r_res - other) > 0.05:
                flags.append(f"resolution differs from {label}")
        rows.append({
            "PDB ID": pid,
            "In RCSB": pid in rcsb_ids, "In PDBe": pid in pdbe_ids, "In UniProt": pid in uniprot_xrefs,
            "RCSB resolution": r_res, "PDBe resolution": p_res, "UniProt resolution": u_res,
            "RCSB method": "; ".join(x["method"] for x in e.get("exptl") or []),
            "PDBe method": b[0].get("experimental_method") if b else None,
            "UniProt method": u.get("Method"),
            "PDBe chains (UniProt range)": "; ".join(
                f"{x['chain_id']}={x['unp_start']}-{x['unp_end']}" for x in b),
            "UniProt chains": u.get("Chains"),
            "Flags": "; ".join(flags) or "consistent",
        })
    return pd.DataFrame(rows)


def analyse_entry_extended(pdb: str, ctx: dict, sites: dict[int, str], useq: str,
                           sess: requests.Session, cfg: Config, rcsb_val: dict) -> dict:
    """Validation, binding sites, glycosylation, active site and CA extraction for one entry.

    Each part is isolated: a failure is recorded in ``warnings`` and the
    other parts still run.
    """
    json_kw, _ = http_kwargs(cfg)
    out: dict[str, Any] = {"validation": {"PDB ID": pdb}, "binding": [], "glyco": [],
                           "active": [], "ca": {}, "warnings": []}
    out["validation"].update({f"RCSB {k}": v for k, v in (rcsb_val.get(pdb) or {}).items()})
    outliers: dict = {}
    try:
        pct, outliers = pdbe_validation(sess, pdb, cfg.cache_dir, cfg.endpoint("pdbe_val_percentiles"),
                                        cfg.endpoint("pdbe_val_outliers"), **json_kw)
        out["validation"].update(pct)
        out["validation"]["Residues with outliers (PDBe)"] = len(outliers)
        out["validation"]["Calculation status"] = Status.OK.value if pct else Status.WARNING.value
        out["validation"]["Warning"] = "" if pct else "no PDBe percentiles available"
    except Exception as ex:  # noqa: BLE001
        out["warnings"].append(f"PDBe validation: {ex}")
        out["validation"]["Calculation status"] = Status.FAIL.value
        out["validation"]["Warning"] = f"PDBe validation: {ex}"
    if not ctx.get("coords"):
        return out
    classes, contacts = cfg.ligand_classes, cfg.require("contacts")
    backbone = set(contacts.get("backbone_atoms", ["N", "CA", "C", "O", "OXT"]))
    parts = (
        ("binding sites", "binding",
         lambda: binding_site_analysis(pdb, ctx, outliers, classes, contacts, useq)),
        ("glycosylation", "glyco", lambda: find_glycosylation_sites(pdb, ctx, classes, contacts)),
        ("active site", "active", lambda: active_site_analysis(pdb, ctx, sites, useq, outliers, backbone)),
        ("CA coordinates", "ca", lambda: ca_coordinates(pdb, ctx)),
    )
    for name, key, fn in parts:
        try:
            out[key] = fn()
        except Exception as ex:  # noqa: BLE001
            out["warnings"].append(f"{name}: {ex}")
    return out


def merge_into_summary(df_sum: pd.DataFrame, df_val: pd.DataFrame, df_active: pd.DataFrame,
                       df_glyco: pd.DataFrame, df_binding: pd.DataFrame) -> pd.DataFrame:
    """Add headline validation, active-site and glycosylation columns to Summary.

    Active-site residues whose model amino acid differs from UniProt
    (``Matches UniProt`` is False, e.g. an E424A mutant) are excluded from
    ``Active-site mean normalised B``.
    """
    df = df_sum.copy()
    if len(df_val):
        v = df_val.set_index("PDB ID")

        def pick(rcsb: str, pdbe: str) -> pd.Series | None:
            a = pd.to_numeric(v.get(rcsb), errors="coerce") if rcsb in v else None
            b = pd.to_numeric(v.get(pdbe), errors="coerce") if pdbe in v else None
            if a is None:
                return b
            return a.fillna(b) if b is not None else a

        for col, rc, pe in (("Clashscore", "RCSB clashscore", "PDBe clashscore"),
                            ("Ramachandran outliers (%)", "RCSB rama_outliers_pct", "PDBe rama_outliers_pct"),
                            ("Rotamer outliers (%)", "RCSB rotamer_outliers_pct", "PDBe rotamer_outliers_pct"),
                            ("RSRZ outliers (%)", "RCSB rsrz_outliers_pct", "PDBe rsrz_outliers_pct")):
            series = pick(rc, pe)
            if series is not None:
                df[col] = df["PDB ID"].map(series)
        pc = "PDBe clashscore percentile (vs similar resolution)"
        if pc in v:
            df["Clashscore percentile (similar resolution)"] = df["PDB ID"].map(v[pc])
    if len(df_active) and "Modelled" in df_active:
        g = df_active.groupby("PDB ID")
        df["Active-site residues modelled (best chain)"] = df["PDB ID"].map(
            df_active.groupby(["PDB ID", "Chain"])["Modelled"].mean().groupby("PDB ID").max().round(2))
        if "Normalised B (z)" in df_active:
            ok = df_active
            if "Matches UniProt" in df_active:
                ok = df_active[df_active["Matches UniProt"].fillna(True).astype(bool)]
            df["Active-site mean normalised B"] = df["PDB ID"].map(
                ok.groupby("PDB ID")["Normalised B (z)"].mean().round(2))
        if "Min occupancy" in df_active:
            df["Active-site partial occupancy"] = df["PDB ID"].map(
                g["Min occupancy"].apply(lambda x: bool((x.dropna() < 1).any())))
        if "Matches UniProt" in df_active:
            mm = df_active[df_active["Matches UniProt"] == False]  # noqa: E712
            df["Active-site residues differing from UniProt"] = df["PDB ID"].map(
                mm.groupby("PDB ID").apply(
                    lambda x: ", ".join(sorted(set(f"{r}/{c}" for r, c in zip(x["UniProt residue"], x["Chain"])))),
                    include_groups=False) if len(mm) else {})
    if len(df_glyco):
        df["Glycosylated Asn (UniProt)"] = df["PDB ID"].map(
            df_glyco.groupby("PDB ID")["Asn (UniProt)"].apply(
                lambda x: ", ".join(f"N{int(n)}" for n in sorted(set(x.dropna())))))
    if len(df_binding):
        lig = df_binding[df_binding["Class"] == "ligand of interest"]
        if len(lig):
            df["Max ligand B ratio"] = df["PDB ID"].map(lig.groupby("PDB ID")["B ratio ligand/environment"].max())
    return df


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
@dataclass
class SurveyResults:
    """Everything produced by :func:`run_survey`.

    ``tables`` holds the workbook sheets in output order; the other fields are
    intermediate objects reused by later notebooks (CA coordinates, sites,
    sequence, metadata) and by the manifest.
    """

    tables: dict[str, pd.DataFrame]
    ids: list[str]
    useq: str
    uniprot_name: str
    sites: dict[int, str]
    ca: dict
    meta: dict
    lid_info: dict
    lid_reference: tuple[str, str] | None
    validation_variant: str
    issues: IssueLog
    input_files: list = field(default_factory=list)


def run_survey(cfg: Config, ids: list[str] | None = None, issues: IssueLog | None = None,
               sess: requests.Session | None = None) -> SurveyResults:
    """Run the full structure survey and return all tables.

    Parameters
    ----------
    cfg:
        Loaded configuration.
    ids:
        Explicit PDB IDs (e.g. the validation set). If None, ``run.ids`` from
        the config is used, otherwise every RCSB entry for the accession
        (limited by ``run.limit``).
    issues:
        Issue log to append to (a new one is created if None).

    Raises
    ------
    FetchError
        If the RCSB search, UniProt entry or metadata cannot be retrieved
        (nothing useful can be produced without them).
    """
    issues = issues if issues is not None else IssueLog()
    sess = sess or session_from_config(cfg)
    acc, cache = cfg.accession, cfg.cache_dir
    json_kw, _ = http_kwargs(cfg)
    use_coords = bool(cfg.get("run.use_coordinates", True))
    if use_coords:
        try:
            import gemmi  # noqa: F401
        except ImportError:
            issues.add("setup", Status.NOT_RUN, "gemmi not installed: coordinate analyses skipped")
            use_coords = False
    if cfg.get("structure.model_policy", "first") == "all":
        issues.add("setup", Status.WARNING, "model_policy 'all' is not implemented until Milestone 2; "
                   "model 1 is analysed for every entry")

    # 1. entry list ---------------------------------------------------------------
    searched = rcsb_search(sess, acc, cfg.endpoint("rcsb_search"))
    if ids is None:
        ids = cfg.get("run.ids") or searched
        if cfg.get("run.limit"):
            ids = ids[: int(cfg.get("run.limit"))]
    ids = [i.upper() for i in ids]
    for missing in sorted(set(ids) - set(searched)):
        issues.add("search", Status.WARNING, "requested entry is not returned by the RCSB search "
                   "for this accession", pdb_id=missing)
    log.info("%d RCSB entries for %s; processing %d", len(searched), acc, len(ids))

    # 2. reference data ---------------------------------------------------------
    entry = uniprot_entry(sess, acc, cache, cfg.endpoint("uniprot_json"), **json_kw)
    useq, uname, uxrefs = uniprot_info(entry)
    best = fetch_best_structures(sess, acc, cache, cfg.endpoint("pdbe_best"), **json_kw)
    sites, site_def = active_site_residues(entry, useq, cfg.active_site, issues)

    # 3. metadata + validation summaries -------------------------------------------
    meta = rcsb_metadata(sess, ids, cache, cfg.endpoint("rcsb_graphql"), **json_kw)
    rcsb_val, variant = rcsb_validation(sess, ids, cache, cfg.endpoint("rcsb_graphql"), **json_kw)
    if variant == "none":
        issues.add("validation", Status.WARNING, "RCSB validation summaries unavailable; PDBe values used")

    # 4. per entry ------------------------------------------------------------------
    summary, chains, ligands, assemblies = [], [], [], []
    validation, binding, glyco, active, ca = [], [], [], [], {}
    for k, pdb in enumerate(ids, 1):
        e = meta.get(pdb)
        if e is None:
            summary.append({"PDB ID": pdb, "Warnings": "no RCSB metadata returned"})
            issues.add("metadata", Status.FAIL, "no RCSB metadata returned", pdb_id=pdb)
            continue
        try:
            sifts = fetch_sifts(sess, pdb, cache, cfg.endpoint("pdbe_sifts"), **json_kw)
            rows = process_entry(pdb, e, sifts, cfg, useq, sess, use_coords)
        except Exception as ex:  # noqa: BLE001 - one entry must not stop the survey
            summary.append({"PDB ID": pdb, "Warnings": f"processing failed: {ex}"})
            issues.add("process_entry", Status.FAIL, f"processing failed: {ex}", pdb_id=pdb)
            continue
        ctx = rows.pop("_ctx")
        ext = analyse_entry_extended(pdb, ctx, sites, useq, sess, cfg, rcsb_val)
        if ext["warnings"]:
            rows["summary"]["Warnings"] = "; ".join(
                w for w in [rows["summary"].get("Warnings")] + ext["warnings"] if w)
            for w in ext["warnings"]:
                issues.add("extended analysis", Status.WARNING, w, pdb_id=pdb)
        if rows["summary"].get("Warnings", "").startswith("coordinate analysis failed"):
            issues.add("coordinates", Status.FAIL, rows["summary"]["Warnings"], pdb_id=pdb)
        summary.append(rows["summary"])
        chains += rows["chains"]
        ligands += rows["ligands"]
        assemblies += rows["assemblies"]
        validation.append(ext["validation"])
        binding += ext["binding"]
        glyco += ext["glyco"]
        active += ext["active"]
        ca.update(ext["ca"])
        if k % 10 == 0 or k == len(ids):
            log.info("%d/%d processed", k, len(ids))

    # 5. cross-entry analyses ---------------------------------------------------------
    df_chains, df_ligands, df_assemblies = map(pd.DataFrame, (chains, ligands, assemblies))
    df_val, df_binding, df_glyco, df_active = map(pd.DataFrame, (validation, binding, glyco, active))
    df_summary = merge_into_summary(pd.DataFrame(summary), df_val, df_active, df_glyco, df_binding)
    if "Resolution (Å)" in df_summary:
        df_summary = df_summary.sort_values(["Resolution (Å)", "PDB ID"], na_position="last")

    modes = cfg.get("ranking.modes") or {}
    if "global" not in modes:
        issues.add("ranking", Status.FAIL, "ranking mode 'global' missing from configuration")
    crit = criteria_table(df_summary, df_chains, df_active)
    df_rank = rank_templates(crit, modes.get("global", {"criteria": []}), "global",
                             extra=df_summary.set_index("PDB ID"))

    df_var, ref = pd.DataFrame(), None
    for pid in df_rank["PDB ID"]:          # reference = best-ranked entry that has coordinates
        ref = next((key for key in ca if key[0] == pid), None)
        if ref:
            df_var = structural_variability(ca, ref, cfg.flexible_regions, list(sites),
                                            int(cfg.get("variability.min_common_ca", 50)))
            break
    lid_cfg = cfg.lid
    df_lid, lid_info = lid_conformation(ca, ref, lid_cfg, construct_ranges(df_chains))
    if len(df_lid):
        low = float(lid_cfg.get("classifier", {}).get("low_confidence_resolution", 3.5))
        df_summary = df_summary.merge(lid_entry_summary(df_lid, df_summary, low), on="PDB ID", how="left")
    lid_tab = pd.DataFrame()
    if lid_cfg:
        df_summary, lid_tab = lid_ligand_association(df_summary, df_binding, lid_cfg, useq)
    elif ca:
        issues.add("entrance lid", Status.NOT_APPLICABLE, "no entrance_lid section in regions config")

    idset = set(ids)
    limited = set(ids) != set(searched)
    df_xcheck = cross_check(idset, meta,
                            [b for b in best if not limited or b["pdb_id"].upper() in idset],
                            {k: v for k, v in uxrefs.items() if not limited or k in idset})

    tables = {
        "Summary": df_summary, "TemplateRanking": df_rank, "Chains": df_chains,
        "Ligands": df_ligands, "BindingSites": df_binding, "Glycosylation": df_glyco,
        "ActiveSite": df_active, "Validation": df_val, "Variability": df_var,
        "EntranceLid": df_lid, "LidVsLigand": lid_tab.reset_index() if len(lid_tab) else lid_tab,
        "Assemblies": df_assemblies, "CrossCheck": df_xcheck,
        "ActiveSiteDefinition": site_def,
    }
    res = SurveyResults(tables=tables, ids=ids, useq=useq, uniprot_name=uname, sites=sites, ca=ca,
                        meta=meta, lid_info=lid_info, lid_reference=ref, validation_variant=variant,
                        issues=issues)
    res.tables["Notes"] = notes_table(cfg, res)
    res.input_files = sorted((cache / "mmcif").glob("*.cif.gz")) if use_coords else []
    res.input_files = [p for p in res.input_files if p.name.split(".")[0] in idset]
    return res


def notes_table(cfg: Config, res: SurveyResults) -> pd.DataFrame:
    """The ``Notes`` sheet: parameters and definitions (same items as before, plus provenance)."""
    df = res.tables["Summary"]
    counts = df.get("Monomer / dimer", pd.Series(dtype=str)).value_counts().to_dict()
    c = cfg.require("contacts")
    import json
    rows = [
        ("UniProt accession", cfg.accession),
        ("Protein", res.uniprot_name),
        ("Canonical length", len(res.useq)),
        ("Entries processed", len(res.ids)),
        ("Oligomeric state counts", json.dumps(counts)),
        ("Coordinate analysis (gemmi)", "yes" if cfg.get("run.use_coordinates", True) and res.ca else "no"),
        ("Assembly choice", "Preference " + " → ".join(cfg.get("structure.assembly_preference", []))
         + "; the reason is recorded per entry (Summary → Assembly choice reason)."),
        ("Monomer / dimer", "Copies of the target in the chosen biological assembly, from the mmCIF "
                            "assembly generators (gemmi), not the ASU chain count."),
        ("Missing residues", "Entity residues without coordinates in model 1, UniProt numbering via SIFTS; "
                             "'internal' = between first and last modelled residue."),
        ("Construct differences", "From SIFTS segments: extensions = residues outside UniProt-mapped "
                                  "segments; substitutions = entity vs canonical UniProt."),
        ("Ligand classes", "ion of interest = " + "/".join(cfg.ligand_classes["ions_of_interest"])
         + "; glycan; additive/buffer; ligand of interest = the rest (config/ligand_classes.yaml)."),
        ("Active-site residues", ", ".join(f"{res.useq[n - 1]}{n}" for n in res.sites)),
        ("Binding sites", f"Contacts within {c['ligand']} Å (ligands), {c['metal']} Å (metals), "
                          f"{c['anion']} Å (anions); symmetry mates included and marked *; "
                          "B ratio = ligand mean B / contacting-atom mean B (an indirect indicator, "
                          "not a substitute for map-model fit)."),
        ("Normalised B", "(residue mean B - chain mean B) / chain SD. Not meaningful for many cryo-EM models."),
        ("Template ranking", "Mode 'global': mean scaled rank of resolution, clashscore, Ramachandran "
                             "outliers, internal missing residues, coverage and active-site completeness "
                             "(weights in config; components shown per entry)."),
        ("Variability", "Kabsch CA superposition onto the top-ranked chain; region RMSD after the global fit."),
        ("Entrance lid", "Majority of three votes (CA i,i+3 helicity; clustering if Ashman D ≥ threshold; "
                         "nearest calibration structure). Too few modelled lid residues = disordered."),
        ("Entrance lid run diagnostics", json.dumps(res.lid_info, default=str)),
        ("RCSB validation schema variant", res.validation_variant),
        ("Missing values", "Blank cell = not available / not applicable; see Calculation status columns."),
        ("UniProt canonical sequence", res.useq),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def fetch_entry_metadata(cfg: Config, sess: requests.Session | None = None) -> tuple[list[str], dict]:
    """RCSB search + GraphQL metadata for every entry of the accession (cheap; cached).

    Used to choose the validation set before the expensive coordinate run.
    """
    sess = sess or session_from_config(cfg)
    json_kw, _ = http_kwargs(cfg)
    ids = rcsb_search(sess, cfg.accession, cfg.endpoint("rcsb_search"))
    meta = rcsb_metadata(sess, ids, cfg.cache_dir, cfg.endpoint("rcsb_graphql"), **json_kw)
    return ids, meta


def save_survey(cfg: Config, res: SurveyResults, extra_sheets: dict[str, pd.DataFrame] | None = None,
                workbook: Path | None = None, tag: str = "") -> dict[str, Path]:
    """Write the survey workbook, CSV/Parquet tables and the reproducibility manifest.

    Sheet order: the 14 original sheets, then ``extra_sheets`` (e.g.
    SanityChecks, ValidationSet), ``ActiveSiteDefinition``, ``RunIssues``,
    ``Software``, ``Reproducibility`` and ``DataDictionary``.

    Returns the written paths (``workbook``, ``manifest``, ``tables_dir``).
    """
    from .excel_output import write_tables, write_workbook
    from .manifest import build_manifest, detect_software, manifest_table, write_manifest

    software = detect_software()
    sheets = {name: res.tables.get(name, pd.DataFrame()) for name in ORIGINAL_SHEETS}
    sheets.update(extra_sheets or {})
    sheets["ActiveSiteDefinition"] = res.tables.get("ActiveSiteDefinition", pd.DataFrame())
    sheets["RunIssues"] = res.issues.to_frame()
    sheets["Software"] = software
    manifest = build_manifest(cfg, structure_ids=res.ids, input_files=res.input_files, software=software,
                              extra={"lid_reference": res.lid_reference, "lid_diagnostics": res.lid_info,
                                     "validation_schema_variant": res.validation_variant,
                                     "issue_counts": {s.value: res.issues.count(s) for s in Status}})
    sheets["Reproducibility"] = manifest_table(manifest)
    suffix = f"_{tag}" if tag else ""
    wb = workbook or cfg.output_file("survey_workbook").with_name(
        cfg.output_file("survey_workbook").stem + suffix + ".xlsx")
    write_workbook(wb, sheets)
    mpath = write_manifest(manifest, cfg.output_file("manifest").with_name(
        cfg.output_file("manifest").stem + suffix + ".json"))
    tdir = cfg.outputs_dir / f"survey_tables{suffix}"
    write_tables(tdir, sheets, csv=bool(cfg.get("outputs.write_csv", True)),
                 parquet=bool(cfg.get("outputs.write_parquet", True)))
    log.info("wrote %s, %s and tables in %s", wb, mpath, tdir)
    return {"workbook": wb, "manifest": mpath, "tables_dir": tdir}


__all__ = ["ORIGINAL_SHEETS", "SurveyResults", "run_survey", "process_entry", "cross_check",
           "analyse_entry_extended", "merge_into_summary", "notes_table", "fetch_entry_metadata",
           "save_survey", "FetchError"]
