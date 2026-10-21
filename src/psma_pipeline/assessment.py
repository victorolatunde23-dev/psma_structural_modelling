"""Modelling assessment (notebook 02, Milestone 2).

Input: the survey workbook written by notebook 01 (or the in-memory survey
tables). Output: ``<ACC>_modelling_assessment.xlsx`` with

* **Interfaces** – one row per chain–chain interface (biological assembly);
* **CrystalContacts** – lattice interfaces of crystal structures;
* **InterfaceContacts** – one row per residue–residue contact pair;
* **InterfaceComparison** – pairwise similarity of biological interfaces;
* **LocalModelFit**, **LigandModelFit**, **EMMapFit** – map–model fit;
* **TemplateRankings** – every ranking mode (``RankingMode`` column) and
  **RankingTop** (top entries per mode);
* **AssessmentSummary** – survey summary + interface and local-fit columns.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .cache import FetchError, cached_json, session_from_config
from .config import Config
from .interfaces import (analyse_interface, compare_interfaces, jaccard, lattice_interfaces, pair_set,
                         residue_set)
from .ligands import classify_component
from .local_fit import (em_entry_row, group_fit, ligand_fit_rows, method_kind, residue_fit_table,
                        support_score, worst_category)
from .pdbe import fetch_sifts, sifts_segments
from .ranking import criteria_table, rank_templates
from .rcsb import load_structure
from .status import IssueLog, Status
from .structure_atoms import (assembly_generators, asu_as_assembly, atoms_from_gemmi, build_assembly,
                              lattice_operators, target_protomers)
from .survey import http_kwargs
from .validation import fetch_validation_xml, parse_validation_xml, pdbe_validation

log = logging.getLogger("psma_pipeline.assessment")

LABEL = re.compile(r"^([A-Z])(\d+)/([^*\s]+)(\*)?")


@dataclass
class AssessmentResults:
    """Tables and intermediate objects of :func:`run_assessment`."""

    tables: dict[str, pd.DataFrame]
    jaccard_matrix: np.ndarray
    matrix_ids: list[str]
    canonical: dict
    issues: IssueLog
    ids: list[str]
    input_files: list = field(default_factory=list)


def load_survey_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read a survey workbook written by notebook 01."""
    return pd.read_excel(path, sheet_name=None)


def _members_from_labels(text: Any, unp_map: dict) -> list[tuple]:
    """``"H377/A, D387/B*"`` -> author-numbered members via ``{(chain, unp): (resnum, icode)}``."""
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        m = LABEL.match(part.strip())
        if m:
            key = (m.group(3), int(m.group(2)))
            if key in unp_map:
                out.append((m.group(3), *unp_map[key]))
    return out


def _entry_inputs(pdb: str, tables: dict[str, pd.DataFrame]) -> dict:
    s = tables["Summary"].set_index("PDB ID").loc[pdb]
    chains = tables["Chains"]
    ch = chains[chains["PDB ID"] == pdb]
    asm = str(s.get("Assembly used") or "none").split()[0]
    return {"summary": s, "target_entities": {str(e) for e in ch["Entity"].dropna()},
            "assembly": None if asm == "none" else asm, "method": s.get("Method"),
            "resolution": s.get("Resolution (Å)")}


def _emdb_maps(sess: requests.Session, emdb_id: str, cfg: Config) -> dict:
    """Map / half-map availability from the EMDB API (recorded as unknown if the fields are absent)."""
    if not emdb_id:
        return {"Primary map": "unknown", "Half maps": "unknown"}
    json_kw, _ = http_kwargs(cfg)
    try:
        data = cached_json(sess, cfg.endpoint("emdb_entry").format(emdb=emdb_id),
                           cfg.cache_dir / "emdb" / f"{emdb_id}.json", allow_404=True, **json_kw)
    except FetchError as ex:
        return {"Primary map": f"NOT RUN ({ex})", "Half maps": "NOT RUN"}
    interp = data.get("interpretation") or {}
    hm = (interp.get("half_map_list") or {}).get("half_map") if isinstance(interp, dict) else None
    mp = data.get("map")
    return {"Primary map": "available" if mp else "unknown (not in EMDB API response)",
            "Half maps": f"{len(hm)} available" if hm else "unknown (not in EMDB API response)"}


def run_assessment(cfg: Config, tables: dict[str, pd.DataFrame], ids: list[str] | None = None,
                   sess: requests.Session | None = None, issues: IssueLog | None = None) -> AssessmentResults:
    """Interfaces, crystal contacts, local model fit and ranking modes for survey entries.

    Parameters
    ----------
    tables:
        Survey tables (``load_survey_tables`` or ``SurveyResults.tables``).
    ids:
        Subset of PDB IDs (default: every entry in Summary).
    """
    import gemmi  # noqa: F401 - required here; fail early with a clear ImportError

    issues = issues if issues is not None else IssueLog()
    sess = sess or session_from_config(cfg)
    json_kw, dl_kw = http_kwargs(cfg)
    icfg = dict(cfg.require("interfaces"))
    icfg["_glycans"] = cfg.ligand_classes["glycans"]
    fcfg = cfg.require("local_fit")
    acc = cfg.accession
    summary = tables["Summary"]
    ids = [i.upper() for i in (summary["PDB ID"].tolist() if ids is None else ids)]
    sites = sorted(int(n) for n in tables.get("ActiveSiteDefinition", pd.DataFrame()).query(
        "Status == 'OK'")["UniProt number"].dropna().unique()) if "ActiveSiteDefinition" in tables else []
    regions = cfg.flexible_regions
    lid_range = cfg.lid.get("range") if cfg.lid else None
    classify = lambda c: classify_component(c, cfg.ligand_classes)  # noqa: E731

    iface_rows, lattice_rows, contact_tabs, local_rows, ligand_rows, em_rows = [], [], [], [], [], []
    ca_by_iface: dict[str, dict] = {}
    pairs_by_iface: dict[str, set] = {}
    modelled: dict[str, dict[str, set]] = {}
    main_iface: dict[str, str] = {}
    entry_extra: dict[str, dict] = {}
    files = []

    for k, pdb in enumerate(ids, 1):
        extra: dict[str, Any] = {}
        entry_extra[pdb] = extra
        try:
            inp = _entry_inputs(pdb, tables)
        except KeyError:
            issues.add("assessment", Status.FAIL, "entry not in survey Summary", pdb_id=pdb)
            continue
        kind = method_kind(inp["method"])
        # ---- structure and atoms -----------------------------------------------------
        try:
            st = load_structure(sess, pdb, cfg.cache_dir / "mmcif", cfg.endpoint("rcsb_cif"), **dl_kw)
            files.append(cfg.cache_dir / "mmcif" / f"{pdb}.cif.gz")
            segs = sifts_segments(fetch_sifts(sess, pdb, cfg.cache_dir, cfg.endpoint("pdbe_sifts"), **json_kw),
                                  pdb, acc)
            atoms = atoms_from_gemmi(st, 0, segs, inp["target_entities"], icfg.get("altloc", "first"))
        except Exception as ex:  # noqa: BLE001
            issues.add("assessment", Status.FAIL, f"structure could not be loaded: {ex}", pdb_id=pdb)
            continue
        unp_map = {}
        tgt = atoms[atoms["target"] & atoms["polymer"] & atoms["unp"].notna()]
        for r in tgt.drop_duplicates(["chain", "unp"]).itertuples():
            unp_map[(r.chain, int(r.unp))] = (int(r.resnum), r.icode or "")
        # ---- biological assembly interfaces -------------------------------------------
        gens = assembly_generators(st, inp["assembly"]) if inp["assembly"] else None
        if gens is None:
            issues.add("interfaces", Status.WARNING, f"assembly {inp['assembly']} not found; ASU used",
                       pdb_id=pdb)
            asm = asu_as_assembly(atoms)
            asm_label = "ASU"
        else:
            asm = build_assembly(atoms, gens)
            asm_label = inp["assembly"]
        prots = target_protomers(asm)
        modelled[pdb] = {p: set(asm[(asm["protomer"] == p) & (asm["atom"] == "CA")]["unp"].dropna().astype(int))
                         for p in prots}
        best = None
        try:
            outliers = pdbe_validation(sess, pdb, cfg.cache_dir, cfg.endpoint("pdbe_val_percentiles"),
                                       cfg.endpoint("pdbe_val_outliers"), **json_kw)[1]
        except Exception as ex:  # noqa: BLE001
            outliers = {}
            issues.add("interfaces", Status.WARNING, f"validation outliers unavailable: {ex}", pdb_id=pdb)
        for p1, p2 in itertools.combinations(prots, 2):
            try:
                row, con, ca = analyse_interface(pdb, asm, p1, p2, icfg, outliers, assembly=asm_label)
            except Exception as ex:  # noqa: BLE001
                issues.add("interfaces", Status.FAIL, f"{p1}-{p2}: {ex}", pdb_id=pdb)
                continue
            iid = f"{pdb}:{p1}-{p2}"
            row["Interface ID"] = iid
            if row.get("Residue contacts", 0) == 0:
                issues.add("interfaces", Status.WARNING, f"{p1}-{p2}: dimer with zero interface contacts",
                           pdb_id=pdb)
            iface_rows.append(row)
            if len(con):
                contact_tabs.append(con.assign(**{"Interface ID": iid}))
                pairs_by_iface[iid] = pair_set(con)
                ca_by_iface[iid] = ca
                if best is None or (row.get("BSA total (Å²)") or 0) > (best[1] or 0):
                    best = (iid, row.get("BSA total (Å²)"))
        if best:
            main_iface[pdb] = best[0]
        # ---- crystal-lattice contacts --------------------------------------------------
        if icfg.get("lattice", {}).get("enabled", True) and kind == "xray":
            try:
                ops = lattice_operators(st, int(icfg["lattice"].get("max_cell_shift", 1)))
                lrows, ltabs, lca = lattice_interfaces(pdb, atoms, ops, icfg)
                for r, t in zip(lrows, ltabs):
                    iid = f"{pdb}:lattice:{r['Protomer 1']}-{r['Protomer 2']}"
                    r["Interface ID"] = iid
                    pairs_by_iface[iid] = pair_set(t)
                    contact_tabs.append(t.assign(**{"Interface ID": iid}))
                lattice_rows += lrows
            except Exception as ex:  # noqa: BLE001
                issues.add("crystal contacts", Status.WARNING, f"lattice search failed: {ex}", pdb_id=pdb)
        # ---- local model fit -----------------------------------------------------------
        fit = pd.DataFrame()
        try:
            xml = parse_validation_xml(fetch_validation_xml(sess, pdb, cfg.cache_dir, cfg.endpoint("validation_xml"),
                                                            cfg.get("endpoints.validation_xml_fallback"), **dl_kw))
            fit = residue_fit_table(xml["residues"], kind, inp["resolution"] if kind == "xray" else
                                    _num(xml["entry"].get("EMDB-resolution")) or inp["resolution"], fcfg)
            extra["Validation XML schema"] = xml["schema"]
            if kind == "em":
                em = em_entry_row(pdb, xml["entry"], fit, xml["schema"])
                em.update(_emdb_maps(sess, str(xml["entry"].get("emdb_id") or ""), cfg))
                em_rows.append(em)
                extra["EM Q-score"] = em.get("Q_score")
        except Exception as ex:  # noqa: BLE001
            issues.add("local fit", Status.NOT_RUN, f"validation XML unavailable: {ex}", pdb_id=pdb)
            extra["Validation XML schema"] = "unavailable"
        res_for_q = _num((em_rows[-1] if kind == "em" and em_rows and em_rows[-1]["PDB ID"] == pdb else {}).get(
            "EMDB-resolution")) or inp["resolution"]
        groups: dict[str, list[tuple]] = {}
        tchains = sorted({c for c, _ in unp_map})
        groups["active site"] = [(c, *unp_map[(c, n)]) for c in tchains for n in sites if (c, n) in unp_map]
        bs = tables.get("BindingSites", pd.DataFrame())
        bs = bs[bs["PDB ID"] == pdb] if len(bs) else bs
        metal = bs[bs["Component"].isin(cfg.ligand_classes["metals"])] if len(bs) else bs
        groups["metal-coordinating residues"] = [m for t in (metal.get("Donor atoms (N/O/S: Å)", pd.Series(dtype=str))
                                                             if len(metal) else []) for m in _members_from_labels(t, unp_map)]
        lig = bs[bs["Class"] == "ligand of interest"] if len(bs) else bs
        groups["ligand-contact residues"] = [m for t in (lig["Protein residues (UniProt/chain)"] if len(lig) else [])
                                             for m in _members_from_labels(t, unp_map)]
        for name, (a, b) in regions.items():
            groups[name] = [(c, *unp_map[(c, n)]) for c in tchains for n in range(a, b + 1) if (c, n) in unp_map]
        if pdb in main_iface:
            r = next(x for x in iface_rows if x["Interface ID"] == main_iface[pdb])
            mem = []
            for side in ("1", "2"):
                c = r[f"Chain {side}"]
                for n in sorted({int(x) for x in str(r.get(f"Interface residues {side} (UniProt)") or "").split(", ") if x}):
                    if (c, n) in unp_map:
                        mem.append((c, *unp_map[(c, n)]))
            groups["dimer-interface residues"] = mem
        gl = tables.get("Glycosylation", pd.DataFrame())
        gl = gl[gl["PDB ID"] == pdb] if len(gl) else gl
        groups["glycosylation sites"] = [(r["Chain"], int(r["Asn (author)"]), "") for _, r in gl.iterrows()] if len(gl) else []
        if len(fit):
            het = fit[fit["label_seq"].isna() & (fit["resname"] != "HOH")]
            cls = het["resname"].map(classify)
            groups["ligands"] = [(r.chain, r.resnum, r.icode) for r in het[cls == "ligand of interest"].itertuples()]
            groups["ions"] = [(r.chain, r.resnum, r.icode) for r in het[cls == "ion of interest"].itertuples()]
        for gname, mem in groups.items():
            local_rows.append(group_fit(pdb, gname, mem, fit, kind, res_for_q, fcfg))
        local_rows.append({"PDB ID": pdb, "Group": "conserved waters", "Method kind": kind,
                           "Calculation status": Status.NOT_RUN.value, "Warning": "conserved waters: Milestone 3"})
        ligand_rows += ligand_fit_rows(pdb, fit, kind, res_for_q, fcfg, classify, tables.get("BindingSites"))
        if k % 10 == 0 or k == len(ids):
            log.info("assessment %d/%d", k, len(ids))

    # ---- interface comparison -------------------------------------------------------------
    df_if = pd.DataFrame(iface_rows)
    df_lat = pd.DataFrame(lattice_rows)
    reasons = summary.set_index("PDB ID")["Assembly choice reason"] if "Assembly choice reason" in summary else pd.Series(dtype=str)
    states = summary.set_index("PDB ID")["Monomer / dimer"]
    canonical_ids = [main_iface[p] for p in main_iface
                     if str(reasons.get(p, "")).startswith("author") and states.get(p) == "dimer"]
    rank_order = tables.get("TemplateRanking", pd.DataFrame()).get("PDB ID", pd.Series(dtype=str)).tolist()
    reference_id = next((main_iface[p] for p in rank_order if p in main_iface), None)
    main_rows = [r for r in iface_rows if r["Interface ID"] in set(main_iface.values())]
    comp, pairs_tab, J, J_ids, can_info = compare_interfaces(main_rows, pairs_by_iface, ca_by_iface, icfg,
                                                            canonical_ids, reference_id)
    if len(df_if) and len(comp):
        df_if = df_if.merge(comp, on="Interface ID", how="left")
        df_if["Main interface"] = df_if["Interface ID"].isin(set(main_iface.values()))
    can_res = set(can_info["canonical residues"])
    if len(df_lat):
        df_lat["Canonical interface similarity"] = [round(jaccard(residue_set(r), can_res) or 0, 3)
                                                    for r in lattice_rows]
        df_lat["Matches canonical dimer"] = df_lat["Canonical interface similarity"] >= float(
            icfg.get("lattice", {}).get("canonical_match_jaccard", 0.5))
    # interface completeness against the canonical residue set
    if len(df_if):
        miss1, miss2, complete = [], [], []
        for _, r in df_if.iterrows():
            m = modelled.get(r["PDB ID"], {})
            s1, s2 = m.get(r["Protomer 1"], set()), m.get(r["Protomer 2"], set())
            a, b = sorted(can_res - s1), sorted(can_res - s2)
            miss1.append(", ".join(map(str, a)))
            miss2.append(", ".join(map(str, b)))
            complete.append(bool(can_res) and not a and not b)
        df_if["Missing canonical interface residues 1"] = miss1
        df_if["Missing canonical interface residues 2"] = miss2
        df_if["Interface complete"] = complete

    # ---- summary columns -------------------------------------------------------------------
    df_local = pd.DataFrame(local_rows)
    asum = _assessment_summary(summary, tables, df_if, df_lat, df_local, pd.DataFrame(ligand_rows),
                               entry_extra, main_iface, cfg)
    rankings, top = rank_all_modes(cfg, asum, tables, issues)
    out = {
        "AssessmentSummary": asum, "Interfaces": df_if, "CrystalContacts": df_lat,
        "InterfaceContacts": pd.concat(contact_tabs, ignore_index=True) if contact_tabs else pd.DataFrame(),
        "InterfaceComparison": pairs_tab, "LocalModelFit": df_local,
        "LigandModelFit": pd.DataFrame(ligand_rows), "EMMapFit": pd.DataFrame(em_rows),
        "TemplateRankings": rankings, "RankingTop": top,
        "CanonicalInterface": pd.DataFrame([{"Item": k, "Value": ", ".join(map(str, v)) if isinstance(v, list) else v}
                                            for k, v in can_info.items()]),
    }
    return AssessmentResults(out, J, J_ids, can_info, issues, ids, files)


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _assessment_summary(summary: pd.DataFrame, tables: dict, df_if: pd.DataFrame, df_lat: pd.DataFrame,
                        df_local: pd.DataFrame, df_lig: pd.DataFrame, extra: dict, main_iface: dict,
                        cfg: Config) -> pd.DataFrame:
    """Survey Summary plus interface, local-fit and ranking-helper columns."""
    s = summary.copy()
    fcfg = cfg.require("local_fit")
    main = df_if[df_if["Main interface"]].set_index("PDB ID") if len(df_if) and "Main interface" in df_if else pd.DataFrame()
    for col in ("Dimer interface area (Å²)", "Interface residue count", "Canonical interface similarity",
                "Interface complete", "Noncanonical interface flag", "Interface cluster"):
        s[col] = s["PDB ID"].map(main[col]) if len(main) and col in main else None
    if len(main):
        s["Interface validation warning"] = s["PDB ID"].map(main["Validation outliers at interface"].fillna("")
                                                            .map(lambda x: f"outliers: {x}" if x else ""))
    if len(df_lat):
        hit = df_lat[df_lat["Matches canonical dimer"]].groupby("PDB ID")["Lattice operator"].apply(
            lambda x: ", ".join(sorted(set(x))))
        s["Canonical dimer in crystal lattice"] = s["PDB ID"].map(hit)
    # local support per group
    if len(df_local) and "Group support" in df_local:
        g = df_local.dropna(subset=["Group support"]).set_index(["PDB ID", "Group"])
        for group, col in (("active site", "Active-site local support"), ("ligands", "Ligand local support"),
                           ("dimer-interface residues", "Interface local support"),
                           ("entrance lid 541-548", "Entrance-lid local support")):
            vals = {p: g.loc[(p, group), "Group support"] for p, gg in g.index if gg == group}
            s[col] = s["PDB ID"].map(vals)
            s[col.replace("support", "support score")] = s[col].map(lambda c: support_score(c, fcfg) if isinstance(c, str) else None)
        lows = df_local.dropna(subset=["Lowest-supported residue"]) if "Lowest-supported residue" in df_local else pd.DataFrame()
        if len(lows):
            lows = lows.assign(_v=pd.to_numeric(lows.get("Min RSCC"), errors="coerce").fillna(
                pd.to_numeric(lows.get("Min Q-score"), errors="coerce")))
            worst = lows.loc[lows.groupby("PDB ID")["_v"].idxmin()].set_index("PDB ID")
            s["Lowest-supported important residue"] = s["PDB ID"].map(
                worst["Group"] + ": " + worst["Lowest-supported residue"])
        avail = df_local.groupby("PDB ID")["Calculation status"].apply(
            lambda x: "available" if (x == Status.OK.value).any() or (x == Status.WARNING.value).any() else "unavailable")
        s["Local-fit availability"] = s["PDB ID"].map(avail)
        weakish = df_local[df_local.get("Group support", pd.Series(dtype=str)).isin(["weak local support"])]
        s["Local-fit warning"] = s["PDB ID"].map(weakish.groupby("PDB ID")["Group"].apply(
            lambda x: "weak support: " + ", ".join(sorted(set(x)))))
    # ligand worst support
    if len(df_lig):
        lig = df_lig[df_lig["Class"] == "ligand of interest"]
        if len(lig):
            s["Ligand local support (worst ligand)"] = s["PDB ID"].map(lig.groupby("PDB ID")["Local support"].apply(worst_category))
    # helpers for ranking modes
    ch = tables["Chains"]
    subs = ch.groupby("PDB ID")["Substitutions vs UniProt"].apply(
        lambda x: ", ".join(sorted({y for v in x.dropna() for y in str(v).split(", ") if y})))
    s["Substitutions vs UniProt (all chains)"] = s["PDB ID"].map(subs).fillna("")
    s["N substitutions vs UniProt"] = s["Substitutions vs UniProt (all chains)"].map(lambda x: len([y for y in x.split(", ") if y]))
    s["Has other polymer chains"] = s["Other polymer entities"].fillna("").astype(str).str.strip().ne("")
    gl = tables.get("Glycosylation", pd.DataFrame())
    s["Glycosylated Asn count"] = s["PDB ID"].map(gl.groupby("PDB ID")["Asn (UniProt)"].nunique()).fillna(0).astype(int) if len(gl) else 0
    exp = int(cfg.active_site.get("metal_site", {}).get("expected_zn_per_protomer", 2))
    if "Zn assembly" in s:
        s["Zn site complete"] = (pd.to_numeric(s["Zn assembly"], errors="coerce")
                                 == exp * pd.to_numeric(s["Target copies in assembly"], errors="coerce"))
    for p, e in extra.items():
        for k, v in e.items():
            s.loc[s["PDB ID"] == p, k] = v
    for c in ("Interface complete", "Zn site complete", "Has other polymer chains", "Noncanonical interface flag"):
        if c in s:
            s[c] = s[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v))
    return s


def rank_all_modes(cfg: Config, asum: pd.DataFrame, tables: dict, issues: IssueLog,
                   top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every configured ranking mode on the assessment summary (``RankingMode`` column)."""
    crit = criteria_table(asum, tables["Chains"], tables.get("ActiveSite", pd.DataFrame()))
    extra = asum.set_index("PDB ID").copy()
    for c in extra.columns:
        if extra[c].dtype == bool or set(extra[c].dropna().unique()) <= {True, False}:
            extra[c] = extra[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else float(bool(v)))
    outs, tops = [], []
    for name, mode in (cfg.get("ranking.modes") or {}).items():
        m = dict(mode)
        for key in ("include", "exclude"):
            m[key] = {c: (float(v) if isinstance(v, bool) else v) for c, v in (mode.get(key) or {}).items()}
        try:
            r = rank_templates(crit, m, name, extra=extra)
        except KeyError as ex:
            issues.add("ranking", Status.FAIL, f"mode {name}: {ex}")
            continue
        missing = [c["column"] for c in mode.get("criteria", []) if c["column"] not in crit and c["column"] not in extra]
        if missing:
            issues.add("ranking", Status.WARNING, f"mode {name}: criteria not available: {', '.join(missing)}")
        r.insert(0, "RankingMode", name)
        r["Mode description"] = mode.get("description", "")
        outs.append(r)
        t = r.head(top_n)[["RankingMode", "PDB ID", "Rank", "Score (0-1, higher = better)"]].copy()
        t["Entries eligible"] = len(r)
        tops.append(t)
    return (pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(),
            pd.concat(tops, ignore_index=True) if tops else pd.DataFrame())


def save_assessment(cfg: Config, res: AssessmentResults, tag: str = "") -> dict[str, Path]:
    """Write the assessment workbook, CSV/Parquet tables and manifest."""
    from .excel_output import write_tables, write_workbook
    from .manifest import build_manifest, detect_software, manifest_table, write_manifest

    software = detect_software()
    sheets = dict(res.tables)
    sheets["AssessmentIssues"] = res.issues.to_frame()
    sheets["Software"] = software
    manifest = build_manifest(cfg, structure_ids=res.ids, input_files=res.input_files, software=software,
                              extra={"stage": "modelling assessment (notebook 02)", "canonical_interface": res.canonical,
                                     "issue_counts": {s.value: res.issues.count(s) for s in Status}})
    sheets["Reproducibility"] = manifest_table(manifest)
    suffix = f"_{tag}" if tag else ""
    wb = cfg.output_file("assessment_workbook")
    wb = wb.with_name(wb.stem + suffix + ".xlsx")
    write_workbook(wb, sheets)
    mpath = write_manifest(manifest, cfg.outputs_dir / f"{cfg.accession}_assessment_manifest{suffix}.json")
    tdir = cfg.outputs_dir / f"assessment_tables{suffix}"
    write_tables(tdir, sheets, csv=bool(cfg.get("outputs.write_csv", True)),
                 parquet=bool(cfg.get("outputs.write_parquet", True)))
    np.savetxt(tdir / "interface_jaccard_matrix.csv", res.jaccard_matrix, delimiter=",",
               header=",".join(res.matrix_ids), comments="")
    return {"workbook": wb, "manifest": mpath, "tables_dir": tdir}
