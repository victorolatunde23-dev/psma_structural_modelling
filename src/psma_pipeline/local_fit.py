"""Local experimental support (map–model fit) for residues, ligands and ions.

Method-specific rules (``local_fit`` in the config)
    * **X-ray** – per-residue RSCC and RSRZ from the wwPDB validation report:
      strong (RSCC ≥ 0.90 and RSRZ ≤ 2), acceptable (RSCC ≥ 0.80 and RSRZ ≤ 3;
      a missing RSRZ, as for most ligands, is not held against the residue),
      otherwise weak.
    * **Cryo-EM** – per-residue Q-score compared with the Q-score expected at
      the map resolution, Q_exp = −0.1775·d + 1.1192 (Pintilie et al. 2020):
      strong (Q ≥ Q_exp), acceptable (Q ≥ Q_exp − 0.15), otherwise weak.
    * Missing metrics → ``unsupported or unavailable``.

    RSCC and Q-score are never compared directly; only the categories, each
    derived with its own rule, are combined.

The ligand/environment B-factor ratio is kept elsewhere as an *indirect*
indicator; it is not a substitute for these map-fit metrics.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .status import Status

STRONG, ACCEPTABLE, WEAK, NONE = ("strong local support", "acceptable local support",
                                  "weak local support", "unsupported or unavailable")
ORDER = [NONE, WEAK, ACCEPTABLE, STRONG]


def method_kind(method: str | None) -> str:
    """``xray`` / ``em`` / ``other`` from an RCSB method string."""
    m = str(method or "").upper()
    if "X-RAY" in m:
        return "xray"
    if "ELECTRON MICROSCOPY" in m:
        return "em"
    return "other"


def expected_q(resolution: float | None, cfg: dict) -> float | None:
    """Q-score expected at a given map resolution (Å)."""
    if resolution is None or pd.isna(resolution):
        return None
    em = cfg["em"]
    return float(em["expected_q_slope"]) * float(resolution) + float(em["expected_q_intercept"])


def categorise(kind: str, rscc: float | None = None, rsrz: float | None = None, q: float | None = None,
               resolution: float | None = None, cfg: dict | None = None) -> str:
    """Support category for one residue/ligand (or a group statistic)."""
    cfg = cfg or {}
    if kind == "xray":
        if rscc is None or pd.isna(rscc):
            return NONE
        x = cfg["xray"]
        ok = lambda lim: rsrz is None or pd.isna(rsrz) or rsrz <= lim  # noqa: E731
        if rscc >= x["strong"]["rscc_min"] and ok(x["strong"]["rsrz_max"]):
            return STRONG
        if rscc >= x["acceptable"]["rscc_min"] and ok(x["acceptable"]["rsrz_max"]):
            return ACCEPTABLE
        return WEAK
    if kind == "em":
        qe = expected_q(resolution, cfg)
        if q is None or pd.isna(q) or qe is None:
            return NONE
        if q >= qe + float(cfg["em"]["strong_delta"]):
            return STRONG
        if q >= qe + float(cfg["em"]["acceptable_delta"]):
            return ACCEPTABLE
        return WEAK
    return NONE


def support_score(category: str, cfg: dict) -> int:
    """Numeric score of a category for ranking (config ``local_fit.scores``)."""
    return int(cfg.get("scores", {}).get(category, ORDER.index(category) if category in ORDER else 0))


def residue_fit_table(xml_residues: pd.DataFrame, kind: str, resolution: float | None, cfg: dict,
                      model: int = 1) -> pd.DataFrame:
    """Add ``Support`` (category) to every validation-report residue of one model."""
    if xml_residues is None or xml_residues.empty:
        return pd.DataFrame()
    r = xml_residues[xml_residues["model"] == model].copy()
    r["Support"] = [categorise(kind, a, b, c, resolution, cfg)
                    for a, b, c in zip(r.get("rscc"), r.get("rsrz"), r.get("Q_score"))]
    return r


def group_fit(pdb: str, group: str, members: Iterable[tuple[str, int, str]], fit: pd.DataFrame,
              kind: str, resolution: float | None, cfg: dict, labels: dict | None = None) -> dict:
    """Summarise local support for one residue group of one entry.

    Parameters
    ----------
    members:
        ``(author chain, author residue number, insertion code)`` triples.
    labels:
        Optional ``{member: display label}`` (e.g. ``H377/A``).
    """
    members = sorted(set(members))
    base = {"PDB ID": pdb, "Group": group, "Method kind": kind, "Residues in group": len(members)}
    if not members:
        return {**base, "Calculation status": Status.NOT_APPLICABLE.value, "Warning": "group empty"}
    if fit is None or fit.empty:
        return {**base, "Group support": NONE, "Calculation status": Status.NOT_RUN.value,
                "Warning": "no validation-report residues"}
    idx = fit.set_index(["chain", "resnum", "icode"])
    idx = idx[~idx.index.duplicated()]
    sub = idx.reindex(members)
    metric = "Q_score" if kind == "em" else "rscc"
    vals = pd.to_numeric(sub[metric], errors="coerce")
    have = vals.notna()
    stat = cfg.get("group_summary", "median")
    rec = {**base, "Residues with metric": int(have.sum()),
           "Metric": "Q-score" if kind == "em" else "RSCC"}
    if not have.any():
        return {**rec, "Group support": NONE, "Calculation status": Status.WARNING.value,
                "Warning": "no local-fit metric for these residues"}
    agg = vals[have].median() if stat == "median" else vals[have].min()
    rsrz = pd.to_numeric(sub.get("rsrz"), errors="coerce")
    rsrz_agg = None
    if kind == "xray" and rsrz is not None and rsrz.notna().any():
        rsrz_agg = rsrz.median() if stat == "median" else rsrz.max()
    cat = categorise(kind, agg if kind == "xray" else None, rsrz_agg,
                     agg if kind == "em" else None, resolution, cfg)
    worst_key = vals[have].idxmin()
    worst_label = (labels or {}).get(worst_key, f"{worst_key[0]}:{sub.at[worst_key, 'resname']}{worst_key[1]}{worst_key[2]}")
    counts = sub["Support"].value_counts() if "Support" in sub else pd.Series(dtype=int)
    missing_members = int((~have).sum())
    return {**rec, f"{stat.capitalize()} {rec['Metric']}": round(float(agg), 3),
            f"Min {rec['Metric']}": round(float(vals[have].min()), 3),
            "Median RSRZ": round(float(rsrz.median()), 2) if kind == "xray" and rsrz is not None and rsrz.notna().any() else None,
            "Lowest-supported residue": f"{worst_label} ({vals[worst_key]:.2f})",
            "Residues strong": int(counts.get(STRONG, 0)), "Residues acceptable": int(counts.get(ACCEPTABLE, 0)),
            "Residues weak": int(counts.get(WEAK, 0)),
            "Group support": cat, "Group support score": support_score(cat, cfg),
            "Calculation status": (Status.WARNING if missing_members else Status.OK).value,
            "Warning": f"{missing_members} residues without a metric (unmodelled or not assessed)" if missing_members else ""}


def ligand_fit_rows(pdb: str, fit: pd.DataFrame, kind: str, resolution: float | None, cfg: dict,
                    classify: Any, binding: pd.DataFrame | None = None) -> list[dict]:
    """One row per ligand / ion instance in the validation report (waters excluded)."""
    if fit is None or fit.empty:
        return []
    het = fit[fit["label_seq"].isna() & (fit["resname"] != "HOH")]
    br = {}
    if binding is not None and len(binding):
        b = binding[binding["PDB ID"] == pdb]
        br = {(r["Chain"], str(r["Residue no."])): r.get("B ratio ligand/environment") for _, r in b.iterrows()}
    rows = []
    for _, r in het.iterrows():
        cls = classify(r["resname"])
        cat = r["Support"]
        geo = int(sum(r.get(k, 0) or 0 for k in ("n_bond_outliers", "n_angle_outliers", "n_chiral_outliers",
                                                  "n_mogul_bond_outliers", "n_mogul_angle_outliers")))
        warn = []
        if cat == NONE:
            warn.append("no experimental support metric: pose cannot be called well supported")
        if (r.get("n_chiral_outliers") or 0) > 0:
            warn.append("chirality outlier")
        rows.append({
            "PDB ID": pdb, "Model": int(r["model"]), "Chain": r["chain"], "Chain (label)": r["label_asym"],
            "Component": r["resname"], "Class": cls, "Residue no. (author)": f"{r['resnum']}{r['icode']}",
            "Alt. conformation": r["altcode"], "RSCC": r.get("rscc"), "RSRZ": r.get("rsrz"),
            "Q-score": r.get("Q_score"), "Occupancy (avg)": r.get("avgoccu"),
            "Clashes": int(r.get("n_clashes", 0) or 0), "Geometry outliers": geo,
            "Chirality outliers": int(r.get("n_chiral_outliers", 0) or 0),
            "B ratio ligand/environment (indirect)": br.get((r["chain"], f"{r['resnum']}{r['icode']}")),
            "Local support": cat, "Local support score": support_score(cat, cfg),
            "Calculation status": (Status.WARNING if warn else Status.OK).value, "Warning": "; ".join(warn),
        })
    return rows


EM_ENTRY_FIELDS = ["emdb_id", "EMDB-resolution", "Q_score", "atom_inclusion_backbone", "atom_inclusion_all_atoms",
                   "contour_level_primary_map", "calculated_fsc_resolution_by_cutoff_0.143",
                   "calculated_fsc_resolution_by_cutoff_halfbit", "author_provided_fsc_resolution_by_cutoff_0.143",
                   "relative-percentile-qrelative", "absolute-percentile-qrelative"]


def em_entry_row(pdb: str, entry: dict, fit: pd.DataFrame, schema: str) -> dict:
    """EMMapFit row: entry-level EM metrics plus per-chain mean Q-score."""
    row = {"PDB ID": pdb, "Schema": schema}
    for k in EM_ENTRY_FIELDS:
        v = entry.get(k)
        row[k] = _maybe_num(v)
    if fit is not None and len(fit) and "Q_score" in fit:
        prot = fit[fit["label_seq"].notna()]
        row["Per-chain mean Q-score"] = "; ".join(
            f"{c}: {g['Q_score'].mean():.3f}" for c, g in prot.groupby("chain") if g["Q_score"].notna().any())
    missing = [k for k in ("emdb_id", "Q_score", "EMDB-resolution") if row.get(k) in (None, "")]
    row["Calculation status"] = (Status.WARNING if missing else Status.OK).value
    row["Warning"] = f"missing in report: {', '.join(missing)}" if missing else ""
    return row


def _maybe_num(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def worst_category(cats: Iterable[str]) -> str:
    cats = [c for c in cats if c in ORDER]
    return min(cats, key=ORDER.index) if cats else NONE


__all__ = ["STRONG", "ACCEPTABLE", "WEAK", "NONE", "method_kind", "expected_q", "categorise", "support_score",
           "residue_fit_table", "group_fit", "ligand_fit_rows", "em_entry_row", "worst_category"]
