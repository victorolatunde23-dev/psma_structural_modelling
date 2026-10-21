"""Transparent, configuration-driven template rankings.

A ranking *mode* (``ranking.modes.<name>`` in the config) declares:

* ``include`` / ``exclude`` filters  – ``{column: value | [values] | {min: x, max: y}}``;
* ``criteria`` – list of ``{column, direction: lower|higher, weight}``;
* ``missing`` – ``ignore`` (average over the criteria an entry has) or
  ``worst`` (a missing value counts as the worst rank).

For each criterion, entries are ranked and the rank is scaled to 0 (best)
… 1 (worst). The score is ``1 − weighted mean of the scaled ranks``. Raw
criteria, scaled components and a plain-text explanation are kept in the
output so every position can be audited.

The ``global`` mode in ``default_config.yaml`` reproduces the original
notebook's equal-weight ranking exactly. Milestone 2 adds the other modes
(active site, ligand-specific, apo, open/closed lid, dimer interface, MD
start, glycosylated, cryo-EM reference).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def criteria_table(df_summary: pd.DataFrame, df_chains: pd.DataFrame,
                   df_active: pd.DataFrame) -> pd.DataFrame:
    """Raw per-entry ranking criteria (index = PDB ID).

    Columns: ``Method``, ``Resolution (Å)``, ``Clashscore``,
    ``Ramachandran outliers (%)``, ``Best UniProt coverage``,
    ``Fewest internal missing``, ``Active-site completeness``, plus any other
    Summary column a mode refers to (looked up at ranking time).
    """
    s = df_summary.set_index("PDB ID")
    crit = pd.DataFrame(index=s.index)
    crit["Method"] = s.get("Method")
    crit["Resolution (Å)"] = s.get("Resolution (Å)")
    crit["Clashscore"] = s.get("Clashscore")
    crit["Ramachandran outliers (%)"] = s.get("Ramachandran outliers (%)")
    if len(df_chains):
        g = df_chains.groupby("PDB ID")
        crit["Best UniProt coverage"] = g["UniProt coverage"].max()
        if "N missing internal" in df_chains:
            crit["Fewest internal missing"] = g["N missing internal"].min()
    if len(df_active) and "Modelled" in df_active:
        crit["Active-site completeness"] = (df_active.groupby(["PDB ID", "Chain"])["Modelled"].mean()
                                            .groupby("PDB ID").max())
    return crit


def _apply_filter(df: pd.DataFrame, spec: dict, keep: bool) -> pd.DataFrame:
    if not spec:
        return df
    mask = pd.Series(True, index=df.index)
    for col, cond in spec.items():
        if col not in df:
            raise KeyError(f"ranking filter refers to unknown column {col!r}")
        v = df[col]
        if isinstance(cond, dict):
            m = pd.Series(True, index=df.index)
            num = pd.to_numeric(v, errors="coerce")
            if "min" in cond:
                m &= num >= cond["min"]
            if "max" in cond:
                m &= num <= cond["max"]
            if "contains" in cond:
                m &= v.astype(str).str.contains(str(cond["contains"]), case=False, na=False)
        elif isinstance(cond, list):
            m = v.isin(cond)
        else:
            m = v == cond
        mask &= m.fillna(False)
    return df[mask] if keep else df[~mask]


def rank_templates(crit: pd.DataFrame, mode: dict, mode_name: str = "global",
                   extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank entries for one mode.

    Parameters
    ----------
    crit:
        Output of :func:`criteria_table` (index = PDB ID).
    mode:
        Mode definition from the configuration.
    extra:
        Optional table (index = PDB ID) with further columns referenced by
        the mode's criteria or filters (e.g. Summary columns).

    Returns
    -------
    DataFrame with ``Rank``, ``PDB ID``, the raw criteria, ``Score (0-1,
    higher = better)``, then ``Ranking mode``, one ``component: <criterion>``
    column per criterion (0 = best, 1 = worst), ``Criteria used`` and
    ``Score explanation``.
    """
    df = crit.copy()
    referenced = ({c["column"] for c in mode.get("criteria") or []}
                  | set(mode.get("include") or {}) | set(mode.get("exclude") or {}))
    if extra is not None and len(extra):
        for col in sorted(referenced - set(df.columns)):
            if col in extra:
                df[col] = extra[col].reindex(df.index)
    df = _apply_filter(df, mode.get("include") or {}, keep=True)
    df = _apply_filter(df, mode.get("exclude") or {}, keep=False)
    missing = mode.get("missing", "ignore")

    comps: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for c in mode.get("criteria") or []:
        col, w = c["column"], float(c.get("weight", 1))
        if col not in df or not df[col].notna().any() or w == 0:
            continue
        r = pd.to_numeric(df[col], errors="coerce").rank(ascending=c["direction"] == "lower")
        n = r.notna().sum()
        comp = (r - 1) / (n - 1) if n > 1 else r * 0          # 0 = best, 1 = worst
        if missing == "worst":
            comp = comp.fillna(1.0)
        comps[col], weights[col] = comp, w

    score_col = "Score (0-1, higher = better)"
    if comps:
        C = pd.DataFrame(comps)
        W = pd.DataFrame({k: np.where(C[k].notna(), w, 0.0) for k, w in weights.items()}, index=C.index)
        wsum = W.sum(axis=1)
        df[score_col] = (1 - (C.fillna(0) * W).sum(axis=1) / wsum.replace(0, np.nan)).round(3)
    else:
        df[score_col] = None
    df = df.sort_values(score_col, ascending=False, na_position="last")
    df.insert(0, "Rank", range(1, len(df) + 1))
    df.index.name = "PDB ID"
    out = df.reset_index()

    raw_cols = [c for c in crit.columns]
    out = out[["PDB ID", "Rank"] + raw_cols + [score_col]
              + [c for c in out.columns if c not in raw_cols + ["Rank", "PDB ID", score_col]]]
    out["Ranking mode"] = mode_name
    for col, comp in comps.items():
        out[f"component: {col}"] = out["PDB ID"].map(comp.round(3))
    out["Criteria used"] = out["PDB ID"].map(
        lambda p: ", ".join(f"{k} (w={weights[k]:g})" for k in comps if pd.notna(comps[k].get(p))))

    def explain(row: Any) -> str:
        parts = []
        for col in comps:
            v = row.get(f"component: {col}")
            if pd.notna(v):
                parts.append(f"{col}={row.get(col)} → {v:.2f}")
            else:
                parts.append(f"{col}: missing ({missing})")
        return "; ".join(parts)

    out["Score explanation"] = out.apply(explain, axis=1)
    return out


def template_ranking(df_summary: pd.DataFrame, df_chains: pd.DataFrame,
                     df_active: pd.DataFrame, mode: dict | None = None,
                     mode_name: str = "global") -> pd.DataFrame:
    """Original single ranking (``global`` mode); kept for backwards compatibility."""
    if mode is None:
        mode = {"criteria": [
            {"column": "Resolution (Å)", "direction": "lower", "weight": 1},
            {"column": "Clashscore", "direction": "lower", "weight": 1},
            {"column": "Ramachandran outliers (%)", "direction": "lower", "weight": 1},
            {"column": "Fewest internal missing", "direction": "lower", "weight": 1},
            {"column": "Best UniProt coverage", "direction": "higher", "weight": 1},
            {"column": "Active-site completeness", "direction": "higher", "weight": 1},
        ], "missing": "ignore"}
    crit = criteria_table(df_summary, df_chains, df_active)
    return rank_templates(crit, mode, mode_name, extra=df_summary.set_index("PDB ID"))
