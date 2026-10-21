"""wwPDB validation metrics from RCSB (entry summaries) and PDBe (percentiles,
residue-level outliers).

Milestone 2 adds parsing of the full wwPDB validation XML (RSCC, RSRZ,
Q-score, ligand fit) in this module.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from .cache import cached_json

log = logging.getLogger("psma_pipeline.validation")

# The RCSB schema for wwPDB validation summaries has changed over time, so two
# query variants are tried in order.
RCSB_VALIDATION_QUERIES = {
    "split": """
query($ids: [String!]!) { entries(entry_ids: $ids) { rcsb_id
  pdbx_vrpt_summary_geometry { clashscore percent_ramachandran_outliers percent_rotamer_outliers }
  pdbx_vrpt_summary_diffraction { percent_RSRZ_outliers DCC_Rfree }
} }""",
    "legacy": """
query($ids: [String!]!) { entries(entry_ids: $ids) { rcsb_id
  pdbx_vrpt_summary { clashscore percent_ramachandran_outliers percent_rotamer_outliers percent_RSRZ_outliers DCC_Rfree }
} }""",
}

OutlierMap = dict[tuple[str, int, str], set[str]]


def _first(x: Any) -> dict:
    if isinstance(x, list):
        return x[0] if x else {}
    return x or {}


def rcsb_validation(sess: requests.Session, ids: list[str], cache: Path, url: str,
                    batch: int = 50, **http: Any) -> tuple[dict[str, dict], str]:
    """Entry-level validation summaries from RCSB GraphQL.

    Returns
    -------
    (values, variant)
        ``values``: ``{PDB ID: {clashscore, rama_outliers_pct,
        rotamer_outliers_pct, rsrz_outliers_pct, dcc_rfree}}``.
        ``variant``: the schema variant that worked, or ``"none"`` (in which
        case ``values`` is empty and PDBe values are used instead).
    """
    for variant, query in RCSB_VALIDATION_QUERIES.items():
        out: dict[str, dict] = {}
        try:
            for i in range(0, len(ids), batch):
                chunk = ids[i:i + batch]
                cf = cache / "validation_rcsb" / f"{variant}_{chunk[0]}_{chunk[-1]}_{len(chunk)}.json"
                data = cached_json(sess, url, cf, method="POST",
                                   payload={"query": query, "variables": {"ids": chunk}}, **http)
                if data.get("errors"):
                    cf.unlink(missing_ok=True)
                    raise RuntimeError(data["errors"][:1])
                for e in data["data"]["entries"] or []:
                    merged: dict = {}
                    for k, v in e.items():
                        if k.startswith("pdbx_vrpt_summary"):
                            merged.update({kk: vv for kk, vv in _first(v).items() if vv is not None})
                    out[e["rcsb_id"].upper()] = {
                        "clashscore": merged.get("clashscore"),
                        "rama_outliers_pct": merged.get("percent_ramachandran_outliers"),
                        "rotamer_outliers_pct": merged.get("percent_rotamer_outliers"),
                        "rsrz_outliers_pct": merged.get("percent_RSRZ_outliers"),
                        "dcc_rfree": merged.get("DCC_Rfree"),
                    }
            log.info("RCSB validation: schema variant '%s'", variant)
            return out, variant
        except Exception as ex:  # noqa: BLE001 - recorded and the next variant tried
            log.warning("RCSB validation variant '%s' failed (%s); trying next", variant, ex)
    return {}, "none"


def metric_key(name: str) -> str | None:
    """Normalise a PDBe percentile metric name to the keys used in tables."""
    n = "".join(ch for ch in name.lower() if ch.isalnum())
    if "clash" in n:
        return "clashscore"
    if "rama" in n:
        return "rama_outliers_pct"
    if "rota" in n or "sidechain" in n:
        return "rotamer_outliers_pct"
    if "rsrz" in n:
        return "rsrz_outliers_pct"
    if "rfree" in n:
        return "rfree"
    return None


def pdbe_validation(sess: requests.Session, pdb: str, cache: Path, url_percentiles: str,
                    url_outliers: str, model_number: int = 1,
                    **http: Any) -> tuple[dict[str, Any], OutlierMap]:
    """Percentiles and residue-level outliers from the PDBe validation API.

    Returns
    -------
    (percentiles, outliers)
        ``outliers`` maps ``(author chain, author residue number, insertion
        code)`` to a set of outlier types, for model ``model_number``.
    """
    pct: dict[str, Any] = {}
    data = cached_json(sess, url_percentiles.format(pdb=pdb.lower()),
                       cache / "validation_pdbe" / f"{pdb}_pct.json", allow_404=True, **http)
    for metric, v in (data.get(pdb.lower()) or {}).items():
        key = metric_key(metric)
        v = _first(v)
        if key and isinstance(v, dict):
            pct[f"PDBe {key}"] = v.get("rawvalue")
            pct[f"PDBe {key} percentile (vs similar resolution)"] = v.get("relative")
            pct[f"PDBe {key} percentile (whole PDB)"] = v.get("absolute")

    outliers: OutlierMap = defaultdict(set)
    data = cached_json(sess, url_outliers.format(pdb=pdb.lower()),
                       cache / "validation_pdbe" / f"{pdb}_outliers.json", allow_404=True, **http)
    for mol in (data.get(pdb.lower()) or {}).get("molecules", []):
        for ch in mol.get("chains", []):
            for mdl in ch.get("models", []):
                if mdl.get("model_id", 1) != model_number:
                    continue
                for r in mdl.get("residues", []):
                    key = (ch.get("chain_id"), r.get("author_residue_number"),
                           (r.get("author_insertion_code") or "").strip())
                    outliers[key].update(r.get("outlier_types") or [])
    return pct, outliers
