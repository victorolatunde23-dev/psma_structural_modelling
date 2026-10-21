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

import pandas as pd
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


# ---------------------------------------------------------------------------
# wwPDB validation XML (Milestone 2)
# ---------------------------------------------------------------------------
RESIDUE_NUMERIC = ["rscc", "rsrz", "rsr", "owab", "avgoccu", "Q_score", "residue_inclusion", "NatomsEDS",
                   "EDIAm", "OPIA"]
CHILD_COUNTS = {"clash": "n_clashes", "bond-outlier": "n_bond_outliers", "angle-outlier": "n_angle_outliers",
                "chiral-outlier": "n_chiral_outliers", "plane-outlier": "n_plane_outliers",
                "mog-bond-outlier": "n_mogul_bond_outliers", "mog-angle-outlier": "n_mogul_angle_outliers",
                "mog-torsion-outlier": "n_mogul_torsion_outliers", "mog-ring-outlier": "n_mogul_ring_outliers",
                "symm-clash": "n_symm_clashes"}


def validation_xml_url(pdb: str, template: str) -> str:
    """Fill ``{pdb}`` (lower case) and ``{mid}`` (characters 2-3) in a URL template."""
    p = pdb.lower()
    return template.format(pdb=p, mid=p[1:3])


def fetch_validation_xml(sess: requests.Session, pdb: str, cache: Path, url_template: str,
                         fallback_template: str | None = None, **http: Any) -> Path:
    """Download (or reuse) the wwPDB validation XML; tries the fallback URL on failure.

    Raises
    ------
    FetchError
        If neither source provides the file.
    """
    from .cache import FetchError, cached_download
    folder = cache / "validation_xml"
    errors = []
    for tmpl in [url_template] + ([fallback_template] if fallback_template else []):
        url = validation_xml_url(pdb, tmpl)
        path = folder / (f"{pdb.upper()}_validation.xml" + (".gz" if url.endswith(".gz") else ""))
        try:
            return cached_download(sess, url, path, **http)
        except FetchError as ex:
            errors.append(str(ex))
    raise FetchError("; ".join(errors))


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_validation_xml(path: Path) -> dict:
    """Parse a wwPDB validation report (any schema version).

    Returns
    -------
    dict
        ``schema`` (schema file name from the root element, or "unknown"),
        ``entry`` (all Entry attributes as strings), ``residues`` (DataFrame,
        one row per ModelledSubgroup, with numeric metrics where present and
        counts of outlier child elements), ``status`` and ``note``.
        Attributes absent from a given schema version are simply missing.
    """
    import gzip
    import xml.etree.ElementTree as ET

    opener = gzip.open if str(path).endswith(".gz") else open
    rows: list[dict] = []
    entry: dict[str, str] = {}
    schema = "unknown"
    with opener(path, "rb") as fh:
        for event, el in ET.iterparse(fh, events=("start", "end")):
            tag = el.tag.split("}")[-1]
            if event == "start" and tag == "wwPDB-validation-information":
                for k, v in el.attrib.items():
                    if k.endswith("noNamespaceSchemaLocation"):
                        schema = v.rsplit("/", 1)[-1]
            elif event == "end" and tag == "Entry":
                entry = dict(el.attrib)
                el.clear()
            elif event == "end" and tag == "ModelledSubgroup":
                a = el.attrib
                row: dict[str, Any] = {
                    "model": int(a.get("model", 1) or 1), "chain": a.get("chain"),
                    "resnum": int(a["resnum"]) if str(a.get("resnum", "")).lstrip("-").isdigit() else None,
                    "icode": (a.get("icode") or "").strip(), "resname": a.get("resname"),
                    "label_asym": a.get("said"), "entity": a.get("ent"),
                    "label_seq": int(a["seq"]) if str(a.get("seq", "")).isdigit() else None,
                    "altcode": (a.get("altcode") or "").strip(),
                    "rama": a.get("rama"), "rota": a.get("rota"),
                }
                for k in RESIDUE_NUMERIC:
                    row[k] = _num(a.get(k))
                for k, v in a.items():
                    if "rmsz" in k.lower() or "mogul" in k.lower():
                        row[k] = _num(v)
                counts = dict.fromkeys(CHILD_COUNTS.values(), 0)
                for child in el:
                    ct = child.tag.split("}")[-1]
                    if ct in CHILD_COUNTS:
                        counts[CHILD_COUNTS[ct]] += 1
                row.update(counts)
                rows.append(row)
                el.clear()
    df = pd.DataFrame(rows)
    note = "" if len(df) else "no ModelledSubgroup elements"
    return {"schema": schema, "entry": entry, "residues": df,
            "status": "OK" if len(df) else "WARNING", "note": note}
