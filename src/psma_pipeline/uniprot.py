"""UniProt: canonical sequence, PDB cross-references and sequence features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from .cache import cached_json


def uniprot_entry(sess: requests.Session, acc: str, cache: Path, url_template: str,
                  **http: Any) -> dict:
    """Raw UniProtKB JSON entry (cached as ``uniprot_<ACC>.json``)."""
    return cached_json(sess, url_template.format(acc=acc), cache / f"uniprot_{acc}.json", **http)


def uniprot_info(entry: dict) -> tuple[str, str, dict[str, dict]]:
    """Return ``(canonical sequence, recommended name, {PDB ID: xref properties})``.

    Raises
    ------
    KeyError
        If the entry has no sequence (a malformed or obsolete accession).
    """
    seq = entry["sequence"]["value"]
    name = (entry.get("proteinDescription", {}).get("recommendedName", {})
            .get("fullName", {}).get("value", ""))
    xrefs: dict[str, dict] = {}
    for x in entry.get("uniProtKBCrossReferences", []):
        if x.get("database") == "PDB":
            props = {p["key"]: p["value"] for p in x.get("properties", [])}
            xrefs[x["id"].upper()] = props
    return seq, name, xrefs


def site_features(entry: dict, feature_types: list[str], max_span: int) -> list[dict]:
    """Short site-like features (UniProt numbering) of the requested types.

    Returns one dict per feature: ``type``, ``start``, ``end``, ``label``.
    Features longer than ``max_span`` residues are regions, not sites, and
    are skipped.
    """
    out = []
    for feat in entry.get("features", []):
        ftype = feat.get("type")
        if ftype not in feature_types:
            continue
        loc = feat.get("location", {})
        s, e = loc.get("start", {}).get("value"), loc.get("end", {}).get("value")
        if s is None or e is None or e - s > max_span - 1:
            continue
        lig = (feat.get("ligand") or {}).get("name") or feat.get("description") or ""
        out.append({"type": ftype, "start": s, "end": e, "label": lig})
    return out
