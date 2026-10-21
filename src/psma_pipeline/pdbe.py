"""PDBe: SIFTS residue mappings and the ``best_structures`` entry list.

SIFTS numbering contract
    ``start.residue_number`` / ``end.residue_number`` in the PDBe mappings
    API are **label_seq_id** values (mmCIF entity numbering, starting at 1),
    and ``struct_asym_id`` is the **label_asym_id**. ``chain_id`` is the
    **author** chain. ``unp_start`` / ``unp_end`` are UniProt numbers.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from .cache import cached_json


def fetch_sifts(sess: requests.Session, pdb: str, cache: Path, url_template: str,
                **http: Any) -> dict:
    """SIFTS UniProt mapping JSON for one entry (``{}`` if PDBe has none)."""
    return cached_json(sess, url_template.format(pdb=pdb.lower()),
                       cache / "sifts" / f"{pdb}.json", allow_404=True, **http)


def fetch_best_structures(sess: requests.Session, acc: str, cache: Path, url_template: str,
                          **http: Any) -> list[dict]:
    """PDBe ``best_structures`` list for a UniProt accession (``[]`` if none)."""
    data = cached_json(sess, url_template.format(acc=acc), cache / f"pdbe_best_{acc}.json",
                       allow_404=True, **http)
    return data.get(acc, [])


def sifts_segments(sifts: dict, pdb: str, acc: str) -> dict[str, list[dict]]:
    """``label_asym_id -> [segment, ...]`` sorted by ``seq_start``.

    Each segment: ``entity_id``, ``chain_id`` (author), ``seq_start``/``seq_end``
    (label_seq_id) and ``unp_start``/``unp_end`` (UniProt).
    """
    block = sifts.get(pdb.lower(), {}).get("UniProt", {}).get(acc, {})
    by_asym: dict[str, list[dict]] = defaultdict(list)
    for m in block.get("mappings", []):
        by_asym[m["struct_asym_id"]].append({
            "entity_id": str(m["entity_id"]),
            "chain_id": m["chain_id"],
            "seq_start": m["start"]["residue_number"],   # label_seq_id
            "seq_end": m["end"]["residue_number"],
            "unp_start": m["unp_start"],
            "unp_end": m["unp_end"],
        })
    for v in by_asym.values():
        v.sort(key=lambda s: s["seq_start"])
    return dict(by_asym)
