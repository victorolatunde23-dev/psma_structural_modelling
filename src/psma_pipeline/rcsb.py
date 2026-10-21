"""RCSB PDB: entry search by UniProt accession, GraphQL metadata, mmCIF download.

Cache file names are identical to the original notebook, so an existing
``pdb_cache`` folder can be reused by pointing ``paths.cache`` at it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from .cache import FetchError, cached_download, cached_json

log = logging.getLogger("psma_pipeline.rcsb")

GRAPHQL_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    exptl { method }
    rcsb_accession_info { initial_release_date }
    rcsb_entry_info { resolution_combined deposited_polymer_monomer_count }
    refine { ls_R_factor_R_free }
    symmetry { space_group_name_H_M }
    assemblies {
      rcsb_id
      pdbx_struct_assembly { id details method_details oligomeric_details oligomeric_count }
      rcsb_struct_symmetry { kind type oligomeric_state stoichiometry symbol }
    }
    polymer_entities {
      rcsb_id
      entity_poly { type pdbx_seq_one_letter_code_can rcsb_sample_sequence_length }
      rcsb_polymer_entity { pdbx_description pdbx_mutation }
      rcsb_polymer_entity_container_identifiers {
        entity_id asym_ids auth_asym_ids
        reference_sequence_identifiers { database_name database_accession }
      }
    }
    nonpolymer_entities {
      pdbx_entity_nonpoly { comp_id name }
      rcsb_nonpolymer_entity_container_identifiers { asym_ids auth_asym_ids }
    }
    branched_entities {
      rcsb_branched_entity { pdbx_description }
      rcsb_branched_entity_container_identifiers { asym_ids auth_asym_ids }
    }
  }
}
"""


def rcsb_search(sess: requests.Session, acc: str, url: str,
                timeout: float = 60) -> list[str]:
    """Return sorted PDB IDs whose polymer entities reference UniProt ``acc``.

    Uses the curated reference-sequence cross-reference (exact match), so
    homologues and text mentions are not picked up.

    Raises
    ------
    FetchError
        If the search request fails.
    """
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers."
                             "reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": acc,
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    try:
        r = sess.post(url, json=query, timeout=timeout)
        if r.status_code == 204:  # no hits
            return []
        r.raise_for_status()
    except requests.RequestException as ex:
        raise FetchError(f"RCSB search failed: {ex}") from ex
    return sorted(h["identifier"].upper() for h in r.json().get("result_set", []))


def rcsb_metadata(sess: requests.Session, ids: list[str], cache: Path, url: str,
                  batch: int = 25, **http: Any) -> dict[str, dict]:
    """GraphQL metadata for ``ids`` in batches; returns ``{PDB ID: entry dict}``.

    Raises
    ------
    RuntimeError
        If GraphQL reports an error (the failing cache file is removed so the
        next run retries).
    """
    out: dict[str, dict] = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        cf = cache / "graphql" / f"{chunk[0]}_{chunk[-1]}_{len(chunk)}.json"
        data = cached_json(sess, url, cf, method="POST",
                           payload={"query": GRAPHQL_QUERY, "variables": {"ids": chunk}}, **http)
        if data.get("errors"):
            cf.unlink(missing_ok=True)
            raise RuntimeError(f"GraphQL error: {data['errors'][:2]}")
        for e in data["data"]["entries"] or []:
            if e:
                out[e["rcsb_id"].upper()] = e
        log.info("metadata %d/%d", min(i + batch, len(ids)), len(ids))
    return out


def entity_is_target(ent: dict, acc: str) -> bool:
    """True if a polymer entity's reference sequences include UniProt ``acc``."""
    ids = (ent.get("rcsb_polymer_entity_container_identifiers") or {})
    for ref in ids.get("reference_sequence_identifiers") or []:
        if (ref.get("database_accession") or "").upper() == acc.upper():
            return True
    return False


def download_mmcif(sess: requests.Session, pdb: str, cif_dir: Path, url_template: str,
                   **http: Any) -> Path:
    """Download (or reuse) ``<cif_dir>/<PDB>.cif.gz`` and return its path."""
    return cached_download(sess, url_template.format(pdb=pdb), cif_dir / f"{pdb}.cif.gz", **http)


def load_structure(sess: requests.Session, pdb: str, cif_dir: Path, url_template: str,
                   **http: Any):
    """Download if needed and read an entry with gemmi (entities set up).

    Raises
    ------
    ImportError
        If gemmi is not installed.
    """
    import gemmi  # local import: the rest of the module works without gemmi
    path = download_mmcif(sess, pdb, cif_dir, url_template, **http)
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st
