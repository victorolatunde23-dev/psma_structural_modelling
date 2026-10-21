"""Biological assemblies: choice, generator expansion and per-model composition.

Scientific rule: the oligomeric state is taken from the chosen *biological
assembly* by applying its generators (sub-chains × operators), never from the
number of chains in the asymmetric unit.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

log = logging.getLogger("psma_pipeline.assemblies")

OLIGOMER_NAMES = {0: "none", 1: "monomer", 2: "dimer", 3: "trimer", 4: "tetramer"}


def oligomer_label(n: Any) -> str:
    """``2 -> "dimer"``; other integers -> ``"<n>-mer"``; non-integers -> ``"unknown"``."""
    return OLIGOMER_NAMES.get(n, f"{n}-mer" if isinstance(n, int) else "unknown")


def _details(a: dict) -> str:
    return ((a.get("pdbx_struct_assembly") or {}).get("details") or "").lower()


def choose_assembly(assemblies: list[dict],
                    preference: list[str] | tuple[str, ...] = ("author", "software", "first")
                    ) -> tuple[dict | None, str]:
    """Pick the biological assembly to analyse and say why.

    Preference order (configurable, ``structure.assembly_preference``):

    1. ``author`` – first assembly whose ``details`` mention the author
       (``author_defined_assembly`` or ``author_and_software_defined_assembly``);
    2. ``software`` – first software-only assembly (e.g. PISA);
    3. ``first`` – the assembly with the lowest ID.

    Returns
    -------
    (assembly or None, reason)
    """
    if not assemblies:
        return None, "no assembly defined"
    ordered = sorted(assemblies, key=_assembly_order)   # deterministic: lowest assembly ID first
    for pref in preference:
        if pref == "author":
            hit = next((a for a in ordered if "author" in _details(a)), None)
            if hit:
                return hit, f"author-defined ({_details(hit)})"
        elif pref == "software":
            hit = next((a for a in ordered
                        if "software" in _details(a) and "author" not in _details(a)), None)
            if hit:
                return hit, f"software-defined only ({_details(hit)}); no author-defined assembly"
        elif pref == "first":
            hit = ordered[0]
            return hit, f"fallback to first assembly ({_details(hit) or 'details missing'})"
    return ordered[0], "fallback to first assembly (no preference matched)"


def _assembly_order(a: dict) -> tuple:
    """Sort key: numeric assembly ID when possible (so 2 < 10), then the RCSB ID."""
    aid = str((a.get("pdbx_struct_assembly") or {}).get("id") or a.get("rcsb_id", "").split("-")[-1])
    return (0, int(aid), "") if aid.isdigit() else (1, 0, aid)


def global_symmetry(a: dict) -> dict:
    """RCSB global symmetry record of an assembly (first record if none is global)."""
    for s in a.get("rcsb_struct_symmetry") or []:
        if s.get("kind") == "Global Symmetry":
            return s
    return (a.get("rcsb_struct_symmetry") or [{}])[0]


def coordinate_analysis(st: Any, target_entity_ids: set[str], model_index: int = 0) -> dict:
    """Per-model composition of a gemmi Structure and assembly expansion helpers.

    Parameters
    ----------
    st:
        ``gemmi.Structure`` with entities set up (``st.setup_entities()``).
    target_entity_ids:
        Entity IDs (strings) of the target protein.
    model_index:
        0-based model index (0 = model 1).

    Returns
    -------
    dict with keys
        ``observed`` (label_asym -> set of observed label_seq), ``seq_len``,
        ``ent_of_sub`` (label_asym -> entity), ``auth_of_sub`` (label_asym ->
        author chain), ``target_subs``, ``asu_het`` (Counter of het residue
        names in this model), ``expand(assembly_id) -> (target copies, het
        Counter)``, ``count_subs(assembly_id, subs) -> copies``, ``st``,
        ``model``, ``model_number`` (1-based).
    """
    import gemmi

    model = st[model_index]
    ent_of_sub: dict[str, str] = {}
    seq_len: dict[str, int] = {}
    for ent in st.entities:
        for sub in ent.subchains:
            ent_of_sub[sub] = ent.name
        if ent.name in target_entity_ids:
            seq_len[ent.name] = len(ent.full_sequence)

    observed: dict[str, set[int]] = defaultdict(set)
    auth_of_sub: dict[str, str] = {}
    het_by_sub: dict[str, Counter] = defaultdict(Counter)
    for ch in model:
        for res in ch:
            sub = res.subchain
            auth_of_sub.setdefault(sub, ch.name)
            if res.is_water():
                continue
            if ent_of_sub.get(sub) in target_entity_ids and res.label_seq is not None:
                observed[sub].add(res.label_seq)
            elif res.entity_type != gemmi.EntityType.Polymer:
                het_by_sub[sub][res.name] += 1

    target_subs = {s for s, e in ent_of_sub.items() if e in target_entity_ids}

    def _assembly(assembly_id: str):
        return next((a for a in st.assemblies if a.name == assembly_id), None)

    def expand(assembly_id: str) -> tuple[int | None, Counter | None]:
        """Copies of the target and het-residue counts in an assembly."""
        asm = _assembly(assembly_id)
        if asm is None:
            return None, None
        copies, het = 0, Counter()
        for gen in asm.generators:
            n_ops = len(gen.operators)
            for sub in gen.subchains:
                if sub in target_subs:
                    copies += n_ops
                for name, n in het_by_sub.get(sub, {}).items():
                    het[name] += n * n_ops
        return copies, het

    def count_subs(assembly_id: str, subs: Any) -> int | None:
        """Copies of the given sub-chains (e.g. glycan chains) in an assembly."""
        asm = _assembly(assembly_id)
        if asm is None:
            return None
        subs = set(subs)
        return sum(len(gen.operators) for gen in asm.generators
                   for sub in gen.subchains if sub in subs)

    asu_het: Counter = Counter()
    for c in het_by_sub.values():
        asu_het.update(c)
    return {
        "observed": observed, "seq_len": seq_len, "ent_of_sub": ent_of_sub,
        "auth_of_sub": auth_of_sub, "target_subs": target_subs,
        "asu_het": asu_het, "expand": expand, "count_subs": count_subs,
        "st": st, "model": model, "model_number": model_index + 1,
    }


def assembly_note(asu_copies: int, asm_copies: int | None) -> str:
    """Human-readable relation between ASU and assembly content."""
    if asm_copies is None:
        return "assembly could not be evaluated"
    if asm_copies == asu_copies:
        return "assembly = asymmetric-unit content"
    if asm_copies > asu_copies:
        return f"assembly generated by symmetry ({asu_copies} in ASU -> {asm_copies})"
    return f"ASU contains {asu_copies} copies; assembly uses {asm_copies}"

