"""Residue numbering: label_seq_id ↔ UniProt, construct differences, range strings.

Three numbering schemes appear in this project and are never mixed without a
label:

* **label_seq** – mmCIF ``label_seq_id``: 1-based position in the entity
  sequence, including tags and unmodelled residues.
* **author** – ``auth_seq_id`` (+ insertion code): the depositor's numbering,
  as shown in most viewers; not guaranteed to equal UniProt numbering.
* **UniProt** – position in the canonical UniProt sequence, obtained only
  through SIFTS segments (never by assuming author == UniProt).
"""

from __future__ import annotations

from typing import Iterable


def to_ranges(nums: Iterable[int]) -> str:
    """Compress integers into a range string, e.g. ``[1,2,3,7,9,10] -> "1-3, 7, 9-10"``."""
    nums = sorted(set(nums))
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = n
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(out)


def parse_ranges(text: str) -> list[int]:
    """Inverse of :func:`to_ranges` (``"1-3, 7" -> [1, 2, 3, 7]``)."""
    out: list[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def segment_is_consistent(seg: dict) -> bool:
    """True if a SIFTS segment spans the same number of label_seq and UniProt residues."""
    return seg["seq_end"] - seg["seq_start"] == seg["unp_end"] - seg["unp_start"]


def label_to_unp(segments: list[dict]) -> dict[int, int]:
    """Map label_seq_id -> UniProt number from SIFTS segments.

    Segments whose two spans differ in length are skipped rather than
    mis-numbered (they are reported by :func:`construct_differences` and by
    the ``sifts_segment_integrity`` sanity check).
    """
    m: dict[int, int] = {}
    for s in segments:
        if not segment_is_consistent(s):
            continue
        for k in range(s["seq_end"] - s["seq_start"] + 1):
            m[s["seq_start"] + k] = s["unp_start"] + k
    return m


def unp_to_label(segments: list[dict]) -> dict[int, int]:
    """Inverse of :func:`label_to_unp` (UniProt number -> label_seq_id)."""
    return {u: l for l, u in label_to_unp(segments).items()}


def construct_differences(seq: str, segs: list[dict], useq: str) -> dict:
    """Compare an entity sequence with the canonical UniProt sequence.

    Parameters
    ----------
    seq:
        Entity sequence (one-letter, label_seq order).
    segs:
        SIFTS segments of one chain of that entity, sorted by ``seq_start``.
    useq:
        Canonical UniProt sequence.

    Returns
    -------
    dict
        ``unp_range``, ``unp_start``, ``unp_end``, ``n_ext``/``c_ext``
        (non-UniProt terminal residues, e.g. tags), ``mutations``
        (``<UniProt aa><UniProt number><entity aa>``), ``gaps`` (deletions,
        insertions, inconsistent segments) and ``coverage`` (fraction of the
        UniProt sequence covered).
    """
    if not segs:
        return {"unp_range": "no SIFTS mapping", "n_ext": "", "c_ext": "",
                "mutations": "", "gaps": "", "coverage": None}
    first, last = segs[0], segs[-1]
    n_ext = seq[: first["seq_start"] - 1]
    c_ext = seq[last["seq_end"]:]
    muts, gaps = [], []
    for s in segs:
        if not segment_is_consistent(s):
            gaps.append(f"length mismatch in segment {s['unp_start']}-{s['unp_end']}")
            continue
        for k in range(s["seq_end"] - s["seq_start"] + 1):
            p = seq[s["seq_start"] - 1 + k] if s["seq_start"] - 1 + k < len(seq) else "?"
            u = useq[s["unp_start"] - 1 + k] if s["unp_start"] - 1 + k < len(useq) else "?"
            if p != u:
                muts.append(f"{u}{s['unp_start'] + k}{p}")
    for a, b in zip(segs, segs[1:]):
        if b["unp_start"] > a["unp_end"] + 1:
            gaps.append(f"UniProt {a['unp_end'] + 1}-{b['unp_start'] - 1} absent from construct")
        if b["seq_start"] > a["seq_end"] + 1:
            ins = seq[a["seq_end"]: b["seq_start"] - 1]
            gaps.append(f"non-UniProt insertion after {a['unp_end']}: {ins}")
    covered = sum(s["unp_end"] - s["unp_start"] + 1 for s in segs)
    return {
        "unp_range": "; ".join(f"{s['unp_start']}-{s['unp_end']}" for s in segs),
        "unp_start": first["unp_start"], "unp_end": last["unp_end"],
        "n_ext": n_ext, "c_ext": c_ext,
        "mutations": ", ".join(muts), "gaps": "; ".join(gaps),
        "coverage": round(covered / len(useq), 3),
    }


# ---------------------------------------------------------------------------
# Sequence-based numbering (Milestone 4: predicted models have no SIFTS mapping)
# ---------------------------------------------------------------------------
def align_to_uniprot(seq: str, useq: str, match: float = 2.0, mismatch: float = -1.0,
                     gap: float = -2.0) -> dict[int, int]:
    """Map positions of ``seq`` (0-based) to UniProt numbers (1-based) of ``useq``.

    Identical sequences and exact substrings (e.g. an ectodomain construct)
    map directly. Otherwise a local (Smith–Waterman) alignment is used, so an
    expression tag and the unmodelled part of UniProt can both be skipped at
    no cost, while point mutations and internal gaps are tolerated.
    Substituted positions keep their UniProt number; insertions and residues
    outside the aligned segment are unmapped.
    """
    if not seq or not useq:
        return {}
    if seq == useq:
        return {i: i + 1 for i in range(len(seq))}
    off = useq.find(seq)
    if off >= 0:
        return {i: off + i + 1 for i in range(len(seq))}
    import numpy as np

    A = np.frombuffer(seq.encode(), np.uint8)
    B = np.frombuffer(useq.encode(), np.uint8)
    n, m = len(A), len(B)
    H = np.zeros((n + 1, m + 1))
    ptr = np.full((n + 1, m + 1), 3, np.int8)        # 0 diagonal, 1 up (gap in useq), 2 left (gap in seq), 3 start
    j = np.arange(1, m + 1)
    for i in range(1, n + 1):
        diag = H[i - 1, :-1] + np.where(B == A[i - 1], match, mismatch)
        up = H[i - 1, 1:] + gap
        T = np.maximum(np.maximum(diag, up), 0.0)
        tptr = np.where(T <= 0, 3, np.where(diag >= up, 0, 1)).astype(np.int8)
        # left moves: H[j] = max_k<=j (T[k] + gap*(j-k)); H[i,0] = 0
        base = np.concatenate([[0.0], T]) - gap * np.arange(m + 1)
        row = np.maximum(T, np.maximum.accumulate(base)[1:] + gap * j)
        H[i, 1:] = row
        ptr[i, 1:] = np.where(row > T, 2, tptr)
    i, k = np.unravel_index(int(np.argmax(H)), H.shape)
    out = {}
    while i > 0 and k > 0 and H[i, k] > 0:
        p = ptr[i, k]
        if p == 0:
            out[int(i) - 1] = int(k)
            i, k = i - 1, k - 1
        elif p == 1:
            i -= 1
        elif p == 2:
            k -= 1
        else:
            break
    return out
