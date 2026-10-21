"""Calculation-status vocabulary and an issue log that ends up in result tables.

Rule for every module: nothing fails silently. A failed or skipped calculation
produces a row (or a column value) with a status from :class:`Status` and a
human-readable note, and the same note is logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

log = logging.getLogger("psma_pipeline")


class Status(str, Enum):
    """Allowed values of every ``Calculation status`` / check-status column."""

    OK = "OK"                          # calculation completed
    PASS = "PASS"                      # a check passed
    WARNING = "WARNING"                # completed, but something needs attention
    FAIL = "FAIL"                      # calculation or check failed
    NOT_APPLICABLE = "NOT APPLICABLE"  # does not apply to this entry (e.g. RSRZ for cryo-EM)
    NOT_RUN = "NOT RUN"                # skipped (tool unavailable, disabled, upstream failure)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass
class Issue:
    """One recorded problem."""

    stage: str
    status: Status
    message: str
    pdb_id: str | None = None
    chain: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class IssueLog:
    """Collects warnings and failures from every stage of a run.

    Use :meth:`add` instead of ``print``/``warnings.warn``; the log becomes the
    ``RunIssues`` sheet so a returning reader sees every problem in one place.
    """

    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def add(self, stage: str, status: Status | str, message: str, *,
            pdb_id: str | None = None, chain: str | None = None,
            **detail: Any) -> Issue:
        """Record an issue and log it at a matching level."""
        st = Status(status) if not isinstance(status, Status) else status
        issue = Issue(stage, st, message, pdb_id, chain, dict(detail))
        self.issues.append(issue)
        where = " ".join(x for x in (pdb_id, chain) if x)
        text = f"[{stage}] {where + ': ' if where else ''}{message}"
        if st is Status.FAIL:
            log.error(text)
        elif st in (Status.WARNING, Status.NOT_RUN):
            log.warning(text)
        else:
            log.info(text)
        return issue

    def to_frame(self) -> pd.DataFrame:
        """Return the issues as a table (one row per issue)."""
        cols = ["Stage", "Status", "PDB ID", "Chain", "Message", "Detail"]
        if not self.issues:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([{
            "Stage": i.stage, "Status": i.status.value, "PDB ID": i.pdb_id,
            "Chain": i.chain, "Message": i.message,
            "Detail": "; ".join(f"{k}={v}" for k, v in i.detail.items()),
        } for i in self.issues], columns=cols)

    def count(self, status: Status) -> int:
        """Number of issues with the given status."""
        return sum(i.status is status for i in self.issues)

    def __len__(self) -> int:
        return len(self.issues)
