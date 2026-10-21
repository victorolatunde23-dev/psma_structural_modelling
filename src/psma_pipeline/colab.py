"""Shared storage for notebooks running in Google Colab.

Colab's local disk (``/content``) is wiped whenever the runtime restarts, so
results written by one notebook are not available to the next. Every
notebook therefore calls :func:`storage_overrides`, which (in Colab, by
default) mounts Google Drive and points ``paths.cache`` and
``paths.outputs`` at one Drive folder shared by all notebooks:

    MyDrive/psma_structural_modelling/cache     downloads (API responses, mmCIF, reports)
    MyDrive/psma_structural_modelling/outputs   workbooks, tables, manifests

Outside Colab nothing changes (the project's ``cache/`` and ``outputs/`` are used).
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_DRIVE_FOLDER = "/content/drive/MyDrive/psma_structural_modelling"


def in_colab() -> bool:
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules


def storage_overrides(use_drive: bool = True, folder: str = DEFAULT_DRIVE_FOLDER) -> dict:
    """Configuration overrides that put cache and outputs on Google Drive (Colab only).

    Parameters
    ----------
    use_drive:
        Set False to keep everything on the temporary Colab disk.
    folder:
        Drive folder shared by all notebooks.

    Returns
    -------
    dict
        ``{"paths": {"cache": ..., "outputs": ...}}`` or ``{}`` when not in
        Colab / Drive not requested.
    """
    if not (use_drive and in_colab()):
        return {}
    from google.colab import drive  # type: ignore[import-not-found]
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    base = Path(folder)
    (base / "cache").mkdir(parents=True, exist_ok=True)
    (base / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"Results and downloads are stored on Google Drive: {base}")
    return {"paths": {"cache": str(base / "cache"), "outputs": str(base / "outputs")}}
