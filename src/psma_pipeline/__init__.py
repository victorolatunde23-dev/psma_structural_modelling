"""PSMA structural-modelling pipeline.

Importable modules behind the notebooks in ``notebooks/``. Scientific
calculations live here; notebooks only orchestrate, explain and plot.

Quick start::

    from psma_pipeline import load_config, setup_logging
    from psma_pipeline.survey import run_survey

    setup_logging()
    cfg = load_config()                 # config/default_config.yaml (Q04609)
    results = run_survey(cfg)
"""

from __future__ import annotations

import logging

__version__ = "0.2.0"

from .config import Config, ConfigError, load_config  # noqa: E402

__all__ = ["Config", "ConfigError", "load_config", "setup_logging", "__version__"]


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the ``psma_pipeline`` logger once and return it.

    Parameters
    ----------
    level:
        Logging level for the package logger (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        The package logger. Messages are also recorded in result tables
        (``Warnings`` / ``Calculation status`` columns); logging is for the
        interactive reader only.
    """
    logger = logging.getLogger("psma_pipeline")
    logger.setLevel(level)
    if not any(getattr(h, "_psma", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                                               datefmt="%H:%M:%S"))
        handler._psma = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger
