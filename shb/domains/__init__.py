"""Importing this package registers every domain's task families.

Add a new domain by writing ``shb/domains/<name>.py`` that calls
``shb.registry.register(...)`` at import time, then adding its name to
``DOMAIN_MODULES`` below. Imports are defensive: one broken or not-yet-written
domain logs a warning instead of taking down the whole registry, which keeps the
benchmark robust as contributors add domains.
"""
from __future__ import annotations

import importlib
import os
import warnings

# In evaluation mode (SHB_STRICT=1) a domain that fails to import is fatal — an
# official run must not silently drop a domain. Development is tolerant.
STRICT = os.environ.get("SHB_STRICT", "") not in ("", "0", "false")

DOMAIN_MODULES = [
    "chemistry",
    "statistics",
    "physics",
    "biology",
    "climate",
    "astronomy",
    "neuroscience",
    "materials",
]

LOAD_ERRORS: dict[str, str] = {}

for _name in DOMAIN_MODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except Exception as exc:  # noqa: BLE001 — a contributor's domain shouldn't break others
        if STRICT:
            raise
        LOAD_ERRORS[_name] = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"[shb] domain '{_name}' failed to load: {exc}")
