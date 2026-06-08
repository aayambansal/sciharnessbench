"""Global registry of task families.

Each domain module under :mod:`shb.domains` calls :func:`register` at import
time. :func:`ensure_loaded` imports the domains package so the registry is
populated before a run.
"""
from __future__ import annotations

from typing import Optional

from .types import Family

_REGISTRY: dict[str, Family] = {}
_LOADED = False


def register(family: Family) -> Family:
    if family.family_id in _REGISTRY:
        raise ValueError(f"duplicate family_id: {family.family_id}")
    _REGISTRY[family.family_id] = family
    return family


def ensure_loaded() -> None:
    global _LOADED
    if not _LOADED:
        _LOADED = True
        import shb.domains  # noqa: F401  (import triggers registration)


def get(family_id: str) -> Family:
    ensure_loaded()
    return _REGISTRY[family_id]


def all_families() -> list[Family]:
    ensure_loaded()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def families_in(domain: str) -> list[Family]:
    return [f for f in all_families() if f.domain == domain]


def domains() -> list[str]:
    return sorted({f.domain for f in all_families()})


def select(domains: Optional[list[str]] = None,
           families: Optional[list[str]] = None) -> list[Family]:
    fams = all_families()
    if domains:
        fams = [f for f in fams if f.domain in set(domains)]
    if families:
        want = set(families)
        fams = [f for f in fams if f.family_id in want]
    return fams
