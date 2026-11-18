"""Compatibility shim for running the unmodified CellOT repository (written for Python 3.9 / numpy 1.19) on Python 3.10:
aliases removed in newer Python / numpy. Loaded through PYTHONPATH by run_learned.py; changes no CellOT logic."""
import collections
import collections.abc
for _n in ("MutableMapping", "Mapping", "Sequence", "Iterable"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))
