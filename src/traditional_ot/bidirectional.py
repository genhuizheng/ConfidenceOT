"""Deprecated compatibility shim for the former M4/UOT module.

The implementation moved to :mod:`confidenceot._cpu_uot`.  This module keeps
historical scripts importable without maintaining a second copy of M4.
"""

from confidenceot import _cpu_uot as _implementation

globals().update(
    {
        name: getattr(_implementation, name)
        for name in dir(_implementation)
        if not name.startswith("__")
    }
)

del _implementation

