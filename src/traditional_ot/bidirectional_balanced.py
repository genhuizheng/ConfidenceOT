"""Deprecated compatibility shim for the former balanced M4 module."""

from confidenceot import _cpu_balanced as _implementation

globals().update(
    {
        name: getattr(_implementation, name)
        for name in dir(_implementation)
        if not name.startswith("__")
    }
)

del _implementation

