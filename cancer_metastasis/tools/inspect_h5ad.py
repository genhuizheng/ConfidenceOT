"""Describe a large H5AD without loading its matrices.

A Seurat export usually carries counts, normalised data and a dense
scale.data, and the dense one is what makes the file large. Reading the
structure first says which layer to extract and how much can be discarded.
"""

import sys

import h5py
import numpy as np


def human(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"


path = sys.argv[1]
with h5py.File(path, "r") as handle:
    print(f"{path}\ntop level: {list(handle)}\n")

    sizes: dict[str, int] = {}

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            sizes[name] = obj.nbytes

    handle.visititems(walk)
    print("largest datasets:")
    for name, size in sorted(sizes.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {human(size):>8}  {name}")

    for group in ("X", "raw/X"):
        if group in handle:
            node = handle[group]
            kind = "sparse" if isinstance(node, h5py.Group) else "dense"
            shape = node.attrs.get("shape", getattr(node, "shape", None))
            print(f"\n{group}: {kind}  shape={shape}")
            if isinstance(node, h5py.Group):
                for key in node:
                    print(f"    {key}: {node[key].shape} {node[key].dtype}")

    for group in ("layers", "obsm", "varm", "obsp"):
        if group in handle:
            print(f"\n{group}: {list(handle[group])}")

    for axis in ("obs", "var"):
        if axis not in handle:
            continue
        node = handle[axis]
        keys = [k for k in node if not k.startswith("_")]
        print(f"\n{axis} columns ({len(keys)}):")
        for key in keys:
            item = node[key]
            if isinstance(item, h5py.Group) and "categories" in item:
                cats = item["categories"][:]
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
                shown = ", ".join(cats[:12])
                print(f"  {key:34s} categorical n={len(cats):<6} {shown}")
            elif isinstance(item, h5py.Dataset):
                if item.dtype.kind in "OS" or h5py.check_string_dtype(item.dtype):
                    sample = item[: min(4, item.shape[0])]
                    sample = [s.decode() if isinstance(s, bytes) else str(s) for s in sample]
                    print(f"  {key:34s} string        {sample}")
                else:
                    sample = np.asarray(item[: min(4, item.shape[0])])
                    print(f"  {key:34s} {str(item.dtype):<13} {sample}")
