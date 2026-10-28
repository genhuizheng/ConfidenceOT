"""The bin-to-bin couplings of a MOSTA pair: what 23_ saves, and how any block is rebuilt.

The blockwise solver never forms the N x M coupling; it ends with the two dual
potentials of its final consistency solve.  With the coordinates, the cost
scale, the final gates, the weights, the rejection cost and epsilon they fix
every entry:

    cost[i, j]  = max(|s_i|^2 + |t_j|^2 - 2 s_i.t_j, 0) / cost_scale
    opt[i, j]   = cost[i, j]       if source_gate[i] and target_gate[j]
                  rejection_cost   otherwise
    P[i, j]     = exp(log_u[i] + log a[i] + log b[j] - opt[i, j] / epsilon + log_v[j])

which is ``confidenceot.blockwise._GatedBlocks.coupling_block``, the formula the
solver itself iterates; the rebuild calls that code, it does not restate it.
``s``/``t`` are the float32 rows of representations/<pair_id>/representation.npz,
taken to the solver's dtype.  Row i is equalised_row i of
slices/<source>_bins.csv.gz, i.e. ``source_bin_ids[i]``; the same for targets.

**Saved per pair** (by :func:`save_coupling`, called from 23_):

* ``potentials.npz``: ``source_bin_ids``, ``target_bin_ids``, ``cost_scale``,
  and for each method prefix (``m4e``, ``m4r``, ``balanced``)
  ``<p>_log_u``, ``<p>_log_v``, ``<p>_source_gate``, ``<p>_target_gate``,
  ``<p>_source_weights``, ``<p>_target_weights``, ``<p>_rejection_cost``,
  ``<p>_epsilon``, all float64 / bool, exactly the solver's values;
* ``coupling.json``: the representation it refers to (path relative to the run
  root, SHA-256, shapes), the cost definition, every solver setting of each
  method, its convergence record, and the SHA-256 of potentials.npz.

**Rebuilt** by :class:`PairCoupling` / :func:`reconstruct_coupling`: any
source x target subset of P (by row index, boolean mask or bin id), on the full
support or restricted to retained x retained bins (``support='active'``, the
support the ``co_`` summaries use), and the conditional matching
probabilities ``P_ij / sum_j P_ij`` (``normalise='source'``) or
``P_ij / sum_i P_ij`` (``normalise='target'``), whose denominators run over
every bin of the partner slice on the same support, not over the subset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

FORMAT = "mosta-coupling-v1"
POTENTIALS = "potentials.npz"
RECORD = "coupling.json"
METHODS = ("M4-E", "M4-R", "balanced")
SUPPORTS = ("all", "active")
COST_DEFINITION = (
    "cost[i,j] = max(|s_i|^2 + |t_j|^2 - 2 s_i.t_j, 0) / cost_scale on the representation rows "
    "(float32 on disk, taken to the solver dtype); opt[i,j] = cost[i,j] where source_gate[i] and "
    "target_gate[j], rejection_cost elsewhere; P[i,j] = exp(log_u[i] + log a[i] + log b[j] - "
    "opt[i,j] / epsilon + log_v[j])"
)


def prefix(method: str) -> str:
    """The key prefix of a method, the one 23_ uses for its columns."""
    return method.lower().replace("-", "")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_coupling(directory: Path, fits: dict, models: dict, *, pair_id: str, representation_path: str,
                  representation_sha256: str, representation_shapes: tuple, source_bin_ids: np.ndarray,
                  target_bin_ids: np.ndarray, cost_scale: float, block_rows: int) -> list[str]:
    """Write potentials.npz and coupling.json for the fitted methods; returns the file names."""
    arrays: dict[str, np.ndarray] = {
        "source_bin_ids": np.asarray(source_bin_ids, dtype=str),
        "target_bin_ids": np.asarray(target_bin_ids, dtype=str),
        "cost_scale": np.float64(cost_scale),
    }
    methods = {}
    for method in METHODS:
        fit, model = fits[method], models[method]
        p = prefix(method)
        arrays.update({
            f"{p}_log_u": np.asarray(fit.log_u, dtype=np.float64),
            f"{p}_log_v": np.asarray(fit.log_v, dtype=np.float64),
            f"{p}_source_gate": np.asarray(fit.source_gate, dtype=bool),
            f"{p}_target_gate": np.asarray(fit.target_gate, dtype=bool),
            f"{p}_source_weights": np.asarray(fit.source_weights, dtype=np.float64),
            f"{p}_target_weights": np.asarray(fit.target_weights, dtype=np.float64),
            f"{p}_rejection_cost": np.float64(fit.rejection_cost),
            f"{p}_epsilon": np.float64(fit.epsilon),
        })
        settings = {key: value for key, value in model._solver_kwargs().items()
                    if not key.endswith("_weights") and not key.startswith("initial_")}
        methods[method] = {
            "prefix": p,
            "solver_settings": {**settings, "cuda_dtype": model.cuda_dtype, "device": model.device,
                                "block_rows": int(block_rows)},
            "result": {
                "backbone": fit.backbone, "variant": fit.variant, "dtype": fit.dtype, "device": fit.device,
                "backend": fit.backend, "block_rows": int(fit.block_rows),
                "rejection_cost": float(fit.rejection_cost), "epsilon": float(fit.epsilon),
                "inner_converged": bool(fit.inner_converged), "outer_converged": bool(fit.outer_converged),
                "cycle_detected": bool(fit.cycle_detected), "cycle_length": int(fit.cycle_length),
                "n_outer_iterations": int(fit.n_outer_iterations),
                "total_inner_iterations": int(fit.total_inner_iterations),
                "objective": float(fit.objective),
                "source_retained": int(np.sum(fit.source_gate)), "target_retained": int(np.sum(fit.target_gate)),
                "source_rejection_bounds": list(fit.source_rejection_bounds),
                "target_rejection_bounds": list(fit.target_rejection_bounds),
            },
        }
    np.savez(directory / POTENTIALS, **arrays)
    record = {
        "format": FORMAT,
        "pair_id": pair_id,
        "representation": {
            "path": representation_path, "sha256": representation_sha256,
            "source_shape": list(representation_shapes[0]), "target_shape": list(representation_shapes[1]),
            "dtype_on_disk": "float32",
        },
        "bin_order": ("row i of the representation's source/target = equalised_row i of "
                      "slices/<sample>_bins.csv.gz = source_bin_ids[i] / target_bin_ids[i] of potentials.npz"),
        "cost": {"definition": COST_DEFINITION, "cost_scale": float(cost_scale)},
        "methods": methods,
        "files": {POTENTIALS: sha256(directory / POTENTIALS)},
    }
    (directory / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return [POTENTIALS, RECORD]


def _selection(selector: Any, ids: np.ndarray, lookup: dict[str, int] | None) -> tuple[np.ndarray, dict | None]:
    """Row positions from None (all), integer positions, a boolean mask, or bin ids."""
    count = len(ids)
    if selector is None:
        return np.arange(count), lookup
    values = np.asarray(selector)
    if values.dtype == bool:
        if values.shape != (count,):
            raise ValueError(f"a boolean selection needs {count} flags, got {values.shape}")
        return np.flatnonzero(values), lookup
    if values.dtype.kind in "iu":
        positions = values.astype(np.int64).ravel()
        if positions.size and (positions.min() < 0 or positions.max() >= count):
            raise IndexError(f"row positions must lie in [0, {count})")
        return positions, lookup
    if values.dtype.kind in "UO":
        if lookup is None:
            lookup = {str(name): position for position, name in enumerate(ids)}
        missing = [str(name) for name in values.ravel() if str(name) not in lookup]
        if missing:
            raise KeyError(f"{len(missing)} bin ids not in this pair, e.g. {missing[:3]}")
        return np.array([lookup[str(name)] for name in values.ravel()], dtype=np.int64), lookup
    raise TypeError("select bins by None, integer positions, a boolean mask or bin ids")


class PairCoupling:
    """The saved couplings of one pair, rebuilt block by block on demand.

    ``pair_dir`` is pairs/<pair_id>; ``root`` (default: two levels up) is the run
    root the representation path is relative to; ``representation`` overrides
    that path.  The representation's SHA-256 is checked against coupling.json
    and the potentials' against the record, unless ``verify`` is False.
    """

    def __init__(self, pair_dir: Path, *, root: Path | None = None, representation: Path | None = None,
                 device: str = "cpu", verify: bool = True) -> None:
        from confidenceot.blockwise import _import_torch

        self.pair_dir = Path(pair_dir)
        self.record = json.loads((self.pair_dir / RECORD).read_text(encoding="utf-8"))
        if self.record.get("format") != FORMAT:
            raise ValueError(f"{self.pair_dir / RECORD}: format {self.record.get('format')!r}, expected {FORMAT}")
        root = Path(root) if root is not None else self.pair_dir.parent.parent
        self.representation_path = Path(representation) if representation is not None else root / self.record["representation"]["path"]
        if verify:
            for path, expected in ((self.pair_dir / POTENTIALS, self.record["files"][POTENTIALS]),
                                   (self.representation_path, self.record["representation"]["sha256"])):
                found = sha256(path)
                if found != expected:
                    raise ValueError(f"{path}: SHA-256 {found} differs from the recorded {expected}")
        with np.load(self.pair_dir / POTENTIALS) as stored:
            self.arrays = {key: stored[key] for key in stored.files}
        with np.load(self.representation_path) as stored:
            self.source_coordinates = np.asarray(stored["source"])
            self.target_coordinates = np.asarray(stored["target"])
        self.source_bin_ids = self.arrays["source_bin_ids"]
        self.target_bin_ids = self.arrays["target_bin_ids"]
        if (len(self.source_bin_ids), len(self.target_bin_ids)) != (len(self.source_coordinates), len(self.target_coordinates)):
            raise ValueError("the potentials and the representation disagree on the number of bins")
        self.cost_scale = float(self.arrays["cost_scale"])
        self.shape = (len(self.source_bin_ids), len(self.target_bin_ids))
        self.device = device
        self.torch = _import_torch(device)
        self._lookups: dict[str, dict | None] = {"source": None, "target": None}

    # -- inputs ---------------------------------------------------------------
    def method(self, method: str) -> dict[str, Any]:
        if method not in self.record["methods"]:
            raise KeyError(f"{method!r} not saved; saved: {sorted(self.record['methods'])}")
        p = prefix(method)
        result = self.record["methods"][method]["result"]
        return {
            "log_u": self.arrays[f"{p}_log_u"], "log_v": self.arrays[f"{p}_log_v"],
            "source_gate": self.arrays[f"{p}_source_gate"], "target_gate": self.arrays[f"{p}_target_gate"],
            "source_weights": self.arrays[f"{p}_source_weights"], "target_weights": self.arrays[f"{p}_target_weights"],
            "rejection_cost": float(self.arrays[f"{p}_rejection_cost"]), "epsilon": float(self.arrays[f"{p}_epsilon"]),
            "dtype": result["dtype"],
        }

    def rows(self, source: Any = None) -> np.ndarray:
        positions, self._lookups["source"] = _selection(source, self.source_bin_ids, self._lookups["source"])
        return positions

    def columns(self, target: Any = None) -> np.ndarray:
        positions, self._lookups["target"] = _selection(target, self.target_bin_ids, self._lookups["target"])
        return positions

    # -- blocks ---------------------------------------------------------------
    def _blocks(self, method: str, rows: np.ndarray, columns: np.ndarray, block_rows: int | None):
        """The solver's gated blocks, on the selected rows and columns only."""
        from confidenceot.blockwise import CoordinateCost, _GatedBlocks, _Geometry, resolve_block_rows

        torch = self.torch
        inputs = self.method(method)
        dtype = {"float32": torch.float32, "float64": torch.float64}[inputs["dtype"]]
        device = torch.device(self.device)
        cost = CoordinateCost(self.source_coordinates[rows], self.target_coordinates[columns], self.cost_scale)
        geometry = _Geometry(torch, cost, device, dtype)
        as_tensor = lambda values: torch.as_tensor(np.asarray(values), device=device, dtype=dtype)  # noqa: E731
        blocks = _GatedBlocks(
            torch, geometry, inputs["source_gate"][rows], inputs["target_gate"][columns],
            rejection_cost=inputs["rejection_cost"], epsilon=inputs["epsilon"],
            block_rows=resolve_block_rows(block_rows, len(rows), len(columns)),
            log_a=torch.log(as_tensor(inputs["source_weights"][rows])),
            log_b=torch.log(as_tensor(inputs["target_weights"][columns])),
        )
        return blocks, as_tensor(inputs["log_u"][rows]), as_tensor(inputs["log_v"][columns]), inputs

    def _tensor_blocks(self, method: str, rows: np.ndarray, columns: np.ndarray, support: str,
                       block_rows: int | None) -> Iterator[tuple[int, int, Any]]:
        """(start, stop, block of P) as tensors on the device, the support applied."""
        from confidenceot.blockwise import _full_precision_matmul

        if support not in SUPPORTS:
            raise ValueError(f"support must be one of {SUPPORTS}")
        if rows.size == 0 or columns.size == 0:
            return
        torch = self.torch
        blocks, log_u, log_v, inputs = self._blocks(method, rows, columns, block_rows)
        if support == "active":
            source_gate = torch.as_tensor(inputs["source_gate"][rows], device=blocks.geometry.device)
            target_gate = torch.as_tensor(inputs["target_gate"][columns], device=blocks.geometry.device)
        with _full_precision_matmul(torch):
            for start, stop in blocks.ranges:
                _, coupling = blocks.coupling_block(start, stop, log_u, log_v)
                coupling = coupling.to(torch.float64)
                if support == "active":
                    coupling = coupling * (source_gate[start:stop, None] & target_gate[None, :])
                yield start, stop, coupling

    def iter_blocks(self, method: str, source: Any = None, target: Any = None, *, support: str = "all",
                    block_rows: int | None = None) -> Iterator[tuple[int, int, np.ndarray]]:
        """(start, stop, P[rows[start:stop]][:, columns]) in float64, block by block over the selected rows."""
        for start, stop, coupling in self._tensor_blocks(method, self.rows(source), self.columns(target),
                                                         support, block_rows):
            yield start, stop, coupling.detach().cpu().numpy()

    def marginals(self, method: str, source: Any = None, target: Any = None, *, support: str = "all",
                  block_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Row sums over the selected columns and column sums over the selected rows, float64, on the device."""
        torch = self.torch
        rows, columns = self.rows(source), self.columns(target)
        device = torch.device(self.device)
        row_mass = torch.zeros(len(rows), device=device, dtype=torch.float64)
        column_mass = torch.zeros(len(columns), device=device, dtype=torch.float64)
        for start, stop, coupling in self._tensor_blocks(method, rows, columns, support, block_rows):
            row_mass[start:stop] = coupling.sum(dim=1)
            column_mass += coupling.sum(dim=0)
        return row_mass.cpu().numpy(), column_mass.cpu().numpy()

    def mass(self, method: str, source: Any = None, target: Any = None, *, support: str = "all",
             normalise: str | None = None, block_rows: int | None = None,
             max_elements: int = 1 << 28) -> np.ndarray:
        """P on the selected source x target bins, or a conditional matching probability.

        ``normalise=None``: the transport mass P_ij.  ``'source'``: P_ij / sum_j P_ij,
        the sum over every target bin on the same support; ``'target'``: P_ij /
        sum_i P_ij over every source bin.  A bin with no mass on the support gives NaN.
        """
        rows, columns = self.rows(source), self.columns(target)
        if len(rows) * len(columns) > max_elements:
            raise MemoryError(f"{len(rows)} x {len(columns)} entries exceeds max_elements={max_elements}; "
                              "use iter_blocks or a smaller selection")
        out = np.empty((len(rows), len(columns)))
        for start, stop, values in self.iter_blocks(method, rows, columns, support=support, block_rows=block_rows):
            out[start:stop] = values
        if normalise is None:
            return out
        with np.errstate(divide="ignore", invalid="ignore"):
            if normalise == "source":
                total, _ = self.marginals(method, rows, None, support=support, block_rows=block_rows)
                return np.where(total[:, None] > 0, out / total[:, None], np.nan)
            if normalise == "target":
                _, total = self.marginals(method, None, columns, support=support, block_rows=block_rows)
                return np.where(total[None, :] > 0, out / total[None, :], np.nan)
        raise ValueError("normalise must be None, 'source' or 'target'")


def reconstruct_coupling(pair_dir: Path, method: str = "M4-E", source: Any = None, target: Any = None, *,
                         support: str = "all", normalise: str | None = None, root: Path | None = None,
                         representation: Path | None = None, device: str = "cpu",
                         block_rows: int | None = None, max_elements: int = 1 << 28) -> np.ndarray:
    """One call: P (or P_ij / sum_j P_ij, or P_ij / sum_i P_ij) on any source x target subset of a saved pair.

    ``source`` / ``target``: None (every bin), integer row positions, a boolean
    mask, or bin ids (``source_bin_ids`` of potentials.npz).  ``support``:
    ``'all'`` is the coupling the solver formed; ``'active'`` keeps only
    retained x retained entries of that method's gates.
    """
    coupling = PairCoupling(pair_dir, root=root, representation=representation, device=device)
    return coupling.mass(method, source, target, support=support, normalise=normalise,
                         block_rows=block_rows, max_elements=max_elements)


def group_mass(coupling: PairCoupling, method: str, source_codes: Sequence[int], n_source: int,
               target_codes: Sequence[int], n_target: int, *, support: str = "all",
               block_rows: int | None = None) -> np.ndarray:
    """sum of P over each (source group, target group), every bin, block by block."""
    source_codes = np.asarray(source_codes, dtype=np.int64)
    target_hot = np.zeros((len(target_codes), n_target))
    target_hot[np.arange(len(target_codes)), np.asarray(target_codes, dtype=np.int64)] = 1.0
    out = np.zeros((n_source, n_target))
    for start, stop, values in coupling.iter_blocks(method, support=support, block_rows=block_rows):
        np.add.at(out, source_codes[start:stop], values @ target_hot)
    return out
