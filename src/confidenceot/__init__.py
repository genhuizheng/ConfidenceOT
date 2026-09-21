"""ConfidenceOT: confidence-filtered optimal transport.

The public package exposes M4-E (exact) and M4-R (reversible) without placing
the proposed method under the ``traditional_ot`` name.  ``Preprocessing``
carries everything between counts and a cost matrix -- the transform, read
equalisation and the cost geometry -- as one recordable configuration, and is
importable from here so a caller never has to assemble those choices itself.  ``cellot`` and
``traditional_ot`` remain compatibility namespaces for existing experiments.
"""

from confidenceot.api import ConfidenceOT, m4_exact, m4_reversible
from confidenceot.calibration import (
    NullCalibrationResult,
    NullValidationRecord,
    calibrate_confidence_cost,
    rotation_null_costs,
    within_side_null_costs,
)
from confidenceot.cuda import CUDAUnavailableError, cuda_available, cuda_device_name
from confidenceot.preprocessing import (
    COSTS,
    NORMALISATIONS,
    CostMatrix,
    Preprocessing,
    Representation,
    equalise_depth,
    median_pair_scale,
    rank_value_encode,
    squared_euclidean,
    unit_rows,
)
from confidenceot.result import BinConfidence, ConfidenceOTResult

__all__ = [
    "ConfidenceOT",
    "COSTS",
    "CostMatrix",
    "NORMALISATIONS",
    "Preprocessing",
    "Representation",
    "ConfidenceOTResult",
    "BinConfidence",
    "CUDAUnavailableError",
    "cuda_available",
    "cuda_device_name",
    "m4_exact",
    "m4_reversible",
    "NullCalibrationResult",
    "NullValidationRecord",
    "calibrate_confidence_cost",
    "equalise_depth",
    "median_pair_scale",
    "rank_value_encode",
    "rotation_null_costs",
    "squared_euclidean",
    "unit_rows",
    "within_side_null_costs",
]
