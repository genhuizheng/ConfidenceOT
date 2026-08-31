"""ConfidenceOT: confidence-filtered optimal transport.

The public package exposes M4-E (exact) and M4-R (reversible) without placing
the proposed method under the ``traditional_ot`` name.  ``cellot`` and
``traditional_ot`` remain compatibility namespaces for existing experiments.
"""

from confidenceot.api import ConfidenceOT, m4_exact, m4_reversible
from confidenceot.calibration import (
    NullCalibrationResult,
    NullValidationRecord,
    calibrate_confidence_cost,
    rotation_null_costs,
)
from confidenceot.cuda import CUDAUnavailableError, cuda_available, cuda_device_name
from confidenceot.result import BinConfidence, ConfidenceOTResult

__all__ = [
    "ConfidenceOT",
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
    "rotation_null_costs",
]
