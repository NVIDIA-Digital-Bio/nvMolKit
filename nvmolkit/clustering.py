import torch

from nvmolkit import _clustering
from nvmolkit._arrayHelpers import *  # noqa: F403
from nvmolkit.types import AsyncGpuResult

def butina(distance_matrix: AsyncGpuResult | torch.Tensor, cutoff: float) -> AsyncGpuResult:
    return AsyncGpuResult(_clustering.butina(distance_matrix.__cuda_array_interface__, cutoff))