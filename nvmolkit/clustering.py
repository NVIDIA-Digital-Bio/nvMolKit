import torch

from nvmolkit import _clustering
from nvmolkit._arrayHelpers import *  # noqa: F403
from nvmolkit.types import AsyncGpuResult

def butina(distance_matrix: AsyncGpuResult | torch.Tensor, cutoff: float) -> AsyncGpuResult:
    """
    Perform Butina clustering on a distance matrix.
    
    The Butina algorithm is a deterministic clustering method that groups items based
    on distance thresholds. It iteratively:
    1. Finds the item with the most neighbors within the cutoff distance
    2. Forms a cluster with that item and all its neighbors
    3. Removes clustered items from consideration
    4. Repeats until all items are clustered
    
    Args:
        distance_matrix: Square distance matrix of shape (N, N) where N is the number
                        of items. Can be an AsyncGpuResult or torch.Tensor on GPU.
        cutoff: Distance threshold for clustering. Items are neighbors if their
                distance is less than this cutoff.
    
    Returns:
        AsyncGpuResult containing cluster assignments as integers. Each element i
        contains the cluster ID for item i. Cluster IDs are sequential integers
        starting from 0.
    
    Note:
        The distance matrix should be symmetric and have zeros on the diagonal.
    """
    return AsyncGpuResult(_clustering.butina(distance_matrix.__cuda_array_interface__, cutoff))