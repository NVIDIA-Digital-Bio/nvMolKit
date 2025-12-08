# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Contains GPU-accelerated Butina clustering implementation. """

import torch

from nvmolkit import _clustering
from nvmolkit._arrayHelpers import *  # noqa: F403
from nvmolkit.types import AsyncGpuResult

def butina(distance_matrix: AsyncGpuResult | torch.Tensor, cutoff: float, enforce_strict_indexing: bool = True) -> AsyncGpuResult:
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
        enforce_strict_indexing: If True, cluster IDs are assigned in strict largest-first
                                order (slower but deterministic ordering). If False, allows
                                parallel assignment for better performance (faster but
                                non-deterministic cluster ID ordering). Clusters will be the
                                same but the cluster IDs will not be in the same order and larger
                                clusters may have higher IDs.
    
    Returns:
        AsyncGpuResult containing cluster assignments as integers. Each element i
        contains the cluster ID for item i. Cluster IDs are sequential integers
        starting from 0.
    
    Note:
        The distance matrix should be symmetric and have zeros on the diagonal.
    """
    return AsyncGpuResult(_clustering.butina(distance_matrix.__cuda_array_interface__, cutoff, enforce_strict_indexing))