// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NVMOLKIT_BUTINA_H
#define NVMOLKIT_BUTINA_H

#include "device_vector.h"
namespace nvMolKit {

/**
 * @brief Perform Butina clustering on a distance matrix with automatic thresholding.
 *
 * This is a convenience wrapper that converts the distance matrix into a binary hit
 * matrix by thresholding at the specified cutoff, then performs Butina clustering.
 * The Butina algorithm is a deterministic clustering method that iteratively selects
 * the item with the most unclustered neighbors and forms clusters.
 *
 * @param distanceMatrix Square distance matrix of size NxN where distanceMatrix[i*N+j]
 *                       contains the distance between items i and j.
 * @param clusters Output array of size N. Each element will contain the cluster ID for
 *                 that item. Modified in-place.
 * @param cutoff Distance threshold for clustering. Items with distance < cutoff are
 *               considered neighbors.
 * @param stream CUDA stream to execute operations on. Defaults to stream 0.
 */
void butinaGpu(cuda::std::span<const double> distanceMatrix,
               cuda::std::span<int>          clusters,
               double                        cutoff,
               cudaStream_t                  stream = nullptr);

/**
 * @brief Perform Butina clustering on a precomputed hit matrix.
 *
 * This is the core GPU implementation of the Butina clustering algorithm. It takes
 * a binary hit matrix where element (i,j) indicates whether items i and j are neighbors.
 * The algorithm iteratively:
 * 1. Counts cluster sizes for each unclustered item
 * 2. Selects the item with the largest cluster
 * 3. Assigns all neighbors to that cluster
 * 4. Repeats until clusters are too small (< 3 items)
 * 5. Pairs doublets (2-item clusters)
 * 6. Assigns singleton clusters
 *
 * @param hitMatrix Binary matrix of size NxN where hitMatrix[i*N+j] = 1 if items i and j
 *                  are neighbors (distance < cutoff), 0 otherwise.
 * @param clusters Output array of size N. Each element will contain the cluster ID for
 *                 that item. Modified in-place.
 * @param stream CUDA stream to execute operations on. Defaults to stream 0.
 */
void butinaGpu(cuda::std::span<const uint8_t> hitMatrix, cuda::std::span<int> clusters, cudaStream_t stream = nullptr);
}  // namespace nvMolKit

#endif  // NVMOLKIT_BUTINA_H