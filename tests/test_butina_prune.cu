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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "butina.h"
#include "device.h"
#include "device_vector.h"
#include "host_vector.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::detail::kAssignedAsSingletonSentinel;
using nvMolKit::detail::kMinLoopSizeForAssignment;

class ButinaPruneFixture : public ::testing::TestWithParam<int> {
 protected:
  nvMolKit::ScopedStream scopedStream_;
  cudaStream_t           stream() { return scopedStream_.stream(); }
};

// Test that prune kernel correctly removes assigned neighbors and compacts
TEST_P(ButinaPruneFixture, PruneRemovesAssignedNeighbors) {
  constexpr int neighborlistMaxSize = 8;
  constexpr int numPoints           = 10;

  // Setup: Point 0 has neighbors [1, 2, 3, 4, 5] (count=6 including self)
  // After assigning point 2 to a cluster, prune should produce [1, 3, 4, 5] (count=5)

  AsyncDeviceVector<int> clusters(numPoints, stream());
  AsyncDeviceVector<int> clusterSizes(numPoints, stream());
  AsyncDeviceVector<int> neighborList(neighborlistMaxSize * numPoints, stream());

  // Initialize all as unassigned
  std::vector<int> clustersHost(numPoints, -1);
  std::vector<int> clusterSizesHost(numPoints, 0);
  std::vector<int> neighborListHost(neighborlistMaxSize * numPoints, -1);

  // Point 0 has 5 neighbors (plus self = count of 6)
  clusterSizesHost[0] = 6;
  neighborListHost[0] = 0;  // self
  neighborListHost[1] = 1;
  neighborListHost[2] = 2;
  neighborListHost[3] = 3;
  neighborListHost[4] = 4;
  neighborListHost[5] = 5;

  // Point 1 has 3 neighbors
  clusterSizesHost[1]                       = 3;
  neighborListHost[neighborlistMaxSize + 0] = 1;
  neighborListHost[neighborlistMaxSize + 1] = 0;
  neighborListHost[neighborlistMaxSize + 2] = 2;

  // Now mark point 2 as assigned to cluster 0
  clustersHost[2] = 0;

  clusters.copyFromHost(clustersHost);
  clusterSizes.copyFromHost(clusterSizesHost);
  neighborList.copyFromHost(neighborListHost);
  cudaStreamSynchronize(stream());

  // Run prune kernel
  nvMolKit::detail::launchPruneNeighborlistKernel<neighborlistMaxSize>(toSpan(clusters),
                                                                       toSpan(clusterSizes),
                                                                       toSpan(neighborList),
                                                                       numPoints,
                                                                       stream());
  cudaStreamSynchronize(stream());

  // Check results
  clusterSizes.copyToHost(clusterSizesHost);
  neighborList.copyToHost(neighborListHost);
  clusters.copyToHost(clustersHost);
  cudaStreamSynchronize(stream());

  // Point 0: was 6, point 2 removed, should be 5
  EXPECT_EQ(clusterSizesHost[0], 5) << "Point 0 count should decrease by 1";

  // Point 1: was 3, point 2 removed, should be 2 (which is < 3, so might be marked singleton)
  // Actually count=2 means doublet, not singleton. Let's check the value.
  EXPECT_EQ(clusterSizesHost[1], 2) << "Point 1 count should decrease by 1";

  // Check compaction for point 0: neighbors should be [0, 1, 3, 4, 5, -1, -1, -1]
  std::vector<int> expectedPoint0 = {0, 1, 3, 4, 5, -1, -1, -1};
  for (int i = 0; i < neighborlistMaxSize; i++) {
    if (i < 5) {
      EXPECT_GE(neighborListHost[i], 0) << "First 5 entries should be valid neighbors";
    } else {
      EXPECT_EQ(neighborListHost[i], -1) << "Remaining entries should be -1";
    }
  }

  // Verify point 2 is not in point 0's neighborlist
  for (int i = 0; i < 5; i++) {
    EXPECT_NE(neighborListHost[i], 2) << "Point 2 should have been pruned from point 0's neighborlist";
  }
}

// Test that prune kernel marks singletons correctly
TEST_P(ButinaPruneFixture, PruneMarksSingletons) {
  constexpr int neighborlistMaxSize = 8;
  const int     numPoints           = 5;

  AsyncDeviceVector<int> clusters(numPoints, stream());
  AsyncDeviceVector<int> clusterSizes(numPoints, stream());
  AsyncDeviceVector<int> neighborList(neighborlistMaxSize * numPoints, stream());

  std::vector<int> clustersHost(numPoints, -1);
  std::vector<int> clusterSizesHost(numPoints, 0);
  std::vector<int> neighborListHost(neighborlistMaxSize * numPoints, -1);

  // Point 0 has 2 neighbors: itself and point 1
  clusterSizesHost[0] = 2;
  neighborListHost[0] = 0;
  neighborListHost[1] = 1;

  // Point 1 is assigned
  clustersHost[1] = 0;

  clusters.copyFromHost(clustersHost);
  clusterSizes.copyFromHost(clusterSizesHost);
  neighborList.copyFromHost(neighborListHost);
  cudaStreamSynchronize(stream());

  nvMolKit::detail::launchPruneNeighborlistKernel<neighborlistMaxSize>(toSpan(clusters),
                                                                       toSpan(clusterSizes),
                                                                       toSpan(neighborList),
                                                                       numPoints,
                                                                       stream());
  cudaStreamSynchronize(stream());

  clusterSizes.copyToHost(clusterSizesHost);
  clusters.copyToHost(clustersHost);
  cudaStreamSynchronize(stream());

  // Point 0 should now have count=1 and be marked as singleton
  EXPECT_EQ(clusterSizesHost[0], 1) << "Point 0 should have count=1";
  EXPECT_EQ(clustersHost[0], kAssignedAsSingletonSentinel) << "Point 0 should be marked as singleton";
}

// Test that build kernel produces correct neighborlists
TEST_P(ButinaPruneFixture, BuildNeighborlistProducesCorrectCounts) {
  constexpr int neighborlistMaxSize = 8;
  const int     numPoints           = 5;

  // Create a simple hit matrix where:
  // Point 0 neighbors: 0, 1, 2 (count=3)
  // Point 1 neighbors: 0, 1, 3 (count=3)
  // Point 2 neighbors: 0, 2 (count=2)
  // Point 3 neighbors: 1, 3 (count=2)
  // Point 4 neighbors: 4 (count=1, singleton)

  std::vector<uint8_t> hitMatrixHost(numPoints * numPoints, 0);
  auto                 setHit = [&](int i, int j) {
    hitMatrixHost[i * numPoints + j] = 1;
    hitMatrixHost[j * numPoints + i] = 1;
  };

  // Diagonal (self)
  for (int i = 0; i < numPoints; i++) {
    hitMatrixHost[i * numPoints + i] = 1;
  }

  setHit(0, 1);
  setHit(0, 2);
  setHit(1, 3);

  AsyncDeviceVector<uint8_t> hitMatrix(numPoints * numPoints, stream());
  AsyncDeviceVector<int>     clusters(numPoints, stream());
  AsyncDeviceVector<int>     clusterSizes(numPoints, stream());
  AsyncDeviceVector<int>     neighborList(neighborlistMaxSize * numPoints, stream());

  std::vector<int> clustersHost(numPoints, -1);
  hitMatrix.copyFromHost(hitMatrixHost);
  clusters.copyFromHost(clustersHost);
  clusterSizes.zero();
  neighborList.zero();
  cudaStreamSynchronize(stream());

  nvMolKit::detail::launchBuildNeighborlistKernel<neighborlistMaxSize>(toSpan(hitMatrix),
                                                                       toSpan(clusters),
                                                                       toSpan(clusterSizes),
                                                                       toSpan(neighborList),
                                                                       numPoints,
                                                                       stream());
  cudaStreamSynchronize(stream());

  std::vector<int> clusterSizesHost(numPoints);
  clusterSizes.copyToHost(clusterSizesHost);
  clusters.copyToHost(clustersHost);
  cudaStreamSynchronize(stream());

  EXPECT_EQ(clusterSizesHost[0], 3) << "Point 0 should have 3 neighbors";
  EXPECT_EQ(clusterSizesHost[1], 3) << "Point 1 should have 3 neighbors";
  EXPECT_EQ(clusterSizesHost[2], 2) << "Point 2 should have 2 neighbors";
  EXPECT_EQ(clusterSizesHost[3], 2) << "Point 3 should have 2 neighbors";
  EXPECT_EQ(clusterSizesHost[4], 1) << "Point 4 should have 1 neighbor (singleton)";

  // Point 4 should be marked as singleton
  EXPECT_EQ(clustersHost[4], kAssignedAsSingletonSentinel) << "Point 4 should be marked as singleton";
}

// Test argmax kernel
TEST_P(ButinaPruneFixture, ArgMaxFindsMaximum) {
  const int numPoints = 10;

  AsyncDeviceVector<int>        values(numPoints, stream());
  nvMolKit::AsyncDevicePtr<int> outVal(0, stream());
  nvMolKit::AsyncDevicePtr<int> outIdx(-1, stream());

  std::vector<int> valuesHost = {3, 7, 2, 9, 5, 1, 8, 4, 6, 0};
  values.copyFromHost(valuesHost);
  cudaStreamSynchronize(stream());

  nvMolKit::detail::launchArgMaxKernel(toSpan(values), outVal.data(), outIdx.data(), stream());
  cudaStreamSynchronize(stream());

  int maxVal, maxIdx;
  cudaMemcpy(&maxVal, outVal.data(), sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&maxIdx, outIdx.data(), sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(maxVal, 9) << "Maximum value should be 9";
  EXPECT_EQ(maxIdx, 3) << "Maximum index should be 3";
}

// Test prune + build consistency: build, assign some, prune, compare with fresh build
TEST_P(ButinaPruneFixture, PruneMatchesFreshBuild) {
  constexpr int neighborlistMaxSize = 8;
  constexpr int numPoints           = 20;
  constexpr int neighborlistSize    = neighborlistMaxSize * numPoints;

  // Create a random-ish hit matrix
  std::vector<uint8_t> hitMatrixHost(numPoints * numPoints, 0);
  for (int i = 0; i < numPoints; i++) {
    hitMatrixHost[i * numPoints + i] = 1;  // diagonal
    for (int j = i + 1; j < numPoints; j++) {
      if ((i + j) % 3 == 0) {
        hitMatrixHost[i * numPoints + j] = 1;
        hitMatrixHost[j * numPoints + i] = 1;
      }
    }
  }

  AsyncDeviceVector<uint8_t> hitMatrix(numPoints * numPoints, stream());
  AsyncDeviceVector<int>     clusters1(numPoints, stream());
  AsyncDeviceVector<int>     clusterSizes1(numPoints, stream());
  AsyncDeviceVector<int>     neighborList1(neighborlistSize, stream());
  AsyncDeviceVector<int>     clusters2(numPoints, stream());
  AsyncDeviceVector<int>     clusterSizes2(numPoints, stream());
  AsyncDeviceVector<int>     neighborList2(neighborlistSize, stream());

  std::vector<int> clustersHost(numPoints, -1);
  hitMatrix.copyFromHost(hitMatrixHost);

  // Path 1: Build, assign point 0 and its neighbors, prune
  clusters1.copyFromHost(clustersHost);
  clusterSizes1.zero();
  neighborList1.zero();
  cudaStreamSynchronize(stream());

  nvMolKit::detail::launchBuildNeighborlistKernel<neighborlistMaxSize>(toSpan(hitMatrix),
                                                                       toSpan(clusters1),
                                                                       toSpan(clusterSizes1),
                                                                       toSpan(neighborList1),
                                                                       numPoints,
                                                                       stream());
  cudaStreamSynchronize(stream());

  // Assign cluster 0 to point 0 and all its hit neighbors
  std::vector<int> clusters1Host(numPoints);
  clusters1.copyToHost(clusters1Host);
  cudaStreamSynchronize(stream());

  for (int i = 0; i < numPoints; i++) {
    if (hitMatrixHost[0 * numPoints + i]) {
      clusters1Host[i] = 0;
    }
  }
  clusters1.copyFromHost(clusters1Host);
  cudaStreamSynchronize(stream());

  // Prune
  nvMolKit::detail::launchPruneNeighborlistKernel<neighborlistMaxSize>(toSpan(clusters1),
                                                                       toSpan(clusterSizes1),
                                                                       toSpan(neighborList1),
                                                                       numPoints,
                                                                       stream());
  cudaStreamSynchronize(stream());

  // Path 2: Build fresh with same assignments
  clusters2.copyFromHost(clusters1Host);
  clusterSizes2.zero();
  neighborList2.zero();
  cudaStreamSynchronize(stream());

  nvMolKit::detail::launchBuildNeighborlistKernel<neighborlistMaxSize>(toSpan(hitMatrix),
                                                                       toSpan(clusters2),
                                                                       toSpan(clusterSizes2),
                                                                       toSpan(neighborList2),
                                                                       numPoints,
                                                                       stream());
  cudaStreamSynchronize(stream());

  // Compare clusterSizes - they should match
  std::vector<int> clusterSizes1Host(numPoints);
  std::vector<int> clusterSizes2Host(numPoints);
  clusterSizes1.copyToHost(clusterSizes1Host);
  clusterSizes2.copyToHost(clusterSizes2Host);
  cudaStreamSynchronize(stream());

  for (int i = 0; i < numPoints; i++) {
    EXPECT_EQ(clusterSizes1Host[i], clusterSizes2Host[i])
      << "Cluster sizes should match for point " << i << " (assigned=" << clusters1Host[i] << ")";
  }
}

INSTANTIATE_TEST_SUITE_P(ButinaPruneTest, ButinaPruneFixture, ::testing::Values(8, 16, 24, 32));
