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
#include <set>
#include <type_traits>
#include <vector>

#include "butina.h"
#include "device.h"
#include "device_vector.h"
#include "host_vector.h"

using nvMolKit::AsyncDeviceVector;
using nvMolKit::detail::kAssignedAsSingletonSentinel;
using nvMolKit::detail::kMinLoopSizeForAssignment;

template <int N>
struct NeighborlistSize : std::integral_constant<int, N> {};

using NeighborlistSizes =
  ::testing::Types<NeighborlistSize<8>, NeighborlistSize<16>, NeighborlistSize<24>, NeighborlistSize<32>,
                   NeighborlistSize<64>, NeighborlistSize<128>>;

template <typename T>
class ButinaPruneFixture : public ::testing::Test {
 protected:
  static constexpr int   kNeighborlistMaxSize = T::value;
  nvMolKit::ScopedStream scopedStream_;
  cudaStream_t           stream() { return scopedStream_.stream(); }
};

TYPED_TEST_SUITE(ButinaPruneFixture, NeighborlistSizes);

// Test that prune kernel correctly removes assigned neighbors and compacts
TYPED_TEST(ButinaPruneFixture, PruneRemovesAssignedNeighbors) {
  constexpr int neighborlistMaxSize = TestFixture::kNeighborlistMaxSize;
  constexpr int numPoints           = 10;

  // Setup: Point 0 has neighbors [1, 2, 3, 4, 5] (count=6 including self)
  // After assigning point 2 to a cluster, prune should produce [1, 3, 4, 5] (count=5)

  AsyncDeviceVector<int> clusters(numPoints, this->stream());
  AsyncDeviceVector<int> clusterSizes(numPoints, this->stream());
  AsyncDeviceVector<int> neighborList(neighborlistMaxSize * numPoints, this->stream());

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
  cudaStreamSynchronize(this->stream());

  // Run prune kernel
  nvMolKit::detail::launchPruneNeighborlistKernel<neighborlistMaxSize>(toSpan(clusters),
                                                                       toSpan(clusterSizes),
                                                                       toSpan(neighborList),
                                                                       numPoints,
                                                                       this->stream());
  cudaStreamSynchronize(this->stream());

  // Check results
  clusterSizes.copyToHost(clusterSizesHost);
  neighborList.copyToHost(neighborListHost);
  clusters.copyToHost(clustersHost);
  cudaStreamSynchronize(this->stream());

  // Point 0: was 6, point 2 removed, should be 5
  EXPECT_EQ(clusterSizesHost[0], 5) << "Point 0 count should decrease by 1";

  // Point 1: was 3, point 2 removed, should be 2 (which is < 3, so might be marked singleton)
  // Actually count=2 means doublet, not singleton. Let's check the value.
  EXPECT_EQ(clusterSizesHost[1], 2) << "Point 1 count should decrease by 1";

  // Check compaction for point 0: first 5 entries should be valid, rest should be -1
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
TYPED_TEST(ButinaPruneFixture, PruneMarksSingletons) {
  constexpr int neighborlistMaxSize = TestFixture::kNeighborlistMaxSize;
  const int     numPoints           = 5;

  AsyncDeviceVector<int> clusters(numPoints, this->stream());
  AsyncDeviceVector<int> clusterSizes(numPoints, this->stream());
  AsyncDeviceVector<int> neighborList(neighborlistMaxSize * numPoints, this->stream());

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
  cudaStreamSynchronize(this->stream());

  nvMolKit::detail::launchPruneNeighborlistKernel<neighborlistMaxSize>(toSpan(clusters),
                                                                       toSpan(clusterSizes),
                                                                       toSpan(neighborList),
                                                                       numPoints,
                                                                       this->stream());
  cudaStreamSynchronize(this->stream());

  clusterSizes.copyToHost(clusterSizesHost);
  clusters.copyToHost(clustersHost);
  cudaStreamSynchronize(this->stream());

  // Point 0 should now have count=1 and be marked as singleton
  EXPECT_EQ(clusterSizesHost[0], 1) << "Point 0 should have count=1";
  EXPECT_EQ(clustersHost[0], kAssignedAsSingletonSentinel) << "Point 0 should be marked as singleton";
}

// Test that build kernel produces correct neighborlists
TYPED_TEST(ButinaPruneFixture, BuildNeighborlistProducesCorrectCounts) {
  constexpr int neighborlistMaxSize = TestFixture::kNeighborlistMaxSize;
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

  AsyncDeviceVector<uint8_t> hitMatrix(numPoints * numPoints, this->stream());
  AsyncDeviceVector<int>     clusters(numPoints, this->stream());
  AsyncDeviceVector<int>     clusterSizes(numPoints, this->stream());
  AsyncDeviceVector<int>     neighborList(neighborlistMaxSize * numPoints, this->stream());

  std::vector<int> clustersHost(numPoints, -1);
  hitMatrix.copyFromHost(hitMatrixHost);
  clusters.copyFromHost(clustersHost);
  clusterSizes.zero();
  neighborList.zero();
  cudaStreamSynchronize(this->stream());

  nvMolKit::detail::launchBuildNeighborlistKernel<neighborlistMaxSize>(toSpan(hitMatrix),
                                                                       toSpan(clusters),
                                                                       toSpan(clusterSizes),
                                                                       toSpan(neighborList),
                                                                       numPoints,
                                                                       this->stream());
  cudaStreamSynchronize(this->stream());

  std::vector<int> clusterSizesHost(numPoints);
  clusterSizes.copyToHost(clusterSizesHost);
  clusters.copyToHost(clustersHost);
  cudaStreamSynchronize(this->stream());

  EXPECT_EQ(clusterSizesHost[0], 3) << "Point 0 should have 3 neighbors";
  EXPECT_EQ(clusterSizesHost[1], 3) << "Point 1 should have 3 neighbors";
  EXPECT_EQ(clusterSizesHost[2], 2) << "Point 2 should have 2 neighbors";
  EXPECT_EQ(clusterSizesHost[3], 2) << "Point 3 should have 2 neighbors";
  EXPECT_EQ(clusterSizesHost[4], 1) << "Point 4 should have 1 neighbor (singleton)";

  // Point 4 should be marked as singleton
  EXPECT_EQ(clustersHost[4], kAssignedAsSingletonSentinel) << "Point 4 should be marked as singleton";
}

// Test argmax kernel
TYPED_TEST(ButinaPruneFixture, ArgMaxFindsMaximum) {
  const int numPoints = 10;

  AsyncDeviceVector<int>        values(numPoints, this->stream());
  nvMolKit::AsyncDevicePtr<int> outVal(0, this->stream());
  nvMolKit::AsyncDevicePtr<int> outIdx(-1, this->stream());

  std::vector<int> valuesHost = {3, 7, 2, 9, 5, 1, 8, 4, 6, 0};
  values.copyFromHost(valuesHost);
  cudaStreamSynchronize(this->stream());

  nvMolKit::detail::launchArgMaxKernel(toSpan(values), outVal.data(), outIdx.data(), this->stream());
  cudaStreamSynchronize(this->stream());

  int maxVal, maxIdx;
  cudaMemcpy(&maxVal, outVal.data(), sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&maxIdx, outIdx.data(), sizeof(int), cudaMemcpyDeviceToHost);

  EXPECT_EQ(maxVal, 9) << "Maximum value should be 9";
  EXPECT_EQ(maxIdx, 3) << "Maximum index should be 3";
}

// Test prune + build consistency: build, assign some, prune, compare with fresh build
TYPED_TEST(ButinaPruneFixture, PruneMatchesFreshBuild) {
  constexpr int neighborlistMaxSize = TestFixture::kNeighborlistMaxSize;
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

  AsyncDeviceVector<uint8_t> hitMatrix(numPoints * numPoints, this->stream());
  AsyncDeviceVector<int>     clusters1(numPoints, this->stream());
  AsyncDeviceVector<int>     clusterSizes1(numPoints, this->stream());
  AsyncDeviceVector<int>     neighborList1(neighborlistSize, this->stream());
  AsyncDeviceVector<int>     clusters2(numPoints, this->stream());
  AsyncDeviceVector<int>     clusterSizes2(numPoints, this->stream());
  AsyncDeviceVector<int>     neighborList2(neighborlistSize, this->stream());

  std::vector<int> clustersHost(numPoints, -1);
  hitMatrix.copyFromHost(hitMatrixHost);

  // Path 1: Build, assign point 0 and its neighbors, prune
  clusters1.copyFromHost(clustersHost);
  clusterSizes1.zero();
  neighborList1.zero();
  cudaStreamSynchronize(this->stream());

  nvMolKit::detail::launchBuildNeighborlistKernel<neighborlistMaxSize>(toSpan(hitMatrix),
                                                                       toSpan(clusters1),
                                                                       toSpan(clusterSizes1),
                                                                       toSpan(neighborList1),
                                                                       numPoints,
                                                                       this->stream());
  cudaStreamSynchronize(this->stream());

  // Assign cluster 0 to point 0 and all its hit neighbors
  std::vector<int> clusters1Host(numPoints);
  clusters1.copyToHost(clusters1Host);
  cudaStreamSynchronize(this->stream());

  for (int i = 0; i < numPoints; i++) {
    if (hitMatrixHost[0 * numPoints + i]) {
      clusters1Host[i] = 0;
    }
  }
  clusters1.copyFromHost(clusters1Host);
  cudaStreamSynchronize(this->stream());

  // Prune
  nvMolKit::detail::launchPruneNeighborlistKernel<neighborlistMaxSize>(toSpan(clusters1),
                                                                       toSpan(clusterSizes1),
                                                                       toSpan(neighborList1),
                                                                       numPoints,
                                                                       this->stream());
  cudaStreamSynchronize(this->stream());

  // Path 2: Build fresh with same assignments
  clusters2.copyFromHost(clusters1Host);
  clusterSizes2.zero();
  neighborList2.zero();
  cudaStreamSynchronize(this->stream());

  nvMolKit::detail::launchBuildNeighborlistKernel<neighborlistMaxSize>(toSpan(hitMatrix),
                                                                       toSpan(clusters2),
                                                                       toSpan(clusterSizes2),
                                                                       toSpan(neighborList2),
                                                                       numPoints,
                                                                       this->stream());
  cudaStreamSynchronize(this->stream());

  // Compare clusterSizes - they should match
  std::vector<int> clusterSizes1Host(numPoints);
  std::vector<int> clusterSizes2Host(numPoints);
  clusterSizes1.copyToHost(clusterSizes1Host);
  clusterSizes2.copyToHost(clusterSizes2Host);
  cudaStreamSynchronize(this->stream());

  for (int i = 0; i < numPoints; i++) {
    EXPECT_EQ(clusterSizes1Host[i], clusterSizes2Host[i])
      << "Cluster sizes should match for point " << i << " (assigned=" << clusters1Host[i] << ")";
  }
}

// Test with many neighbors - uses numPoints neighbors to stress larger sizes
using LargeNeighborlistSizes = ::testing::Types<NeighborlistSize<64>, NeighborlistSize<128>>;

template <typename T>
class ButinaPruneLargeFixture : public ::testing::Test {
 protected:
  static constexpr int   kNeighborlistMaxSize = T::value;
  nvMolKit::ScopedStream scopedStream_;
  cudaStream_t           stream() { return scopedStream_.stream(); }
};

TYPED_TEST_SUITE(ButinaPruneLargeFixture, LargeNeighborlistSizes);

TYPED_TEST(ButinaPruneLargeFixture, PruneLargeNeighborlist) {
  constexpr int neighborlistMaxSize = TestFixture::kNeighborlistMaxSize;
  const int     numPoints           = 100;

  AsyncDeviceVector<int> clusters(numPoints, this->stream());
  AsyncDeviceVector<int> clusterSizes(numPoints, this->stream());
  AsyncDeviceVector<int> neighborList(neighborlistMaxSize * numPoints, this->stream());

  std::vector<int> clustersHost(numPoints, -1);
  std::vector<int> clusterSizesHost(numPoints, 0);
  std::vector<int> neighborListHost(neighborlistMaxSize * numPoints, -1);

  // Point 0 has many neighbors (up to neighborlistMaxSize or numPoints, whichever is smaller)
  const int numNeighbors = std::min(neighborlistMaxSize, numPoints);
  clusterSizesHost[0]    = numNeighbors;
  neighborListHost[0]    = 0;  // self
  for (int i = 1; i < numNeighbors; i++) {
    neighborListHost[i] = i;
  }

  // Assign half of them to a cluster (even indices except 0)
  for (int i = 2; i < numNeighbors; i += 2) {
    clustersHost[i] = 0;
  }

  clusters.copyFromHost(clustersHost);
  clusterSizes.copyFromHost(clusterSizesHost);
  neighborList.copyFromHost(neighborListHost);
  cudaStreamSynchronize(this->stream());

  nvMolKit::detail::launchPruneNeighborlistKernel<neighborlistMaxSize>(toSpan(clusters),
                                                                       toSpan(clusterSizes),
                                                                       toSpan(neighborList),
                                                                       numPoints,
                                                                       this->stream());
  cudaStreamSynchronize(this->stream());

  clusterSizes.copyToHost(clusterSizesHost);
  neighborList.copyToHost(neighborListHost);
  clusters.copyToHost(clustersHost);
  cudaStreamSynchronize(this->stream());

  // Count expected remaining neighbors (odd indices + 0)
  int expectedCount = 0;
  for (int i = 0; i < numNeighbors; i++) {
    if (i == 0 || i % 2 == 1) {
      expectedCount++;
    }
  }

  EXPECT_EQ(clusterSizesHost[0], expectedCount) << "Point 0 count should match expected after pruning";

  // Verify all remaining neighbors are valid (not assigned to a real cluster)
  // Note: neighbors may be marked as singletons (kAssignedAsSingletonSentinel) if they have no
  // neighborlist set up, which is fine - we just check they're not assigned to a real cluster
  for (int i = 0; i < clusterSizesHost[0]; i++) {
    const int  neighbor   = neighborListHost[i];
    const int  clusterVal = clustersHost[neighbor];
    const bool unassigned = (clusterVal < 0) || (clusterVal == kAssignedAsSingletonSentinel);
    EXPECT_GE(neighbor, 0) << "Valid neighbor slot should have valid index";
    EXPECT_TRUE(unassigned) << "Remaining neighbor " << neighbor << " should be unassigned or singleton, got "
                            << clusterVal;
  }

  // Verify pruned slots are -1
  for (int i = clusterSizesHost[0]; i < neighborlistMaxSize; i++) {
    EXPECT_EQ(neighborListHost[i], -1) << "Pruned slots should be -1";
  }
}

TYPED_TEST(ButinaPruneLargeFixture, BuildLargeNeighborlist) {
  constexpr int neighborlistMaxSize = TestFixture::kNeighborlistMaxSize;
  const int     numPoints           = 100;

  // Create a hit matrix where point 0 is connected to many others
  std::vector<uint8_t> hitMatrixHost(numPoints * numPoints, 0);
  for (int i = 0; i < numPoints; i++) {
    hitMatrixHost[i * numPoints + i] = 1;  // diagonal
  }

  // Point 0 connected to points 1 through min(neighborlistMaxSize-1, numPoints-1)
  const int numNeighbors = std::min(neighborlistMaxSize, numPoints);
  for (int i = 1; i < numNeighbors; i++) {
    hitMatrixHost[0 * numPoints + i] = 1;
    hitMatrixHost[i * numPoints + 0] = 1;
  }

  AsyncDeviceVector<uint8_t> hitMatrix(numPoints * numPoints, this->stream());
  AsyncDeviceVector<int>     clusters(numPoints, this->stream());
  AsyncDeviceVector<int>     clusterSizes(numPoints, this->stream());
  AsyncDeviceVector<int>     neighborList(neighborlistMaxSize * numPoints, this->stream());

  std::vector<int> clustersHost(numPoints, -1);
  hitMatrix.copyFromHost(hitMatrixHost);
  clusters.copyFromHost(clustersHost);
  clusterSizes.zero();
  neighborList.zero();
  cudaStreamSynchronize(this->stream());

  nvMolKit::detail::launchBuildNeighborlistKernel<neighborlistMaxSize>(toSpan(hitMatrix),
                                                                       toSpan(clusters),
                                                                       toSpan(clusterSizes),
                                                                       toSpan(neighborList),
                                                                       numPoints,
                                                                       this->stream());
  cudaStreamSynchronize(this->stream());

  std::vector<int> clusterSizesHost(numPoints);
  std::vector<int> neighborListHost(neighborlistMaxSize * numPoints);
  clusterSizes.copyToHost(clusterSizesHost);
  neighborList.copyToHost(neighborListHost);
  cudaStreamSynchronize(this->stream());

  EXPECT_EQ(clusterSizesHost[0], numNeighbors) << "Point 0 should have " << numNeighbors << " neighbors";

  // Verify neighborlist contains expected neighbors
  std::set<int> foundNeighbors;
  for (int i = 0; i < clusterSizesHost[0]; i++) {
    foundNeighbors.insert(neighborListHost[i]);
  }
  EXPECT_EQ(foundNeighbors.size(), static_cast<size_t>(numNeighbors));
  EXPECT_TRUE(foundNeighbors.count(0) > 0) << "Point 0 should include itself";
}
