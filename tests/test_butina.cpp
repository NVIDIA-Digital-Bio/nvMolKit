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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "butina.h"
#include "device.h"
#include "host_vector.h"

using nvMolKit::AsyncDeviceVector;

namespace {

std::vector<double> makeSymmetricDifferenceMatrix(const int nPts, std::mt19937 &rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<double> distances(nPts * nPts, 0.0);
  for (int row = 0; row < nPts; ++row) {
    // row + 1 to ignore diagonal.
    for (int col = row + 1; col < nPts; ++col) {
      const double value = dist(rng);
      distances[(row * nPts) + col] = value;
      distances[(col * nPts) + row] = value;
    }
  }
  return distances;
}

std::vector<uint8_t> makeAdjacency(const std::vector<double> &distances, int nPts, double cutoff) {
  std::vector<uint8_t> adjacency(distances.size(), 0);
  for (size_t idx = 0; idx < distances.size(); ++idx) {
    adjacency[idx] = distances[idx] < cutoff ? 1U : 0U;
  }
  return adjacency;
}

std::vector<int> runButina(const std::vector<double> &distances, const int nPts, const double cutoff, cudaStream_t stream) {
  AsyncDeviceVector<double> distancesDev(distances.size(), stream);
  AsyncDeviceVector<int>    resultDev(nPts, stream);
  distancesDev.copyFromHost(distances);
  nvMolKit::butinaGpu(toSpan(distancesDev), toSpan(resultDev), cutoff, stream);
  std::vector<int> got(nPts);
  resultDev.copyToHost(got);
  cudaStreamSynchronize(stream);
  return got;
}

void checkButinaCorrectness(const std::vector<uint8_t> &adjacency, const std::vector<int> &labels) {
  const int nPts = static_cast<int>(labels.size());
  ASSERT_EQ(adjacency.size(), static_cast<size_t>(nPts) * static_cast<size_t>(nPts));

  std::vector<uint8_t> working = adjacency;
  std::vector<int>     counts(nPts, 0);
  for (int row = 0; row < nPts; ++row) {
    const size_t base = static_cast<size_t>(row) * nPts;
    for (int col = 0; col < nPts; ++col) {
      if (working[base + col] != 0U) {
        ++counts[row];
      }
    }
  }

  std::vector<bool> seen(nPts, false);
  int               seenCount = 0;

  const int maxLabelId = *std::ranges::max_element(labels.begin(), labels.end());
  for (int label = 0; label <= maxLabelId; ++label) {
    std::vector<int> cluster;
    cluster.reserve(nPts);
    for (int idx = 0; idx < nPts; ++idx) {
      if (labels[idx] == label) {
        cluster.push_back(idx);
      }
    }
    const auto clusterSize = static_cast<int>(cluster.size());
    ASSERT_GT(clusterSize, 0) << "Cluster ID " << label << " has no members";

    const int maxCount = *std::ranges::max_element(counts.begin(), counts.end());
    ASSERT_EQ(clusterSize, maxCount);

    for (const int member : cluster) {
      ASSERT_FALSE(seen[member]);
    }

    for (const int member : cluster) {
      seen[member] = true;
      const size_t rowBase = static_cast<size_t>(member) * nPts;
      for (int col = 0; col < nPts; ++col) {
        const size_t idx       = rowBase + col;
        const size_t mirrorIdx = static_cast<size_t>(col) * nPts + member;
        if (working[idx] != 0U) {
          working[idx] = 0U;
          --counts[member];
        }
        if (working[mirrorIdx] != 0U) {
          working[mirrorIdx] = 0U;
          --counts[col];
        }
      }
      counts[member] = 0;
    }
    seenCount += clusterSize;
  }

  ASSERT_EQ(seenCount, nPts);
}

} // namespace


TEST(ButinaClusterTest, HandlesSinglePoint) {
  constexpr int          nPts   = 1;
  constexpr double       cutoff = 0.2;
  nvMolKit::ScopedStream const scopedStream;
  cudaStream_t           stream = scopedStream.stream();

  AsyncDeviceVector<double> distancesDev(nPts * nPts, stream);
  AsyncDeviceVector<int>    resultDev(nPts, stream);
  distancesDev.copyFromHost(std::vector<double>{0.0});

  nvMolKit::butinaGpu(toSpan(distancesDev), toSpan(resultDev), cutoff, stream);
  std::vector<int> got(nPts);
  resultDev.copyToHost(got);
  cudaStreamSynchronize(stream);
  EXPECT_THAT(got, ::testing::ElementsAre(0));
}

class ButinaClusterTestFixture : public ::testing::TestWithParam<int> {};
TEST_P(ButinaClusterTestFixture, ClusteringMatchesReference) {
  nvMolKit::ScopedStream const scopedStream;
  cudaStream_t                 stream = scopedStream.stream();
  std::mt19937                 rng(42);

  const int nPts = GetParam();
  constexpr double cutoff = 0.1;
  const auto                distances = makeSymmetricDifferenceMatrix(nPts, rng);
  const auto                adjacency = makeAdjacency(distances, nPts, cutoff);
  const std::vector<int>    labels    = runButina(distances, nPts, cutoff, stream);
  SCOPED_TRACE(::testing::Message() << "nPts=" << nPts);
  checkButinaCorrectness(adjacency, labels);
}

INSTANTIATE_TEST_SUITE_P(ButinaClusterTest, ButinaClusterTestFixture, ::testing::Values(1, 10, 100, 1000));


TEST(ButinaClusterEdgeTest, EdgeOneCluster) {
  constexpr int          nPts   = 10;
  constexpr double       cutoff = 100.0;
  nvMolKit::ScopedStream const scopedStream;
  cudaStream_t           stream = scopedStream.stream();

  std::vector<double> distances(static_cast<size_t>(nPts) * nPts, 0.5);
  for (int i = 0; i < nPts; ++i) {
    distances[static_cast<size_t>(i) * nPts + i] = 0.0;
  }

  const std::vector<int> labels = runButina(distances, nPts, cutoff, stream);
  EXPECT_THAT(labels, ::testing::Each(0));
}

TEST(ButinaClusterEdgeTest, EdgeNClusters) {
  constexpr int          nPts   = 10;
  constexpr double       cutoff = 1e-8;
  nvMolKit::ScopedStream const scopedStream;
  cudaStream_t           stream = scopedStream.stream();

  std::vector<double> distances(static_cast<size_t>(nPts) * nPts, 1.0);
  for (int i = 0; i < nPts; ++i) {
    distances[static_cast<size_t>(i) * nPts + i] = 0.0;
  }

  const std::vector<int> labels = runButina(distances, nPts, cutoff, stream);
  std::vector<int>       sorted = labels;
  std::ranges::sort(sorted);
  std::vector<int> want(nPts);
  std::iota(want.begin(), want.end(), 0);
  EXPECT_THAT(sorted, ::testing::ElementsAreArray(want));
}

