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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "cuda_error_check.h"
#include "fire_minimizer.h"

namespace {

constexpr int kDim = 4;

__device__ int findSystemIndexForAtom(const int atomIdx, const int* atomStarts, const int numSystems) {
  for (int sysIdx = 0; sysIdx < numSystems; ++sysIdx) {
    if (atomIdx >= atomStarts[sysIdx] && atomIdx < atomStarts[sysIdx + 1]) {
      return sysIdx;
    }
  }
  return numSystems - 1;
}

__global__ void quarticGradientKernel(const int   totalCoords,
                                      const int*  atomStarts,
                                      const int   numSystems,
                                      const double* positions,
                                      double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalCoords) {
    return;
  }

  const int atomIdx    = idx / kDim;
  const int systemIdx  = findSystemIndexForAtom(atomIdx, atomStarts, numSystems);
  const int atomOffset  = atomIdx - atomStarts[systemIdx];
  const int coordOffset = atomOffset * kDim + (idx % kDim);
  const double wantVal  = static_cast<double>(coordOffset);
  const double diff     = positions[idx] - wantVal;
  grad[idx]            = 4.0 * diff * diff * diff;
}

double quarticEnergyAtIndex(const double value, const double wantVal) {
  const double diff = value - wantVal;
  return diff * diff * diff * diff;
}

class FireMinimizerQuarticTest : public ::testing::Test {
 protected:
  void setUpSystems(const int seed = 1337) {
    atomStarts_  = {0, 3, 10, 12};
    numSystems_  = static_cast<int>(atomStarts_.size()) - 1;
    totalAtoms_  = atomStarts_.back();
    totalCoords_ = totalAtoms_ * kDim;
    hostPositions_.resize(totalCoords_);

    std::mt19937                           rng(seed);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (int sysIdx = 0; sysIdx < numSystems_; ++sysIdx) {
      for (int atomIdx = atomStarts_[sysIdx]; atomIdx < atomStarts_[sysIdx + 1]; ++atomIdx) {
        const int atomOffset = atomIdx - atomStarts_[sysIdx];
        for (int dim = 0; dim < kDim; ++dim) {
          const int coordIdx        = atomIdx * kDim + dim;
          const int coordOffset     = atomOffset * kDim + dim;
          hostPositions_[coordIdx] = static_cast<double>(coordOffset) + dist(rng);
        }
      }
    }

    atomStartsDevice_.setFromVector(atomStarts_);
    positionsDevice_.setFromVector(hostPositions_);
    gradDevice_.resize(totalCoords_);
    gradDevice_.zero();
    energyOutsDevice_.resize(numSystems_);
    energyOutsDevice_.zero();
    energyBufferDevice_.resize(totalCoords_);
    energyBufferDevice_.zero();
  }

  std::function<void()> gradientFunctor() const {
    return [this]() {
      const int     totalCoords = totalCoords_;
      constexpr int blockSize   = 128;
      const int     numBlocks   = (totalCoords + blockSize - 1) / blockSize;
      quarticGradientKernel<<<numBlocks, blockSize>>>(totalCoords,
                                                      atomStartsDevice_.data(),
                                                      numSystems_,
                                                      positionsDevice_.data(),
                                                      gradDevice_.data());
      cudaCheckError(cudaGetLastError());
    };
  }

  std::vector<double> copyPositionsFromDevice() const {
    std::vector<double> positions(totalCoords_);
    positionsDevice_.copyToHost(positions);
    cudaCheckError(cudaDeviceSynchronize());
    return positions;
  }

  std::vector<double> computeGradientHost(const std::vector<double>& positions) const {
    std::vector<double> grad(totalCoords_);
    for (int i = 0; i < totalCoords_; ++i) {
      const double wantVal = expectedCoordinateValue(i);
      const double diff    = positions[i] - wantVal;
      grad[i]           = 4.0 * diff * diff * diff;
    }
    return grad;
  }

  double computeEnergyHost(const std::vector<double>& positions) const {
    double energy = 0.0;
    for (int i = 0; i < totalCoords_; ++i) {
      const double wantVal = expectedCoordinateValue(i);
      energy += quarticEnergyAtIndex(positions[i], wantVal);
    }
    return energy;
  }

  double computeSystemEnergyHost(const std::vector<double>& positions, int systemIdx) const {
    const int start = atomStarts_[systemIdx] * kDim;
    const int end   = atomStarts_[systemIdx + 1] * kDim;
    double    energy = 0.0;
    for (int i = start; i < end; ++i) {
      const double wantVal = expectedCoordinateValue(i);
      energy += quarticEnergyAtIndex(positions[i], wantVal);
    }
    return energy;
  }

  int findSystemIndexForAtomHost(const int atomIdx) const {
    auto it = std::upper_bound(atomStarts_.begin() + 1, atomStarts_.end(), atomIdx);
    return static_cast<int>(std::distance(atomStarts_.begin(), it) - 1);
  }

  double expectedCoordinateValue(const int coordIdx) const {
    const int atomIdx      = coordIdx / kDim;
    const int systemIdx    = findSystemIndexForAtomHost(atomIdx);
    const int atomOffset   = atomIdx - atomStarts_[systemIdx];
    const int coordOffset  = atomOffset * kDim + (coordIdx % kDim);
    return static_cast<double>(coordOffset);
  }

  std::vector<int>                    atomStarts_;
  int                                 numSystems_  = 0;
  int                                 totalAtoms_  = 0;
  int                                 totalCoords_ = 0;
  std::vector<double>                 hostPositions_;
  nvMolKit::AsyncDeviceVector<int>    atomStartsDevice_;
  nvMolKit::AsyncDeviceVector<double> positionsDevice_;
  nvMolKit::AsyncDeviceVector<double> gradDevice_;
  nvMolKit::AsyncDeviceVector<double> energyOutsDevice_;
  nvMolKit::AsyncDeviceVector<double> energyBufferDevice_;
};

}  // namespace

TEST_F(FireMinimizerQuarticTest, QuarticPotentialConvergesToTargets) {
  setUpSystems();
  std::vector<double> initialSystemEnergies(numSystems_);
  for (int sysIdx = 0; sysIdx < numSystems_; ++sysIdx) {
    initialSystemEnergies[sysIdx] = computeSystemEnergyHost(hostPositions_, sysIdx);
    EXPECT_GT(initialSystemEnergies[sysIdx], 1e-6) << "System " << sysIdx << " unexpectedly minimized";
  }

  nvMolKit::FireBatchMinimizer minimizer(kDim);
  auto                         gradFunc   = gradientFunctor();
  auto                         energyFunc = [](const double*) {};

  const bool converged = minimizer.minimize(2000,
                                            1e-6,
                                            atomStarts_,
                                            atomStartsDevice_,
                                            positionsDevice_,
                                            gradDevice_,
                                            energyOutsDevice_,
                                            energyBufferDevice_,
                                            energyFunc,
                                            gradFunc);

  EXPECT_TRUE(converged);

  const std::vector<double> finalPositions = copyPositionsFromDevice();
  const std::vector<double> finalGradient  = computeGradientHost(finalPositions);
  std::vector<double>       finalSystemEnergies(numSystems_);

  for (int i = 0; i < totalCoords_; ++i) {
    const double want = expectedCoordinateValue(i);
    EXPECT_NEAR(finalPositions[i], want, 1e-2) << "Mismatch at coordinate " << i;
  }

  const double maxAbsGrad = *std::max_element(finalGradient.begin(), finalGradient.end(), [](double a, double b) {
    return std::abs(a) < std::abs(b);
  });

  for (int sysIdx = 0; sysIdx < numSystems_; ++sysIdx) {
    finalSystemEnergies[sysIdx] = computeSystemEnergyHost(finalPositions, sysIdx);
    EXPECT_NEAR(finalSystemEnergies[sysIdx], 0.0, 1e-5) << "System " << sysIdx << " energy mismatch";
  }
  EXPECT_LT(std::abs(maxAbsGrad), 1e-6);
}

TEST_F(FireMinimizerQuarticTest, RespectsActiveSystemMask) {
  setUpSystems();
  const std::vector<double> initialPositions = copyPositionsFromDevice();
  std::vector<double>       initialSystemEnergies(numSystems_);
  for (int sysIdx = 0; sysIdx < numSystems_; ++sysIdx) {
    initialSystemEnergies[sysIdx] = computeSystemEnergyHost(initialPositions, sysIdx);
  }

  std::vector<uint8_t> activeMask = {1, 0, 1};
  ASSERT_EQ(static_cast<int>(activeMask.size()), numSystems_);

  nvMolKit::FireBatchMinimizer minimizer(kDim);
  minimizer.initialize(atomStarts_, activeMask.data());

  auto gradFunc = gradientFunctor();
  for (int iter = 0; iter < 2000; ++iter) {
    minimizer.step(1e-6, atomStartsDevice_, positionsDevice_, gradDevice_, gradFunc);
  }

  const std::vector<double> finalPositions = copyPositionsFromDevice();
  const std::vector<double> finalGrad      = computeGradientHost(finalPositions);

  const int inactiveSystemIdx = 1;
  const int inactiveStart     = atomStarts_[inactiveSystemIdx] * kDim;
  const int inactiveEnd       = atomStarts_[inactiveSystemIdx + 1] * kDim;

  for (int i = inactiveStart; i < inactiveEnd; ++i) {
    EXPECT_NEAR(finalPositions[i], initialPositions[i], 1e-9) << "Inactive coordinate changed at index " << i;
  }

  auto expectSystemConverged = [&](int systemIdx) {
    const int start = atomStarts_[systemIdx] * kDim;
    const int end   = atomStarts_[systemIdx + 1] * kDim;
    for (int i = start; i < end; ++i) {
      const double want = expectedCoordinateValue(i);
      EXPECT_NEAR(finalPositions[i], want, 1e-2) << "Active coordinate mismatch at index " << i;
      EXPECT_LT(std::abs(finalGrad[i]), 1e-3) << "Active gradient too large at index " << i;
    }
    EXPECT_NEAR(computeSystemEnergyHost(finalPositions, systemIdx), 0.0, 1e-5);
  };

  expectSystemConverged(0);
  expectSystemConverged(2);

  const double inactiveEnergy = computeSystemEnergyHost(finalPositions, inactiveSystemIdx);
  EXPECT_NEAR(inactiveEnergy, initialSystemEnergies[inactiveSystemIdx], 1e-6);
  EXPECT_GT(inactiveEnergy, 1.0);
}
