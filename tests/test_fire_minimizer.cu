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

__global__ void quarticGradientKernel(const int totalCoords, const double* positions, double* grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalCoords) {
    return;
  }

  const double wantVal = static_cast<double>(idx);
  const double diff    = positions[idx] - wantVal;
  grad[idx]            = 4.0 * diff * diff * diff;
}

double quarticEnergyAtIndex(const int idx, const double value) {
  const double wantVal = static_cast<double>(idx);
  const double diff    = value - wantVal;
  return diff * diff * diff * diff;
}

class FireMinimizerQuarticTest : public ::testing::Test {
 protected:
  void setUpSystems(const int seed = 1337) {
    atomStarts_   = {0, 3, 10, 12};
    numSystems_   = static_cast<int>(atomStarts_.size()) - 1;
    totalAtoms_   = atomStarts_.back();
    totalCoords_  = totalAtoms_ * kDim;
    hostPositions_.resize(totalCoords_);

    std::mt19937                    rng(seed);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (int i = 0; i < totalCoords_; ++i) {
      hostPositions_[i] = static_cast<double>(i) + dist(rng);
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
      const int totalCoords = totalCoords_;
      constexpr int blockSize = 128;
      const int numBlocks     = (totalCoords + blockSize - 1) / blockSize;
      quarticGradientKernel<<<numBlocks, blockSize>>>(totalCoords, positionsDevice_.data(), gradDevice_.data());
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
      const double diff = positions[i] - static_cast<double>(i);
      grad[i]           = 4.0 * diff * diff * diff;
    }
    return grad;
  }

  double computeEnergyHost(const std::vector<double>& positions) const {
    double energy = 0.0;
    for (int i = 0; i < totalCoords_; ++i) {
      energy += quarticEnergyAtIndex(i, positions[i]);
    }
    return energy;
  }

  std::vector<int>                    atomStarts_;
  int                                 numSystems_   = 0;
  int                                 totalAtoms_   = 0;
  int                                 totalCoords_  = 0;
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

  nvMolKit::FireBatchMinimizer minimizer(kDim);
  auto                         gradFunc  = gradientFunctor();
  auto                         energyFunc = [](const double*) {};

  const bool converged = minimizer.minimize(2000,
                                            1e-5,
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
  const double              finalEnergy    = computeEnergyHost(finalPositions);
  const std::vector<double> finalGradient  = computeGradientHost(finalPositions);

  for (int i = 0; i < totalCoords_; ++i) {
    const double want = static_cast<double>(i);
    EXPECT_NEAR(finalPositions[i], want, 1e-3) << "Mismatch at coordinate " << i;
  }

  const double maxAbsGrad = *std::max_element(finalGradient.begin(), finalGradient.end(), [](double a, double b) {
    return std::abs(a) < std::abs(b);
  });

  EXPECT_NEAR(finalEnergy, 0.0, 1e-6);
  EXPECT_LT(std::abs(maxAbsGrad), 1e-3);
}

