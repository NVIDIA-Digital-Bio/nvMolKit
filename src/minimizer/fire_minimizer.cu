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

#include <cub/cub.cuh>
#include <numeric>

#include "fire_minimizer.h"
#include "nvtx.h"

namespace nvMolKit {

namespace {
// TODO - consolidate this
template <typename T> __global__ void setAllKernel(const int numElements, T value, T* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}
template <typename T> void setAll(AsyncDeviceVector<T>& vec, const T& value) {
  const int          numElements = vec.size();
  const cudaStream_t stream      = vec.stream();
  if (numElements == 0) {
    return;
  }
  constexpr int blockSize = 128;
  const int     numBlocks = (numElements + blockSize - 1) / blockSize;
  setAllKernel<<<numBlocks, blockSize, 0, stream>>>(numElements, value, vec.data());
  cudaCheckError(cudaGetLastError());
}

template <typename T>
__device__ __forceinline__ cuda::std::span<T> getSystemSpan(const cuda::std::span<T>         data,
                                                            const cuda::std::span<const int> atomStarts,
                                                            const int                        sysIdx,
                                                            const int                        dataDim) {
  return data.subspan(atomStarts[sysIdx] * dataDim, (atomStarts[sysIdx + 1] - atomStarts[sysIdx]) * dataDim);
}

constexpr int   updatePowerBlockSize = 256;
__global__ void fireV1Kernel(const cuda::std::span<const int> atomStarts,
                             const cuda::std::span<double>    x,
                             const cuda::std::span<double>    v,
                             const cuda::std::span<double>    f,
                             const int                        dataDim,
                             const cuda::std::span<double>    alphas,
                             const cuda::std::span<double>    dt,
                             const cuda::std::span<int>       numStepsWithPositivePower,
                             const int                        positiveStepIncrementDelay,
                             const double                     dtIncrementFactor,
                             const double                     dtDecrementFactor,
                             const double                     maxDt,
                             const double                     alphaStart,
                             const double                     alphaDecrementFactor,
                             const double                     gradTol,
                             uint8_t*                         activeSystems) {
  namespace cg     = cooperative_groups;
  auto      block  = cg::this_thread_block();
  const int sysIdx = blockIdx.x;
  if (activeSystems != nullptr && activeSystems[sysIdx] == 0) {
    return;
  }
  __shared__ bool hadNegativePowerShared[1];
  __shared__ bool metConvergenceCriteria[1];
  if (block.thread_rank() == 0) {
    *hadNegativePowerShared = false;
    *metConvergenceCriteria = false;
  }

  // Compute v * F power.
  const auto   vSys  = getSystemSpan(v, atomStarts, sysIdx, dataDim);
  const auto   fSys  = getSystemSpan(f, atomStarts, sysIdx, dataDim);
  const double alpha = alphas[sysIdx];

  // TODO consolidate dot product implementations.
  // TODO this is just zero on step 0, so could be skipped.
  double power   = 0.0;
  double maxGrad = 0.0;
  for (int i = block.thread_rank(); i < vSys.size(); i += updatePowerBlockSize) {
    power += vSys[i] * -fSys[i];
    maxGrad = fmax(maxGrad, std::abs(fSys[i]));
  }

  using BlockReduce = cub::BlockReduce<double, updatePowerBlockSize>;
  __shared__ BlockReduce::TempStorage tempStorage;
  const double                        powerSum = BlockReduce(tempStorage).Sum(power);
  block.sync();  // To reuse the temp storage.
  const double maxGradReduced = BlockReduce(tempStorage).Reduce(maxGrad, cuda::maximum<>{});

  // -----------------------------------------------------------------
  // Update counting vars, alphas and dt based on powerSum.
  // This set of operations is per system, so only do it on one thread.
  // -----------------------------------------------------------------
  if (block.thread_rank() == 0) {
    if (maxGradReduced <= gradTol) {
      // printf("Converged system %d with maxGrad %f <= %f\n", sysIdx, maxGradReduced, gradTol);
      *metConvergenceCriteria = true;
    }
    if (powerSum >= 0.0) {
      const int numStepsPositive        = numStepsWithPositivePower[sysIdx] + 1;
      // Equivalent to numStepsPositive++ but we saved the new value locally too.
      numStepsWithPositivePower[sysIdx] = numStepsPositive;
      if (numStepsPositive > positiveStepIncrementDelay) {
        alphas[sysIdx] = alpha * alphaDecrementFactor;
        dt[sysIdx]     = fmin(dtIncrementFactor * dt[sysIdx], maxDt);
      }
    } else {
      *hadNegativePowerShared           = true;
      numStepsWithPositivePower[sysIdx] = 0;
      alphas[sysIdx]                    = alphaStart;
      dt[sysIdx] *= dtDecrementFactor;
    }
    // printf("System %d: power=%f, alpha=%f, dt=%f, numPosSteps=%d maxGrad%f\n",
    //        sysIdx,
    //        powerSum,
    //        alphas[sysIdx],
    //        dt[sysIdx],
    //        numStepsWithPositivePower[sysIdx],
    //        maxGradReduced);
  }
  // END per system compute ^^^, all threads now active again (if they were before).
  block.sync();
  if (*metConvergenceCriteria) {
    if (block.thread_rank() == 0 && activeSystems != nullptr) {
      activeSystems[sysIdx] = 0;
    }
    return;
  }
  if (*hadNegativePowerShared) {
    // Reset case.
    for (int i = block.thread_rank(); i < vSys.size(); i += updatePowerBlockSize) {
      vSys[i] = 0.0;
    }
  } else {
    // Note we're using the non-updated alpha that was taken before the increment.
    double vDot = 0.0;
    for (int i = block.thread_rank(); i < vSys.size(); i += updatePowerBlockSize) {
      vDot += vSys[i] * vSys[i];
    }
    const double vDotSum = BlockReduce(tempStorage).Sum(vDot);
    block.sync();  // To reuse the temp storage.
    double fDot = 0.0;
    for (int i = block.thread_rank(); i < vSys.size(); i += updatePowerBlockSize) {
      fDot += fSys[i] * fSys[i];
    }
    const double fDotSum      = BlockReduce(tempStorage).Sum(fDot);
    const double constFactor1 = (1.0 - alpha);
    const double constFactor2 = alpha * sqrt(vDotSum) / sqrt(fDotSum);
    for (int i = block.thread_rank(); i < vSys.size(); i += updatePowerBlockSize) {
      // v = (1-alpha)*v + alpha* v_norm * F_unitvec
      vSys[i] = constFactor1 * vSys[i] + constFactor2 * -fSys[i];
    }
  }

  // Update V with force/dt with or without above mixing
  for (int i = block.thread_rank(); i < vSys.size(); i += updatePowerBlockSize) {
    vSys[i] += -fSys[i] * dt[sysIdx];
  }

  // Now integrate positions
  const auto xSys = getSystemSpan(x, atomStarts, sysIdx, dataDim);
  for (int i = block.thread_rank(); i < xSys.size(); i += updatePowerBlockSize) {
    xSys[i] += vSys[i] * dt[sysIdx];
  }
}

}  // namespace

FireBatchMinimizer::FireBatchMinimizer(const int dataDim, const FireOptions& options, cudaStream_t stream)
    : dataDim_(dataDim),
      fireOptions_(options),
      stream_(stream) {
  velocities_.setStream(stream_);
  prevVelocities_.setStream(stream_);
  statuses_.setStream(stream_);
  dt_.setStream(stream_);
  alpha_.setStream(stream_);
  allSystemIndices_.setStream(stream_);
  activeSystemIndices_.setStream(stream_);
  numStepsWithNegativePower_.setStream(stream_);
  numStepsWithPositivePower_.setStream(stream_);
  countUnfinished_.setStream(stream_);
  countTempStorage_.setStream(stream_);
  powers_.setStream(stream_);
  loopStatusHost_.resize(1);
  loopStatusHost_[0] = 0;
}

void FireBatchMinimizer::fireV1(const double                  gradTol,
                                const AsyncDeviceVector<int>& atomStarts,
                                AsyncDeviceVector<double>&    positions,
                                AsyncDeviceVector<double>&    grad) {
  const int numSystems = atomStarts.size() - 1;
  fireV1Kernel<<<numSystems, updatePowerBlockSize, 0, stream_>>>(toSpan(atomStarts),
                                                                 toSpan(positions),
                                                                 toSpan(velocities_),
                                                                 toSpan(grad),
                                                                 dataDim_,
                                                                 toSpan(alpha_),
                                                                 toSpan(dt_),
                                                                 toSpan(numStepsWithPositivePower_),
                                                                 fireOptions_.nMinForIncrease,
                                                                 fireOptions_.timeStepIncrement,
                                                                 fireOptions_.timeStepDecrement,
                                                                 fireOptions_.dtMax,
                                                                 fireOptions_.alphaInit,
                                                                 fireOptions_.alphaDecrement,
                                                                 gradTol,
                                                                 statuses_.data());
  cudaCheckError(cudaGetLastError());
}

void FireBatchMinimizer::initialize(const std::vector<int>& atomStartsHost, const uint8_t* activeSystems) {
  const int totalAtoms = atomStartsHost.back();
  const int numSystems = atomStartsHost.size() - 1;

  // Resize datadim * N buffers.
  velocities_.resize(totalAtoms * dataDim_);
  prevVelocities_.resize(totalAtoms * dataDim_);
  velocities_.zero();
  prevVelocities_.zero();

  // Resize and set per-system buffers.
  statuses_.resize(numSystems);
  if (activeSystems != nullptr) {
    // Copy activeThisStage to statuses_ with type conversion
    cudaMemcpyAsync(statuses_.data(), activeSystems, numSystems * sizeof(uint8_t), cudaMemcpyDefault, stream_);
  } else {
    setAll(statuses_, static_cast<uint8_t>(1));
  }

  // Note that if activeSystem above has inactive systems, they won't be pruned until the end of
  // the first step, but the actual minimization won't run on step 0 due to in-kernel checks.
  activeSystemIndices_.resize(numSystems);
  allSystemIndices_.resize(numSystems);
  std::vector<int> activeSystemIndicesHost(numSystems);
  std::iota(activeSystemIndicesHost.begin(), activeSystemIndicesHost.end(), 0);
  allSystemIndices_.setFromVector(activeSystemIndicesHost);
  activeSystemIndices_.setFromVector(activeSystemIndicesHost);
  powers_.resize(numSystems);
  powers_.zero();
  numStepsWithNegativePower_.resize(numSystems);
  numStepsWithNegativePower_.zero();
  numStepsWithPositivePower_.resize(numSystems);
  numStepsWithPositivePower_.zero();
  alpha_.resize(numSystems);
  setAll(alpha_, fireOptions_.alphaInit);
  dt_.resize(numSystems);
  setAll(dt_, fireOptions_.dtInit);

  // Compute CUB temp storage requirements and allocate.
  size_t tempStorageBytes = 0;
  cudaCheckError(cub::DeviceSelect::Flagged(nullptr,
                                            tempStorageBytes,
                                            allSystemIndices_.data(),
                                            countTempStorage_.data(),
                                            activeSystemIndices_.data(),
                                            countUnfinished_.data(),
                                            allSystemIndices_.size(),
                                            stream_));

  if (tempStorageBytes > countTempStorage_.size()) {
    countTempStorage_.resize(tempStorageBytes);
  }
}

bool FireBatchMinimizer::step(const double                  gradTol,
                              const AsyncDeviceVector<int>& atomStarts,
                              AsyncDeviceVector<double>&    positions,
                              AsyncDeviceVector<double>&    grad,
                              const GradFunctor&            gFunc) {
  const int numSystems = atomStarts.size() - 1;
  gFunc();
  fireV1(gradTol, atomStarts, positions, grad);
  const int numFinished = compactAndCountConverged();
  return numFinished == numSystems;
}

bool FireBatchMinimizer::minimize(const int                     numIters,
                                  const double                  gradTol,
                                  const std::vector<int>&       atomStartsHost,
                                  const AsyncDeviceVector<int>& atomStarts,
                                  AsyncDeviceVector<double>&    positions,
                                  AsyncDeviceVector<double>&    grad,
                                  AsyncDeviceVector<double>&    energyOuts,
                                  AsyncDeviceVector<double>&    energyBuffer,
                                  EnergyFunctor                 eFunc,
                                  const GradFunctor             gFunc,
                                  const uint8_t*                activeThisStage) {
  initialize(atomStartsHost);

  for (int i = 0; i < numIters; ++i) {
    if (step(gradTol, atomStarts, positions, grad, gFunc)) {
      return true;
    }
  }
  return false;
}

int FireBatchMinimizer::compactAndCountConverged() {
  const ScopedNvtxRange fireCompact("FireBatchMinimizer::compactAndCountConverged");
  size_t                storageBytes = countTempStorage_.size();
  cudaCheckError(cub::DeviceSelect::Flagged(countTempStorage_.data(),
                                            storageBytes,
                                            allSystemIndices_.data(),
                                            statuses_.data(),
                                            activeSystemIndices_.data(),
                                            countUnfinished_.data(),
                                            allSystemIndices_.size(),
                                            stream_));

  int& unfinishedHost = loopStatusHost_[0];
  countUnfinished_.get(unfinishedHost);
  cudaStreamSynchronize(stream_);
  return allSystemIndices_.size() - unfinishedHost;
}

}  // namespace nvMolKit
