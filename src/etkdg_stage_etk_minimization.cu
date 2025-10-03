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

#include "dist_geom_flattened_builder.h"
#include "etkdg_stage_etk_minimization.h"
#include "minimizer/bfgs_minimize.h"

namespace nvMolKit {
namespace detail {

constexpr int dim = 4;

namespace {

// TODO: Only run on active systems.
__global__ void updateReferencePositionsKernel(const int      numTerms,
                                               const double*  refPos,
                                               const int*     idx1,
                                               const int*     idx2,
                                               double*        lowerBound,
                                               double*        upperBound,
                                               const uint8_t* isImproperConstrainedTerm = nullptr) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTerms) {
    if (isImproperConstrainedTerm != nullptr && isImproperConstrainedTerm[idx]) {
      // Skip improper constraints.
      return;
    }
    const int    i1  = idx1[idx];
    const int    i2  = idx2[idx];
    const double p1x = refPos[dim * i1];
    const double p1y = refPos[dim * i1 + 1];
    const double p1z = refPos[dim * i1 + 2];
    const double p2x = refPos[dim * i2];
    const double p2y = refPos[dim * i2 + 1];
    const double p2z = refPos[dim * i2 + 2];

    // For long distance, the constraint can be tighter. Get it from the previous bounds rather than constants.
    const double lowerBoundValue = lowerBound[idx];
    const double upperBoundValue = upperBound[idx];
    const double boundDelta      = (upperBoundValue - lowerBoundValue) / 2.0;

    const double dist = sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y) + (p1z - p2z) * (p1z - p2z));

    lowerBound[idx] = dist - boundDelta;
    upperBound[idx] = dist + boundDelta;
  }
}

__global__ void planarToleranceCheck(const int      numSystems,
                                     const double*  energies,
                                     const int*     numImpropers,
                                     const uint8_t* activeThisStage,
                                     uint8_t*       failedThisStage) {
  const int sysIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (sysIdx >= numSystems) {
    return;
  }
  if (!activeThisStage[sysIdx]) {
    return;  // Skip inactive systems.
  }
  const int        numTermsForSystem = numImpropers[sysIdx];
  constexpr double toleranceFactor   = 0.7;
  const double     tolerance         = toleranceFactor * numTermsForSystem;
  const double     e                 = energies[sysIdx];

  if (e > tolerance) {
    // If the energy is too high, mark the system as failed.
    failedThisStage[sysIdx] = 1;
  }
}

}  // namespace

ETKMinimizationStage::ETKMinimizationStage(const std::vector<const RDKit::ROMol*>&     mols,
                                           const std::vector<EmbedArgs>&               eargs,
                                           const RDKit::DGeomHelpers::EmbedParameters& embedParam,
                                           const ETKDGContext&                         ctx,
                                           cudaStream_t                                stream)
    : embedParam_(embedParam),
      stream_(stream) {
  setStreams(molSystemDevice, stream);

  const int totalNumAtoms = ctx.systemHost.atomStarts.back();

  std::vector<double> positions(totalNumAtoms * dim, 0.0);
  for (size_t i = 0; i < mols.size(); ++i) {
    if (eargs[i].dim != 4) {
      throw std::runtime_error("ETKDG minimization stage only supports 4D coordinates");
    }
    const auto& etkdgDetails = eargs[i].etkdgDetails;
    const auto& mmat         = eargs[i].mmat;
    // Set up GPU system. NOTE: Regardless of 3 or 4D system, the setup uses 3D.
    // Note if we actually cared about the positions for this setup, the 3D/4D stride would be off, but
    // we override the reference positions at execute time.
    // TODO: Fix 3D/4D stride issue anyway for clarity.
    auto        ffParams     = nvMolKit::DistGeom::construct3DForceFieldContribs(*mmat,
                                                                      etkdgDetails,
                                                                      positions,
                                                                      /*dim=*/3,
                                                                      embedParam.useBasicKnowledge);
    addMoleculeToMolecularSystem3D(ffParams, ctx.systemHost.atomStarts, molSystemHost);
  }
  setupDeviceBuffers3D(molSystemHost, molSystemDevice, positions, mols.size());
  DistGeom::sendContribsAndIndicesToDevice3D(molSystemHost, molSystemDevice);
}

void ETKMinimizationStage::setReferenceValues(const ETKDGContext& ctx) {
  const int numTerms12 = molSystemDevice.contribs.dist12Terms.idx1.size();
  const int numTerms13 = molSystemDevice.contribs.dist13Terms.idx1.size();

  if (numTerms12 > 0) {
    updateReferencePositionsKernel<<<(numTerms12 + 255) / 256, 256, 0, stream_>>>(
      numTerms12,
      ctx.systemDevice.positions.data(),
      molSystemDevice.contribs.dist12Terms.idx1.data(),
      molSystemDevice.contribs.dist12Terms.idx2.data(),
      molSystemDevice.contribs.dist12Terms.minLen.data(),
      molSystemDevice.contribs.dist12Terms.maxLen.data());
    cudaCheckError(cudaGetLastError());
  }
  if (numTerms13 > 0) {
    updateReferencePositionsKernel<<<(numTerms13 + 255) / 256, 256, 0, stream_>>>(
      numTerms13,
      ctx.systemDevice.positions.data(),
      molSystemDevice.contribs.dist13Terms.idx1.data(),
      molSystemDevice.contribs.dist13Terms.idx2.data(),
      molSystemDevice.contribs.dist13Terms.minLen.data(),
      molSystemDevice.contribs.dist13Terms.maxLen.data(),
      molSystemDevice.contribs.dist13Terms.isImproperConstrained.data());

    cudaCheckError(cudaGetLastError());
  }
}

void ETKMinimizationStage::execute(ETKDGContext& ctx) {
  // 1. Update reference positions for start of loop.
  setReferenceValues(ctx);
  // 2. Minimize via BFGS.
  // Create energy and gradient functions
  // Use PLAIN mode for ETDG (useBasicKnowledge=false), ALL mode for ETKDG/KDG (useBasicKnowledge=true)
  const auto etkTerm = embedParam_.useBasicKnowledge ? DistGeom::ETKTerm::ALL : DistGeom::ETKTerm::PLAIN;

  auto eFunc = [&](const double* pos) {
    computeEnergyETK(molSystemDevice,
                     ctx.systemDevice.atomStarts,
                     ctx.systemDevice.positions,
                     ctx.activeThisStage.data(),
                     pos,
                     etkTerm,
                     stream_);
  };

  auto gFunc = [&]() {
    computeGradientsETK(molSystemDevice,
                        ctx.systemDevice.atomStarts,
                        ctx.systemDevice.positions,
                        ctx.activeThisStage.data(),
                        etkTerm,
                        stream_);
  };

  // Create and configure BFGS minimizer
  // TODO: Reuse between iterations.
  BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/dim, nvMolKit::DebugLevel::NONE, true, stream_);

  // Run minimization
  constexpr int maxIters = 300;  // Taken from hard-coded RDKit value.
  bfgsMinimizer.minimize(maxIters,
                         embedParam_.optimizerForceTol,
                         ctx.systemHost.atomStarts,
                         ctx.systemDevice.atomStarts,
                         ctx.systemDevice.positions,
                         molSystemDevice.grad,
                         molSystemDevice.energyOuts,
                         molSystemDevice.energyBuffer,
                         eFunc,
                         gFunc,
                         ctx.activeThisStage.data());

  // 3. Check planar tolerance (only if useBasicKnowledge is true - ETKDG/KDG variants)
  if (embedParam_.useBasicKnowledge) {
    DistGeom::computePlanarEnergy(molSystemDevice,
                                  ctx.systemDevice.atomStarts,
                                  ctx.systemDevice.positions,
                                  ctx.activeThisStage.data(),
                                  nullptr,
                                  stream_);
    const int numSystems = ctx.systemHost.atomStarts.size() - 1;
    planarToleranceCheck<<<(numSystems + 255) / 256, 256, 0, stream_>>>(
      numSystems,
      molSystemDevice.energyOuts.data(),
      molSystemDevice.contribs.improperTorsionTerms.numImpropers.data(),
      ctx.activeThisStage.data(),
      ctx.failedThisStage.data());
    cudaCheckError(cudaGetLastError());
  }
}

}  // namespace detail
}  // namespace nvMolKit
