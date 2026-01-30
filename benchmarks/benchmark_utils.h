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

#pragma once

#include <GraphMol/ROMol.h>

#include <memory>
#include <string>
#include <vector>

namespace BenchUtils {

// Returns lowercase file extension including dot, e.g. ".sdf" or ".smi". Returns empty string if none.
std::string getFileExtensionLower(const std::string& filePath);

// Read molecules for embedding workflows. Ensures molecules are sanitized, hydrogens are added, and
// all conformers are cleared so that callers can embed without retaining pre-existing coordinates.
std::vector<std::unique_ptr<RDKit::RWMol>> readMoleculesForEmbedding(const std::string& filePath,
                                                                     unsigned int       count,
                                                                     unsigned int       maxAtoms = 256);

// Read molecules while preserving any existing conformers (e.g., for MMFF starting structures from SDF).
// For SMILES-like inputs, behaves like readMoleculesForEmbedding (no conformers).
std::vector<std::unique_ptr<RDKit::RWMol>> readMoleculesKeepConfs(const std::string& filePath,
                                                                  unsigned int       count,
                                                                  unsigned int       maxAtoms = 256);

// Ensure a molecule has at least numConfs conformers by copying the first conformer as needed.
// No perturbation is applied here.
void ensureNumConformersByCopying(RDKit::ROMol& mol, int numConfs);

// Perturb all atom positions of all conformers in all molecules by a random offset in [-delta, delta].
void perturbAllConformers(std::vector<RDKit::ROMol*>& mols, float delta, int seedBase = 0);

// Embed numConfs conformers for each molecule using RDKit ETKDGv3. Existing conformers are retained
// unless clearExisting is true.
void embedConformersRDKit(std::vector<RDKit::ROMol*>& mols,
                          int                         numConfs,
                          int                         maxIterations = 1000,
                          bool                        clearExisting = false);

// For SMILES inputs: embed exactly one conformer per molecule (OpenMP over molecules), then duplicate
// to reach numConfs. Existing conformers are cleared first. RDKit EmbedParameters::numThreads is set
// to rdkitNumThreads.
void embedOneConfThenDuplicate(std::vector<RDKit::ROMol*>& mols,
                               int                         numConfs,
                               int                         rdkitNumThreads,
                               int                         maxIterations = 1000);

/**
 * @brief Timing result containing average and standard deviation in milliseconds.
 */
struct TimingResult {
  double avgMs;
  double stdMs;
};

/**
 * @brief Time a callable function with warmups and multiple runs.
 * @param func Callable to benchmark
 * @param runs Number of timing runs
 * @param warmups Number of warmup runs
 * @return TimingResult with average and standard deviation in milliseconds
 */
template <typename Func>
TimingResult timeIt(Func&& func, int runs = 3, int warmups = 1) {
  for (int i = 0; i < warmups; ++i) {
    func();
  }

  std::vector<double> times;
  times.reserve(runs);

  for (int i = 0; i < runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    times.push_back(static_cast<double>(duration) / 1.0e6);
  }

  double sum = 0.0;
  for (double t : times) {
    sum += t;
  }
  const double avgMs = sum / runs;

  double variance = 0.0;
  for (double t : times) {
    const double diff = t - avgMs;
    variance += diff * diff;
  }
  const double stdMs = std::sqrt(variance / runs);

  return TimingResult{avgMs, stdMs};
}


}  // namespace BenchUtils