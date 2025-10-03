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

#include <getopt.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <nanobench.h>
#include <omp.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "../tests/test_utils.h"
#include "benchmark_utils.h"
#include "bfgs_mmff.h"

namespace {

bool parseBoolArg(const std::string& arg) {
  std::string s = arg;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

std::optional<nvMolKit::MMFF::OptimizerOptions::Backend> parseMinimizerArg(const std::string& arg) {
  std::string s = arg;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "bfgs") {
    return nvMolKit::MMFF::OptimizerOptions::Backend::BFGS;
  }
  if (s == "fire") {
    return nvMolKit::MMFF::OptimizerOptions::Backend::FIRE;
  }
  return std::nullopt;
}

void printHelp(const char* progName) {
  std::cout << "Usage: " << progName << " [options]\n\n";
  std::cout << "Options:\n";
  std::cout
    << "  -f, --file_path <path>              Path to input file (.sdf, .smi, .smiles) [default: benchmarks/data/MMFF94_hypervalent.sdf]\n";
  std::cout << "  -n, --num_mols <int>                Number of molecules to process [default: 20]\n";
  std::cout << "  -c, --confs_per_mol <int>           Number of conformers per molecule [default: 20]\n";
  std::cout << "  -r, --do_rdkit <bool>               Run RDKit MMFF optimization for comparison [default: true]\n";
  std::cout << "  -w, --do_warmup <bool>              Run warmup before benchmarking [default: true]\n";
  std::cout
    << "  -e, --do_energy_check <bool>        Compare energies between RDKit and nvMolKit results [default: true]\n";
  std::cout << "  -B, --num_concurrent_batches <int>  Number of concurrent batches per GPU [default: 10]\n";
  std::cout << "  -b, --batch_size <int>              Batch size for processing [default: 1000]\n";
  std::cout << "  -g, --num_gpus <int>                Number of GPUs to use (IDs 0..n-1). If omitted, use all GPUs.\n";
  std::cout
    << "  -t, --num_threads <int>             RDKit MMFF optimize threads (per-molecule conformer threads) [default: OMP max]\n";
  std::cout
    << "  -p, --perturbation_factor <float>    Random displacement magnitude for starting structures [default: 0.5]\n";
  std::cout << "  -m, --minimizer <BFGS|FIRE>          Minimizer to use [default: BFGS]\n";
  std::cout << "  -h, --help                          Show this help message\n\n";
  std::cout << "Boolean values can be: true/false, 1/0, yes/no, on/off (case insensitive)\n";
}

std::vector<std::vector<double>> runRDKit(std::vector<RDKit::ROMol*>& molsPtrs, int maxIters, int rdkitThreads) {
  std::vector<std::vector<double>> allEnergies;
  std::string                      benchName = "RDKit MMFF, num_mols=" + std::to_string(molsPtrs.size());
  ankerl::nanobench::Bench().epochIterations(1).epochs(1).run(benchName, [&]() {
    allEnergies.clear();
    allEnergies.reserve(molsPtrs.size());
    for (auto* mol : molsPtrs) {
      try {
        std::vector<std::pair<int, double>> res(mol->getNumConformers());
        RDKit::MMFF::MMFFOptimizeMoleculeConfs(*mol, res, rdkitThreads, maxIters, "MMFF94", 100.0);
        std::vector<double> energies;
        energies.reserve(res.size());
        for (const auto& r : res) {
          energies.push_back(r.second);
        }
        allEnergies.push_back(std::move(energies));
      } catch (const std::exception& e) {
        std::cerr << "Warning: RDKit MMFF failed for a molecule and will be skipped: " << e.what() << std::endl;
      }
    }
  });
  return allEnergies;
}

std::vector<std::vector<double>> runNvMolKit(std::vector<RDKit::ROMol*>&               molsPtrs,
                                             int                                       maxIters,
                                             int                                       batchSize,
                                             int                                       batchesPerGpu,
                                             int                                       numGpus,
                                             nvMolKit::MMFF::OptimizerOptions::Backend minimizer) {
  nvMolKit::BatchHardwareOptions perfOptions;
  perfOptions.batchesPerGpu = batchesPerGpu;
  perfOptions.batchSize     = batchSize;
  if (numGpus > 0) {
    perfOptions.gpuIds.clear();
    perfOptions.gpuIds.reserve(static_cast<size_t>(numGpus));
    for (int i = 0; i < numGpus; ++i) {
      perfOptions.gpuIds.push_back(i);
    }
  }
  nvMolKit::MMFF::OptimizerOptions optimizerOptions;
  optimizerOptions.backend = minimizer;
  std::vector<std::vector<double>> energies;
  std::string                      benchName = "nvMolKit MMFF, minimizer=" +
                          std::string(minimizer == nvMolKit::MMFF::OptimizerOptions::Backend::BFGS ? "BFGS" : "FIRE") +
                          ", num_mols=" + std::to_string(molsPtrs.size()) +
                          ", batch_size=" + std::to_string(batchSize) +
                          ", num_concurrent_batches=" + std::to_string(batchesPerGpu);
  ankerl::nanobench::Bench().epochIterations(1).epochs(1).run(benchName, [&]() {
    energies = nvMolKit::MMFF::MMFFOptimizeMoleculesConfsBfgs(molsPtrs, maxIters, 100.0, perfOptions, optimizerOptions);
  });
  return energies;
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string                               filePath           = "benchmarks/data/MMFF94_hypervalent.sdf";
  int                                       numMols            = 20;
  int                                       confsPerMol        = 20;
  bool                                      doRdkit            = true;
  bool                                      doWarmup           = true;
  bool                                      doEnergyCheck      = true;
  int                                       batchSize          = 1000;
  int                                       batchesPerGpu      = 10;
  int                                       maxIters           = 1000;
  int                                       numGpus            = -1;  // If <0, use all GPUs
  float                                     perturbationFactor = 0.5f;
  int                                       rdkitThreads       = -1;  // If <0, use OMP max
  nvMolKit::MMFF::OptimizerOptions::Backend minimizer          = nvMolKit::MMFF::OptimizerOptions::Backend::BFGS;

  static struct option long_options[] = {
    {             "file_path", required_argument, 0, 'f'},
    {              "num_mols", required_argument, 0, 'n'},
    {         "confs_per_mol", required_argument, 0, 'c'},
    {              "do_rdkit", required_argument, 0, 'r'},
    {             "do_warmup", required_argument, 0, 'w'},
    {       "do_energy_check", required_argument, 0, 'e'},
    {"num_concurrent_batches", required_argument, 0, 'B'},
    {            "batch_size", required_argument, 0, 'b'},
    {              "num_gpus", required_argument, 0, 'g'},
    {           "num_threads", required_argument, 0, 't'},
    {   "perturbation_factor", required_argument, 0, 'p'},
    {             "minimizer", required_argument, 0, 'm'},
    {                  "help",       no_argument, 0, 'h'},
    {                       0,                 0, 0,   0}
  };

  int option_index = 0;
  int c;
  while ((c = getopt_long(argc, argv, "f:n:c:r:w:e:B:b:g:t:p:m:h", long_options, &option_index)) != -1) {
    switch (c) {
      case 'f':
        filePath = optarg;
        break;
      case 'n':
        try {
          numMols = std::stoi(optarg);
          if (numMols <= 0) {
            std::cerr << "Error: num_mols must be positive\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for num_mols: " << optarg << "\n";
          return 1;
        }
        break;
      case 'c':
        try {
          confsPerMol = std::stoi(optarg);
          if (confsPerMol <= 0) {
            std::cerr << "Error: confs_per_mol must be positive\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for confs_per_mol: " << optarg << "\n";
          return 1;
        }
        break;
      case 'r':
        doRdkit = parseBoolArg(optarg);
        break;
      case 'w':
        doWarmup = parseBoolArg(optarg);
        break;
      case 'e':
        doEnergyCheck = parseBoolArg(optarg);
        break;
      case 'B':
        try {
          batchesPerGpu = std::stoi(optarg);
          if (batchesPerGpu <= 0) {
            std::cerr << "Error: num_concurrent_batches must be positive\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for num_concurrent_batches: " << optarg << "\n";
          return 1;
        }
        break;
      case 'b':
        try {
          batchSize = std::stoi(optarg);
          if (batchSize <= 0) {
            std::cerr << "Error: batch_size must be positive\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for batch_size: " << optarg << "\n";
          return 1;
        }
        break;
      case 'g':
        try {
          numGpus = std::stoi(optarg);
          if (numGpus <= 0) {
            std::cerr << "Error: num_gpus must be positive\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for num_gpus: " << optarg << "\n";
          return 1;
        }
        break;
      case 't':
        try {
          rdkitThreads = std::stoi(optarg);
          if (rdkitThreads == 0) {
            std::cerr << "Error: num_threads must be non-zero\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for num_threads: " << optarg << "\n";
          return 1;
        }
        break;
      case 'p':
        try {
          perturbationFactor = std::stof(optarg);
          if (perturbationFactor < 0.0f) {
            std::cerr << "Error: perturbation_factor must be non-negative\n";
            return 1;
          }
        } catch (...) {
          std::cerr << "Error: Invalid value for perturbation_factor: " << optarg << "\n";
          return 1;
        }
        break;
      case 'm': {
        const auto minimizerParsed = parseMinimizerArg(optarg);
        if (!minimizerParsed) {
          std::cerr << "Error: Invalid value for minimizer: " << optarg << " (expected BFGS or FIRE)\n";
          return 1;
        }
        minimizer = *minimizerParsed;
        break;
      }
      case 'h':
        printHelp(argv[0]);
        return 0;
      case '?':
        std::cerr << "\nUse --help for usage information.\n";
        return 1;
      default:
        std::cerr << "Unknown option\n";
        return 1;
    }
  }

  if (optind < argc) {
    std::cerr << "Error: Unexpected non-option arguments: ";
    while (optind < argc) {
      std::cerr << argv[optind++] << " ";
    }
    std::cerr << "\nUse --help for usage information.\n";
    return 1;
  }

  if (!std::filesystem::exists(filePath)) {
    std::cerr << "Error: File does not exist: " << filePath << std::endl;
    return 1;
  }

  std::cout << "Configuration:\n";
  std::cout << "  File path: " << filePath << "\n";
  std::cout << "  Number of molecules: " << numMols << "\n";
  std::cout << "  Conformers per molecule: " << confsPerMol << "\n";
  std::cout << "  Run RDKit comparison: " << (doRdkit ? "yes" : "no") << "\n";
  std::cout << "  Run warmup: " << (doWarmup ? "yes" : "no") << "\n";
  std::cout << "  Compare energies: " << (doEnergyCheck ? "yes" : "no") << "\n";
  std::cout << "  Batch size: " << batchSize << "\n";
  std::cout << "  Number of concurrent batches: " << batchesPerGpu << "\n";
  std::cout << "  Number of GPUs: " << (numGpus > 0 ? std::to_string(numGpus) : std::string("all")) << "\n";
  std::cout << "  RDKit MMFF threads: " << (rdkitThreads > 0 ? std::to_string(rdkitThreads) : std::string("OMP max"))
            << "\n";
  std::cout << "  Perturbation factor: " << perturbationFactor << "\n\n";
  std::cout << "  Minimizer: " << (minimizer == nvMolKit::MMFF::OptimizerOptions::Backend::BFGS ? "BFGS" : "FIRE")
            << "\n\n";

  const std::string ext          = BenchUtils::getFileExtensionLower(filePath);
  const bool        isSmilesLike = (ext == ".smi" || ext == ".smiles" || ext == ".cxsmiles");

  // Determine RDKit threads for embedding
  int rdkitThreadsResolved = rdkitThreads;
  if (rdkitThreadsResolved <= 0) {
#ifdef _OPENMP
    rdkitThreadsResolved = omp_get_max_threads();
#else
    rdkitThreadsResolved = 1;
#endif
  }
  // Embedding threads always use OMP max, independent of -t
  int embedThreads = 1;
#ifdef _OPENMP
  embedThreads = omp_get_max_threads();
#endif

  // Warmup (use fewer conformers to avoid large allocations during warmup)
  const int warmupConfs = std::min(confsPerMol, 10);
  if (doWarmup) {
    std::cout << "Warming up..." << std::endl;
    auto                       warmupMolsOwning = isSmilesLike ? BenchUtils::readMoleculesForEmbedding(filePath, 1) :
                                                                 BenchUtils::readMoleculesKeepConfs(filePath, 1);
    std::vector<RDKit::ROMol*> warmupPtrs;
    for (auto& m : warmupMolsOwning)
      warmupPtrs.push_back(static_cast<RDKit::ROMol*>(m.get()));

    if (isSmilesLike) {
      BenchUtils::embedOneConfThenDuplicate(warmupPtrs, warmupConfs, embedThreads, 1000);
    } else {
      BenchUtils::ensureNumConformersByCopying(*warmupPtrs[0], warmupConfs);
    }

    BenchUtils::perturbAllConformers(warmupPtrs, perturbationFactor, 123);

    (void)runNvMolKit(warmupPtrs, maxIters, batchSize, batchesPerGpu, numGpus, minimizer);
    if (doRdkit) {
      (void)runRDKit(warmupPtrs, maxIters, rdkitThreadsResolved);
    }
    std::cout << "Warmed up" << std::endl;
  } else {
    std::cout << "Skipping warmup" << std::endl;
  }

  // Prepare molecules for the main run
  auto molsOwning = isSmilesLike ? BenchUtils::readMoleculesForEmbedding(filePath, static_cast<unsigned int>(numMols)) :
                                   BenchUtils::readMoleculesKeepConfs(filePath, static_cast<unsigned int>(numMols));
  std::vector<RDKit::ROMol*> molsPtrs;
  molsPtrs.reserve(molsOwning.size());
  for (auto& m : molsOwning)
    molsPtrs.push_back(static_cast<RDKit::ROMol*>(m.get()));

  if (isSmilesLike) {
    // Generate single conformer and duplicate (multi-threaded across molecules)
    BenchUtils::embedOneConfThenDuplicate(molsPtrs, confsPerMol, embedThreads, 10);
  } else {
    // Ensure requested number of conformers exists by cloning
    for (auto* m : molsPtrs) {
      BenchUtils::ensureNumConformersByCopying(*m, confsPerMol);
    }
  }

  // Apply perturbation factor to all conformers to move starting structures
  BenchUtils::perturbAllConformers(molsPtrs, perturbationFactor, 999);

  // Filter out molecules that fail MMFF setup (e.g., kekulization issues)
  std::vector<std::unique_ptr<RDKit::RWMol>> filteredOwning;
  filteredOwning.reserve(molsPtrs.size());
  for (auto* m : molsPtrs) {
    try {
      RDKit::MMFF::MMFFMolProperties mmffProps(*m);
      // Test FF construction on the first conformer
      const int                      confId = m->getConformer().getId();
      auto                           ff     = RDKit::MMFF::constructForceField(*m, &mmffProps, confId);
      if (ff) {
        filteredOwning.push_back(std::make_unique<RDKit::RWMol>(*m));
      }
    } catch (const std::exception& e) {
    }
  }
  if (filteredOwning.empty()) {
    std::cerr << "Error: No valid molecules remain after MMFF filtering." << std::endl;
    return 1;
  }
  if (filteredOwning.size() < static_cast<size_t>(numMols)) {
    const size_t originalSize = filteredOwning.size();
    for (size_t i = 0; i < static_cast<size_t>(numMols) - originalSize; ++i) {
      filteredOwning.push_back(std::make_unique<RDKit::RWMol>(*filteredOwning[i % originalSize]));
    }
  }

  // Rebuild pointers from filtered set (limit to requested numMols)
  std::vector<RDKit::ROMol*> filteredPtrs;
  filteredPtrs.reserve(static_cast<size_t>(numMols));
  for (int i = 0; i < numMols && i < static_cast<int>(filteredOwning.size()); ++i) {
    filteredPtrs.push_back(static_cast<RDKit::ROMol*>(filteredOwning[i].get()));
  }

  // Duplicate molecule sets for fair comparison
  std::vector<std::unique_ptr<RDKit::RWMol>> rdkitCopies;
  std::vector<std::unique_ptr<RDKit::RWMol>> nvmolkitCopies;
  std::vector<RDKit::ROMol*>                 rdkitPtrs;
  std::vector<RDKit::ROMol*>                 nvmolkitPtrs;
  rdkitCopies.reserve(filteredPtrs.size());
  nvmolkitCopies.reserve(filteredPtrs.size());
  for (auto* m : filteredPtrs) {
    rdkitCopies.push_back(std::make_unique<RDKit::RWMol>(*m));
    nvmolkitCopies.push_back(std::make_unique<RDKit::RWMol>(*m));
  }
  for (auto& m : rdkitCopies)
    rdkitPtrs.push_back(m.get());
  for (auto& m : nvmolkitCopies)
    nvmolkitPtrs.push_back(m.get());

  // Run benchmarks
  auto nvmolkitRes = runNvMolKit(nvmolkitPtrs, maxIters, batchSize, batchesPerGpu, numGpus, minimizer);
  std::vector<std::vector<double>> rdkitRes;
  if (doRdkit) {
    rdkitRes = runRDKit(rdkitPtrs, maxIters, rdkitThreadsResolved);
  }

  if (doEnergyCheck && doRdkit) {
    int totalDiffs = 0;
    int totalConfs = 0;
    for (size_t i = 0; i < nvmolkitRes.size(); ++i) {
      const auto&  a = nvmolkitRes[i];
      const auto&  b = rdkitRes[i];
      const size_t n = std::min(a.size(), b.size());
      for (size_t j = 0; j < n; ++j) {
        totalConfs++;
        if (std::abs(a[j] - b[j]) > 1e-2)
          totalDiffs++;
      }
    }
    if (totalDiffs > 0) {
      std::cout << "Differences found: " << totalDiffs << "/" << totalConfs << " conformers differ" << std::endl;
    } else {
      std::cout << "Perfect match (" << totalConfs << " conformers)" << std::endl;
    }
  }

  return 0;
}
