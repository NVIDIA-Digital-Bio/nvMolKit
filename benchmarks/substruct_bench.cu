// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cublas_v2.h>
#include <getopt.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "benchmark_utils.h"
#include "cuda_error_check.h"
#include "device.h"
#include "substruct/substruct_search.h"
#include "testutils/substruct_validation.h"

using nvMolKit::algorithmName;
using nvMolKit::checkReturnCode;
using nvMolKit::countCudaDevices;
using nvMolKit::getRDKitSubstructMatches;
using nvMolKit::getSubstructMatches;
using nvMolKit::hasSubstructMatch;
using nvMolKit::HasSubstructMatchResults;
using nvMolKit::printValidationResult;
using nvMolKit::ScopedStream;
using nvMolKit::SubstructAlgorithm;
using nvMolKit::SubstructSearchConfig;
using nvMolKit::SubstructSearchResults;
using nvMolKit::SubstructValidationResult;
using nvMolKit::validateAgainstRDKit;

namespace {

std::unique_ptr<RDKit::ROMol> makeMolFromSmiles(const std::string& smiles) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmilesToMol(smiles));
}

std::unique_ptr<RDKit::ROMol> makeMolFromSmarts(const std::string& smarts) {
  return std::unique_ptr<RDKit::ROMol>(RDKit::SmartsToMol(smarts));
}

/**
 * @brief Read and parse molecules from SMILES file until we get enough valid ones.
 *
 * Supports .smi, .smiles, .cxsmiles file formats.
 * Lines starting with # are treated as comments.
 * Continues reading until either maxCount valid molecules are parsed or EOF.
 *
 * @param filePath Path to SMILES file
 * @param maxCount Maximum number of valid molecules to parse
 * @param asQuery If true, parse as SMARTS; otherwise parse as SMILES
 * @param maxAtoms Maximum number of atoms allowed (0 = no limit)
 * @param smilesOut Optional output vector to store the original SMILES/SMARTS strings
 * @return Vector of parsed molecules
 */
std::vector<std::unique_ptr<RDKit::ROMol>> readAndParseMolecules(const std::string&        filePath,
                                                                 unsigned int              maxCount,
                                                                 bool                      asQuery,
                                                                 unsigned int              maxAtoms  = 0,
                                                                 std::vector<std::string>* smilesOut = nullptr) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filePath);
  }

  std::vector<std::unique_ptr<RDKit::ROMol>> mols;
  mols.reserve(maxCount);
  std::string  line;
  unsigned int linesRead       = 0;
  unsigned int parseFailures   = 0;
  unsigned int filteredByAtoms = 0;

  while (std::getline(file, line) && mols.size() < maxCount) {
    linesRead++;

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::string smiles = line.substr(0, line.find_first_of(" \t"));
    if (smiles.empty()) {
      continue;
    }

    auto mol = asQuery ? makeMolFromSmarts(smiles) : makeMolFromSmiles(smiles);
    if (!mol) {
      parseFailures++;
      continue;
    }

    if (maxAtoms > 0 && mol->getNumAtoms() > maxAtoms) {
      filteredByAtoms++;
      continue;
    }

    mols.push_back(std::move(mol));
    if (smilesOut) {
      smilesOut->push_back(smiles);
    }
  }

  if (mols.empty()) {
    throw std::runtime_error("No valid molecules found in file: " + filePath);
  }

  std::cout << "  Read " << linesRead << " lines from " << filePath << "\n";
  std::cout << "  Parsed " << mols.size() << " valid " << (asQuery ? "queries" : "targets");
  if (parseFailures > 0) {
    std::cout << " (" << parseFailures << " parse failures";
    if (filteredByAtoms > 0) {
      std::cout << ", " << filteredByAtoms << " filtered by atom cap";
    }
    std::cout << ")";
  } else if (filteredByAtoms > 0) {
    std::cout << " (" << filteredByAtoms << " filtered by atom cap)";
  }
  std::cout << "\n";

  return mols;
}

/**
 * @brief Benchmark RDKit substructure matching.
 */
void benchRDKit(const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                int&                                              totalMatches,
                BenchUtils::TimingResult&                         timingOut,
                int                                               iterations = 3,
                int                                               warmups    = 1) {
  totalMatches = 0;

  timingOut = BenchUtils::timeIt(
    [&]() {
      RDKit::SubstructMatchParameters params;
      params.uniquify  = false;
      int localMatches = 0;

      for (const auto& target : targetMols) {
        for (const auto& query : queryMols) {
          auto matches = RDKit::SubstructMatch(*target, *query, params);
          localMatches += static_cast<int>(matches.size());
        }
      }
      totalMatches = localMatches;
    },
    iterations,
    warmups);

  std::cout << "RDKit SubstructMatch, targets=" << targetMols.size() << ", queries=" << queryMols.size() << ": "
            << timingOut.avgMs << " ms (±" << timingOut.stdMs << " ms)\n";
}

/**
 * @brief Benchmark nvMolKit GPU hasSubstructMatch (boolean mode).
 */
void benchNvMolKitHasMatch(const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                           const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                           SubstructAlgorithm                                algorithm,
                           const SubstructSearchConfig&                      config,
                           int&                                              totalMatches,
                           HasSubstructMatchResults&                         resultsOut,
                           BenchUtils::TimingResult&                         timingOut,
                           int                                               iterations = 3,
                           int                                               warmups    = 1) {
  std::string algoStr = algorithmName(algorithm);

  ScopedStream stream;

  std::vector<const RDKit::ROMol*> targetPtrs;
  std::vector<const RDKit::ROMol*> queryPtrs;
  targetPtrs.reserve(targetMols.size());
  queryPtrs.reserve(queryMols.size());
  for (const auto& mol : targetMols) {
    targetPtrs.push_back(mol.get());
  }
  for (const auto& mol : queryMols) {
    queryPtrs.push_back(mol.get());
  }

  timingOut = BenchUtils::timeIt(
    [&]() { hasSubstructMatch(targetPtrs, queryPtrs, resultsOut, algorithm, stream.stream(), config); },
    iterations,
    warmups);

  totalMatches = 0;
  for (int t = 0; t < resultsOut.numTargets; ++t) {
    for (int q = 0; q < resultsOut.numQueries; ++q) {
      if (resultsOut.matches(t, q)) {
        totalMatches++;
      }
    }
  }

  std::string threadingStr = std::to_string(config.workerThreads) + " workers";
  std::cout << "nvMolKit HasSubstructMatch (" << algoStr << ", " << threadingStr << "), targets=" << targetMols.size()
            << ", queries=" << queryMols.size() << ": " << timingOut.avgMs << " ms (±" << timingOut.stdMs << " ms)\n";
}

/**
 * @brief Benchmark nvMolKit GPU substructure matching.
 */
void benchNvMolKit(const std::vector<std::unique_ptr<RDKit::ROMol>>& targetMols,
                   const std::vector<std::unique_ptr<RDKit::ROMol>>& queryMols,
                   SubstructAlgorithm                                algorithm,
                   const SubstructSearchConfig&                      config,
                   int&                                              totalMatches,
                   SubstructSearchResults&                           resultsOut,
                   BenchUtils::TimingResult&                         timingOut,
                   int                                               iterations = 3,
                   int                                               warmups    = 1) {
  std::string algoStr = algorithmName(algorithm);

  ScopedStream stream;

  std::vector<const RDKit::ROMol*> targetPtrs;
  std::vector<const RDKit::ROMol*> queryPtrs;
  targetPtrs.reserve(targetMols.size());
  queryPtrs.reserve(queryMols.size());
  for (const auto& mol : targetMols) {
    targetPtrs.push_back(mol.get());
  }
  for (const auto& mol : queryMols) {
    queryPtrs.push_back(mol.get());
  }

  timingOut = BenchUtils::timeIt(
    [&]() { getSubstructMatches(targetPtrs, queryPtrs, resultsOut, algorithm, stream.stream(), config); },
    iterations,
    warmups);

  totalMatches = 0;
  for (int t = 0; t < resultsOut.numTargets; ++t) {
    for (int q = 0; q < resultsOut.numQueries; ++q) {
      totalMatches += resultsOut.matchCount(t, q);
    }
  }

  std::string threadingStr = std::to_string(config.workerThreads) + " workers";
  std::cout << "nvMolKit SubstructMatch (" << algoStr << ", " << threadingStr << "), targets=" << targetMols.size()
            << ", queries=" << queryMols.size() << ": " << timingOut.avgMs << " ms (±" << timingOut.stdMs << " ms)\n";
}

/**
 * @brief Warm up GPU(s) with a simple memcpy and cuBLAS operation.
 *
 * This initializes CUDA contexts and wakes GPUs from power-saving mode.
 */
void warmupGpus(const std::vector<int>& gpuIds) {
  std::vector<int> devices = gpuIds.empty() ? std::vector<int>{0} : gpuIds;

  constexpr int      N = 256;
  std::vector<float> hostData(N, 1.0f);

  for (int gpuId : devices) {
    cudaCheckError(cudaSetDevice(gpuId));

    float* devData = nullptr;
    cudaCheckError(cudaMalloc(&devData, N * sizeof(float)));
    cudaCheckError(cudaMemcpy(devData, hostData.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    float result = 0.0f;
    cublasSnrm2(handle, N, devData, 1, &result);
    cublasDestroy(handle);

    cudaCheckError(cudaFree(devData));
    cudaCheckError(cudaDeviceSynchronize());
  }
}

bool parseBoolArg(const std::string& arg) {
  std::string s = arg;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

SubstructAlgorithm parseAlgorithmArg(const std::string& arg) {
  std::string s = arg;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "vf2" || s == "0") {
    return SubstructAlgorithm::VF2;
  } else if (s == "gsi" || s == "1") {
    return SubstructAlgorithm::GSI;
  } else {
    throw std::runtime_error("Invalid algorithm. Use 'vf2' or 'gsi'");
  }
}

void printHelp(const char* progName) {
  std::cout << "Usage: " << progName << " [options]\n\n";
  std::cout << "Substructure matching benchmark comparing nvMolKit GPU algorithms to RDKit.\n\n";
  std::cout << "Options:\n";
  std::cout << "  -t, --targets <path>      Path to targets SMILES file [required]\n";
  std::cout << "  -q, --queries <path>      Path to queries SMARTS file [required]\n";
  std::cout << "  -n, --num_targets <int>   Max number of target molecules [default: 100]\n";
  std::cout << "  -m, --num_queries <int>   Max number of query molecules [default: 10]\n";
  std::cout << "  -a, --algorithm <str>     Algorithm: vf2 or gsi [default: gsi]\n";
  std::cout << "  -b, --batch_size <int>    GPU batch size for matching [default: 1024]\n";
  std::cout << "  -c, --cap <int>           Max atoms per molecule (filter larger) [default: 128]\n";
  std::cout << "  -p, --num_runners <int>   Number of GPU runner threads per GPU [default: 2]\n";
  std::cout << "  -e, --num_preproc <int>   Number of CPU preprocessor threads (0 = inline) [default: 0]\n";
  std::cout << "  -S, --slots <int>         Slots per runner for inline mode (1-8) [default: 3]\n";
  std::cout << "  -G, --multi_gpu <bool>    Use all available GPUs [default: false]\n";
  std::cout << "  -r, --do_rdkit <bool>     Run RDKit benchmark comparison [default: true]\n";
  std::cout << "  -w, --do_warmup <bool>    Run warmup before benchmarking [default: true]\n";
  std::cout << "  -v, --validate <bool>     Validate GPU results against RDKit [default: false]\n";
  std::cout << "  -P, --profile <bool>      Profile mode: run 1 iteration only [default: false]\n";
  std::cout << "  -d, --debug <int>         Debug/verbosity level (0-2) [default: 0]\n";
  std::cout << "                            0: No debug output\n";
  std::cout << "                            1: Print validation failures summary\n";
  std::cout << "                            2: Print detailed validation failures with SMILES\n";
  std::cout << "  -M, --max_matches <int>   Max matches to find per pair (-1 = no limit) [default: -1]\n";
  std::cout << "  -H, --has_match_only <bool> Only check for existence of match [default: false]\n";
  std::cout << "  -h, --help                Show this help message\n\n";
  std::cout << "Boolean values can be: true/false, 1/0, yes/no, on/off (case insensitive)\n";
  std::cout << "\nExamples:\n";
  std::cout << "  " << progName << " --targets targets.smi --queries queries.smi --num_targets 1000 --algorithm gsi\n";
  std::cout << "  " << progName << " -t targets.smi -q queries.smi -n 500 -m 20 -b 512 -c 64 -v true -d 2\n";
  std::cout << "  " << progName << " -t targets.smi -q queries.smi -p 2 -e 4  # 2 runners, 4 preprocessors\n";
  std::cout << "  " << progName << " -t targets.smi -q queries.smi -G true    # Use all GPUs\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string        targetsPath;
  std::string        queriesPath;
  int                numTargets         = 100;
  int                numQueries         = 10;
  SubstructAlgorithm algorithm          = SubstructAlgorithm::GSI;
  int                batchSize          = 1024;
  unsigned int       maxAtoms           = 128;
  int                numRunners         = 2;
  int                numPreprocessors   = 0;
  int                executorsPerRunner = 3;
  bool               useMultiGpu        = false;
  bool               doRdkit            = true;
  bool               doWarmup           = true;
  bool               doValidate         = false;
  bool               doProfile          = false;
  int                debugLevel         = 0;
  int                maxMatches         = 0;
  bool               hasMatchOnly       = false;

  static struct option long_options[] = {
    {       "targets", required_argument, 0, 't'},
    {       "queries", required_argument, 0, 'q'},
    {   "num_targets", required_argument, 0, 'n'},
    {   "num_queries", required_argument, 0, 'm'},
    {     "algorithm", required_argument, 0, 'a'},
    {    "batch_size", required_argument, 0, 'b'},
    {           "cap", required_argument, 0, 'c'},
    {   "num_runners", required_argument, 0, 'p'},
    {   "num_preproc", required_argument, 0, 'e'},
    {         "slots", required_argument, 0, 'S'},
    {     "multi_gpu", required_argument, 0, 'G'},
    {      "do_rdkit", required_argument, 0, 'r'},
    {     "do_warmup", required_argument, 0, 'w'},
    {      "validate", required_argument, 0, 'v'},
    {       "profile", required_argument, 0, 'P'},
    {         "debug", required_argument, 0, 'd'},
    {   "max_matches", required_argument, 0, 'M'},
    {"has_match_only", required_argument, 0, 'H'},
    {          "help",       no_argument, 0, 'h'},
    {               0,                 0, 0,   0}
  };

  int option_index = 0;
  int c;

  while ((c = getopt_long(argc, argv, "t:q:n:m:a:b:c:p:e:S:G:r:w:v:P:d:M:H:h", long_options, &option_index)) != -1) {
    switch (c) {
      case 't':
        targetsPath = optarg;
        break;
      case 'q':
        queriesPath = optarg;
        break;
      case 'n':
        try {
          numTargets = std::stoi(optarg);
          if (numTargets <= 0) {
            std::cerr << "Error: num_targets must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_targets: " << optarg << "\n";
          return 1;
        }
        break;
      case 'm':
        try {
          numQueries = std::stoi(optarg);
          if (numQueries <= 0) {
            std::cerr << "Error: num_queries must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_queries: " << optarg << "\n";
          return 1;
        }
        break;
      case 'a':
        try {
          algorithm = parseAlgorithmArg(optarg);
        } catch (const std::exception& e) {
          std::cerr << "Error: " << e.what() << "\n";
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
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for batch_size: " << optarg << "\n";
          return 1;
        }
        break;
      case 'c':
        try {
          maxAtoms = std::stoul(optarg);
          if (maxAtoms == 0) {
            std::cerr << "Error: cap must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for cap: " << optarg << "\n";
          return 1;
        }
        break;
      case 'p':
        try {
          numRunners = std::stoi(optarg);
          if (numRunners <= 0) {
            std::cerr << "Error: num_runners must be positive\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_runners: " << optarg << "\n";
          return 1;
        }
        break;
      case 'e':
        try {
          numPreprocessors = std::stoi(optarg);
          if (numPreprocessors < 0) {
            std::cerr << "Error: num_preproc must be non-negative\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for num_preproc: " << optarg << "\n";
          return 1;
        }
        break;
      case 'S':
        try {
          executorsPerRunner = std::stoi(optarg);
          if (executorsPerRunner < 1 || executorsPerRunner > 8) {
            std::cerr << "Error: executors must be between 1 and 8\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for slots: " << optarg << "\n";
          return 1;
        }
        break;
      case 'G':
        useMultiGpu = parseBoolArg(optarg);
        break;
      case 'r':
        doRdkit = parseBoolArg(optarg);
        break;
      case 'w':
        doWarmup = parseBoolArg(optarg);
        break;
      case 'v':
        doValidate = parseBoolArg(optarg);
        break;
      case 'P':
        doProfile = parseBoolArg(optarg);
        break;
      case 'd':
        try {
          debugLevel = std::stoi(optarg);
          if (debugLevel < 0 || debugLevel > 2) {
            std::cerr << "Error: debug level must be 0, 1, or 2\n";
            return 1;
          }
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for debug: " << optarg << "\n";
          return 1;
        }
        break;
      case 'M':
        try {
          maxMatches = std::stoi(optarg);
        } catch (const std::exception& e) {
          std::cerr << "Error: Invalid value for max_matches: " << optarg << "\n";
          return 1;
        }
        break;
      case 'H':
        hasMatchOnly = parseBoolArg(optarg);
        break;
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

  if (targetsPath.empty() || queriesPath.empty()) {
    std::cerr << "Error: Both --targets and --queries are required.\n";
    std::cerr << "Use --help for usage information.\n";
    return 1;
  }

  if (!std::filesystem::exists(targetsPath)) {
    std::cerr << "Error: Targets file does not exist: " << targetsPath << std::endl;
    return 1;
  }
  if (!std::filesystem::exists(queriesPath)) {
    std::cerr << "Error: Queries file does not exist: " << queriesPath << std::endl;
    return 1;
  }

  const int        numGpus = useMultiGpu ? countCudaDevices() : 1;
  std::vector<int> gpuIds;
  if (useMultiGpu) {
    for (int i = 0; i < numGpus; ++i) {
      gpuIds.push_back(i);
    }
  }

  std::cout << "Configuration:\n";
  std::cout << "  Targets file: " << targetsPath << "\n";
  std::cout << "  Queries file: " << queriesPath << "\n";
  std::cout << "  Max targets: " << numTargets << "\n";
  std::cout << "  Max queries: " << numQueries << "\n";
  std::cout << "  Algorithm: " << algorithmName(algorithm) << "\n";
  std::cout << "  Batch size: " << batchSize << "\n";
  std::cout << "  Atom cap: " << maxAtoms << "\n";
  std::cout << "  Runner threads: " << numRunners << " per GPU\n";
  std::cout << "  Preprocessing threads: " << numPreprocessors << "\n";
  std::cout << "  Executors per runner: " << executorsPerRunner << "\n";
  std::cout << "  Multi-GPU: " << (useMultiGpu ? "yes" : "no") << " (" << numGpus << " GPU(s))\n";
  std::cout << "  Run RDKit comparison: " << (doRdkit ? "yes" : "no") << "\n";
  std::cout << "  Run warmup: " << (doWarmup ? "yes" : "no") << "\n";
  std::cout << "  Validate results: " << (doValidate ? "yes" : "no") << "\n";
  std::cout << "  Profile mode: " << (doProfile ? "yes" : "no") << "\n";
  std::cout << "  Debug level: " << debugLevel << "\n";
  std::cout << "  Max matches: " << (maxMatches < 0 ? "unlimited" : std::to_string(maxMatches)) << "\n";
  std::cout << "  Has match only: " << (hasMatchOnly ? "yes" : "no") << "\n\n";

  std::cout << "Loading and parsing molecules...\n";
  std::vector<std::string> targetSmiles;
  std::vector<std::string> querySmarts;
  auto                     targetMols =
    readAndParseMolecules(targetsPath, numTargets, false, maxAtoms, debugLevel >= 2 ? &targetSmiles : nullptr);
  auto queryMols =
    readAndParseMolecules(queriesPath, numQueries, true, maxAtoms, debugLevel >= 2 ? &querySmarts : nullptr);
  std::cout << "\n";

  if (targetMols.empty()) {
    std::cerr << "Error: No valid target molecules parsed\n";
    return 1;
  }
  if (queryMols.empty()) {
    std::cerr << "Error: No valid query molecules parsed\n";
    return 1;
  }

  if (static_cast<int>(queryMols.size()) < numQueries) {
    std::cout << "Note: Requested " << numQueries << " queries but only " << queryMols.size()
              << " available. Duplicating to reach " << numQueries << " queries.\n";
    const size_t originalSize = queryMols.size();
    for (int i = static_cast<int>(originalSize); i < numQueries; ++i) {
      queryMols.push_back(std::make_unique<RDKit::ROMol>(*queryMols[i % originalSize]));
    }
  }

  std::cout << "Using " << targetMols.size() << " targets and " << queryMols.size() << " queries for benchmark\n\n";

  if (doWarmup) {
    std::cout << "Warming up GPU(s)...\n";
    warmupGpus(gpuIds);
    std::cout << "Warmed up\n\n";
  }

  std::cout << "Running benchmarks...\n";

  const int benchIterations = doProfile ? 1 : 3;
  const int benchWarmups    = doProfile ? 0 : 1;

  SubstructSearchConfig benchConfig;
  benchConfig.batchSize            = batchSize;
  benchConfig.workerThreads        = numRunners;
  benchConfig.preprocessingThreads = numPreprocessors;
  benchConfig.executorsPerRunner   = executorsPerRunner;
  benchConfig.gpuIds               = gpuIds;
  benchConfig.maxMatches           = maxMatches;

  int                      nvmolkitMatches = 0;
  SubstructSearchResults   nvmolkitResults;
  HasSubstructMatchResults nvmolkitHasMatchResults;
  BenchUtils::TimingResult nvmolkitTiming;

  if (hasMatchOnly) {
    benchNvMolKitHasMatch(targetMols,
                          queryMols,
                          algorithm,
                          benchConfig,
                          nvmolkitMatches,
                          nvmolkitHasMatchResults,
                          nvmolkitTiming,
                          benchIterations,
                          benchWarmups);
    std::cout << "nvMolKit pairs with matches: " << nvmolkitMatches << "\n";
  } else {
    benchNvMolKit(targetMols,
                  queryMols,
                  algorithm,
                  benchConfig,
                  nvmolkitMatches,
                  nvmolkitResults,
                  nvmolkitTiming,
                  benchIterations,
                  benchWarmups);
    std::cout << "nvMolKit total matches: " << nvmolkitMatches << "\n";
  }

  int                      rdkitMatches = 0;
  BenchUtils::TimingResult rdkitTiming{0.0, 0.0};

  if (doRdkit) {
    benchRDKit(targetMols, queryMols, rdkitMatches, rdkitTiming, benchIterations, benchWarmups);
    std::cout << "RDKit total matches: " << rdkitMatches << "\n";

    if (!hasMatchOnly) {
      if (nvmolkitMatches == rdkitMatches) {
        std::cout << "Match counts EQUAL\n";
      } else {
        std::cout << "Match counts DIFFER by " << std::abs(nvmolkitMatches - rdkitMatches) << "\n";
      }
    }

    std::cout << "\nSpeedup: " << (rdkitTiming.avgMs / nvmolkitTiming.avgMs) << "x\n";
  }

  if (doValidate) {
    if (hasMatchOnly) {
      std::cout << "\nValidation not supported for has_match_only mode\n";
    } else {
      std::cout << "\nValidating against RDKit (per-pair comparison)...\n";
      auto validation = validateAgainstRDKit(nvmolkitResults, targetMols, queryMols);

      if (debugLevel == 0) {
        printValidationResult(validation, algorithmName(algorithm));
      } else if (debugLevel == 1) {
        printValidationResult(validation, algorithmName(algorithm));
      } else if (debugLevel >= 2) {
        const int maxDetails = (debugLevel == 2) ? 10 : 100;
        printValidationResultDetailed(validation,
                                      nvmolkitResults,
                                      targetMols,
                                      queryMols,
                                      targetSmiles,
                                      querySmarts,
                                      algorithmName(algorithm),
                                      maxDetails);
      }
    }
  }

  std::cout << "\n\nCSV Results:\n";
  std::cout
    << "algorithm,num_targets,num_queries,batch_size,num_runners,num_preproc,slots,num_gpus,max_matches,has_match_only,nvmolkit_time_ms,nvmolkit_std_ms";
  if (doRdkit) {
    std::cout << ",rdkit_time_ms,rdkit_std_ms";
  }
  std::cout << "\n";

  std::cout << algorithmName(algorithm) << "," << targetMols.size() << "," << queryMols.size() << "," << batchSize
            << "," << numRunners << "," << numPreprocessors << "," << executorsPerRunner << "," << numGpus << ","
            << maxMatches << "," << (hasMatchOnly ? 1 : 0) << "," << nvmolkitTiming.avgMs << ","
            << nvmolkitTiming.stdMs;
  if (doRdkit) {
    std::cout << "," << rdkitTiming.avgMs << "," << rdkitTiming.stdMs;
  }
  std::cout << "\n";

  return 0;
}
