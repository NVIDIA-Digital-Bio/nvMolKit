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

#include <GraphMol/ROMol.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cuda_error_check.h"
#include "graph_labeler.cuh"
#include "host_vector.h"
#include "molecules_device.cuh"
#include "nvtx.h"
#include "pinned_host_allocator.h"
#include "sm_shared_mem_config.cuh"
#include "substruct_algos.cuh"
#include "substruct_debug.h"
#include "substruct_kernels.h"
#include "substruct_launch_config.h"
#include "substruct_search.h"
#include "substruct_search_internal.cuh"
#include "thread_safe_queue.h"

namespace nvMolKit {

struct QueryPreprocessContext;

namespace {

void runPipelinedSubstructSearch(const std::vector<const RDKit::ROMol*>& targets,
                                 const MoleculesHost&                    queriesHost,
                                 const MoleculesDevice&                  queriesDevice,
                                 const LeafSubpatterns&                  leafSubpatterns,
                                 const QueryPreprocessContext&           queryContext,
                                 SubstructSearchResults&                 results,
                                 SubstructAlgorithm                      algorithm,
                                 cudaStream_t                            stream,
                                 const SubstructSearchConfig&            config,
                                 int                                     effectivePreprocessingThreads,
                                 RDKitFallbackQueue*                     fallbackQueue,
                                 HasSubstructMatchResults*               boolResults  = nullptr,
                                 std::vector<int>*                       countResults = nullptr);

}  // anonymous namespace

// =============================================================================
// RDKit Fallback Implementation
// =============================================================================

void processWithRDKitFallback(const RDKit::ROMol*       target,
                              const RDKit::ROMol*       query,
                              int                       targetIdx,
                              int                       queryIdx,
                              SubstructSearchResults&   results,
                              std::mutex&               resultsMutex,
                              int                       maxMatches,
                              HasSubstructMatchResults* boolResults,
                              std::vector<int>*         countResults) {
  RDKit::SubstructMatchParameters params;
  params.uniquify             = false;
  params.maxMatches           = (maxMatches > 0) ? static_cast<unsigned int>(maxMatches) : 0;
  params.useChirality         = false;
  params.useQueryQueryMatches = false;

  std::vector<RDKit::MatchVectType> rdkitMatches = RDKit::SubstructMatch(*target, *query, params);

  const int matchCount = static_cast<int>(rdkitMatches.size());
  if (matchCount == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(resultsMutex);

  if (boolResults) {
    boolResults->setMatch(targetIdx, queryIdx, true);
  } else if (countResults) {
    const int pairIdx        = targetIdx * results.numQueries + queryIdx;
    (*countResults)[pairIdx] = matchCount;
  } else {
    std::vector<std::vector<int>> convertedMatches;
    convertedMatches.reserve(rdkitMatches.size());
    for (const auto& match : rdkitMatches) {
      std::vector<int> mapping(match.size());
      for (size_t i = 0; i < match.size(); ++i) {
        mapping[i] = match[i].second;
      }
      convertedMatches.push_back(std::move(mapping));
    }

    auto& targetMatches = results.getMatchesMut(targetIdx, queryIdx);
    targetMatches.insert(targetMatches.end(),
                         std::make_move_iterator(convertedMatches.begin()),
                         std::make_move_iterator(convertedMatches.end()));
  }
}

RDKitFallbackQueue::RDKitFallbackQueue(const std::vector<const RDKit::ROMol*>* targets,
                                       const std::vector<const RDKit::ROMol*>* queries,
                                       SubstructSearchResults*                 results,
                                       std::mutex*                             resultsMutex,
                                       int                                     maxMatches,
                                       HasSubstructMatchResults*               boolResults,
                                       std::vector<int>*                       countResults)
    : targets_(targets),
      queries_(queries),
      results_(results),
      boolResults_(boolResults),
      countResults_(countResults),
      resultsMutex_(resultsMutex),
      maxMatches_(maxMatches) {}

void RDKitFallbackQueue::enqueue(const std::vector<RDKitFallbackEntry>& entries) {
  if (entries.empty())
    return;
  queue_.pushBatch(entries);
}

void RDKitFallbackQueue::enqueue(const RDKitFallbackEntry& entry) {
  queue_.push(entry);
}

void RDKitFallbackQueue::registerProducer() {
  std::lock_guard<std::mutex> lock(producerMutex_);
  ++activeProducers_;
}

void RDKitFallbackQueue::unregisterProducer() {
  std::lock_guard<std::mutex> lock(producerMutex_);
  --activeProducers_;
  closeQueueIfDone();
}

void RDKitFallbackQueue::shutdown() {
  std::lock_guard<std::mutex> lock(producerMutex_);
  shutdown_ = true;
  closeQueueIfDone();
}

void RDKitFallbackQueue::closeQueueIfDone() {
  if (shutdown_ || activeProducers_ == 0) {
    queue_.close();
  }
}

void RDKitFallbackQueue::workerLoop() {
  while (true) {
    auto optEntry = queue_.pop();
    if (!optEntry) {
      return;
    }
    processEntry(*optEntry);
  }
}

size_t RDKitFallbackQueue::processedCount() const {
  return processedCount_.load(std::memory_order_relaxed);
}

std::vector<RDKitFallbackEntry> RDKitFallbackQueue::drainToVector() {
  std::vector<RDKitFallbackEntry> result;
  while (auto opt = queue_.tryPop()) {
    result.push_back(std::move(*opt));
  }
  return result;
}

std::mutex& RDKitFallbackQueue::getResultsMutex() {
  return *resultsMutex_;
}

bool RDKitFallbackQueue::tryProcessOne() {
  auto optEntry = queue_.tryPop();
  if (!optEntry) {
    return false;
  }
  processEntry(*optEntry);
  return true;
}

bool RDKitFallbackQueue::hasWork() const {
  return !queue_.empty();
}

void RDKitFallbackQueue::processEntry(const RDKitFallbackEntry& entry) {
  ScopedNvtxRange pairRange("RDKit fallback T" + std::to_string(entry.originalTargetIdx) + "/Q" +
                            std::to_string(entry.originalQueryIdx));

  const RDKit::ROMol* target = (*targets_)[entry.originalTargetIdx];
  const RDKit::ROMol* query  = (*queries_)[entry.originalQueryIdx];

  const int effectiveMaxMatches = boolResults_ ? 1 : maxMatches_;
  processWithRDKitFallback(target,
                           query,
                           entry.originalTargetIdx,
                           entry.originalQueryIdx,
                           *results_,
                           *resultsMutex_,
                           effectiveMaxMatches,
                           boolResults_,
                           countResults_);

  processedCount_.fetch_add(1, std::memory_order_relaxed);
}

// =============================================================================
// Pipelined Batch Processing Types (internal, but needs external linkage for forward decl)
// =============================================================================

std::pair<int, int> getStreamPriorityRange();

static constexpr size_t kPinnedHostAlignment = 256;

static size_t alignPinnedOffset(const size_t offset) {
  return (offset + kPinnedHostAlignment - 1) & ~(kPinnedHostAlignment - 1);
}

static size_t computePinnedHostBufferBytes(int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth) {
  size_t offset = 0;

  auto addBlock = [&](const size_t bytes) {
    offset = alignPinnedOffset(offset);
    offset += bytes;
  };

  addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));
  addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize + 1));
  addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));
  addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));
  addBlock(sizeof(int16_t) * static_cast<size_t>(maxMatchIndicesEstimate));

  for (int i = 0; i <= kMaxRecursionDepth; ++i) {
    addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));
    addBlock(sizeof(int) * static_cast<size_t>(maxBatchSize));
  }

  for (int i = 0; i < 2; ++i) {
    addBlock(sizeof(BatchedPatternEntry) * static_cast<size_t>(maxPatternsPerDepth));
  }

  return offset;
}

struct PinnedHostBuffer {
  PinnedHostView<int>     pairIndices;
  PinnedHostView<int>     miniBatchPairMatchStarts;
  PinnedHostView<int>     matchCounts;
  PinnedHostView<int>     reportedCounts;
  PinnedHostView<int16_t> matchIndices;

  std::array<PinnedHostView<int>, kMaxRecursionDepth + 1> matchGlobalPairIndicesHost = {};
  std::array<PinnedHostView<int>, kMaxRecursionDepth + 1> matchBatchLocalIndicesHost = {};
  std::array<PinnedHostView<BatchedPatternEntry>, 2>      patternsAtDepthHost        = {};
};

class PinnedHostBufferPool {
 public:
  void initialize(int poolSize, int maxBatchSize, int maxMatchIndicesEstimate, int maxPatternsPerDepth) {
    buffers_.clear();
    available_ = std::make_unique<ThreadSafeQueue<PinnedHostBuffer*>>();

    buffers_.reserve(static_cast<size_t>(poolSize));
    for (int i = 0; i < poolSize; ++i) {
      auto buffer = createBuffer(maxBatchSize, maxMatchIndicesEstimate, maxPatternsPerDepth);
      available_->push(buffer.get());
      buffers_.push_back(std::move(buffer));
    }
  }

  PinnedHostBuffer* acquire() {
    auto opt = available_->pop();
    return opt.value_or(nullptr);
  }

  void release(PinnedHostBuffer* buffer) {
    if (buffer != nullptr) {
      available_->push(buffer);
    }
  }

  void shutdown() { available_->close(); }

 private:
  static std::unique_ptr<PinnedHostBuffer> createBuffer(int maxBatchSize,
                                                        int maxMatchIndicesEstimate,
                                                        int maxPatternsPerDepth) {
    const size_t bufferBytes = computePinnedHostBufferBytes(maxBatchSize, maxMatchIndicesEstimate, maxPatternsPerDepth);
    PinnedHostAllocator allocator(bufferBytes);
    auto                buffer = std::make_unique<PinnedHostBuffer>();

    buffer->pairIndices              = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
    buffer->miniBatchPairMatchStarts = allocator.allocate<int>(static_cast<size_t>(maxBatchSize + 1));
    buffer->matchCounts              = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
    buffer->reportedCounts           = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
    if (maxMatchIndicesEstimate > 0) {
      buffer->matchIndices = allocator.allocate<int16_t>(static_cast<size_t>(maxMatchIndicesEstimate));
    }

    for (int i = 0; i <= kMaxRecursionDepth; ++i) {
      buffer->matchGlobalPairIndicesHost[i] = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
      buffer->matchBatchLocalIndicesHost[i] = allocator.allocate<int>(static_cast<size_t>(maxBatchSize));
    }

    for (int i = 0; i < 2; ++i) {
      buffer->patternsAtDepthHost[i] =
        allocator.allocate<BatchedPatternEntry>(static_cast<size_t>(maxPatternsPerDepth));
    }

    return buffer;
  }

  std::vector<std::unique_ptr<PinnedHostBuffer>>      buffers_;
  std::unique_ptr<ThreadSafeQueue<PinnedHostBuffer*>> available_;
};

struct GpuExecutor {
  int miniBatchPairOffset = 0;
  int numPairsInMiniBatch = 0;
  int totalMatchIndices   = 0;

  // Precomputed recursive mini-batch setup (populated by prepareRecursiveMiniBatchOnCPU)
  int                                                                  recursiveMaxDepth      = 0;
  int                                                                  firstTargetInMiniBatch = 0;
  int                                                                  numTargetsInMiniBatch  = 0;
  std::array<std::vector<BatchedPatternEntry>, kMaxRecursionDepth + 1> patternsAtDepth;

  // Streams and events (declared first so they're destroyed last)
  ScopedStream    computeStream;
  ScopedCudaEvent copyDoneEvent;
  ScopedCudaEvent allocDoneEvent;
  ScopedCudaEvent targetsReadyEvent;

  // Recursive pipeline (inlined from RecursivePipelineContext)
  ScopedStreamWithPriority                                   recursiveStream;
  ScopedStreamWithPriority                                   postRecursionStream;
  std::array<ScopedCudaEvent, kMaxRecursionDepth>            depthEvents;
  ScopedCudaEvent                                            recursiveDoneEvent;
  ScopedCudaEvent                                            postRecursionDoneEvent;
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchGlobalPairIndices;
  std::array<AsyncDeviceVector<int>, kMaxRecursionDepth + 1> matchMiniBatchLocalIndices;
  std::array<int, kMaxRecursionDepth + 1>                    matchPairsCounts    = {};
  int                                                        maxDepthInMiniBatch = 0;

  RecursiveScratchBuffers recursiveScratch;
  MiniBatchResultsDevice  deviceResults;
  AsyncDeviceVector<int>  pairIndicesDev;
  MoleculesDevice         targetsDevice;

  int deviceId = 0;  ///< GPU device ID this executor is assigned to

  GpuExecutor(int executorIdx, int gpuDeviceId)
      : computeStream(("executor" + std::to_string(executorIdx) + "_mainStream").c_str()),
        recursiveStream(getStreamPriorityRange().first,
                        ("executor" + std::to_string(executorIdx) + "_priorityRecursiveStream").c_str()),
        postRecursionStream(getStreamPriorityRange().second,
                            ("executor" + std::to_string(executorIdx) + "_postRecursionStream").c_str()),
        recursiveScratch(nullptr),
        deviceId(gpuDeviceId) {}

  void initializeForStream() {
    cudaStream_t s         = computeStream.stream();
    cudaStream_t recStream = recursiveStream.stream();
    deviceResults.setStream(s);
    pairIndicesDev.setStream(s);
    recursiveScratch.setStream(recStream);
  }

  cudaStream_t stream() const { return computeStream.stream(); }
};

struct MiniBatchPlan {
  int                                                                  miniBatchPairOffset    = 0;
  int                                                                  numPairsInMiniBatch    = 0;
  int                                                                  totalMatchIndices      = 0;
  int                                                                  recursiveMaxDepth      = 0;
  int                                                                  firstTargetInMiniBatch = 0;
  int                                                                  numTargetsInMiniBatch  = 0;
  int                                                                  maxDepthInMiniBatch    = 0;
  std::array<std::vector<BatchedPatternEntry>, kMaxRecursionDepth + 1> patternsAtDepth;
  std::array<int, kMaxRecursionDepth + 1>                              matchPairsCounts = {};
};

struct PreparedMiniBatch {
  std::shared_ptr<MoleculesHost>    targetsHost;
  std::shared_ptr<std::vector<int>> targetOriginalIndices;
  std::shared_ptr<std::vector<int>> targetAtomCounts;
  ThreadWorkerContext               ctx;
  MiniBatchPlan                     plan;
  PinnedHostBuffer*                 pinnedBuffer = nullptr;
};

using PreparedBatchQueue = ThreadSafeQueue<std::unique_ptr<PreparedMiniBatch>>;

struct QueryPreprocessContext {
  PinnedHostVector<int> queryAtomCounts;
  std::vector<int>      queryDepths;
  std::vector<int>      queryMaxDepths;
  std::vector<int8_t>   queryHasPatterns;
  int                   numQueries    = 0;
  int                   maxQueryAtoms = 0;
};

// =============================================================================
// LeafSubpatterns Implementation
// =============================================================================

void LeafSubpatterns::buildAllPatterns(const MoleculesHost& queriesHost) {
  ScopedNvtxRange buildRange("LeafSubpatterns::buildAllPatterns");

  const int numQueries = static_cast<int>(queriesHost.numMolecules());

  // First pass: build pattern molecules and register in patternIndexMap
  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
    if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      LeafSubpatternKey key{queryIdx, entry.patternId};
      if (patternIndexMap.find(key) != patternIndexMap.end()) {
        continue;
      }

      int molIdx = static_cast<int>(patternsHost.numMolecules());

      std::vector<std::pair<int, int>> childrenByLocalId;
      for (const auto& p : recursiveInfo.patterns) {
        if (p.parentPatternId == entry.patternId) {
          childrenByLocalId.emplace_back(p.localIdInParent, p.patternId);
        }
      }
      std::sort(childrenByLocalId.begin(), childrenByLocalId.end());

      std::vector<int> childPatternIds;
      for (const auto& [localId, childId] : childrenByLocalId) {
        childPatternIds.push_back(childId);
      }

      if constexpr (kDebugPaintRecursive) {
        printf("[LeafSubpatterns] buildAllPatterns: queryIdx=%d, patternId=%d, found %zu children: [",
               queryIdx,
               entry.patternId,
               childPatternIds.size());
        for (size_t i = 0; i < childPatternIds.size(); ++i) {
          printf("%d%s", childPatternIds[i], i + 1 < childPatternIds.size() ? "," : "");
        }
        printf("]\n");
      }

      if (childPatternIds.empty()) {
        addQueryToBatch(entry.queryMol, patternsHost);
      } else {
        addQueryToBatch(entry.queryMol, patternsHost, childPatternIds);
      }

      patternIndexMap[key] = molIdx;
    }
  }

  // Second pass: build precomputed BatchedPatternEntry structures
  perQueryPatterns.resize(numQueries);
  perQueryMaxDepth.resize(numQueries, 0);

  for (int queryIdx = 0; queryIdx < numQueries; ++queryIdx) {
    if (queryIdx >= static_cast<int>(queriesHost.recursivePatterns.size())) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    perQueryMaxDepth[queryIdx] = recursiveInfo.maxDepth;

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      const int patternMolIdx = getPatternIndex(queryIdx, entry.patternId);
      if (patternMolIdx < 0) {
        continue;
      }

      BatchedPatternEntry batchEntry;
      batchEntry.mainQueryIdx    = queryIdx;
      batchEntry.patternId       = entry.patternId;
      batchEntry.patternMolIdx   = patternMolIdx;
      batchEntry.depth           = entry.depth;
      batchEntry.localIdInParent = entry.localIdInParent;

      perQueryPatterns[queryIdx][entry.depth].push_back(batchEntry);
    }
  }
}

void LeafSubpatterns::syncToDevice(cudaStream_t stream) {
  ScopedNvtxRange syncRange("LeafSubpatterns::syncToDevice");

  if (!patternsHost.numMolecules()) {
    return;
  }
  patternsDevice.copyFromHost(patternsHost, stream);
}

// =============================================================================
// Stream Priority Helper
// =============================================================================

std::pair<int, int> getStreamPriorityRange() {
  int leastPriority    = 0;
  int greatestPriority = 0;
  cudaCheckError(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  return {greatestPriority, leastPriority};
}

// =============================================================================
// MiniBatchResultsDevice Implementation
// =============================================================================

void MiniBatchResultsDevice::setStream(cudaStream_t stream) {
  stream_ = stream;
  matchCounts_.setStream(stream);
  reportedCounts_.setStream(stream);
  pairMatchStarts_.setStream(stream);
  matchIndices_.setStream(stream);
  queryAtomCounts_.setStream(stream);
  overflowBuffer_.setStream(stream);
  recursiveMatchBits_.setStream(stream);
  labelMatrixBuffer_.setStream(stream);
}

void MiniBatchResultsDevice::allocateMiniBatch(int        miniBatchSize,
                                               const int* miniBatchPairMatchStarts,
                                               int        totalMiniBatchMatchIndices,
                                               int        numQueries,
                                               int        maxTargetAtoms,
                                               int        numBuffersPerBlock,
                                               int        maxMatchesToFind,
                                               bool       countOnly) {
  ScopedNvtxRange allocRange("MiniBatchResultsDevice::allocateMiniBatch");

  miniBatchSize_              = miniBatchSize;
  numQueries_                 = numQueries;
  maxTargetAtoms_             = maxTargetAtoms;
  totalMiniBatchMatchIndices_ = countOnly ? 0 : totalMiniBatchMatchIndices;
  overflowBuffersPerBlock_    = numBuffersPerBlock;
  maxMatchesToFind_           = maxMatchesToFind;
  countOnly_                  = countOnly;

  if (matchCounts_.size() < static_cast<size_t>(miniBatchSize)) {
    matchCounts_.resize(static_cast<size_t>(miniBatchSize * 1.5));
  }

  if (!countOnly) {
    if (reportedCounts_.size() < static_cast<size_t>(miniBatchSize)) {
      reportedCounts_.resize(static_cast<size_t>(miniBatchSize * 1.5));
    }

    if (pairMatchStarts_.size() < static_cast<size_t>(miniBatchSize + 1)) {
      pairMatchStarts_.resize(static_cast<size_t>((miniBatchSize + 1) * 1.5));
    }
    pairMatchStarts_.copyFromHost(miniBatchPairMatchStarts, miniBatchSize + 1);

    if (matchIndices_.size() < static_cast<size_t>(totalMiniBatchMatchIndices)) {
      matchIndices_.resize(static_cast<size_t>(totalMiniBatchMatchIndices) * 3 / 2);
    }
  }

  const int overflowEntries = miniBatchSize * numBuffersPerBlock * kOverflowEntriesPerBuffer;
  if (overflowBuffer_.size() < static_cast<size_t>(overflowEntries)) {
    overflowBuffer_.resize(static_cast<size_t>(overflowEntries * 1.5));
  }

  const size_t recursiveBitsSize = static_cast<size_t>(miniBatchSize) * maxTargetAtoms;
  if (recursiveMatchBits_.size() < recursiveBitsSize) {
    recursiveMatchBits_.resize(static_cast<size_t>(recursiveBitsSize * 1.5));
  }
  recursiveMatchBits_.zero();

  const size_t labelMatrixSize = static_cast<size_t>(miniBatchSize) * kLabelMatrixWords;
  if (labelMatrixBuffer_.size() < labelMatrixSize) {
    labelMatrixBuffer_.resize(static_cast<size_t>(labelMatrixSize * 1.5));
  }
}

void MiniBatchResultsDevice::setQueryAtomCounts(const int* queryAtomCounts, size_t count) {
  if (queryAtomCounts_.size() < count) {
    queryAtomCounts_.resize(static_cast<size_t>(count * 1.5));
  }
  queryAtomCounts_.copyFromHost(queryAtomCounts, count);
}

void MiniBatchResultsDevice::zeroRecursiveBits() {
  recursiveMatchBits_.zero();
}

void MiniBatchResultsDevice::copyMiniBatchToHost(int*     hostMatchCounts,
                                                 int*     hostReportedCounts,
                                                 int16_t* hostMatchIndices) const {
  matchCounts_.copyToHost(hostMatchCounts, miniBatchSize_);
  reportedCounts_.copyToHost(hostReportedCounts, miniBatchSize_);
  matchIndices_.copyToHost(hostMatchIndices, totalMiniBatchMatchIndices_);
}

void MiniBatchResultsDevice::copyCountsOnlyToHost(int* hostMatchCounts) const {
  matchCounts_.copyToHost(hostMatchCounts, miniBatchSize_);
}

// =============================================================================
// Pipelined Batch Processing Implementation
// =============================================================================

namespace {

/**
 * @brief Precompute the pipeline schedule for a mini-batch.
 *
 * Groups pairs by their query's recursion depth and populates the host-side
 * index vectors for the recursive stream and match stream.
 *
 * @param plan Mini-batch plan to populate with schedule
 * @param ctx Worker context with cached query depths
 * @param buffer Pinned buffer holding host-side index arrays
 */
void precomputePipelineSchedule(MiniBatchPlan& plan, const ThreadWorkerContext& ctx, PinnedHostBuffer& buffer) {
  ScopedNvtxRange scheduleRange("CPU: precomputePipelineSchedule");
  int             maxDepth = 0;

  plan.matchPairsCounts.fill(0);

  int queryIdx = plan.miniBatchPairOffset % ctx.numQueries;
  for (int i = 0; i < plan.numPairsInMiniBatch; ++i) {
    const int depth  = ctx.queryDepths[queryIdx];
    const int offset = plan.matchPairsCounts[depth]++;

    buffer.matchGlobalPairIndicesHost[depth][offset] = buffer.pairIndices[i];
    buffer.matchBatchLocalIndicesHost[depth][offset] = i;

    if (depth > maxDepth) {
      maxDepth = depth;
    }

    if (++queryIdx >= ctx.numQueries) {
      queryIdx = 0;
    }
  }
  plan.maxDepthInMiniBatch = maxDepth;
}

void prepareRecursiveMiniBatchOnCPU(MiniBatchPlan&             plan,
                                    const ThreadWorkerContext& ctx,
                                    const LeafSubpatterns&     leafSubpatterns,
                                    PinnedHostBuffer&          buffer) {
  ScopedNvtxRange prepRecRange("prepareRecursiveMiniBatchOnCPU");

  precomputePipelineSchedule(plan, ctx, buffer);

  for (auto& vec : plan.patternsAtDepth) {
    vec.clear();
  }

  const int numUniqueQueries = std::min(plan.numPairsInMiniBatch, ctx.numQueries);

  int maxDepth = 0;
  int queryIdx = plan.miniBatchPairOffset % ctx.numQueries;
  for (int i = 0; i < numUniqueQueries; ++i) {
    if (ctx.queryHasPatterns[queryIdx]) {
      const int queryMaxDepth = ctx.queryMaxDepths[queryIdx];
      if (queryMaxDepth > maxDepth) {
        maxDepth = queryMaxDepth;
      }

      for (int d = 0; d <= queryMaxDepth; ++d) {
        const auto& srcEntries  = leafSubpatterns.perQueryPatterns[queryIdx][d];
        auto&       destEntries = plan.patternsAtDepth[d];
        destEntries.insert(destEntries.end(), srcEntries.begin(), srcEntries.end());
      }
    }

    if (++queryIdx >= ctx.numQueries) {
      queryIdx = 0;
    }
  }
  plan.recursiveMaxDepth = maxDepth;

  plan.firstTargetInMiniBatch     = plan.miniBatchPairOffset / ctx.numQueries;
  const int lastTargetInMiniBatch = (plan.miniBatchPairOffset + plan.numPairsInMiniBatch - 1) / ctx.numQueries;
  plan.numTargetsInMiniBatch      = lastTargetInMiniBatch - plan.firstTargetInMiniBatch + 1;
}

void prepareMiniBatchOnCPU(MiniBatchPlan&             plan,
                           PinnedHostBuffer&          buffer,
                           const ThreadWorkerContext& ctx,
                           const LeafSubpatterns&     leafSubpatterns,
                           int                        miniBatchPairOffset,
                           int                        maxPairsInMiniBatch) {
  ScopedNvtxRange prepRange("prepareMiniBatchOnCPU");

  const int numPairs            = ctx.numTargets * ctx.numQueries;
  const int miniBatchEnd        = std::min(miniBatchPairOffset + maxPairsInMiniBatch, numPairs);
  const int numPairsInMiniBatch = miniBatchEnd - miniBatchPairOffset;

  plan.miniBatchPairOffset = miniBatchPairOffset;
  plan.numPairsInMiniBatch = numPairsInMiniBatch;

  const bool useMaxMatchesLimit = ctx.maxMatches > 0;
  int        targetIdx          = miniBatchPairOffset / ctx.numQueries;
  int        queryIdx           = miniBatchPairOffset % ctx.numQueries;

  buffer.miniBatchPairMatchStarts[0] = 0;
  for (int i = 0; i < numPairsInMiniBatch; ++i) {
    const int targetAtoms  = (*ctx.targetAtomCounts)[targetIdx];
    const int queryAtoms   = ctx.queryAtomCounts[queryIdx];
    const int pairCapacity = useMaxMatchesLimit ? (ctx.maxMatches * queryAtoms) : (targetAtoms * queryAtoms);
    buffer.miniBatchPairMatchStarts[i + 1] = buffer.miniBatchPairMatchStarts[i] + pairCapacity;
    buffer.pairIndices[i]                  = miniBatchPairOffset + i;

    if (++queryIdx >= ctx.numQueries) {
      queryIdx = 0;
      ++targetIdx;
    }
  }
  plan.totalMatchIndices = buffer.miniBatchPairMatchStarts[numPairsInMiniBatch];

  prepareRecursiveMiniBatchOnCPU(plan, ctx, leafSubpatterns, buffer);
}

/**
 * @brief Launch label matrix and match kernels for a subset of pairs.
 */
void launchLabelAndMatch(int                        numPairsInGroup,
                         GpuExecutor&               executor,
                         const ThreadWorkerContext& ctx,
                         MoleculesDevice&           targetsDevice,
                         const MoleculesDevice&     queriesDevice,
                         SubstructAlgorithm         algorithm,
                         cudaStream_t               stream,
                         int                        depthGroupIdx,
                         const PinnedHostBuffer&    hostBuffer) {
  ScopedNvtxRange launchRange("launchLabelAndMatch depth=" + std::to_string(depthGroupIdx));

  if (numPairsInGroup == 0) {
    return;
  }

  const int* globalPairIndicesHost        = hostBuffer.matchGlobalPairIndicesHost[depthGroupIdx].data();
  const int* miniBatchLocalIndicesHostPtr = hostBuffer.matchBatchLocalIndicesHost[depthGroupIdx].data();

  auto& globalPairIndicesDev     = executor.matchGlobalPairIndices[depthGroupIdx];
  auto& miniBatchLocalIndicesDev = executor.matchMiniBatchLocalIndices[depthGroupIdx];

  globalPairIndicesDev.setStream(stream);
  if (globalPairIndicesDev.size() < static_cast<size_t>(numPairsInGroup)) {
    globalPairIndicesDev.resize(static_cast<size_t>(numPairsInGroup * 1.5));
  }
  globalPairIndicesDev.copyFromHost(globalPairIndicesHost, numPairsInGroup);

  miniBatchLocalIndicesDev.setStream(stream);
  if (miniBatchLocalIndicesDev.size() < static_cast<size_t>(numPairsInGroup)) {
    miniBatchLocalIndicesDev.resize(static_cast<size_t>(numPairsInGroup * 1.5));
  }
  miniBatchLocalIndicesDev.copyFromHost(miniBatchLocalIndicesHostPtr, numPairsInGroup);

  launchLabelMatrixKernel(ctx.templateConfig,
                          targetsDevice.view<MoleculeType::Target>(),
                          queriesDevice.view<MoleculeType::Query>(),
                          globalPairIndicesDev.data(),
                          numPairsInGroup,
                          ctx.numQueries,
                          executor.deviceResults.labelMatrixBuffer(),
                          executor.deviceResults.recursiveMatchBits(),
                          executor.deviceResults.maxTargetAtoms(),
                          miniBatchLocalIndicesDev.data(),
                          stream);

  launchSubstructMatchKernel(ctx.templateConfig,
                             algorithm,
                             targetsDevice.view<MoleculeType::Target>(),
                             queriesDevice.view<MoleculeType::Query>(),
                             executor.deviceResults,
                             globalPairIndicesDev.data(),
                             numPairsInGroup,
                             ctx.numQueries,
                             miniBatchLocalIndicesDev.data(),
                             nullptr,
                             stream);
}

void launchRecursivePaintKernels(
  SubstructTemplateConfig                                                     templateConfig,
  const MoleculesDevice&                                                      targetsDevice,
  const LeafSubpatterns&                                                      leafSubpatterns,
  MiniBatchResultsDevice&                                                     miniBatchResults,
  int                                                                         numQueries,
  int                                                                         miniBatchPairOffset,
  int                                                                         miniBatchSize,
  SubstructAlgorithm                                                          algorithm,
  cudaStream_t                                                                stream,
  RecursiveScratchBuffers&                                                    scratch,
  const std::array<std::vector<BatchedPatternEntry>, kMaxRecursionDepth + 1>& patternsAtDepth,
  int                                                                         maxDepth,
  int                                                                         firstTargetInMiniBatch,
  int                                                                         numTargetsInMiniBatch,
  cudaEvent_t*                                                                depthEvents,
  int                                                                         numDepthEvents) {
  ScopedNvtxRange processRecursiveRange("launchRecursivePaintKernels");

  scratch.setStream(stream);

  constexpr int gsiBuffersPerBlock = 2;

  const int maxPaintPairsPerSubBatch = std::max(miniBatchSize, 1024);

  for (int currentDepth = 0; currentDepth <= maxDepth; ++currentDepth) {
    ScopedNvtxRange depthRange("Process recursive depth level " + std::to_string(currentDepth));

    const auto& patternsForDepth = patternsAtDepth[currentDepth];

    if (patternsForDepth.empty()) {
      if (currentDepth < numDepthEvents && depthEvents != nullptr) {
        cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
      }
      continue;
    }

    const size_t numPatterns         = patternsForDepth.size();
    const int    patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInMiniBatch);

    for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
      ScopedNvtxRange subBatchRange("Process sub-batch " + std::to_string(patternStart));

      const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
      const size_t numPatternsInSubBatch = patternEnd - patternStart;
      const size_t numBlocksInSubBatch   = numTargetsInMiniBatch * numPatternsInSubBatch;

      ScopedNvtxRange prepareRange("GPU: Upload pattern entries");
      const int       bufferIdx = scratch.acquireBufferIndex();
      scratch.waitForBuffer(bufferIdx);
      scratch.ensureCapacity(bufferIdx, static_cast<int>(numPatternsInSubBatch));
      for (size_t i = 0; i < numPatternsInSubBatch; ++i) {
        scratch.patternsAtDepthHost[bufferIdx][i] = patternsForDepth[patternStart + i];
      }
      prepareRange.pop();

      const int    buffersPerBlock = gsiBuffersPerBlock;
      const size_t overflowNeeded  = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

      if (scratch.overflow.size() < overflowNeeded) {
        scratch.overflow.resize(static_cast<size_t>(overflowNeeded * 1.5));
      }

      const size_t labelMatrixNeeded = numBlocksInSubBatch * kLabelMatrixWords;
      if (scratch.labelMatrixBuffer.size() < labelMatrixNeeded) {
        scratch.labelMatrixBuffer.resize(static_cast<size_t>(labelMatrixNeeded * 1.5));
      }

      if (scratch.patternEntries.size() < numPatternsInSubBatch) {
        scratch.patternEntries.resize(static_cast<size_t>(numPatternsInSubBatch * 1.5));
      }

      scratch.patternEntries.copyFromHost(scratch.patternsAtDepthHost[bufferIdx].data(), numPatternsInSubBatch);
      scratch.recordCopy(bufferIdx, scratch.patternEntries.stream());

      const uint32_t* recursiveBitsForLabel = (currentDepth > 0) ? miniBatchResults.recursiveMatchBits() : nullptr;

      launchLabelMatrixPaintKernel(templateConfig,
                                   targetsDevice.view<MoleculeType::Target>(),
                                   leafSubpatterns.view(),
                                   scratch.patternEntries.data(),
                                   static_cast<int>(numPatternsInSubBatch),
                                   numBlocksInSubBatch,
                                   numQueries,
                                   miniBatchPairOffset,
                                   miniBatchSize,
                                   scratch.labelMatrixBuffer.data(),
                                   firstTargetInMiniBatch,
                                   recursiveBitsForLabel,
                                   miniBatchResults.maxTargetAtoms(),
                                   stream);

      launchSubstructPaintKernel(templateConfig,
                                 algorithm,
                                 targetsDevice.view<MoleculeType::Target>(),
                                 leafSubpatterns.view(),
                                 scratch.patternEntries.data(),
                                 static_cast<int>(numPatternsInSubBatch),
                                 numBlocksInSubBatch,
                                 miniBatchResults.recursiveMatchBits(),
                                 miniBatchResults.maxTargetAtoms(),
                                 numQueries,
                                 0,
                                 0,
                                 miniBatchPairOffset,
                                 miniBatchSize,
                                 scratch.overflow.data(),
                                 scratch.overflow.data(),
                                 kOverflowEntriesPerBuffer,
                                 scratch.labelMatrixBuffer.data(),
                                 firstTargetInMiniBatch,
                                 stream);
    }

    if (currentDepth < numDepthEvents && depthEvents != nullptr) {
      cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
    }
  }

  cudaCheckError(cudaGetLastError());
}

void uploadAndLaunchMiniBatch(GpuExecutor&               executor,
                              const ThreadWorkerContext& ctx,
                              MoleculesDevice&           targetsDevice,
                              const MoleculesDevice&     queriesDevice,
                              const LeafSubpatterns&     leafSubpatterns,
                              SubstructAlgorithm         algorithm,
                              const PinnedHostBuffer&    hostBuffer) {
  ScopedNvtxRange uploadRange("uploadAndLaunchMiniBatch");

  cudaStream_t executorStream     = executor.stream();
  const int    numBuffersPerBlock = (algorithm == SubstructAlgorithm::GSI) ? 2 : 1;

  if (executor.maxDepthInMiniBatch == 0) {
    ScopedNvtxRange nonRecursiveRange("Non-recursive path");

    const int maxMatchesToFind = ctx.maxMatches > 0 ? ctx.maxMatches : -1;
    executor.deviceResults.allocateMiniBatch(executor.numPairsInMiniBatch,
                                             hostBuffer.miniBatchPairMatchStarts.data(),
                                             executor.totalMatchIndices,
                                             ctx.numQueries,
                                             ctx.maxTargetAtoms,
                                             numBuffersPerBlock,
                                             maxMatchesToFind,
                                             ctx.countOnly);
    executor.deviceResults.setQueryAtomCounts(ctx.queryAtomCounts, ctx.numQueries);

    if (executor.pairIndicesDev.size() < static_cast<size_t>(executor.numPairsInMiniBatch)) {
      executor.pairIndicesDev.resize(static_cast<size_t>(executor.numPairsInMiniBatch * 1.5));
    }
    executor.pairIndicesDev.copyFromHost(hostBuffer.pairIndices.data(), executor.numPairsInMiniBatch);

    launchLabelMatrixKernel(ctx.templateConfig,
                            targetsDevice.view<MoleculeType::Target>(),
                            queriesDevice.view<MoleculeType::Query>(),
                            executor.pairIndicesDev.data(),
                            executor.numPairsInMiniBatch,
                            ctx.numQueries,
                            executor.deviceResults.labelMatrixBuffer(),
                            executor.deviceResults.recursiveMatchBits(),
                            executor.deviceResults.maxTargetAtoms(),
                            nullptr,
                            executorStream);

    launchSubstructMatchKernel(ctx.templateConfig,
                               algorithm,
                               targetsDevice.view<MoleculeType::Target>(),
                               queriesDevice.view<MoleculeType::Query>(),
                               executor.deviceResults,
                               executor.pairIndicesDev.data(),
                               executor.numPairsInMiniBatch,
                               ctx.numQueries,
                               nullptr,
                               nullptr,
                               executorStream);
    return;
  }

  ScopedNvtxRange multiStreamRange("Multi-stream recursive pipeline");

  cudaStream_t recursiveStream = executor.recursiveStream.stream();

  const int maxMatchesToFind = ctx.maxMatches > 0 ? ctx.maxMatches : -1;
  executor.deviceResults.allocateMiniBatch(executor.numPairsInMiniBatch,
                                           hostBuffer.miniBatchPairMatchStarts.data(),
                                           executor.totalMatchIndices,
                                           ctx.numQueries,
                                           ctx.maxTargetAtoms,
                                           numBuffersPerBlock,
                                           maxMatchesToFind,
                                           ctx.countOnly);
  executor.deviceResults.setQueryAtomCounts(ctx.queryAtomCounts, ctx.numQueries);

  cudaCheckError(cudaEventRecord(executor.allocDoneEvent.event(), executorStream));

  ScopedNvtxRange waitAllocRange("Wait: recursiveStream waits for alloc");
  cudaCheckError(cudaStreamWaitEvent(recursiveStream, executor.allocDoneEvent.event(), 0));
  waitAllocRange.pop();

  std::array<cudaEvent_t, kMaxRecursionDepth> depthEventPtrs;
  for (int i = 0; i < kMaxRecursionDepth; ++i) {
    depthEventPtrs[i] = executor.depthEvents[i].event();
  }

  ScopedNvtxRange preprocRange("launchRecursivePaintKernels (recursiveStream)");
  launchRecursivePaintKernels(ctx.templateConfig,
                              targetsDevice,
                              leafSubpatterns,
                              executor.deviceResults,
                              ctx.numQueries,
                              executor.miniBatchPairOffset,
                              executor.numPairsInMiniBatch,
                              algorithm,
                              recursiveStream,
                              executor.recursiveScratch,
                              executor.patternsAtDepth,
                              executor.recursiveMaxDepth,
                              executor.firstTargetInMiniBatch,
                              executor.numTargetsInMiniBatch,
                              depthEventPtrs.data(),
                              kMaxRecursionDepth);
  preprocRange.pop();

  ScopedNvtxRange depth0Range("Match depth-0 pairs (executorStream)");
  launchLabelAndMatch(executor.matchPairsCounts[0],
                      executor,
                      ctx,
                      targetsDevice,
                      queriesDevice,
                      algorithm,
                      executorStream,
                      0,
                      hostBuffer);
  depth0Range.pop();

  cudaStream_t postStream = executor.postRecursionStream.stream();
  cudaCheckError(cudaStreamWaitEvent(postStream, executor.allocDoneEvent.event(), 0));

  for (int depth = 1; depth <= executor.maxDepthInMiniBatch; ++depth) {
    ScopedNvtxRange depthRange("Match depth-" + std::to_string(depth) + " pairs (postRecursionStream)");

    ScopedNvtxRange waitRange("Wait: postRecursionStream waits for depth event");
    cudaCheckError(cudaStreamWaitEvent(postStream, depthEventPtrs[depth - 1], 0));
    waitRange.pop();

    launchLabelAndMatch(executor.matchPairsCounts[depth],
                        executor,
                        ctx,
                        targetsDevice,
                        queriesDevice,
                        algorithm,
                        postStream,
                        depth,
                        hostBuffer);
  }
  cudaCheckError(cudaEventRecord(executor.postRecursionDoneEvent.event(), postStream));

  cudaCheckError(cudaEventRecord(executor.recursiveDoneEvent.event(), recursiveStream));
  cudaCheckError(cudaStreamWaitEvent(executorStream, executor.recursiveDoneEvent.event(), 0));
  cudaCheckError(cudaStreamWaitEvent(executorStream, executor.postRecursionDoneEvent.event(), 0));
}

void initiateResultsCopyToHost(GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
  ScopedNvtxRange copyRange("initiateResultsCopyToHost");
  executor.deviceResults.copyMiniBatchToHost(hostBuffer.matchCounts.data(),
                                             hostBuffer.reportedCounts.data(),
                                             hostBuffer.matchIndices.data());
  cudaCheckError(cudaEventRecord(executor.copyDoneEvent.event(), executor.stream()));
}

struct PairUpdate {
  int targetIdx;
  int queryIdx;
  int miniBatchLocalOffset;
  int reportedMatches;
  int queryAtoms;
};

void accumulateMiniBatchResults(GpuExecutor&               executor,
                                const ThreadWorkerContext& ctx,
                                SubstructSearchResults&    results,
                                std::mutex&                resultsMutex,
                                const PinnedHostBuffer&    hostBuffer,
                                RDKitFallbackQueue*        fallbackQueue = nullptr) {
  ScopedNvtxRange accumRange("accumulateMiniBatchResults");

  // Phase 1: Build update list without any locks

  std::vector<PairUpdate> updates;
  updates.reserve(executor.numPairsInMiniBatch);

  for (int i = 0; i < executor.numPairsInMiniBatch; ++i) {
    const int pairIdxInBatch = hostBuffer.pairIndices[i];
    const int localTargetIdx = pairIdxInBatch / ctx.numQueries;
    const int queryIdx       = pairIdxInBatch % ctx.numQueries;

    const int targetIdx = (*ctx.targetOriginalIndices)[localTargetIdx];

    const int queryAtoms      = ctx.queryAtomCounts[queryIdx];
    const int actualMatches   = hostBuffer.matchCounts[i];
    const int reportedMatches = hostBuffer.reportedCounts[i];

    const bool isBufferOverflow = (actualMatches > reportedMatches) && (ctx.maxMatches == 0);
    if (isBufferOverflow && fallbackQueue != nullptr) {
      fallbackQueue->enqueue({targetIdx, queryIdx});
      continue;
    }

    if (reportedMatches > 0) {
      updates.push_back({targetIdx, queryIdx, hostBuffer.miniBatchPairMatchStarts[i], reportedMatches, queryAtoms});
    }
  }

  if (updates.empty()) {
    return;
  }

  // Phase 2: Single lock acquisition to get all hash map references
  std::vector<std::vector<std::vector<int>>*> matchRefs;
  matchRefs.reserve(updates.size());

  {
    std::lock_guard<std::mutex> lock(resultsMutex);
    for (const auto& u : updates) {
      matchRefs.push_back(&results.getMatchesMut(u.targetIdx, u.queryIdx));
    }
  }
  // Lock released here

  // Phase 3: Copy all match data without holding the lock
  // Safe because mini-batches have non-overlapping pair indices
  for (size_t i = 0; i < updates.size(); ++i) {
    const auto& u             = updates[i];
    auto&       targetMatches = *matchRefs[i];

    targetMatches.reserve(targetMatches.size() + u.reportedMatches);
    const int16_t* src = hostBuffer.matchIndices.data() + u.miniBatchLocalOffset;

    for (int m = 0; m < u.reportedMatches; ++m) {
      auto& match = targetMatches.emplace_back(u.queryAtoms);
      for (int a = 0; a < u.queryAtoms; ++a) {
        match[a] = src[m * u.queryAtoms + a];
      }
    }
  }
}

void initiateCountsOnlyCopyToHost(GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
  ScopedNvtxRange copyRange("initiateCountsOnlyCopyToHost");
  executor.deviceResults.copyCountsOnlyToHost(hostBuffer.matchCounts.data());
  cudaCheckError(cudaEventRecord(executor.copyDoneEvent.event(), executor.stream()));
}

void accumulateMiniBatchResultsBoolean(GpuExecutor&               executor,
                                       const ThreadWorkerContext& ctx,
                                       HasSubstructMatchResults&  results,
                                       std::mutex&                resultsMutex,
                                       const PinnedHostBuffer&    hostBuffer) {
  ScopedNvtxRange accumRange("accumulateMiniBatchResultsBoolean");

  std::lock_guard<std::mutex> lock(resultsMutex);

  for (int i = 0; i < executor.numPairsInMiniBatch; ++i) {
    if (hostBuffer.matchCounts[i] == 0) {
      continue;
    }

    const int pairIdxInBatch = hostBuffer.pairIndices[i];
    const int localTargetIdx = pairIdxInBatch / ctx.numQueries;
    const int queryIdx       = pairIdxInBatch % ctx.numQueries;

    const int targetIdx = (*ctx.targetOriginalIndices)[localTargetIdx];

    results.setMatch(targetIdx, queryIdx, true);
  }
}

void accumulateMiniBatchResultsCounts(GpuExecutor&               executor,
                                      const ThreadWorkerContext& ctx,
                                      std::vector<int>&          counts,
                                      std::mutex&                resultsMutex,
                                      const PinnedHostBuffer&    hostBuffer) {
  ScopedNvtxRange accumRange("accumulateMiniBatchResultsCounts");

  std::lock_guard<std::mutex> lock(resultsMutex);

  for (int i = 0; i < executor.numPairsInMiniBatch; ++i) {
    const int pairIdxInBatch = hostBuffer.pairIndices[i];
    const int localTargetIdx = pairIdxInBatch / ctx.numQueries;
    const int queryIdx       = pairIdxInBatch % ctx.numQueries;

    const int targetIdx = (*ctx.targetOriginalIndices)[localTargetIdx];

    const int pairIdx = targetIdx * ctx.numQueries + queryIdx;
    counts[pairIdx]   = hostBuffer.matchCounts[i];
  }
}

constexpr int kMaxExecutorsPerRunner = 8;

struct InFlightBatch {
  GpuExecutor*                       executor = nullptr;
  std::unique_ptr<PreparedMiniBatch> batch;
};

void applyPreparedMiniBatch(GpuExecutor& executor, PreparedMiniBatch& batch) {
  executor.miniBatchPairOffset    = batch.plan.miniBatchPairOffset;
  executor.numPairsInMiniBatch    = batch.plan.numPairsInMiniBatch;
  executor.totalMatchIndices      = batch.plan.totalMatchIndices;
  executor.recursiveMaxDepth      = batch.plan.recursiveMaxDepth;
  executor.firstTargetInMiniBatch = batch.plan.firstTargetInMiniBatch;
  executor.numTargetsInMiniBatch  = batch.plan.numTargetsInMiniBatch;
  executor.maxDepthInMiniBatch    = batch.plan.maxDepthInMiniBatch;
  executor.matchPairsCounts       = batch.plan.matchPairsCounts;
  executor.patternsAtDepth        = std::move(batch.plan.patternsAtDepth);
}

template <typename InitiateCopyFunc, typename AccumulateFunc>
void runnerWorkerPipeline(int                       workerIdx,
                          const MoleculesDevice&    queriesDevice,
                          const LeafSubpatterns&    leafSubpatterns,
                          SubstructAlgorithm        algorithm,
                          int                       deviceId,
                          std::vector<GpuExecutor*> executors,
                          PreparedBatchQueue&       batchQueue,
                          PinnedHostBufferPool&     bufferPool,
                          InitiateCopyFunc&&        initiateCopy,
                          AccumulateFunc&&          accumulate,
                          RDKitFallbackQueue*       fallbackQueue,
                          std::atomic<bool>&        pipelineAbort,
                          std::exception_ptr&       exceptionPtr) {
  try {
    FallbackQueueProducerGuard producerGuard(fallbackQueue);
    ScopedNvtxRange            workerRange("runnerWorkerPipeline " + std::to_string(workerIdx) + " GPU" +
                                std::to_string(deviceId));
    const WithDevice           setDevice(deviceId);

    const int                                         executorsPerRunner = static_cast<int>(executors.size());
    std::array<InFlightBatch, kMaxExecutorsPerRunner> pending{};
    int                                               pendingHead  = 0;
    int                                               pendingTail  = 0;
    int                                               pendingCount = 0;

    auto drainOne = [&]() {
      InFlightBatch&  slot   = pending[pendingHead];
      GpuExecutor*    oldest = slot.executor;
      ScopedNvtxRange waitRange("Wait for D2H copy", NvtxColor::kRed);
      cudaCheckError(cudaEventSynchronize(oldest->copyDoneEvent.event()));
      waitRange.pop();

      ScopedNvtxRange accumRange("Accumulate mini-batch");
      accumulate(*oldest, slot.batch->ctx, *slot.batch->pinnedBuffer);
      accumRange.pop();

      bufferPool.release(slot.batch->pinnedBuffer);
      slot.batch.reset();

      pendingHead = (pendingHead + 1) % executorsPerRunner;
      --pendingCount;
    };

    while (true) {
      if (pendingCount == executorsPerRunner) {
        drainOne();
        continue;
      }

      std::unique_ptr<PreparedMiniBatch> batch;
      if (pendingCount > 0) {
        auto optBatch = batchQueue.tryPop();
        if (!optBatch) {
          drainOne();
          continue;
        }
        batch = std::move(*optBatch);
      } else {
        std::optional<std::unique_ptr<PreparedMiniBatch>> optBatch;
        {
          ScopedNvtxRange waitRange("Wait for prepared batch", NvtxColor::kRed);
          optBatch = batchQueue.pop();
        }
        if (!optBatch) {
          break;
        }
        batch = std::move(*optBatch);
      }

      GpuExecutor* executor = executors[pendingTail];
      applyPreparedMiniBatch(*executor, *batch);
      executor->recursiveScratch.setPinnedBuffer(batch->pinnedBuffer->patternsAtDepthHost,
                                                 static_cast<int>(batch->pinnedBuffer->patternsAtDepthHost[0].size()));

      executor->targetsDevice.copyFromHost(*batch->targetsHost, executor->stream());
      cudaCheckError(cudaEventRecord(executor->targetsReadyEvent.event(), executor->stream()));
      cudaCheckError(cudaStreamWaitEvent(executor->recursiveStream.stream(), executor->targetsReadyEvent.event(), 0));
      cudaCheckError(
        cudaStreamWaitEvent(executor->postRecursionStream.stream(), executor->targetsReadyEvent.event(), 0));

      ScopedNvtxRange launchRange("GPU launch prepared batch");
      uploadAndLaunchMiniBatch(*executor,
                               batch->ctx,
                               executor->targetsDevice,
                               queriesDevice,
                               leafSubpatterns,
                               algorithm,
                               *batch->pinnedBuffer);
      initiateCopy(*executor, *batch->pinnedBuffer);
      launchRange.pop();

      pending[pendingTail].executor = executor;
      pending[pendingTail].batch    = std::move(batch);
      pendingTail                   = (pendingTail + 1) % executorsPerRunner;
      ++pendingCount;
    }

    while (pendingCount > 0) {
      drainOne();
    }
  } catch (...) {
    exceptionPtr = std::current_exception();
    pipelineAbort.store(true, std::memory_order_release);
    batchQueue.close();
    bufferPool.shutdown();
  }
}

void runnerWorkerPipelineResults(int                       workerIdx,
                                 const MoleculesDevice&    queriesDevice,
                                 const LeafSubpatterns&    leafSubpatterns,
                                 SubstructSearchResults&   results,
                                 std::mutex&               resultsMutex,
                                 SubstructAlgorithm        algorithm,
                                 int                       deviceId,
                                 std::vector<GpuExecutor*> executors,
                                 PreparedBatchQueue&       batchQueue,
                                 PinnedHostBufferPool&     bufferPool,
                                 RDKitFallbackQueue*       fallbackQueue,
                                 std::atomic<bool>&        pipelineAbort,
                                 std::exception_ptr&       exceptionPtr) {
  auto initiateCopy = [&](GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
    initiateResultsCopyToHost(executor, hostBuffer);
  };
  auto accumulate = [&](GpuExecutor& executor, const ThreadWorkerContext& ctx, const PinnedHostBuffer& hostBuffer) {
    accumulateMiniBatchResults(executor, ctx, results, resultsMutex, hostBuffer, fallbackQueue);
  };
  runnerWorkerPipeline(workerIdx,
                       queriesDevice,
                       leafSubpatterns,
                       algorithm,
                       deviceId,
                       std::move(executors),
                       batchQueue,
                       bufferPool,
                       initiateCopy,
                       accumulate,
                       fallbackQueue,
                       pipelineAbort,
                       exceptionPtr);
}

void runnerWorkerPipelineBoolean(int                       workerIdx,
                                 const MoleculesDevice&    queriesDevice,
                                 const LeafSubpatterns&    leafSubpatterns,
                                 HasSubstructMatchResults& results,
                                 std::mutex&               resultsMutex,
                                 SubstructAlgorithm        algorithm,
                                 int                       deviceId,
                                 std::vector<GpuExecutor*> executors,
                                 PreparedBatchQueue&       batchQueue,
                                 PinnedHostBufferPool&     bufferPool,
                                 std::atomic<bool>&        pipelineAbort,
                                 std::exception_ptr&       exceptionPtr) {
  auto initiateCopy = [&](GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
    initiateCountsOnlyCopyToHost(executor, hostBuffer);
  };
  auto accumulate = [&](GpuExecutor& executor, const ThreadWorkerContext& ctx, const PinnedHostBuffer& hostBuffer) {
    accumulateMiniBatchResultsBoolean(executor, ctx, results, resultsMutex, hostBuffer);
  };
  runnerWorkerPipeline(workerIdx,
                       queriesDevice,
                       leafSubpatterns,
                       algorithm,
                       deviceId,
                       std::move(executors),
                       batchQueue,
                       bufferPool,
                       initiateCopy,
                       accumulate,
                       nullptr,
                       pipelineAbort,
                       exceptionPtr);
}

void runnerWorkerPipelineCounts(int                       workerIdx,
                                const MoleculesDevice&    queriesDevice,
                                const LeafSubpatterns&    leafSubpatterns,
                                std::vector<int>&         counts,
                                std::mutex&               resultsMutex,
                                SubstructAlgorithm        algorithm,
                                int                       deviceId,
                                std::vector<GpuExecutor*> executors,
                                PreparedBatchQueue&       batchQueue,
                                PinnedHostBufferPool&     bufferPool,
                                std::atomic<bool>&        pipelineAbort,
                                std::exception_ptr&       exceptionPtr) {
  auto initiateCopy = [&](GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
    initiateCountsOnlyCopyToHost(executor, hostBuffer);
  };
  auto accumulate = [&](GpuExecutor& executor, const ThreadWorkerContext& ctx, const PinnedHostBuffer& hostBuffer) {
    accumulateMiniBatchResultsCounts(executor, ctx, counts, resultsMutex, hostBuffer);
  };
  runnerWorkerPipeline(workerIdx,
                       queriesDevice,
                       leafSubpatterns,
                       algorithm,
                       deviceId,
                       std::move(executors),
                       batchQueue,
                       bufferPool,
                       initiateCopy,
                       accumulate,
                       nullptr,
                       pipelineAbort,
                       exceptionPtr);
}

}  // namespace

// =============================================================================
// Main API
// =============================================================================

namespace {

void runPipelinedSubstructSearch(const std::vector<const RDKit::ROMol*>& targets,
                                 const MoleculesHost&                    queriesHost,
                                 const MoleculesDevice&                  queriesDevice,
                                 const LeafSubpatterns&                  leafSubpatterns,
                                 const QueryPreprocessContext&           queryContext,
                                 SubstructSearchResults&                 results,
                                 SubstructAlgorithm                      algorithm,
                                 cudaStream_t                            stream,
                                 const SubstructSearchConfig&            config,
                                 int                                     effectivePreprocessingThreads,
                                 RDKitFallbackQueue*                     fallbackQueue,
                                 HasSubstructMatchResults*               boolResults,
                                 std::vector<int>*                       countResults) {
  (void)stream;
  const bool      countOnly  = (boolResults != nullptr) || (countResults != nullptr);
  const char*     rangeLabel = boolResults ?
                                 "runPipelinedHasSubstructMatch" :
                                 (countResults ? "runPipelinedCountSubstructMatches" : "runPipelinedSubstructSearch");
  ScopedNvtxRange e2eRange(rangeLabel);

  const int numTargets = static_cast<int>(targets.size());
  const int numQueries = queryContext.numQueries;
  if (numTargets == 0 || numQueries == 0) {
    return;
  }

  // Determine GPU list: empty gpuIds = current device only
  std::vector<int> gpuIds        = config.gpuIds;
  int              currentDevice = 0;
  cudaCheckError(cudaGetDevice(&currentDevice));
  if (gpuIds.empty()) {
    gpuIds.push_back(currentDevice);
  }
  const int numGpus = static_cast<int>(gpuIds.size());

  // Determine runner counts (per GPU, possibly limited by target count).
  const int runnersPerGpu = std::max(1, config.workerThreads);
  int       numRunners    = runnersPerGpu * numGpus;
  if (numRunners > numTargets) {
    numRunners = numTargets;
  }
  if (numRunners == 0) {
    return;
  }

  int executorsPerRunner;
  if (config.executorsPerRunner == -1) {
    executorsPerRunner = (numRunners == 1) ? 3 : 2;
  } else if (config.executorsPerRunner < 1 || config.executorsPerRunner > kMaxExecutorsPerRunner) {
    throw std::invalid_argument("executorsPerRunner must be -1 (auto) or between 1 and " +
                                std::to_string(kMaxExecutorsPerRunner));
  } else {
    executorsPerRunner = config.executorsPerRunner;
  }

  std::vector<int> workersPerGpu(numGpus, numRunners / numGpus);
  for (int i = 0; i < numRunners % numGpus; ++i) {
    workersPerGpu[i]++;
  }

  // Precompute max patterns per depth across all queries for pinned buffer sizing.
  int maxPatternsPerDepth = 256;
  for (int d = 0; d <= kMaxRecursionDepth; ++d) {
    int patternsAtThisDepth = 0;
    for (size_t q = 0; q < leafSubpatterns.perQueryPatterns.size(); ++q) {
      patternsAtThisDepth += static_cast<int>(leafSubpatterns.perQueryPatterns[q][d].size());
    }
    maxPatternsPerDepth = std::max(maxPatternsPerDepth, patternsAtThisDepth);
  }

  const int targetsPerBatch  = std::max(1, config.batchSize / numQueries);
  const int maxPairsPerBatch = std::max(1, config.batchSize);

  size_t maxMatchIndicesPerMiniBatch;
  if (countOnly) {
    maxMatchIndicesPerMiniBatch = 0;
  } else if (config.maxMatches > 0) {
    maxMatchIndicesPerMiniBatch =
      static_cast<size_t>(maxPairsPerBatch) * config.maxMatches * queryContext.maxQueryAtoms;
  } else {
    maxMatchIndicesPerMiniBatch = static_cast<size_t>(maxPairsPerBatch) * kMaxTargetAtoms * queryContext.maxQueryAtoms;
  }

  const int    poolSize = std::max(1, effectivePreprocessingThreads) * 2;
  const size_t perBufferSize =
    computePinnedHostBufferBytes(maxPairsPerBatch, static_cast<int>(maxMatchIndicesPerMiniBatch), maxPatternsPerDepth);
  const size_t totalPinnedBytes = static_cast<size_t>(poolSize) * perBufferSize;

  const long   pages      = sysconf(_SC_PHYS_PAGES);
  const long   pageSize   = sysconf(_SC_PAGE_SIZE);
  const size_t systemRam  = static_cast<size_t>(pages) * static_cast<size_t>(pageSize);
  const size_t maxAllowed = systemRam / 4;
  if (totalPinnedBytes > maxAllowed) {
    throw std::runtime_error("Substructure search would require " + std::to_string(totalPinnedBytes / (1024 * 1024)) +
                             " MB of pinned memory, exceeding 1/4 of system RAM (" +
                             std::to_string(maxAllowed / (1024 * 1024)) +
                             " MB). "
                             "Reduce workerThreads, executorsPerRunner, or batchSize.");
  }

  PinnedHostBufferPool bufferPool;
  bufferPool.initialize(poolSize, maxPairsPerBatch, static_cast<int>(maxMatchIndicesPerMiniBatch), maxPatternsPerDepth);

  PreparedBatchQueue batchQueue;
  std::atomic<int>   nextTargetIdx{0};
  std::atomic<bool>  pipelineAbort{false};

  // Use the fallback queue's mutex if available (ensures GPU batch accumulation
  // and fallback processing use the same mutex to avoid race conditions)
  std::mutex  localResultsMutex;
  std::mutex& resultsMutex = fallbackQueue ? fallbackQueue->getResultsMutex() : localResultsMutex;

  std::vector<std::exception_ptr> exceptions(numRunners);
  std::vector<std::exception_ptr> preprocessExceptions(effectivePreprocessingThreads);

  ScopedNvtxRange          launchRange("CPU: Launch GPU coordinators (pipeline)");
  std::vector<std::thread> gpuThreads;
  gpuThreads.reserve(numGpus);

  int executorOffset = 0;
  int workerIdOffset = 0;
  for (int g = 0; g < numGpus; ++g) {
    const int numWorkersThisGpu = workersPerGpu[g];
    if (numWorkersThisGpu == 0) {
      continue;
    }

    const int deviceId            = gpuIds[g];
    const int startWorkerIdx      = workerIdOffset;
    const int numExecutorsThisGpu = numWorkersThisGpu * executorsPerRunner;
    workerIdOffset += numWorkersThisGpu;
    executorOffset += numExecutorsThisGpu;

    gpuThreads.emplace_back([=,
                             &batchQueue,
                             &bufferPool,
                             &queriesHost,
                             &queriesDevice,
                             &leafSubpatterns,
                             &results,
                             &resultsMutex,
                             &exceptions,
                             &pipelineAbort]() mutable {
      try {
        ScopedNvtxRange  coordRange("GPU" + std::to_string(deviceId) + " coordinator (pipeline)");
        const WithDevice setDevice(deviceId);

        std::vector<std::unique_ptr<GpuExecutor>> executors;
        executors.reserve(static_cast<size_t>(numExecutorsThisGpu));
        for (int i = 0; i < numExecutorsThisGpu; ++i) {
          auto executor = std::make_unique<GpuExecutor>(startWorkerIdx * executorsPerRunner + i, deviceId);
          executor->initializeForStream();
          executors.push_back(std::move(executor));
        }

        std::unique_ptr<MoleculesDevice> localQueries;
        std::unique_ptr<LeafSubpatterns> localLeafPatterns;

        const MoleculesDevice* queriesPtr = &queriesDevice;
        const LeafSubpatterns* leafPtr    = &leafSubpatterns;

        if (deviceId != currentDevice) {
          localQueries = std::make_unique<MoleculesDevice>();
          localQueries->copyFromHost(queriesHost);
          localLeafPatterns = std::make_unique<LeafSubpatterns>();
          localLeafPatterns->buildAllPatterns(queriesHost);
          localLeafPatterns->syncToDevice(nullptr);
          queriesPtr = localQueries.get();
          leafPtr    = localLeafPatterns.get();
        }

        std::vector<std::thread> workers;
        workers.reserve(numWorkersThisGpu);
        for (int w = 0; w < numWorkersThisGpu; ++w) {
          const int                 globalIdx = startWorkerIdx + w;
          std::vector<GpuExecutor*> workerExecutors;
          workerExecutors.reserve(executorsPerRunner);
          for (int s = 0; s < executorsPerRunner; ++s) {
            workerExecutors.push_back(executors[w * executorsPerRunner + s].get());
          }

          auto workerLoop = [&, globalIdx, workerExecutors]() mutable {
            if (boolResults) {
              runnerWorkerPipelineBoolean(globalIdx,
                                          std::cref(*queriesPtr),
                                          std::cref(*leafPtr),
                                          std::ref(*boolResults),
                                          std::ref(resultsMutex),
                                          algorithm,
                                          deviceId,
                                          std::move(workerExecutors),
                                          std::ref(batchQueue),
                                          std::ref(bufferPool),
                                          std::ref(pipelineAbort),
                                          std::ref(exceptions[globalIdx]));
            } else if (countResults) {
              runnerWorkerPipelineCounts(globalIdx,
                                         std::cref(*queriesPtr),
                                         std::cref(*leafPtr),
                                         std::ref(*countResults),
                                         std::ref(resultsMutex),
                                         algorithm,
                                         deviceId,
                                         std::move(workerExecutors),
                                         std::ref(batchQueue),
                                         std::ref(bufferPool),
                                         std::ref(pipelineAbort),
                                         std::ref(exceptions[globalIdx]));
            } else {
              runnerWorkerPipelineResults(globalIdx,
                                          std::cref(*queriesPtr),
                                          std::cref(*leafPtr),
                                          std::ref(results),
                                          std::ref(resultsMutex),
                                          algorithm,
                                          deviceId,
                                          std::move(workerExecutors),
                                          std::ref(batchQueue),
                                          std::ref(bufferPool),
                                          fallbackQueue,
                                          std::ref(pipelineAbort),
                                          std::ref(exceptions[globalIdx]));
            }
          };

          workers.emplace_back(workerLoop);
        }

        for (auto& worker : workers) {
          worker.join();
        }
      } catch (...) {
        exceptions[startWorkerIdx] = std::current_exception();
      }
    });
  }
  launchRange.pop();

  ScopedNvtxRange          preprocessRange("CPU: Preprocess micro-batches");
  std::vector<std::thread> preprocessThreads;
  preprocessThreads.reserve(effectivePreprocessingThreads);

  struct BufferReleaseGuard {
    PinnedHostBufferPool* pool   = nullptr;
    PinnedHostBuffer*     buffer = nullptr;
    ~BufferReleaseGuard() {
      if (pool && buffer) {
        pool->release(buffer);
      }
    }
    void release() { buffer = nullptr; }
  };

  for (int t = 0; t < effectivePreprocessingThreads; ++t) {
    preprocessThreads.emplace_back([&, t]() {
      try {
        ScopedNvtxRange                  threadRange("Preprocess thread " + std::to_string(t));
        std::vector<const RDKit::ROMol*> batchTargets;
        std::vector<int>                 batchOriginalIndices;
        std::vector<RDKitFallbackEntry>  fallbackEntries;
        batchTargets.reserve(static_cast<size_t>(targetsPerBatch));
        batchOriginalIndices.reserve(static_cast<size_t>(targetsPerBatch));

        while (true) {
          if (pipelineAbort.load(std::memory_order_acquire)) {
            break;
          }
          const int start = nextTargetIdx.fetch_add(targetsPerBatch, std::memory_order_relaxed);
          if (start >= numTargets) {
            break;
          }

          const int end = std::min(start + targetsPerBatch, numTargets);
          batchTargets.clear();
          batchOriginalIndices.clear();
          fallbackEntries.clear();

          for (int i = start; i < end; ++i) {
            const RDKit::ROMol* target        = targets[i];
            const unsigned int  atomCount     = target->getNumAtoms();
            const bool          needsFallback = (atomCount > kMaxTargetAtoms) || requiresRDKitFallback(target);
            if (needsFallback) {
              for (int q = 0; q < numQueries; ++q) {
                fallbackEntries.push_back({i, q});
              }
              continue;
            }
            batchTargets.push_back(target);
            batchOriginalIndices.push_back(i);
          }

          if (!fallbackEntries.empty() && fallbackQueue != nullptr) {
            fallbackQueue->enqueue(fallbackEntries);
          }

          if (batchTargets.empty()) {
            if (fallbackQueue != nullptr) {
              fallbackQueue->tryProcessOne();
            }
            continue;
          }

          MoleculesHost    targetsHost;
          std::vector<int> emptySortOrder;
          buildTargetBatchParallelInto(targetsHost, 1, batchTargets, emptySortOrder);

          auto sharedTargetsHost     = std::make_shared<MoleculesHost>(std::move(targetsHost));
          auto sharedOriginalIndices = std::make_shared<std::vector<int>>(std::move(batchOriginalIndices));
          batchOriginalIndices.clear();
          batchOriginalIndices.reserve(static_cast<size_t>(targetsPerBatch));

          const int numBatchTargets  = static_cast<int>(sharedOriginalIndices->size());
          auto      sharedAtomCounts = std::make_shared<std::vector<int>>(static_cast<size_t>(numBatchTargets));

          int localMaxTargetAtoms  = 0;
          int localMaxBondsPerAtom = 0;
          for (int tIdx = 0; tIdx < numBatchTargets; ++tIdx) {
            const int atomStart       = sharedTargetsHost->batchAtomStarts[tIdx];
            const int atomEnd         = sharedTargetsHost->batchAtomStarts[tIdx + 1];
            const int atoms           = atomEnd - atomStart;
            (*sharedAtomCounts)[tIdx] = atoms;
            localMaxTargetAtoms       = std::max(localMaxTargetAtoms, atoms);
            for (int a = atomStart; a < atomEnd; ++a) {
              localMaxBondsPerAtom =
                std::max(localMaxBondsPerAtom, static_cast<int>(sharedTargetsHost->targetAtomBonds[a].degree));
            }
          }

          const int totalPairs = numBatchTargets * numQueries;
          for (int pairOffset = 0; pairOffset < totalPairs; pairOffset += maxPairsPerBatch) {
            if (pipelineAbort.load(std::memory_order_acquire)) {
              break;
            }
            PinnedHostBuffer* buffer = nullptr;
            {
              ScopedNvtxRange waitRange("Wait for pinned buffer", NvtxColor::kRed);
              buffer = bufferPool.acquire();
            }
            if (buffer == nullptr) {
              break;
            }
            BufferReleaseGuard innerReleaseGuard{&bufferPool, buffer};

            auto batch                   = std::make_unique<PreparedMiniBatch>();
            batch->pinnedBuffer          = buffer;
            batch->targetsHost           = sharedTargetsHost;
            batch->targetOriginalIndices = sharedOriginalIndices;
            batch->targetAtomCounts      = sharedAtomCounts;

            batch->ctx.queryAtomCounts       = queryContext.queryAtomCounts.data();
            batch->ctx.queryDepths           = queryContext.queryDepths.data();
            batch->ctx.queryMaxDepths        = queryContext.queryMaxDepths.data();
            batch->ctx.queryHasPatterns      = queryContext.queryHasPatterns.data();
            batch->ctx.targetAtomCounts      = sharedAtomCounts.get();
            batch->ctx.targetOriginalIndices = sharedOriginalIndices.get();
            batch->ctx.numTargets            = numBatchTargets;
            batch->ctx.numQueries            = numQueries;
            batch->ctx.maxTargetAtoms        = localMaxTargetAtoms;
            batch->ctx.maxQueryAtoms         = queryContext.maxQueryAtoms;
            batch->ctx.maxBondsPerAtom       = localMaxBondsPerAtom;
            batch->ctx.maxMatches            = config.maxMatches;
            batch->ctx.countOnly             = countOnly;
            const int templateTargetAtoms    = std::max(localMaxTargetAtoms, queryContext.maxQueryAtoms);
            batch->ctx.templateConfig =
              selectTemplateConfig(templateTargetAtoms, queryContext.maxQueryAtoms, localMaxBondsPerAtom);

            prepareMiniBatchOnCPU(batch->plan, *buffer, batch->ctx, leafSubpatterns, pairOffset, maxPairsPerBatch);

            innerReleaseGuard.release();
            batchQueue.push(std::move(batch));
          }

          if (fallbackQueue != nullptr && fallbackQueue->hasWork()) {
            fallbackQueue->tryProcessOne();
          }
        }
      } catch (...) {
        preprocessExceptions[t] = std::current_exception();
        pipelineAbort.store(true, std::memory_order_release);
        batchQueue.close();
        bufferPool.shutdown();
      }
    });
  }

  for (auto& t : preprocessThreads) {
    t.join();
  }
  preprocessRange.pop();

  batchQueue.close();

  ScopedNvtxRange joinRange("CPU: Join GPU coordinators (pipeline)");
  for (auto& t : gpuThreads) {
    t.join();
  }
  joinRange.pop();

  for (const auto& ex : preprocessExceptions) {
    if (ex) {
      std::rethrow_exception(ex);
    }
  }

  for (const auto& ex : exceptions) {
    if (ex) {
      std::rethrow_exception(ex);
    }
  }

  cudaCheckError(cudaGetLastError());
}

}  // namespace

// =============================================================================
// Recursive SMARTS Preprocessing
// =============================================================================

void preprocessRecursiveSmartsBatchedWithEvents(SubstructTemplateConfig           templateConfig,
                                                const MoleculesDevice&            targetsDevice,
                                                const MoleculesHost&              queriesHost,
                                                const LeafSubpatterns&            leafSubpatterns,
                                                MiniBatchResultsDevice&           miniBatchResults,
                                                const int                         numQueries,
                                                const int                         miniBatchPairOffset,
                                                const int                         miniBatchSize,
                                                const SubstructAlgorithm          algorithm,
                                                cudaStream_t                      stream,
                                                RecursiveScratchBuffers&          scratch,
                                                std::vector<BatchedPatternEntry>& scratchPatternEntries,
                                                cudaEvent_t*                      depthEvents,
                                                int                               numDepthEvents) {
  ScopedNvtxRange processRecursiveRange("Process recursive mini-batch with events");

  // Configure kernels for max shared memory carveout (once per process)
  configureSubstructKernelsSharedMem();

  ScopedNvtxRange processRecursiveRangeSetup("Process recursive mini-batch setup");

  scratch.setStream(stream);

  std::vector<BatchedPatternEntry>& patternEntriesHost = scratchPatternEntries;
  patternEntriesHost.clear();

  const int firstQueryInMiniBatch = miniBatchPairOffset % numQueries;
  const int numUniqueQueries      = std::min(miniBatchSize, numQueries);
  const int recursivePatternsSize = static_cast<int>(queriesHost.recursivePatterns.size());

  int maxDepth = 0;
  for (int i = 0; i < numUniqueQueries; ++i) {
    const int queryIdx = (firstQueryInMiniBatch + i) % numQueries;

    if (queryIdx >= recursivePatternsSize) {
      continue;
    }

    const auto& recursiveInfo = queriesHost.recursivePatterns[queryIdx];
    if (recursiveInfo.empty()) {
      continue;
    }

    maxDepth = std::max(maxDepth, recursiveInfo.maxDepth);

    for (const auto& entry : recursiveInfo.patterns) {
      if (entry.queryMol == nullptr) {
        continue;
      }

      const int patternMolIdx = leafSubpatterns.getPatternIndex(queryIdx, entry.patternId);
      if (patternMolIdx < 0) {
        throw std::runtime_error("Pattern not found in pre-built LeafSubpatterns: queryIdx=" +
                                 std::to_string(queryIdx) + ", patternId=" + std::to_string(entry.patternId));
      }

      BatchedPatternEntry& batchEntry = patternEntriesHost.emplace_back();
      batchEntry.mainQueryIdx         = queryIdx;
      batchEntry.patternId            = entry.patternId;
      batchEntry.patternMolIdx        = patternMolIdx;
      batchEntry.depth                = entry.depth;
      batchEntry.localIdInParent      = entry.localIdInParent;
    }
  }

  if (patternEntriesHost.empty()) {
    return;
  }

  const int firstTargetInMiniBatch = miniBatchPairOffset / numQueries;
  const int lastTargetInMiniBatch  = (miniBatchPairOffset + miniBatchSize - 1) / numQueries;
  const int numTargetsInMiniBatch  = lastTargetInMiniBatch - firstTargetInMiniBatch + 1;

  constexpr int gsiBuffersPerBlock = 2;

  const int maxPaintPairsPerSubBatch = std::max(miniBatchSize, 1024);
  processRecursiveRangeSetup.pop();

  for (int currentDepth = 0; currentDepth <= maxDepth; ++currentDepth) {
    ScopedNvtxRange depthRange("Process recursive depth level " + std::to_string(currentDepth));

    std::vector<BatchedPatternEntry> patternsAtDepth;
    for (const auto& entry : patternEntriesHost) {
      if (entry.depth == currentDepth) {
        patternsAtDepth.push_back(entry);
      }
    }

    if (patternsAtDepth.empty()) {
      if (currentDepth < numDepthEvents && depthEvents != nullptr) {
        cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
      }
      continue;
    }

    const size_t numPatterns         = patternsAtDepth.size();
    const int    patternsPerSubBatch = std::max(1, maxPaintPairsPerSubBatch / numTargetsInMiniBatch);

    for (size_t patternStart = 0; patternStart < numPatterns; patternStart += patternsPerSubBatch) {
      ScopedNvtxRange subBatchRange("Process sub-batch " + std::to_string(patternStart));

      const size_t patternEnd            = std::min(patternStart + patternsPerSubBatch, numPatterns);
      const size_t numPatternsInSubBatch = patternEnd - patternStart;
      const size_t numBlocksInSubBatch   = numTargetsInMiniBatch * numPatternsInSubBatch;

      ScopedNvtxRange prepareRange("CPU: Prepare pattern entries");
      const int       bufferIdx = scratch.acquireBufferIndex();
      scratch.waitForBuffer(bufferIdx);
      scratch.ensureCapacity(bufferIdx, static_cast<int>(numPatternsInSubBatch));
      for (size_t i = 0; i < numPatternsInSubBatch; ++i) {
        scratch.patternsAtDepthHost[bufferIdx][i] = patternsAtDepth[patternStart + i];
      }
      prepareRange.pop();

      const int    buffersPerBlock = gsiBuffersPerBlock;
      const size_t overflowNeeded  = numBlocksInSubBatch * buffersPerBlock * kOverflowEntriesPerBuffer;

      if (scratch.overflow.size() < overflowNeeded) {
        scratch.overflow.resize(static_cast<size_t>(overflowNeeded * 1.5));
      }

      const size_t labelMatrixNeeded = numBlocksInSubBatch * kLabelMatrixWords;
      if (scratch.labelMatrixBuffer.size() < labelMatrixNeeded) {
        scratch.labelMatrixBuffer.resize(static_cast<size_t>(labelMatrixNeeded * 1.5));
      }

      if (scratch.patternEntries.size() < numPatternsInSubBatch) {
        scratch.patternEntries.resize(static_cast<size_t>(numPatternsInSubBatch * 1.5));
      }

      scratch.patternEntries.copyFromHost(scratch.patternsAtDepthHost[bufferIdx].data(), numPatternsInSubBatch);
      scratch.recordCopy(bufferIdx, scratch.patternEntries.stream());

      const uint32_t* recursiveBitsForLabel = (currentDepth > 0) ? miniBatchResults.recursiveMatchBits() : nullptr;

      launchLabelMatrixPaintKernel(templateConfig,
                                   targetsDevice.view<MoleculeType::Target>(),
                                   leafSubpatterns.view(),
                                   scratch.patternEntries.data(),
                                   static_cast<int>(numPatternsInSubBatch),
                                   numBlocksInSubBatch,
                                   numQueries,
                                   miniBatchPairOffset,
                                   miniBatchSize,
                                   scratch.labelMatrixBuffer.data(),
                                   firstTargetInMiniBatch,
                                   recursiveBitsForLabel,
                                   miniBatchResults.maxTargetAtoms(),
                                   stream);

      launchSubstructPaintKernel(templateConfig,
                                 algorithm,
                                 targetsDevice.view<MoleculeType::Target>(),
                                 leafSubpatterns.view(),
                                 scratch.patternEntries.data(),
                                 static_cast<int>(numPatternsInSubBatch),
                                 numBlocksInSubBatch,
                                 miniBatchResults.recursiveMatchBits(),
                                 miniBatchResults.maxTargetAtoms(),
                                 numQueries,
                                 0,
                                 0,
                                 miniBatchPairOffset,
                                 miniBatchSize,
                                 scratch.overflow.data(),
                                 scratch.overflow.data(),
                                 kOverflowEntriesPerBuffer,
                                 scratch.labelMatrixBuffer.data(),
                                 firstTargetInMiniBatch,
                                 stream);
    }

    if (currentDepth < numDepthEvents && depthEvents != nullptr) {
      cudaCheckError(cudaEventRecord(depthEvents[currentDepth], stream));
    }
  }

  cudaCheckError(cudaGetLastError());
}

namespace {

/**
 * @brief Remove duplicate matches that differ only in atom enumeration order.
 *
 * Two matches are considered duplicates if they map query atoms to the same set
 * of target atoms, regardless of the ordering. For example, with query "CCC" on
 * cyclohexane, matches (0,1,2) and (2,1,0) would be considered duplicates since
 * they both involve target atoms {0,1,2}.
 *
 * This is a postprocessing step applied after all matches are collected.
 */
void uniquifyResults(SubstructSearchResults& results) {
  ScopedNvtxRange uniquifyRange("uniquifyResults");

  std::set<std::vector<int>>    seenSorted;
  std::vector<std::vector<int>> uniqueMatches;
  std::vector<int>              sortedMatch;

  for (auto& [pairIdx, matchList] : results.matches) {
    if (matchList.size() <= 1) {
      continue;
    }

    seenSorted.clear();
    uniqueMatches.clear();
    uniqueMatches.reserve(matchList.size());

    for (auto& match : matchList) {
      sortedMatch.assign(match.begin(), match.end());
      std::sort(sortedMatch.begin(), sortedMatch.end());

      if (seenSorted.insert(sortedMatch).second) {
        uniqueMatches.push_back(std::move(match));
      }
    }

    if (uniqueMatches.size() < matchList.size()) {
      matchList = std::move(uniqueMatches);
    }
  }
}

}  // namespace

/**
 * @brief Compute effective thread counts using autoselect logic.
 *
 * When a config value is -1 (autoselect):
 * - preprocessingThreads: uses hardware_concurrency
 * - workerThreads (per GPU): min(4, hardware_concurrency / numGpus)
 */
void computeEffectiveThreadCounts(const SubstructSearchConfig& config,
                                  int                          numGpus,
                                  int&                         effectivePreprocessingThreads,
                                  int&                         effectiveWorkerThreads) {
  const int hwThreads        = static_cast<int>(std::thread::hardware_concurrency());
  const int effectiveNumGpus = std::max(1, numGpus);

  effectivePreprocessingThreads =
    (config.preprocessingThreads == -1) ? hwThreads : std::max(1, config.preprocessingThreads);

  effectiveWorkerThreads = (config.workerThreads == -1) ? std::min(4, std::max(1, hwThreads / effectiveNumGpus)) :
                                                          std::max(1, config.workerThreads);
}

namespace {

void getSubstructMatchesImpl(const std::vector<const RDKit::ROMol*>& targets,
                             const std::vector<const RDKit::ROMol*>& queries,
                             SubstructSearchResults&                 results,
                             SubstructAlgorithm                      algorithm,
                             cudaStream_t                            stream,
                             const SubstructSearchConfig&            config,
                             HasSubstructMatchResults*               boolResults,
                             std::vector<int>*                       countResults) {
  const int numTargets = static_cast<int>(targets.size());
  const int numQueries = static_cast<int>(queries.size());

  if (numTargets == 0 || numQueries == 0) {
    results.resize(numTargets, numQueries);
    return;
  }

  std::vector<int> gpuIds = config.gpuIds;
  if (gpuIds.empty()) {
    int currentDevice = 0;
    cudaCheckError(cudaGetDevice(&currentDevice));
    gpuIds.push_back(currentDevice);
  }
  const int numGpus = static_cast<int>(gpuIds.size());

  int effectivePreprocessingThreads, effectiveWorkerThreads;
  computeEffectiveThreadCounts(config, numGpus, effectivePreprocessingThreads, effectiveWorkerThreads);

  ScopedNvtxRange overloadRange(
    "getSubstructMatches T=" + std::to_string(numTargets) + " Q=" + std::to_string(numQueries) +
    " batch=" + std::to_string(config.batchSize) + " prep=" + std::to_string(effectivePreprocessingThreads) +
    " workers=" + std::to_string(effectiveWorkerThreads) + " gpus=" + std::to_string(numGpus));

  SubstructSearchConfig effectiveConfig = config;
  effectiveConfig.preprocessingThreads  = effectivePreprocessingThreads;
  effectiveConfig.workerThreads         = effectiveWorkerThreads;
  effectiveConfig.gpuIds                = gpuIds;

  {
    ScopedNvtxRange setupRange("Prepare search context");
    // Initialize results for all original targets
    results.resize(numTargets, numQueries);
  }

  ScopedNvtxRange  buildRange2("Build host query data structures");
  std::vector<int> emptySortOrder;
  MoleculesHost    queriesHost = buildQueryBatchParallel(queries, emptySortOrder, effectivePreprocessingThreads);
  buildRange2.pop();

  ScopedNvtxRange buildRange3("Build device query data structures");
  MoleculesDevice queriesDevice(stream);
  buildRange3.pop();

  ScopedNvtxRange buildRange4("Copy queries to device");
  queriesDevice.copyFromHost(queriesHost);
  buildRange4.pop();

  ScopedNvtxRange leafRange("Build LeafSubpatterns");
  LeafSubpatterns leafSubpatterns;
  leafSubpatterns.buildAllPatterns(queriesHost);
  leafSubpatterns.syncToDevice(stream);
  leafRange.pop();

  // Ensure queries and patterns are fully copied before workers start using them.
  // Workers use different streams, so we need an explicit sync here.
  cudaCheckError(cudaStreamSynchronize(stream));

  QueryPreprocessContext queryContext;
  queryContext.numQueries = numQueries;
  queryContext.queryAtomCounts.resize(numQueries);
  queryContext.queryDepths.resize(numQueries);
  queryContext.queryMaxDepths.resize(numQueries);
  queryContext.queryHasPatterns.resize(numQueries);

  const int precomputedSize      = static_cast<int>(leafSubpatterns.perQueryPatterns.size());
  const int perQueryMaxDepthSize = static_cast<int>(leafSubpatterns.perQueryMaxDepth.size());

  int maxQueryAtoms = 0;
  int maxDepthSeen  = 0;

#pragma omp parallel num_threads(effectivePreprocessingThreads) reduction(max : maxQueryAtoms, maxDepthSeen)
  {
#pragma omp for nowait
    for (int q = 0; q < numQueries; ++q) {
      const int atomStart             = queriesHost.batchAtomStarts[q];
      const int atomEnd               = queriesHost.batchAtomStarts[q + 1];
      const int atomCount             = atomEnd - atomStart;
      queryContext.queryAtomCounts[q] = atomCount;

      const int depth             = getQueryRecursionDepth(queriesHost, q);
      queryContext.queryDepths[q] = depth;
      maxDepthSeen                = std::max(maxDepthSeen, depth);

      const int maxDepth             = (q < perQueryMaxDepthSize) ? leafSubpatterns.perQueryMaxDepth[q] : 0;
      queryContext.queryMaxDepths[q] = maxDepth;

      const bool hasPatterns =
        (q < precomputedSize) && (maxDepth > 0 || !leafSubpatterns.perQueryPatterns[q][0].empty());
      queryContext.queryHasPatterns[q] = hasPatterns ? 1 : 0;

      maxQueryAtoms = std::max(maxQueryAtoms, atomCount);
    }
  }

  if (maxDepthSeen > kMaxRecursionDepth) {
    throw std::runtime_error("Recursive SMARTS depth " + std::to_string(maxDepthSeen) +
                             " exceeds maximum supported depth of " + std::to_string(kMaxRecursionDepth));
  }
  queryContext.maxQueryAtoms = maxQueryAtoms;

  // Mutex shared between GPU batch accumulation and fallback queue processing
  std::mutex resultsMutex;

  // Create fallback queue to collect overflow and oversized targets.
  RDKitFallbackQueue
    fallbackQueue(&targets, &queries, &results, &resultsMutex, config.maxMatches, boolResults, countResults);

  runPipelinedSubstructSearch(targets,
                              queriesHost,
                              queriesDevice,
                              leafSubpatterns,
                              queryContext,
                              results,
                              algorithm,
                              stream,
                              effectiveConfig,
                              effectivePreprocessingThreads,
                              &fallbackQueue,
                              boolResults,
                              countResults);

  // Process any remaining fallback entries after GPU work completes.
  while (fallbackQueue.tryProcessOne()) {
  }

  if (!boolResults && config.uniquify) {
    uniquifyResults(results);
  }
}

}  // anonymous namespace

void getSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                         const std::vector<const RDKit::ROMol*>& queries,
                         SubstructSearchResults&                 results,
                         SubstructAlgorithm                      algorithm,
                         cudaStream_t                            stream,
                         const SubstructSearchConfig&            config) {
  getSubstructMatchesImpl(targets, queries, results, algorithm, stream, config, nullptr, nullptr);
}

void countSubstructMatches(const std::vector<const RDKit::ROMol*>& targets,
                           const std::vector<const RDKit::ROMol*>& queries,
                           std::vector<int>&                       counts,
                           SubstructAlgorithm                      algorithm,
                           cudaStream_t                            stream,
                           const SubstructSearchConfig&            config) {
  const int numTargets = static_cast<int>(targets.size());
  const int numQueries = static_cast<int>(queries.size());

  counts.assign(static_cast<size_t>(numTargets) * numQueries, 0);

  SubstructSearchResults matchResults;
  SubstructSearchConfig  countConfig = config;
  countConfig.maxMatches             = 0;

  getSubstructMatchesImpl(targets, queries, matchResults, algorithm, stream, countConfig, nullptr, &counts);
}

void hasSubstructMatch(const std::vector<const RDKit::ROMol*>& targets,
                       const std::vector<const RDKit::ROMol*>& queries,
                       HasSubstructMatchResults&               results,
                       SubstructAlgorithm                      algorithm,
                       cudaStream_t                            stream,
                       const SubstructSearchConfig&            config) {
  const int numTargets = static_cast<int>(targets.size());
  const int numQueries = static_cast<int>(queries.size());

  ScopedNvtxRange overloadRange("hasSubstructMatch T=" + std::to_string(numTargets) +
                                " Q=" + std::to_string(numQueries));

  results.resize(numTargets, numQueries);

  if (numTargets == 0 || numQueries == 0) {
    return;
  }

  SubstructSearchConfig  hasMatchConfig;
  SubstructSearchResults matchResults;
  {
    ScopedNvtxRange setupRange("hasSubstructMatch setup");
    hasMatchConfig            = config;
    hasMatchConfig.maxMatches = 1;
  }
  getSubstructMatchesImpl(targets, queries, matchResults, algorithm, stream, hasMatchConfig, &results, nullptr);

  for (auto& [pairIdx, matches] : matchResults.matches) {
    if (!matches.empty()) {
      results.hasMatch[pairIdx] = 1;
    }
  }
}

}  // namespace nvMolKit
