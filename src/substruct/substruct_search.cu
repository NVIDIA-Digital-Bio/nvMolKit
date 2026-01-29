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
#include <utility>
#include <vector>

#include "cuda_error_check.h"
#include "gpu_executor.h"
#include "graph_labeler.cuh"
#include "host_vector.h"
#include "minibatch_planner.h"
#include "molecules_device.cuh"
#include "nvtx.h"
#include "pinned_buffer_pool.h"
#include "recursive_preprocessor.h"
#include "sm_shared_mem_config.cuh"
#include "substruct_algos.cuh"
#include "substruct_debug.h"
#include "substruct_kernels.h"
#include "substruct_launch_config.h"
#include "substruct_search.h"
#include "substruct_search_internal.h"
#include "thread_safe_queue.h"

namespace nvMolKit {

struct QueryPreprocessContext;

namespace {

void runPipelinedSubstructSearch(const std::vector<const RDKit::ROMol*>& targets,
                                 const MoleculesHost&                    queriesHost,
                                 const MoleculesDevice&                  queriesDevice,
                                 const RecursivePatternPreprocessor&     recursivePreprocessor,
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

void RDKitFallbackQueue::closeQueueIfDone() {
  if (activeProducers_ == 0) {
    queue_.close();
  }
}

size_t RDKitFallbackQueue::processedCount() const {
  return processedCount_.load(std::memory_order_relaxed);
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
  std::vector<int>      queryPipelineDepths;  // Pipeline stage depth (maxRecursiveDepth + 1, 0 if none)
  std::vector<int>      queryMaxDepths;
  std::vector<int8_t>   queryHasPatterns;
  int                   numQueries    = 0;
  int                   maxQueryAtoms = 0;
};

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

  if (overflowFlags_.size() < static_cast<size_t>(miniBatchSize)) {
    overflowFlags_.resize(static_cast<size_t>(miniBatchSize * 1.5));
  }
  overflowFlags_.zero();
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
                                                 int16_t* hostMatchIndices,
                                                 uint8_t* hostOverflowFlags) const {
  matchCounts_.copyToHost(hostMatchCounts, miniBatchSize_);
  reportedCounts_.copyToHost(hostReportedCounts, miniBatchSize_);
  matchIndices_.copyToHost(hostMatchIndices, totalMiniBatchMatchIndices_);
  overflowFlags_.copyToHost(hostOverflowFlags, miniBatchSize_);
}

void MiniBatchResultsDevice::copyCountsOnlyToHost(int* hostMatchCounts) const {
  matchCounts_.copyToHost(hostMatchCounts, miniBatchSize_);
}

// =============================================================================
// Pipelined Batch Processing Implementation
// =============================================================================

namespace {

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

void uploadAndLaunchMiniBatch(GpuExecutor&                        executor,
                              const ThreadWorkerContext&          ctx,
                              MoleculesDevice&                    targetsDevice,
                              const MoleculesDevice&              queriesDevice,
                              const RecursivePatternPreprocessor& recursivePreprocessor,
                              SubstructAlgorithm                  algorithm,
                              const PinnedHostBuffer&             hostBuffer) {
  ScopedNvtxRange uploadRange("uploadAndLaunchMiniBatch");

  cudaStream_t executorStream     = executor.stream();
  const int    numBuffersPerBlock = (algorithm == SubstructAlgorithm::GSI) ? 2 : 1;

  if (executor.plan.maxPipelineDepthInMiniBatch == 0) {
    ScopedNvtxRange nonRecursiveRange("Non-recursive path");

    const int maxMatchesToFind = ctx.maxMatches > 0 ? ctx.maxMatches : -1;
    executor.deviceResults.allocateMiniBatch(executor.plan.numPairsInMiniBatch,
                                             hostBuffer.miniBatchPairMatchStarts.data(),
                                             executor.plan.totalMatchIndices,
                                             ctx.numQueries,
                                             ctx.maxTargetAtoms,
                                             numBuffersPerBlock,
                                             maxMatchesToFind,
                                             ctx.countOnly);
    executor.deviceResults.setQueryAtomCounts(ctx.queryAtomCounts, ctx.numQueries);

    if (executor.pairIndicesDev.size() < static_cast<size_t>(executor.plan.numPairsInMiniBatch)) {
      executor.pairIndicesDev.resize(static_cast<size_t>(executor.plan.numPairsInMiniBatch * 1.5));
    }
    executor.pairIndicesDev.copyFromHost(hostBuffer.pairIndices.data(), executor.plan.numPairsInMiniBatch);

    launchLabelMatrixKernel(ctx.templateConfig,
                            targetsDevice.view<MoleculeType::Target>(),
                            queriesDevice.view<MoleculeType::Query>(),
                            executor.pairIndicesDev.data(),
                            executor.plan.numPairsInMiniBatch,
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
                               executor.plan.numPairsInMiniBatch,
                               ctx.numQueries,
                               nullptr,
                               nullptr,
                               executorStream);
    return;
  }

  ScopedNvtxRange multiStreamRange("Multi-stream recursive pipeline");

  cudaStream_t recursiveStream = executor.recursiveStream.stream();

  const int maxMatchesToFind = ctx.maxMatches > 0 ? ctx.maxMatches : -1;
  executor.deviceResults.allocateMiniBatch(executor.plan.numPairsInMiniBatch,
                                           hostBuffer.miniBatchPairMatchStarts.data(),
                                           executor.plan.totalMatchIndices,
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
  recursivePreprocessor.preprocessMiniBatch(ctx.templateConfig,
                                            targetsDevice,
                                            executor.deviceResults,
                                            ctx.numQueries,
                                            executor.plan.miniBatchPairOffset,
                                            executor.plan.numPairsInMiniBatch,
                                            algorithm,
                                            recursiveStream,
                                            executor.recursiveScratch,
                                            executor.plan.patternsAtDepth,
                                            executor.plan.recursiveMaxDepth,
                                            executor.plan.firstTargetInMiniBatch,
                                            executor.plan.numTargetsInMiniBatch,
                                            depthEventPtrs.data(),
                                            kMaxRecursionDepth);
  preprocRange.pop();

  ScopedNvtxRange depth0Range("Match depth-0 pairs (executorStream)");
  launchLabelAndMatch(executor.plan.matchPairsCounts[0],
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

  for (int depth = 1; depth <= executor.plan.maxPipelineDepthInMiniBatch; ++depth) {
    ScopedNvtxRange depthRange("Match depth-" + std::to_string(depth) + " pairs (postRecursionStream)");

    ScopedNvtxRange waitRange("Wait: postRecursionStream waits for depth event");
    cudaCheckError(cudaStreamWaitEvent(postStream, depthEventPtrs[depth - 1], 0));
    waitRange.pop();

    launchLabelAndMatch(executor.plan.matchPairsCounts[depth],
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
                                             hostBuffer.matchIndices.data(),
                                             hostBuffer.overflowFlags.data());
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
  updates.reserve(executor.plan.numPairsInMiniBatch);

  for (int i = 0; i < executor.plan.numPairsInMiniBatch; ++i) {
    const int pairIdxInBatch = hostBuffer.pairIndices[i];
    const int localTargetIdx = pairIdxInBatch / ctx.numQueries;
    const int queryIdx       = pairIdxInBatch % ctx.numQueries;

    const int targetIdx = (*ctx.targetOriginalIndices)[localTargetIdx];

    const int queryAtoms      = ctx.queryAtomCounts[queryIdx];
    const int actualMatches   = hostBuffer.matchCounts[i];
    const int reportedMatches = hostBuffer.reportedCounts[i];

    // Check for overflow: either output buffer overflow (more matches than stored)
    // or partial match buffer overflow (algorithm couldn't explore all paths)
    const bool isOutputOverflow  = (actualMatches > reportedMatches) && (ctx.maxMatches == 0);
    const bool isPartialOverflow = hostBuffer.overflowFlags[i] != 0;
    if ((isOutputOverflow || isPartialOverflow) && fallbackQueue != nullptr) {
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

  for (int i = 0; i < executor.plan.numPairsInMiniBatch; ++i) {
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

  for (int i = 0; i < executor.plan.numPairsInMiniBatch; ++i) {
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
  executor.applyMiniBatchPlan(std::move(batch.plan));
}

template <typename InitiateCopyFunc, typename AccumulateFunc>
void runnerWorkerPipeline(int                                 workerIdx,
                          const MoleculesDevice&              queriesDevice,
                          const RecursivePatternPreprocessor& recursivePreprocessor,
                          SubstructAlgorithm                  algorithm,
                          int                                 deviceId,
                          std::vector<GpuExecutor*>           executors,
                          PreparedBatchQueue&                 batchQueue,
                          PinnedHostBufferPool&               bufferPool,
                          InitiateCopyFunc&&                  initiateCopy,
                          AccumulateFunc&&                    accumulate,
                          RDKitFallbackQueue*                 fallbackQueue,
                          std::atomic<bool>&                  pipelineAbort,
                          std::exception_ptr&                 exceptionPtr) {
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
                               recursivePreprocessor,
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

void runnerWorkerPipelineResults(int                                 workerIdx,
                                 const MoleculesDevice&              queriesDevice,
                                 const RecursivePatternPreprocessor& recursivePreprocessor,
                                 SubstructSearchResults&             results,
                                 std::mutex&                         resultsMutex,
                                 SubstructAlgorithm                  algorithm,
                                 int                                 deviceId,
                                 std::vector<GpuExecutor*>           executors,
                                 PreparedBatchQueue&                 batchQueue,
                                 PinnedHostBufferPool&               bufferPool,
                                 RDKitFallbackQueue*                 fallbackQueue,
                                 std::atomic<bool>&                  pipelineAbort,
                                 std::exception_ptr&                 exceptionPtr) {
  auto initiateCopy = [&](GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
    initiateResultsCopyToHost(executor, hostBuffer);
  };
  auto accumulate = [&](GpuExecutor& executor, const ThreadWorkerContext& ctx, const PinnedHostBuffer& hostBuffer) {
    accumulateMiniBatchResults(executor, ctx, results, resultsMutex, hostBuffer, fallbackQueue);
  };
  runnerWorkerPipeline(workerIdx,
                       queriesDevice,
                       recursivePreprocessor,
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

void runnerWorkerPipelineBoolean(int                                 workerIdx,
                                 const MoleculesDevice&              queriesDevice,
                                 const RecursivePatternPreprocessor& recursivePreprocessor,
                                 HasSubstructMatchResults&           results,
                                 std::mutex&                         resultsMutex,
                                 SubstructAlgorithm                  algorithm,
                                 int                                 deviceId,
                                 std::vector<GpuExecutor*>           executors,
                                 PreparedBatchQueue&                 batchQueue,
                                 PinnedHostBufferPool&               bufferPool,
                                 std::atomic<bool>&                  pipelineAbort,
                                 std::exception_ptr&                 exceptionPtr) {
  auto initiateCopy = [&](GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
    initiateCountsOnlyCopyToHost(executor, hostBuffer);
  };
  auto accumulate = [&](GpuExecutor& executor, const ThreadWorkerContext& ctx, const PinnedHostBuffer& hostBuffer) {
    accumulateMiniBatchResultsBoolean(executor, ctx, results, resultsMutex, hostBuffer);
  };
  runnerWorkerPipeline(workerIdx,
                       queriesDevice,
                       recursivePreprocessor,
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

void runnerWorkerPipelineCounts(int                                 workerIdx,
                                const MoleculesDevice&              queriesDevice,
                                const RecursivePatternPreprocessor& recursivePreprocessor,
                                std::vector<int>&                   counts,
                                std::mutex&                         resultsMutex,
                                SubstructAlgorithm                  algorithm,
                                int                                 deviceId,
                                std::vector<GpuExecutor*>           executors,
                                PreparedBatchQueue&                 batchQueue,
                                PinnedHostBufferPool&               bufferPool,
                                std::atomic<bool>&                  pipelineAbort,
                                std::exception_ptr&                 exceptionPtr) {
  auto initiateCopy = [&](GpuExecutor& executor, const PinnedHostBuffer& hostBuffer) {
    initiateCountsOnlyCopyToHost(executor, hostBuffer);
  };
  auto accumulate = [&](GpuExecutor& executor, const ThreadWorkerContext& ctx, const PinnedHostBuffer& hostBuffer) {
    accumulateMiniBatchResultsCounts(executor, ctx, counts, resultsMutex, hostBuffer);
  };
  runnerWorkerPipeline(workerIdx,
                       queriesDevice,
                       recursivePreprocessor,
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

void runGpuCoordinator(int                                 deviceId,
                       int                                 startWorkerIdx,
                       int                                 numWorkersThisGpu,
                       int                                 executorsPerRunner,
                       int                                 currentDevice,
                       const MoleculesHost&                queriesHost,
                       const MoleculesDevice&              queriesDevice,
                       const RecursivePatternPreprocessor& recursivePreprocessor,
                       SubstructSearchResults&             results,
                       std::mutex&                         resultsMutex,
                       SubstructAlgorithm                  algorithm,
                       PreparedBatchQueue&                 batchQueue,
                       PinnedHostBufferPool&               bufferPool,
                       RDKitFallbackQueue*                 fallbackQueue,
                       HasSubstructMatchResults*           boolResults,
                       std::vector<int>*                   countResults,
                       std::vector<std::exception_ptr>&    exceptions,
                       std::atomic<bool>&                  pipelineAbort) {
  try {
    ScopedNvtxRange  coordRange("GPU" + std::to_string(deviceId) + " coordinator (pipeline)");
    const WithDevice setDevice(deviceId);

    const int numExecutorsThisGpu = numWorkersThisGpu * executorsPerRunner;

    std::vector<std::unique_ptr<GpuExecutor>> executors;
    executors.reserve(static_cast<size_t>(numExecutorsThisGpu));
    for (int i = 0; i < numExecutorsThisGpu; ++i) {
      auto executor = std::make_unique<GpuExecutor>(startWorkerIdx * executorsPerRunner + i, deviceId);
      executor->initializeForStream();
      executors.push_back(std::move(executor));
    }

    std::unique_ptr<MoleculesDevice>              localQueries;
    std::unique_ptr<RecursivePatternPreprocessor> localPreprocessor;

    const MoleculesDevice*              queriesPtr      = &queriesDevice;
    const RecursivePatternPreprocessor* preprocessorPtr = &recursivePreprocessor;

    if (deviceId != currentDevice) {
      localQueries = std::make_unique<MoleculesDevice>();
      localQueries->copyFromHost(queriesHost);
      localPreprocessor = std::make_unique<RecursivePatternPreprocessor>();
      localPreprocessor->buildPatterns(queriesHost);
      localPreprocessor->syncToDevice(nullptr);
      queriesPtr      = localQueries.get();
      preprocessorPtr = localPreprocessor.get();
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
                                      std::cref(*preprocessorPtr),
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
                                     std::cref(*preprocessorPtr),
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
                                      std::cref(*preprocessorPtr),
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
}

}  // namespace

// =============================================================================
// Main API
// =============================================================================

namespace {

void runPipelinedSubstructSearch(const std::vector<const RDKit::ROMol*>& targets,
                                 const MoleculesHost&                    queriesHost,
                                 const MoleculesDevice&                  queriesDevice,
                                 const RecursivePatternPreprocessor&     recursivePreprocessor,
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

  const int              numTargets      = static_cast<int>(targets.size());
  const int              numQueries      = queryContext.numQueries;
  const LeafSubpatterns& leafSubpatterns = recursivePreprocessor.leafSubpatterns();
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

  MiniBatchPlanner planner;

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

  int workerIdOffset = 0;
  for (int g = 0; g < numGpus; ++g) {
    const int numWorkersThisGpu = workersPerGpu[g];
    if (numWorkersThisGpu == 0) {
      continue;
    }

    const int deviceId       = gpuIds[g];
    const int startWorkerIdx = workerIdOffset;
    workerIdOffset += numWorkersThisGpu;

    gpuThreads.emplace_back([=,
                             &batchQueue,
                             &bufferPool,
                             &queriesHost,
                             &queriesDevice,
                             &recursivePreprocessor,
                             &results,
                             &resultsMutex,
                             &exceptions,
                             &pipelineAbort]() mutable {
      runGpuCoordinator(deviceId,
                        startWorkerIdx,
                        numWorkersThisGpu,
                        executorsPerRunner,
                        currentDevice,
                        queriesHost,
                        queriesDevice,
                        recursivePreprocessor,
                        results,
                        resultsMutex,
                        algorithm,
                        batchQueue,
                        bufferPool,
                        fallbackQueue,
                        boolResults,
                        countResults,
                        exceptions,
                        pipelineAbort);
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
            batch->ctx.queryPipelineDepths   = queryContext.queryPipelineDepths.data();
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

            planner.prepareMiniBatch(batch->plan, *buffer, batch->ctx, leafSubpatterns, pairOffset, maxPairsPerBatch);

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

  ScopedNvtxRange              leafRange("Build LeafSubpatterns");
  RecursivePatternPreprocessor recursivePreprocessor;
  recursivePreprocessor.buildPatterns(queriesHost);
  recursivePreprocessor.syncToDevice(stream);
  leafRange.pop();

  // Ensure queries and patterns are fully copied before workers start using them.
  // Workers use different streams, so we need an explicit sync here.
  cudaCheckError(cudaStreamSynchronize(stream));

  const LeafSubpatterns& leafSubpatterns = recursivePreprocessor.leafSubpatterns();
  QueryPreprocessContext queryContext;
  queryContext.numQueries = numQueries;
  queryContext.queryAtomCounts.resize(numQueries);
  queryContext.queryPipelineDepths.resize(numQueries);
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

      const int depth                     = getQueryPipelineDepth(queriesHost, q);
      queryContext.queryPipelineDepths[q] = depth;
      maxDepthSeen                        = std::max(maxDepthSeen, depth);

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
                              recursivePreprocessor,
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
