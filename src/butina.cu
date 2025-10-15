#include <cub/cub.cuh>

#include "butina.h"
#include "host_vector.h"
namespace nvMolKit {

constexpr int blockSizeCount = 256;

__global__ void butinaKernelCountClusterSize(const cuda::std::span<const double> distanceMatrix,
                                             const cuda::std::span<int>          clusters,
                                             const cuda::std::span<int>          clusterSizes,
                                             const double                        cutoff) {
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);
  const auto numPoints = static_cast<int>(clusters.size());

  const cuda::std::span<const double> distances  = distanceMatrix.subspan(pointIdx * numPoints, numPoints);
  int                                 localCount = 0;
  for (int i = tid; i < numPoints; i += blockSizeCount) {
    const double dist = distances[i];
    if (const int cluster = clusters[i]; dist < cutoff && cluster == 0) {
      localCount++;
    }
  }

  __shared__ cub::BlockReduce<double, blockSizeCount>::TempStorage tempStorage;
  const int totalCount = cub::BlockReduce<double, blockSizeCount>(tempStorage).Sum(localCount);
  if (tid == 0) {
    clusterSizes[pointIdx] = totalCount;
  }
}

__global__ void butinaWriteClusterValue(const cuda::std::span<const double> distanceMatrix,
                                        const cuda::std::span<int>          clusters,
                                        const double                        cutoff,
                                        const int*                          centralIdx,
                                        const int*                          clusterIdx) {
  const auto numPoints  = static_cast<int>(clusters.size());
  const auto tid        = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int  pointIdx   = *centralIdx;
  const int  clusterVal = *clusterIdx;

  const cuda::std::span<const double> distances = distanceMatrix.subspan(pointIdx * numPoints, numPoints);
  if (tid < numPoints) {
    if (const double dist = distances[tid]; dist < cutoff) {
      if (const int cluster = clusters[tid]; cluster == 0) {
        clusters[tid] = clusterVal;
      }
    }
  }
}

__global__ void bumpClusterIdxKernel(int* clusterIdx) {
  if (const auto tid = static_cast<int>(threadIdx.x); tid == 0) {
    clusterIdx[tid] += 1;
  }
}

__global__ void assignSingletonIdsKernel(const cuda::std::span<int> clusters, const int* nextClusterIdx) {
  int clusterIdx = *nextClusterIdx;
  for (int i = 0; i < clusters.size(); i++) {
    if (clusters[i] == 0) {
      clusters[i] = clusterIdx;
      clusterIdx++;
    }
  }
}

void butinaGpu(const AsyncDeviceVector<double>& distanceMatrix,
               AsyncDeviceVector<int>&          clusters,
               const double                     cutoff,
               cudaStream_t                     stream) {
  const size_t numPoints = clusters.size();
  clusters.zero();
  if (const size_t matSize = distanceMatrix.size(); numPoints * numPoints != matSize) {
    throw std::runtime_error("Butina size mismatch" + std::to_string(numPoints) +
                             " points^2 != " + std::to_string(matSize) + " distance matrix size");
  }
  if (clusters.stream() != stream) {
    throw std::runtime_error("Butina stream mismatch");
  }

  AsyncDeviceVector<int> clusterSizes(clusters.size(), stream);
  clusterSizes.zero();

  const AsyncDevicePtr<int> maxIndex(-1, stream);
  const AsyncDevicePtr<int> maxValue(std::numeric_limits<int>::max(), stream);
  const AsyncDevicePtr<int> clusterIdx(1, stream);
  PinnedHostVector<int>     maxCluster(1);
  maxCluster[0]           = std::numeric_limits<int>::max();
  size_t tempStorageBytes = 0;
  cub::DeviceReduce::ArgMax(nullptr,
                            tempStorageBytes,
                            clusterSizes.data(),
                            maxValue.data(),
                            maxIndex.data(),
                            numPoints,
                            stream);
  AsyncDeviceVector<uint8_t> tempStorage(tempStorageBytes, stream);
  const int                  numBlocksFlat = (clusterSizes.size() - 1) / blockSizeCount + 1;
  while (maxCluster[0] > 1) {
    butinaKernelCountClusterSize<<<numPoints, blockSizeCount, 0, stream>>>(toSpan(distanceMatrix),
                                                                           toSpan(clusters),
                                                                           toSpan(clusterSizes),
                                                                           cutoff);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cub::DeviceReduce::ArgMax(tempStorage.data(),
                                             tempStorageBytes,
                                             clusterSizes.data(),
                                             maxValue.data(),
                                             maxIndex.data(),
                                             numPoints,
                                             stream));
    butinaWriteClusterValue<<<numBlocksFlat, blockSizeCount, 0, stream>>>(toSpan(distanceMatrix),
                                                                          toSpan(clusters),
                                                                          cutoff,
                                                                          maxIndex.data(),
                                                                          clusterIdx.data());
    cudaCheckError(cudaGetLastError());
    bumpClusterIdxKernel<<<1, 1, 0, stream>>>(clusterIdx.data());
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaMemcpyAsync(maxCluster.data(), maxValue.data(), sizeof(int), cudaMemcpyDefault, stream));
    cudaStreamSynchronize(stream);
  }
  assignSingletonIdsKernel<<<1, 1, 0, stream>>>(toSpan(clusters), clusterIdx.data());
  cudaStreamSynchronize(stream);
}

}  // namespace nvMolKit