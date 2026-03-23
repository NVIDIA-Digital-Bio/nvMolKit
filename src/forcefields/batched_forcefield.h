#ifndef NVMOLKIT_BATCHED_FORCEFIELD_H
#define NVMOLKIT_BATCHED_FORCEFIELD_H

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "../minimizer/bfgs_types.h"

namespace nvMolKit {

struct BatchedSystemInfo {
  int systemIdx    = -1;
  int moleculeIdx  = -1;
  int conformerIdx = -1;
};

struct BatchedForcefieldMetadata {
  std::vector<int>              systemToMoleculeIdx;
  std::vector<int>              systemToConformerIdx;
  std::vector<std::vector<int>> moleculeToSystemIndices;

  void reserveSystems(const int numSystems) {
    systemToMoleculeIdx.reserve(numSystems);
    systemToConformerIdx.reserve(numSystems);
  }

  void ensureMolecule(const int moleculeIdx) {
    if (moleculeIdx >= static_cast<int>(moleculeToSystemIndices.size())) {
      moleculeToSystemIndices.resize(moleculeIdx + 1);
    }
  }

  BatchedSystemInfo recordSystem(const int moleculeIdx, const int conformerIdx) {
    const int systemIdx = static_cast<int>(systemToMoleculeIdx.size());
    ensureMolecule(moleculeIdx);
    systemToMoleculeIdx.push_back(moleculeIdx);
    systemToConformerIdx.push_back(conformerIdx);
    moleculeToSystemIndices[moleculeIdx].push_back(systemIdx);
    BatchedSystemInfo info;
    info.systemIdx    = systemIdx;
    info.moleculeIdx  = moleculeIdx;
    info.conformerIdx = conformerIdx;
    return info;
  }

  int numSystems() const { return static_cast<int>(systemToMoleculeIdx.size()); }
  int numLogicalMolecules() const { return static_cast<int>(moleculeToSystemIndices.size()); }

  static BatchedForcefieldMetadata identity(const int numSystems) {
    BatchedForcefieldMetadata metadata;
    metadata.reserveSystems(numSystems);
    metadata.moleculeToSystemIndices.resize(numSystems);
    for (int systemIdx = 0; systemIdx < numSystems; ++systemIdx) {
      metadata.systemToMoleculeIdx.push_back(systemIdx);
      metadata.systemToConformerIdx.push_back(0);
      metadata.moleculeToSystemIndices[systemIdx].push_back(systemIdx);
    }
    return metadata;
  }
};

class BatchedForcefield {
 public:
  virtual ~BatchedForcefield() = default;

  virtual cudaError_t computeEnergy(double*        energyOuts,
                                    const double*  positions,
                                    const uint8_t* activeSystemMask = nullptr,
                                    cudaStream_t   stream           = nullptr) = 0;

  virtual cudaError_t computeGradients(double*        grad,
                                       const double*  positions,
                                       const uint8_t* activeSystemMask = nullptr,
                                       cudaStream_t   stream           = nullptr) = 0;

  int                              numMolecules() const { return numMolecules_; }
  int                              dataDim() const { return dataDim_; }
  int                              totalPositions() const { return totalPositions_; }
  const std::vector<int>&          atomStartsHost() const { return atomStartsHost_; }
  const int*                       atomStartsDevice() const { return atomStartsDevice_; }
  ForceFieldType                   type() const { return type_; }
  const BatchedForcefieldMetadata& metadata() const { return metadata_; }
  int                              numLogicalMolecules() const { return metadata_.numLogicalMolecules(); }
  const std::vector<int>&          systemToMoleculeIdx() const { return metadata_.systemToMoleculeIdx; }
  const std::vector<int>&          systemToConformerIdx() const { return metadata_.systemToConformerIdx; }
  const std::vector<int>&          systemsForMolecule(const int moleculeIdx) const {
    return metadata_.moleculeToSystemIndices[moleculeIdx];
  }

 protected:
  BatchedForcefield(ForceFieldType            type,
                    int                       dataDim,
                    std::vector<int>          atomStartsHost,
                    const int*                atomStartsDevice,
                    BatchedForcefieldMetadata metadata = {})
      : numMolecules_(static_cast<int>(atomStartsHost.size()) - 1),
        dataDim_(dataDim),
        totalPositions_(atomStartsHost.empty() ? 0 : atomStartsHost.back() * dataDim),
        atomStartsHost_(std::move(atomStartsHost)),
        atomStartsDevice_(atomStartsDevice),
        metadata_(metadata.numSystems() == 0 ? BatchedForcefieldMetadata::identity(numMolecules_) :
                                               std::move(metadata)),
        type_(type) {}

  void setAtomStartsDevice(const int* atomStartsDevice) { atomStartsDevice_ = atomStartsDevice; }

 private:
  int                       numMolecules_   = 0;
  int                       dataDim_        = 0;
  int                       totalPositions_ = 0;
  std::vector<int>          atomStartsHost_;
  const int*                atomStartsDevice_ = nullptr;
  BatchedForcefieldMetadata metadata_;
  ForceFieldType            type_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_BATCHED_FORCEFIELD_H
