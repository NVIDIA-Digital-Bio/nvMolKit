#include "uff_batched_forcefield.h"

namespace nvMolKit {
namespace {
void allocateEnergyScratch(const UFF::BatchedMolecularSystemHost& molSystemHost,
                           UFF::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}
}  // namespace

UFFBatchedForcefield::UFFBatchedForcefield(const UFF::BatchedMolecularSystemHost& molSystemHost,
                                           BatchedForcefieldMetadata               metadata,
                                           const cudaStream_t                      stream)
    : BatchedForcefield(ForceFieldType::UFF, 3, molSystemHost.indices.atomStarts, nullptr, std::move(metadata)) {
  UFF::setStreams(systemDevice_, stream);
  UFF::sendContribsAndIndicesToDevice(molSystemHost, systemDevice_);
  allocateEnergyScratch(molSystemHost, systemDevice_);
  setAtomStartsDevice(systemDevice_.indices.atomStarts.data());
}

cudaError_t UFFBatchedForcefield::computeEnergy(double*        energyOuts,
                                                const double*  positions,
                                                const uint8_t* activeSystemMask,
                                                cudaStream_t   stream) {
  return UFF::computeEnergy(systemDevice_, energyOuts, positions, activeSystemMask, stream);
}

cudaError_t UFFBatchedForcefield::computeGradients(double*        grad,
                                                   const double*  positions,
                                                   const uint8_t* activeSystemMask,
                                                   cudaStream_t   stream) {
  return UFF::computeGradients(systemDevice_, positions, grad, activeSystemMask, stream);
}

}  // namespace nvMolKit
