#include "mmff_batched_forcefield.h"

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                           MMFF::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}
}  // namespace

MMFFBatchedForcefield::MMFFBatchedForcefield(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                                             BatchedForcefieldMetadata               metadata,
                                             const cudaStream_t                      stream)
    : BatchedForcefield(ForceFieldType::MMFF, 3, molSystemHost.indices.atomStarts, nullptr, std::move(metadata)) {
  MMFF::setStreams(systemDevice_, stream);
  MMFF::sendContribsAndIndicesToDevice(molSystemHost, systemDevice_);
  allocateEnergyScratch(molSystemHost, systemDevice_);
  setAtomStartsDevice(systemDevice_.indices.atomStarts.data());
}

cudaError_t MMFFBatchedForcefield::computeEnergy(double*        energyOuts,
                                                 const double*  positions,
                                                 const uint8_t* activeSystemMask,
                                                 cudaStream_t   stream) {
  return MMFF::computeEnergy(systemDevice_, energyOuts, positions, activeSystemMask, stream);
}

cudaError_t MMFFBatchedForcefield::computeGradients(double*        grad,
                                                    const double*  positions,
                                                    const uint8_t* activeSystemMask,
                                                    cudaStream_t   stream) {
  return MMFF::computeGradients(systemDevice_, positions, grad, activeSystemMask, stream);
}

}  // namespace nvMolKit
