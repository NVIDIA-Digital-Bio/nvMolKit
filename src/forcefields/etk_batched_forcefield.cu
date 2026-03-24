#include "etk_batched_forcefield.h"

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const DistGeom::BatchedMolecularSystem3DHost& molSystemHost,
                           DistGeom::BatchedMolecular3DDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
}
}  // namespace

ETKBatchedForcefield::ETKBatchedForcefield(const DistGeom::BatchedMolecularSystem3DHost& molSystemHost,
                                           const std::vector<int>&                       atomStartsHost,
                                           const bool                                    useBasicKnowledge,
                                           BatchedForcefieldMetadata                     metadata,
                                           const cudaStream_t                            stream)
    : BatchedForcefield(ForceFieldType::ETK, 3, atomStartsHost, nullptr, std::move(metadata)),
      term_(useBasicKnowledge ? DistGeom::ETKTerm::ALL : DistGeom::ETKTerm::PLAIN) {
  atomStartsDevice_.setStream(stream);
  DistGeom::setStreams(systemDevice_, stream);
  DistGeom::sendContribsAndIndicesToDevice3D(molSystemHost, systemDevice_);
  atomStartsDevice_.setFromVector(atomStartsHost);
  allocateEnergyScratch(molSystemHost, systemDevice_);
  setAtomStartsDevice(atomStartsDevice_.data());
}

cudaError_t ETKBatchedForcefield::computeEnergy(double*        energyOuts,
                                                const double*  positions,
                                                const uint8_t* activeSystemMask,
                                                cudaStream_t   stream) {
  return DistGeom::computeEnergyETK(systemDevice_,
                                    energyOuts,
                                    atomStartsDevice_.data(),
                                    positions,
                                    activeSystemMask,
                                    positions,
                                    term_,
                                    stream);
}

cudaError_t ETKBatchedForcefield::computeGradients(double*        grad,
                                                   const double*  positions,
                                                   const uint8_t* activeSystemMask,
                                                   cudaStream_t   stream) {
  return DistGeom::computeGradientsETK(systemDevice_,
                                       grad,
                                       atomStartsDevice_.data(),
                                       positions,
                                       activeSystemMask,
                                       term_,
                                       stream);
}

cudaError_t ETKBatchedForcefield::computePlanarEnergy(double*        energyOuts,
                                                      const double*  positions,
                                                      const uint8_t* activeSystemMask,
                                                      cudaStream_t   stream) {
  return DistGeom::computePlanarEnergy(systemDevice_,
                                       energyOuts,
                                       atomStartsDevice_.data(),
                                       positions,
                                       activeSystemMask,
                                       positions,
                                       stream);
}

}  // namespace nvMolKit
