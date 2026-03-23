#include "dg_batched_forcefield.h"

namespace nvMolKit {

namespace {
void allocateEnergyScratch(const DistGeom::BatchedMolecularSystemHost& molSystemHost,
                           DistGeom::BatchedMolecularDeviceBuffers&    systemDevice) {
  systemDevice.energyBuffer.resize(molSystemHost.indices.energyBufferStarts.back());
  systemDevice.energyBuffer.zero();
  systemDevice.dimension = molSystemHost.dimension;
}
}  // namespace

DGBatchedForcefield::DGBatchedForcefield(const DistGeom::BatchedMolecularSystemHost& molSystemHost,
                                         const std::vector<int>&                     atomStartsHost,
                                         const double                                chiralWeight,
                                         const double                                fourthDimWeight,
                                         BatchedForcefieldMetadata                   metadata,
                                         const cudaStream_t                          stream)
    : BatchedForcefield(ForceFieldType::DG, molSystemHost.dimension, atomStartsHost, nullptr, std::move(metadata)),
      chiralWeight_(chiralWeight),
      fourthDimWeight_(fourthDimWeight) {
  atomStartsDevice_.setStream(stream);
  DistGeom::setStreams(systemDevice_, stream);
  DistGeom::sendContribsAndIndicesToDevice(molSystemHost, systemDevice_);
  atomStartsDevice_.setFromVector(atomStartsHost);
  allocateEnergyScratch(molSystemHost, systemDevice_);
  setAtomStartsDevice(atomStartsDevice_.data());
}

cudaError_t DGBatchedForcefield::computeEnergy(double*        energyOuts,
                                               const double*  positions,
                                               const uint8_t* activeSystemMask,
                                               cudaStream_t   stream) {
  return DistGeom::computeEnergy(systemDevice_,
                                 energyOuts,
                                 atomStartsDevice_.data(),
                                 positions,
                                 chiralWeight_,
                                 fourthDimWeight_,
                                 activeSystemMask,
                                 positions,
                                 stream);
}

cudaError_t DGBatchedForcefield::computeGradients(double*        grad,
                                                  const double*  positions,
                                                  const uint8_t* activeSystemMask,
                                                  cudaStream_t   stream) {
  return DistGeom::computeGradients(systemDevice_,
                                    grad,
                                    atomStartsDevice_.data(),
                                    positions,
                                    chiralWeight_,
                                    fourthDimWeight_,
                                    activeSystemMask,
                                    stream);
}

}  // namespace nvMolKit
