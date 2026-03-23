#ifndef NVMOLKIT_ETK_BATCHED_FORCEFIELD_H
#define NVMOLKIT_ETK_BATCHED_FORCEFIELD_H

#include "batched_forcefield.h"
#include "dist_geom.h"

namespace nvMolKit {

class ETKBatchedForcefield final : public BatchedForcefield {
 public:
  ETKBatchedForcefield(const DistGeom::BatchedMolecularSystem3DHost& molSystemHost,
                       const std::vector<int>&                       atomStartsHost,
                       bool                                          useBasicKnowledge,
                       BatchedForcefieldMetadata                     metadata = {},
                       cudaStream_t                                  stream   = nullptr);

  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;

  cudaError_t computePlanarEnergy(double*        energyOuts,
                                  const double*  positions,
                                  const uint8_t* activeSystemMask = nullptr,
                                  cudaStream_t   stream           = nullptr);

  const DistGeom::Energy3DForceContribsDevice& contribs() const { return systemDevice_.contribs; }

 private:
  DistGeom::BatchedMolecular3DDeviceBuffers systemDevice_;
  AsyncDeviceVector<int>                    atomStartsDevice_;
  DistGeom::ETKTerm                         term_ = DistGeom::ETKTerm::ALL;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_ETK_BATCHED_FORCEFIELD_H
