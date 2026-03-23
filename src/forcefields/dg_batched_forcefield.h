#ifndef NVMOLKIT_DG_BATCHED_FORCEFIELD_H
#define NVMOLKIT_DG_BATCHED_FORCEFIELD_H

#include "batched_forcefield.h"
#include "dist_geom.h"

namespace nvMolKit {

class DGBatchedForcefield final : public BatchedForcefield {
 public:
  DGBatchedForcefield(const DistGeom::BatchedMolecularSystemHost& molSystemHost,
                      const std::vector<int>&                     atomStartsHost,
                      double                                      chiralWeight,
                      double                                      fourthDimWeight,
                      BatchedForcefieldMetadata                   metadata = {},
                      cudaStream_t                                stream   = nullptr);

  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;

 private:
  DistGeom::BatchedMolecularDeviceBuffers systemDevice_;
  AsyncDeviceVector<int>                  atomStartsDevice_;
  double                                  chiralWeight_    = 1.0;
  double                                  fourthDimWeight_ = 0.1;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_DG_BATCHED_FORCEFIELD_H
