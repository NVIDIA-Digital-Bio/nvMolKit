#ifndef NVMOLKIT_UFF_BATCHED_FORCEFIELD_H
#define NVMOLKIT_UFF_BATCHED_FORCEFIELD_H

#include "batched_forcefield.h"
#include "uff.h"

namespace nvMolKit {

class UFFBatchedForcefield final : public BatchedForcefield {
 public:
  explicit UFFBatchedForcefield(const UFF::BatchedMolecularSystemHost& molSystemHost,
                                BatchedForcefieldMetadata              metadata = {},
                                cudaStream_t                           stream   = nullptr);

  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;

 private:
  UFF::BatchedMolecularDeviceBuffers systemDevice_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_UFF_BATCHED_FORCEFIELD_H
