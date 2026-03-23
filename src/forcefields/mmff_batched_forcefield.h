#ifndef NVMOLKIT_MMFF_BATCHED_FORCEFIELD_H
#define NVMOLKIT_MMFF_BATCHED_FORCEFIELD_H

#include "batched_forcefield.h"
#include "mmff.h"

namespace nvMolKit {

class MMFFBatchedForcefield final : public BatchedForcefield {
 public:
  explicit MMFFBatchedForcefield(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                                 BatchedForcefieldMetadata               metadata = {},
                                 cudaStream_t                            stream   = nullptr);

  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;

 private:
  MMFF::BatchedMolecularDeviceBuffers systemDevice_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_BATCHED_FORCEFIELD_H
