#ifndef NVMOLKIT_MMFF_BATCHED_FORCEFIELD_H
#define NVMOLKIT_MMFF_BATCHED_FORCEFIELD_H

#include "batched_forcefield.h"
#include "mmff.h"

namespace nvMolKit {

//! \brief Batched-forcefield adapter for MMFF systems.
//!
//! This wrapper exposes an `MMFF::BatchedMolecularSystemHost` through the
//! generic `BatchedForcefield` interface so host-driven batched BFGS can
//! evaluate MMFF energies and gradients without MMFF-specific dispatch code.
class MMFFBatchedForcefield final : public BatchedForcefield {
 public:
  //! \brief Builds a generic batched-forcefield view over MMFF host data.
  //! \param molSystemHost Flattened MMFF host-side system description.
  //! \param metadata Optional mapping from concrete systems back to logical molecules/conformers.
  //! \param stream CUDA stream used for internal device allocations and uploads.
  explicit MMFFBatchedForcefield(const MMFF::BatchedMolecularSystemHost& molSystemHost,
                                 BatchedForcefieldMetadata               metadata = {},
                                 cudaStream_t                            stream   = nullptr);

  //! \brief Computes MMFF energies through the generic batched-forcefield API.
  cudaError_t computeEnergy(double*        energyOuts,
                            const double*  positions,
                            const uint8_t* activeSystemMask = nullptr,
                            cudaStream_t   stream           = nullptr) override;

  //! \brief Computes MMFF gradients through the generic batched-forcefield API.
  cudaError_t computeGradients(double*        grad,
                               const double*  positions,
                               const uint8_t* activeSystemMask = nullptr,
                               cudaStream_t   stream           = nullptr) override;

 private:
  MMFF::BatchedMolecularDeviceBuffers systemDevice_;
};

}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_BATCHED_FORCEFIELD_H
