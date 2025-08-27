// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NVMOLKIT_MMFF_H
#define NVMOLKIT_MMFF_H

#include <cstdint>
#include <vector>

#include "device_vector.h"

namespace nvMolKit {
namespace MMFF {

// -----------------------------------------------
// Device and host structs for MMFF contrib terms.
//
// For references on the MMFF forcefield equations, see
// https://www.charmm-gui.org/charmmdoc/mmff.html
// or
// https://docs.eyesopen.com/toolkits/python/oefftk/fftheory.html#mmff
// -----------------------------------------------
struct BondStretchContribTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<double> r0;
  std::vector<double> kb;
};

struct AngleBendTerms {
  std::vector<int>          idx1;
  std::vector<int>          idx2;
  std::vector<int>          idx3;
  std::vector<double>       theta0;
  std::vector<double>       ka;
  std::vector<std::uint8_t> isLinear;
};

struct BendStretchTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<int>    idx3;
  std::vector<double> theta0;
  std::vector<double> restLen1;
  std::vector<double> restLen2;
  std::vector<double> forceConst1;
  std::vector<double> forceConst2;
};

struct OutOfPlaneTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<int>    idx3;
  std::vector<int>    idx4;
  std::vector<double> koop;
};

struct TorsionContribTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<int>    idx3;
  std::vector<int>    idx4;
  std::vector<double> V1;
  std::vector<double> V2;
  std::vector<double> V3;
};

struct VdwTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<double> R_ij_star;
  std::vector<double> wellDepth;
};

struct EleTerms {
  std::vector<int>     idx1;
  std::vector<int>     idx2;
  std::vector<double>  chargeTerm;
  std::vector<uint8_t> dielModel;
  std::vector<uint8_t> is1_4;
};

struct EnergyForceContribsHost {
  BondStretchContribTerms bondTerms;
  AngleBendTerms          angleTerms;
  BendStretchTerms        bendTerms;
  OutOfPlaneTerms         oopTerms;
  TorsionContribTerms     torsionTerms;
  VdwTerms                vdwTerms;
  EleTerms                eleTerms;
};

struct BatchedIndicesHost {
  //! Size n_molecules + 1, defines the start and end of each molecule in the batch.
  //! The last element contains the number of atoms in the system.
  std::vector<int> atomStarts         = {0};
  //! Defines the start of each molecule's energy buffer region that will be added to then reduced.
  std::vector<int> energyBufferStarts = {0};
  //! Size total atoms, maps atom index to batch index.
  std::vector<int> atomIdxToBatchIdx;
  //! Size total energy buffer blocks, maps energy buffer block index to batch index.
  std::vector<int> energyBufferBlockIdxToBatchIdx;
  //! Size n_molecules, defines the start and end of each molecule's bond term count
  std::vector<int> bondTermStarts    = {0};
  //! Size n_molecules, defines the start and end of each molecule's angle term count
  std::vector<int> angleTermStarts   = {0};
  //! Size n_molecules, defines the start and end of each molecule's bend term count
  std::vector<int> bendTermStarts    = {0};
  //! Size n_molecules, defines the start and end of each molecule's oop term count
  std::vector<int> oopTermStarts     = {0};
  //! Size n_molecules, defines the start and end of each molecule's torsion term count
  std::vector<int> torsionTermStarts = {0};
  //! Size n_molecules, defines the start and end of each molecule's vdw term count
  std::vector<int> vdwTermStarts     = {0};
  //! Size n_molecules, defines the start and end of each molecule's ele term count
  std::vector<int> eleTermStarts     = {0};
};

struct BatchedMolecularSystemHost {
  EnergyForceContribsHost contribs;
  BatchedIndicesHost      indices;
  //! Size total num atoms * 3
  std::vector<double>     positions;

  //! Size total num atoms
  std::vector<int> atomNumbers;
  //! Largest system size in the batch
  int              maxNumAtoms = 0;
};

struct BondStretchContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<double> r0;
  nvMolKit::AsyncDeviceVector<double> kb;
};

struct AngleBendTermsDevice {
  nvMolKit::AsyncDeviceVector<int>          idx1;
  nvMolKit::AsyncDeviceVector<int>          idx2;
  nvMolKit::AsyncDeviceVector<int>          idx3;
  nvMolKit::AsyncDeviceVector<double>       theta0;
  nvMolKit::AsyncDeviceVector<double>       ka;
  nvMolKit::AsyncDeviceVector<std::uint8_t> isLinear;
};

struct BendStretchTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<int>    idx3;
  nvMolKit::AsyncDeviceVector<double> theta0;
  nvMolKit::AsyncDeviceVector<double> restLen1;
  nvMolKit::AsyncDeviceVector<double> restLen2;
  nvMolKit::AsyncDeviceVector<double> forceConst1;
  nvMolKit::AsyncDeviceVector<double> forceConst2;
};

struct OutOfPlaneTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<int>    idx3;
  nvMolKit::AsyncDeviceVector<int>    idx4;
  nvMolKit::AsyncDeviceVector<double> koop;
};

struct TorsionContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<int>    idx3;
  nvMolKit::AsyncDeviceVector<int>    idx4;
  nvMolKit::AsyncDeviceVector<double> V1;
  nvMolKit::AsyncDeviceVector<double> V2;
  nvMolKit::AsyncDeviceVector<double> V3;
};

struct VdwTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<double> R_ij_star;
  nvMolKit::AsyncDeviceVector<double> wellDepth;
};

struct EleTermsDevice {
  nvMolKit::AsyncDeviceVector<int>     idx1;
  nvMolKit::AsyncDeviceVector<int>     idx2;
  nvMolKit::AsyncDeviceVector<double>  chargeTerm;
  nvMolKit::AsyncDeviceVector<uint8_t> dielModel;
  nvMolKit::AsyncDeviceVector<uint8_t> is1_4;
};

struct EnergyForceContribsDevice {
  BondStretchContribTermsDevice bondTerms;
  AngleBendTermsDevice          angleTerms;
  BendStretchTermsDevice        bendTerms;
  OutOfPlaneTermsDevice         oopTerms;
  TorsionContribTermsDevice     torsionTerms;
  VdwTermsDevice                vdwTerms;
  EleTermsDevice                eleTerms;
};

//! See BatchedIndices for more information on each field.
struct BatchedIndicesDevice {
  nvMolKit::AsyncDeviceVector<int> atomStarts;
  nvMolKit::AsyncDeviceVector<int> atomIdxToBatchIdx;
  nvMolKit::AsyncDeviceVector<int> energyBufferStarts;
  nvMolKit::AsyncDeviceVector<int> energyBufferBlockIdxToBatchIdx;

  nvMolKit::AsyncDeviceVector<int> bondTermStarts;
  nvMolKit::AsyncDeviceVector<int> angleTermStarts;
  nvMolKit::AsyncDeviceVector<int> bendTermStarts;
  nvMolKit::AsyncDeviceVector<int> oopTermStarts;
  nvMolKit::AsyncDeviceVector<int> torsionTermStarts;
  nvMolKit::AsyncDeviceVector<int> vdwTermStarts;
  nvMolKit::AsyncDeviceVector<int> eleTermStarts;
};

//! Buffers for interfacing with the 4D padded L-BFGS minimizer.
//! The minimizer requires homogenous batches, so we need to pad the positions
//! and gradients. It also works in double4s, so we need to convert.
//!
//!  Operation flow between buffers:
//!      In energy calculation:
//!         - Copy interface padded d4 positions to our condensed d3.
//!         - Compute energies
//!         - return enegies, no additional copying needed since 1 per molecule.
//!
//!      In grad calculation:
//!         - Copy interface padded d4 positions to our condensed d3
//!         - Compute gradients
//!         - Copy d3 gradients to interface padded d4 gradients
//!
//!      In output gathering:
//!         - Copy d4 padded positions to our d3 condensed and download
//! TODO: Potential optimization points:
//!    - Kernels that work on d4 padded positions and gradients directly
//!    - One copy of positions for energy and gradient. This might be shaky
struct Dim4PaddedInterfaceBuffers {
  //! Size n_molecules * (max atoms in batch) * 4
  nvMolKit::AsyncDeviceVector<double> positionsD4Padded;
  //! Size n_molecules * (max atoms in batch) * 3
  nvMolKit::AsyncDeviceVector<double> gradD3Padded;
  //! Size n_molecules * (max atoms in batch) * 4, will be -1 for padded or 4th dims.
  nvMolKit::AsyncDeviceVector<int>    writeBackIndices;
  //! Size n_molecules * (max atoms in batch)
  nvMolKit::AsyncDeviceVector<int>    atomNumbers;
};

//! Device buffers for the batched molecular system.
//! Most of the terms are either 1 per molecule or CSR-like format with the BatchedIndicesDevice terms used for
//! indexing.
//!
//! The only nonstandard term is the energyBuffer and associated indices, which is allocated and used the following way.
//! - First, the maximum energy term size for each molecule is calculated. For example, a molecule with 10 bond terms,
//!   20 angle terms, and 5 bend terms would have a max term size of 20.
//! - Next, the energy buffer per molecule is rounded up to the energy reduction block size. See
//!   blockSizeEnergyReduction in kernel_utils.cuh. So, a molecule with a max term of 150 will be allocated
//!   256 energy buffer positions.
//! - Indices are built off of this data. The energyBufferStarts field in BatchedIndicesDevice defines the start of each
//!   buffer region. It is guaranteed to be an offset of blockSizeEnergyReduction but this is not critical to the
//!   algorithm. The energyBufferBlockIdxToBatchIdx maps each block to the molecule it belongs to.
//! - When computing energies, each term adds into term index + energyBufferStarts[moleculeIdx] in the energy buffer.
//!   This means that for smaller terms, the energy buffer will have some unused space, and unless the largest term is
//!   a multiple of the block size, there will be some zero elements. This is fine and expected.
//!   Finally, on reduction, each block does a local summation, then atomically adds to the output energy for the
//!   molecule, using the energyBufferBlockIdxToBatchIdx to map the block to the molecule output index.
struct BatchedMolecularDeviceBuffers {
  EnergyForceContribsDevice           contribs;
  //! Size n_molecules
  BatchedIndicesDevice                indices;
  //! Size total num atoms * 3
  nvMolKit::AsyncDeviceVector<double> positions;
  //! Size total num atoms
  nvMolKit::AsyncDeviceVector<int>    atomNumbers;
  //! Size total num atoms * 3
  nvMolKit::AsyncDeviceVector<double> grad;
  //! Variable size - max terms in each molecule concatenated.
  //! Each molecule has an energy buffer to add to and reduce to energyOuts.
  nvMolKit::AsyncDeviceVector<double> energyBuffer;
  //! Size n_molecules
  nvMolKit::AsyncDeviceVector<double> energyOuts;
  //! Dimension change and padding buffers
  Dim4PaddedInterfaceBuffers          dataFormatInterchangeBuffers;
};

//! Add a molecule to the batched molecular system.
//! Populates the molSystem with the molecule's energy force contribs, and adds the current positions.
void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem,
                        std::vector<int>*              atomNumbers = nullptr);

//! Send the batched molecular system to the device.
void sendContribsAndIndicesToDevice(const BatchedMolecularSystemHost& molSystemHost,
                                    BatchedMolecularDeviceBuffers&    molSystemDevice);

//! Sets all device vector streams
void setStreams(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream);

//! Allocate intermediate buffers on the device for the batched molecular system.
//! These include the gradients, energy buffer, and energy outs.
void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice);

//! Allocate the buffers for the 4D padded interface.
void allocateDim4ConversionBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                   BatchedMolecularDeviceBuffers&    molSystemDevice);

//! Compute the energy of the batched molecular system. This will populate the energyOuts buffer on device.
//! energyOuts and energyBuffer must be zeroed before calling this function.
//! Optionally computes on user-provided coordinates rather than those in molSystemDevice.
//! If not null, coords must be GPU resident. The molSystemDevice intermediate system description,
//! energy accumulator and output buffers are always used, only the coordinates are swappable.
cudaError_t computeEnergy(BatchedMolecularDeviceBuffers& molSystemDevice,
                          const double*                  coords = nullptr,
                          cudaStream_t                   stream = nullptr);
//! Compute the gradients of the batched molecular system. This will populate the grad buffer on device.
//! grad must be zeroed before calling this function.
cudaError_t computeGradients(BatchedMolecularDeviceBuffers& molSystemDevice, cudaStream_t stream = nullptr);

}  // namespace MMFF
}  // namespace nvMolKit

#endif  // NVMOLKIT_MMFF_H
