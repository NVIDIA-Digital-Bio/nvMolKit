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

#ifndef NVMOLKIT_DISTGEOM_H
#define NVMOLKIT_DISTGEOM_H

#include <cstdint>
#include <vector>

#include "device_vector.h"

namespace nvMolKit {
namespace DistGeom {
// ----------------------------
// Distance Geometry (DG) terms
// ----------------------------
struct DistViolationContribTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<double> ub2;  // squared upper bounds
  std::vector<double> lb2;  // squared lower bounds
  std::vector<double> weight;
};

struct ChiralViolationContribTerms {
  std::vector<int>    idx1;
  std::vector<int>    idx2;
  std::vector<int>    idx3;
  std::vector<int>    idx4;
  std::vector<double> volUpper;
  std::vector<double> volLower;
  std::vector<double> weight;
};

struct FourthDimContribTerms {
  std::vector<int>    idx;
  std::vector<double> weight;
};

struct EnergyForceContribsHost {
  DistViolationContribTerms   distTerms;
  ChiralViolationContribTerms chiralTerms;
  FourthDimContribTerms       fourthTerms;
};

// ------------------------------------
// Experimental Torsion Knowledge (ETK)
// ------------------------------------

enum class ETKTerm {
  ALL = 0,
  EXPERIMANTAL_TORSION,
  IMPPROPER_TORSION,
  DISTANCE_12,
  DISTANCE_13,
  ANGLE_13,
  LONGDISTANCE
};

struct TorsionAngleContribTerms {
  std::vector<int>    idx1;            // Index of atom1 of the torsion angle
  std::vector<int>    idx2;            // Index of atom2 of the torsion angle
  std::vector<int>    idx3;            // Index of atom3 of the torsion angle
  std::vector<int>    idx4;            // Index of atom4 of the torsion angle
  std::vector<double> forceConstants;  // Force constants for each torsion (6 values per term)
  std::vector<int>    signs;           // Signs for each torsion (6 values per term)
};

struct InversionContribTerms {
  std::vector<int>    idx1;           // Index of atom1
  std::vector<int>    idx2;           // Index of atom2
  std::vector<int>    idx3;           // Index of atom3
  std::vector<int>    idx4;           // Index of atom4
  std::vector<int>    at2AtomicNum;   // Atomic number for atom 2
  std::vector<bool>   isCBoundToO;    // True if atom 2 is sp2 carbon bound to sp2 oxygen
  std::vector<double> C0;             // Inversion coefficient 0
  std::vector<double> C1;             // Inversion coefficient 1
  std::vector<double> C2;             // Inversion coefficient 2
  std::vector<double> forceConstant;  // Force constant
  std::vector<int>    numImpropers;
};

struct DistanceConstraintContribTerms {
  std::vector<int>     idx1;                   // Index of atom1 of the distance constraint
  std::vector<int>     idx2;                   // Index of atom2 of the distance constraint
  std::vector<double>  minLen;                 // Lower bound of the flat bottom potential
  std::vector<double>  maxLen;                 // Upper bound of the flat bottom potential
  std::vector<double>  forceConstant;          // Force constant for distance constraint
  std::vector<uint8_t> isImproperConstrained;  // True if the angle is an improper torsion. Only used for 1-3 distances.
};

struct AngleConstraintContribTerms {
  std::vector<int>    idx1;      // Index of atom1 of the angle constraint
  std::vector<int>    idx2;      // Index of atom2 of the angle constraint
  std::vector<int>    idx3;      // Index of atom3 of the angle constraint
  std::vector<double> minAngle;  // Lower bound of the flat bottom potential
  std::vector<double> maxAngle;  // Upper bound of the flat bottom potential
};

struct Energy3DForceContribsHost {
  // Experimental torsion terms (from addExperimentalTorsionTerms)
  TorsionAngleContribTerms experimentalTorsionTerms;

  // Improper torsion terms (from addImproperTorsionTerms)
  InversionContribTerms improperTorsionTerms;

  // 1-2 distance terms (from add12Terms)
  DistanceConstraintContribTerms dist12Terms;

  // 1-3 distance terms (from add13Terms)
  DistanceConstraintContribTerms dist13Terms;
  // 1-3 angle terms (from add13Terms)
  AngleConstraintContribTerms    angle13Terms;

  // Long range distance terms (from addLongRangeDistanceConstraints)
  DistanceConstraintContribTerms longRangeDistTerms;
};

struct BatchedIndicesHost {
  //! Defines the start of each molecule's energy buffer region that will be added to then reduced.
  std::vector<int> energyBufferStarts = {0};
  //! Size total atoms, maps atom index to batch index (i.e. molecule index in the batch).
  std::vector<int> atomIdxToBatchIdx;
  //! Size total energy buffer blocks, maps energy buffer block index to batch index.
  std::vector<int> energyBufferBlockIdxToBatchIdx;
  //! Size n_molecules + 1, defines the start and end of each molecule's distance violation term count
  std::vector<int> distTermStarts   = {0};
  //! Size n_molecules + 1, defines the start and end of each molecule's chirality violation term count
  std::vector<int> chiralTermStarts = {0};
  //! Size n_molecules + 1, defines the start and end of each molecule's fourth dimension term count
  std::vector<int> fourthTermStarts = {0};
};

struct BatchedIndices3DHost {
  // Common batch indices
  std::vector<int> energyBufferStarts = {0};        // Defines the start of each molecule's energy buffer region
  std::vector<int> atomIdxToBatchIdx;               // Maps atom index to batch index (molecule index in the batch)
  std::vector<int> energyBufferBlockIdxToBatchIdx;  // Maps energy buffer block index to batch index

  // Term starts for each contribution type
  std::vector<int> experimentalTorsionTermStarts = {0};  // Start indices for experimental torsion terms
  std::vector<int> improperTorsionTermStarts     = {0};  // Start indices for improper torsion terms
  std::vector<int> dist12TermStarts              = {0};  // Start indices for 1-2 distance terms
  std::vector<int> dist13TermStarts              = {0};  // Start indices for 1-3 distance terms
  std::vector<int> angle13TermStarts             = {0};  // Start indices for 1-3 angle terms
  std::vector<int> longRangeDistTermStarts       = {0};  // Start indices for long range distance terms
};

struct BatchedMolecularSystemHost {
  EnergyForceContribsHost contribs;
  BatchedIndicesHost      indices;
  //! Size total num atoms
  std::vector<int>        atomNumbers;
  //! Largest system size in the batch
  int                     maxNumAtoms = 0;
  //! Dimension of all molecules in the batch (3 or 4)
  int                     dimension   = 3;
};

struct BatchedMolecularSystem3DHost {
  Energy3DForceContribsHost contribs;
  BatchedIndices3DHost      indices;
};

struct DistViolationContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<double> ub2;  // squared upper bounds
  nvMolKit::AsyncDeviceVector<double> lb2;  // squared lower bounds
  nvMolKit::AsyncDeviceVector<double> weight;
};

struct ChiralViolationContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;
  nvMolKit::AsyncDeviceVector<int>    idx2;
  nvMolKit::AsyncDeviceVector<int>    idx3;
  nvMolKit::AsyncDeviceVector<int>    idx4;
  nvMolKit::AsyncDeviceVector<double> volUpper;
  nvMolKit::AsyncDeviceVector<double> volLower;
  nvMolKit::AsyncDeviceVector<double> weight;
};

struct FourthDimContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx;
  nvMolKit::AsyncDeviceVector<double> weight;
};

struct TorsionAngleContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;            // Index of atom1 of the torsion angle
  nvMolKit::AsyncDeviceVector<int>    idx2;            // Index of atom2 of the torsion angle
  nvMolKit::AsyncDeviceVector<int>    idx3;            // Index of atom3 of the torsion angle
  nvMolKit::AsyncDeviceVector<int>    idx4;            // Index of atom4 of the torsion angle
  nvMolKit::AsyncDeviceVector<double> forceConstants;  // Force constants for each torsion (6 values per term)
  nvMolKit::AsyncDeviceVector<int>    signs;           // Signs for each torsion (6 values per term)
};

struct InversionContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int> idx1;          // Index of atom1
  nvMolKit::AsyncDeviceVector<int> idx2;          // Index of atom2
  nvMolKit::AsyncDeviceVector<int> idx3;          // Index of atom3
  nvMolKit::AsyncDeviceVector<int> idx4;          // Index of atom4
  nvMolKit::AsyncDeviceVector<int> at2AtomicNum;  // Atomic number for atom 2
  nvMolKit::AsyncDeviceVector<uint8_t>
    isCBoundToO;  // True if atom 2 is sp2 carbon bound to sp2 oxygen (stored as int for device)
  nvMolKit::AsyncDeviceVector<double> C0;             // Inversion coefficient 0
  nvMolKit::AsyncDeviceVector<double> C1;             // Inversion coefficient 1
  nvMolKit::AsyncDeviceVector<double> C2;             // Inversion coefficient 2
  nvMolKit::AsyncDeviceVector<double> forceConstant;  // Force constant
  nvMolKit::AsyncDeviceVector<int>    numImpropers;
};

struct DistanceConstraintContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>     idx1;                   // Index of atom1 of the distance constraint
  nvMolKit::AsyncDeviceVector<int>     idx2;                   // Index of atom2 of the distance constraint
  nvMolKit::AsyncDeviceVector<double>  minLen;                 // Lower bound of the flat bottom potential
  nvMolKit::AsyncDeviceVector<double>  maxLen;                 // Upper bound of the flat bottom potential
  nvMolKit::AsyncDeviceVector<double>  forceConstant;          // Force constant for distance constraint
  nvMolKit::AsyncDeviceVector<uint8_t> isImproperConstrained;  // True if the angle is an improper torsion. Only used
                                                               // for 1-3 distances (stored as int for device)
};

struct AngleConstraintContribTermsDevice {
  nvMolKit::AsyncDeviceVector<int>    idx1;      // Index of atom1 of the angle constraint
  nvMolKit::AsyncDeviceVector<int>    idx2;      // Index of atom2 of the angle constraint
  nvMolKit::AsyncDeviceVector<int>    idx3;      // Index of atom3 of the angle constraint
  nvMolKit::AsyncDeviceVector<double> minAngle;  // Lower bound of the flat bottom potential
  nvMolKit::AsyncDeviceVector<double> maxAngle;  // Upper bound of the flat bottom potential;
};

struct EnergyForceContribsDevice {
  DistViolationContribTermsDevice   distTerms;
  ChiralViolationContribTermsDevice chiralTerms;
  FourthDimContribTermsDevice       fourthTerms;
};

struct Energy3DForceContribsDevice {
  // Experimental torsion terms (from addExperimentalTorsionTerms)
  TorsionAngleContribTermsDevice experimentalTorsionTerms;

  // Improper torsion terms (from addImproperTorsionTerms)
  InversionContribTermsDevice improperTorsionTerms;

  // 1-2 distance terms (from add12Terms)
  DistanceConstraintContribTermsDevice dist12Terms;

  // 1-3 distance terms (from add13Terms)
  DistanceConstraintContribTermsDevice dist13Terms;

  // 1-3 angle terms (from add13Terms)
  AngleConstraintContribTermsDevice angle13Terms;

  // Long range distance terms (from addLongRangeDistanceConstraints)
  DistanceConstraintContribTermsDevice longRangeDistTerms;
};

//! See BatchedIndices for more information on each field.
struct BatchedIndicesDevice {
  nvMolKit::AsyncDeviceVector<int> energyBufferStarts;
  nvMolKit::AsyncDeviceVector<int> atomIdxToBatchIdx;
  nvMolKit::AsyncDeviceVector<int> energyBufferBlockIdxToBatchIdx;

  nvMolKit::AsyncDeviceVector<int> distTermStarts;
  nvMolKit::AsyncDeviceVector<int> chiralTermStarts;
  nvMolKit::AsyncDeviceVector<int> fourthTermStarts;
};

struct BatchedIndices3DDevice {
  // Common batch indices
  nvMolKit::AsyncDeviceVector<int> energyBufferStarts;  // Defines the start of each molecule's energy buffer region
  nvMolKit::AsyncDeviceVector<int> atomIdxToBatchIdx;   // Maps atom index to batch index (molecule index in the batch)
  nvMolKit::AsyncDeviceVector<int> energyBufferBlockIdxToBatchIdx;  // Maps energy buffer block index to batch index

  // Term starts for each contribution type
  nvMolKit::AsyncDeviceVector<int> experimentalTorsionTermStarts;  // Start indices for experimental torsion terms
  nvMolKit::AsyncDeviceVector<int> improperTorsionTermStarts;      // Start indices for improper torsion terms
  nvMolKit::AsyncDeviceVector<int> dist12TermStarts;               // Start indices for 1-2 distance terms
  nvMolKit::AsyncDeviceVector<int> dist13TermStarts;               // Start indices for 1-3 distance terms
  nvMolKit::AsyncDeviceVector<int> angle13TermStarts;              // Start indices for 1-3 angle terms
  nvMolKit::AsyncDeviceVector<int> longRangeDistTermStarts;        // Start indices for long range distance terms
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
//!         - Copy d3 gradients to interface padded d3 gradients
//!
//!      In output gathering:
//!         - Copy d4 padded positions to our d3 condensed and download
//! TODO: Potential optimization points:
//!    - Kernels that work on d4 padded positions and gradients directly
//!    - One copy of positions for energy and gradient. This might be shaky

struct PaddedInterfaceBuffers {
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
  //! Size total num atoms
  nvMolKit::AsyncDeviceVector<int>    atomNumbers;
  //! Size total num positions of all molecules
  nvMolKit::AsyncDeviceVector<double> grad;
  //! Variable size - max terms in each molecule concatenated.
  //! Each molecule has an energy buffer to add to and reduce to energyOuts.
  nvMolKit::AsyncDeviceVector<double> energyBuffer;
  //! Size n_molecules
  nvMolKit::AsyncDeviceVector<double> energyOuts;
  //! Dimension change and padding buffers
  PaddedInterfaceBuffers              dataFormatInterchangeBuffers;
  //! Dimension of all molecules in the batch (3 or 4)
  int                                 dimension = 3;
};

struct BatchedMolecular3DDeviceBuffers {
  Energy3DForceContribsDevice         contribs;
  BatchedIndices3DDevice              indices;
  //! Size total num positions of all molecules
  nvMolKit::AsyncDeviceVector<double> grad;
  //! Variable size - max terms in each molecule concatenated.
  //! Each molecule has an energy buffer to add to and reduce to energyOuts.
  nvMolKit::AsyncDeviceVector<double> energyBuffer;
  //! Size n_molecules
  nvMolKit::AsyncDeviceVector<double> energyOuts;
};

//! Set all DeviceVector streams for the batched molecular device buffers.
void setStreams(BatchedMolecularDeviceBuffers& devBuffers, cudaStream_t stream);
//! Set all DeviceVector streams for the batched 3D molecular device buffers.
void setStreams(BatchedMolecular3DDeviceBuffers& devBuffers, cudaStream_t stream);

//! Add a molecule to the context.
void addMoleculeToContext(int                  dimension,
                          int                  numAtoms,
                          int&                 nTotalSystems,
                          std::vector<int>&    ctxAtomStarts,
                          std::vector<double>& ctxPositions);

//! Add a molecule to the context with positions.
void addMoleculeToContextWithPositions(const std::vector<double>& positions,
                                       int                        dimension,
                                       std::vector<int>&          ctxAtomStarts,
                                       std::vector<double>&       ctxPositions);

//! Add a molecule to the molecular system.
void addMoleculeToMolecularSystem(const EnergyForceContribsHost& contribs,
                                  const int                      numAtoms,
                                  const int                      dimension,
                                  const std::vector<int>&        ctxAtomStarts,
                                  BatchedMolecularSystemHost&    molSystem,
                                  std::vector<int>*              atomNumbers = nullptr);

//! Add a molecule to the molecular system.
void addMoleculeToMolecularSystem3D(const Energy3DForceContribsHost& contribs,
                                    const std::vector<int>&          ctxAtomStarts,
                                    BatchedMolecularSystem3DHost&    molSystem);

//! Add a molecule to the batched molecular system.
//! Populates the molSystem with the molecule's energy force contribs, and adds the current positions.
void addMoleculeToBatch(const EnergyForceContribsHost& contribs,
                        const std::vector<double>&     positions,
                        BatchedMolecularSystemHost&    molSystem,
                        const int                      dimension,
                        std::vector<int>&              ctxAtomStarts,
                        std::vector<double>&           ctxPositions,
                        std::vector<int>*              atomNumbers = nullptr);

//! Add a molecule to the batched molecular system.
//! Populates the molSystem with the molecule's energy force contribs, and adds the current positions.
void addMoleculeToBatch3D(const Energy3DForceContribsHost& contribs,
                          const std::vector<double>&       positions,
                          BatchedMolecularSystem3DHost&    molSystem,
                          std::vector<int>&                ctxAtomStarts,
                          std::vector<double>&             ctxPositions);

//! Send the batched molecular system to the device.
void sendContribsAndIndicesToDevice(const BatchedMolecularSystemHost& molSystemHost,
                                    BatchedMolecularDeviceBuffers&    molSystemDevice);

//! Send the batched molecular system to the device.
void sendContribsAndIndicesToDevice3D(const BatchedMolecularSystem3DHost& molSystemHost,
                                      BatchedMolecular3DDeviceBuffers&    molSystemDevice);

//! Send the context to the device.
void sendContextToDevice(const std::vector<double>&           ctxPositionsHost,
                         nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                         const std::vector<int>&              ctxAtomStartsHost,
                         nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice);

//! Setup the device buffers for the batched molecular system.
void setupDeviceBuffers(BatchedMolecularSystemHost&    molSystemHost,
                        BatchedMolecularDeviceBuffers& molSystemDevice,
                        const std::vector<double>&     ctxPositionsHost,
                        const int                      numMols);

//! Setup the device buffers for the batched molecular system.
void setupDeviceBuffers3D(BatchedMolecularSystem3DHost&    molSystemHost,
                          BatchedMolecular3DDeviceBuffers& molSystemDevice,
                          const std::vector<double>&       ctxPositionsHost,
                          const int                        numMols);

//! Allocate intermediate buffers on the device for the batched molecular system.
//! These include the gradients, energy buffer, and energy outs.
void allocateIntermediateBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                 BatchedMolecularDeviceBuffers&    molSystemDevice);

//! Allocate intermediate buffers on the device for the batched molecular system.
//! These include the gradients, energy buffer, and energy outs.
void allocateIntermediateBuffers3D(const BatchedMolecularSystem3DHost& molSystemHost,
                                   BatchedMolecular3DDeviceBuffers&    molSystemDevice);

//! Allocate the buffers for the 4D padded interface.
void allocateDim4ConversionBuffers(const BatchedMolecularSystemHost& molSystemHost,
                                   BatchedMolecularDeviceBuffers&    molSystemDevice);

//! Compute the energy of the batched molecular system. This will populate the energyOuts buffer on device.
//! energyOuts and energyBuffer must be zeroed before calling this function.
cudaError_t computeEnergy(BatchedMolecularDeviceBuffers&             molSystemDevice,
                          const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                          const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                          const uint8_t*                             activeThisStage = nullptr,
                          const double*                              positions       = nullptr,
                          cudaStream_t                               stream          = nullptr);

//! Compute the energy of the batched molecular system. This will populate the energyOuts buffer on device.
//! energyOuts and energyBuffer must be zeroed before calling this function.
cudaError_t computeEnergyETK(BatchedMolecular3DDeviceBuffers&           molSystemDevice,
                             const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                             const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                             const uint8_t*                             activeThisStage = nullptr,
                             const double*                              positions       = nullptr,
                             ETKTerm                                    term            = ETKTerm::ALL,
                             cudaStream_t                               stream          = nullptr);

//! Compute the gradients of the batched molecular system. This will populate the grad buffer on device.
//! grad must be zeroed before calling this function.
cudaError_t computeGradients(BatchedMolecularDeviceBuffers&             molSystemDevice,
                             const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                             const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                             const uint8_t*                             activeThisStage = nullptr,
                             cudaStream_t                               stream          = nullptr);

//! Compute the gradients of the batched molecular system. This will populate the grad buffer on device.
//! grad must be zeroed before calling this function.
cudaError_t computeGradientsETK(BatchedMolecular3DDeviceBuffers&           molSystemDevice,
                                const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                                const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                                const uint8_t*                             activeThisStage = nullptr,
                                ETKTerm                                    term            = ETKTerm::ALL,
                                cudaStream_t                               stream          = nullptr);

cudaError_t computePlanarEnergy(BatchedMolecular3DDeviceBuffers&           molSystemDevice,
                                const nvMolKit::AsyncDeviceVector<int>&    ctxAtomStartsDevice,
                                const nvMolKit::AsyncDeviceVector<double>& ctxPositionsDevice,
                                const uint8_t*                             activeThisStage,
                                const double*                              positions = nullptr,
                                cudaStream_t                               stream    = nullptr);
}  // namespace DistGeom
}  // namespace nvMolKit

#endif  // NVMOLKIT_DISTGEOM_H
