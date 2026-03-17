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

#include <boost/python.hpp>
#include <boost/python/manage_new_object.hpp>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/Conformer.h>

#include "array_helpers.h"
#include "conformer_rmsd.h"
#include "device.h"
#include "utils/host_vector.h"

namespace {

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

}  // namespace

BOOST_PYTHON_MODULE(_conformerRmsd) {
  boost::python::def(
    "GetConformerRMSMatrixBatch",
    +[](boost::python::list& mols,
        const bool           prealigned,
        std::uintptr_t       streamPtr) -> boost::python::object {
      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      auto stream = *streamOpt;

      const int numMols = boost::python::len(mols);
      if (numMols == 0) {
        return boost::python::list();
      }

      // Extract and validate molecules.
      std::vector<const RDKit::ROMol*> molsVec(numMols);
      for (int i = 0; i < numMols; ++i) {
        molsVec[i] = boost::python::extract<const RDKit::ROMol*>(
            boost::python::object(mols[i]));
        if (molsVec[i] == nullptr) {
          throw std::invalid_argument("Invalid molecule at index " + std::to_string(i));
        }
        if (molsVec[i]->getNumAtoms() == 0 && molsVec[i]->getNumConformers() >= 2) {
          throw std::invalid_argument("Molecule at index " + std::to_string(i) +
                                      " has no atoms");
        }
      }

      // Build per-molecule metadata on the host.
      nvMolKit::PinnedHostVector<int> numConfsArr(numMols);
      nvMolKit::PinnedHostVector<int> numAtomsArr(numMols);
      nvMolKit::PinnedHostVector<int> pairOffsetsArr(numMols + 1);
      nvMolKit::PinnedHostVector<int> coordOffsetsArr(numMols);

      pairOffsetsArr[0] = 0;
      int totalCoords   = 0;
      for (int m = 0; m < numMols; ++m) {
        const int nc    = molsVec[m]->getNumConformers();
        const int na    = molsVec[m]->getNumAtoms();
        numConfsArr[m]  = nc;
        numAtomsArr[m]  = na;
        coordOffsetsArr[m] = totalCoords;
        totalCoords        += nc * na * 3;
        const int64_t numPairs64 = (nc >= 2) ? static_cast<int64_t>(nc) * (nc - 1) / 2 : 0;
        if (numPairs64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
          throw std::overflow_error("Molecule at index " + std::to_string(m) +
                                    " has too many conformer pairs for a single kernel launch");
        }
        pairOffsetsArr[m + 1] = pairOffsetsArr[m] + static_cast<int>(numPairs64);
      }
      const int totalPairs = pairOffsetsArr[numMols];

      // Pack all conformer coordinates into a single flat pinned buffer.
      nvMolKit::PinnedHostVector<double> hostCoords(totalCoords > 0 ? totalCoords : 1);
      for (int m = 0; m < numMols; ++m) {
        const RDKit::ROMol& mol = *molsVec[m];
        const int na            = numAtomsArr[m];
        int confIdx             = 0;
        for (auto it = mol.beginConformers(); it != mol.endConformers(); ++it, ++confIdx) {
          const RDKit::Conformer& conf = **it;
          for (int a = 0; a < na; ++a) {
            const auto& pos = conf.getAtomPos(a);
            const int base  = coordOffsetsArr[m] + confIdx * na * 3 + a * 3;
            hostCoords[base + 0] = pos.x;
            hostCoords[base + 1] = pos.y;
            hostCoords[base + 2] = pos.z;
          }
        }
      }

      // Transfer coordinates and metadata to device.
      nvMolKit::AsyncDeviceVector<double> devCoords(totalCoords > 0 ? totalCoords : 1, stream);
      nvMolKit::AsyncDeviceVector<int>    devNumConfs(numMols, stream);
      nvMolKit::AsyncDeviceVector<int>    devNumAtoms(numMols, stream);
      nvMolKit::AsyncDeviceVector<int>    devPairOffsets(numMols + 1, stream);
      nvMolKit::AsyncDeviceVector<int>    devCoordOffsets(numMols, stream);

      if (totalCoords > 0) hostCoords.copyToDevice(devCoords, stream);
      numConfsArr.copyToDevice(devNumConfs, stream);
      numAtomsArr.copyToDevice(devNumAtoms, stream);
      pairOffsetsArr.copyToDevice(devPairOffsets, stream);
      coordOffsetsArr.copyToDevice(devCoordOffsets, stream);

      // Allocate per-molecule output buffers; collect their raw device pointers.
      std::vector<nvMolKit::AsyncDeviceVector<double>> devRmsdVecs;
      devRmsdVecs.reserve(numMols);
      nvMolKit::PinnedHostVector<double*> hostRmsdPtrs(numMols);
      for (int m = 0; m < numMols; ++m) {
        const int numPairs = pairOffsetsArr[m + 1] - pairOffsetsArr[m];
        devRmsdVecs.emplace_back(numPairs > 0 ? numPairs : 0, stream);
        hostRmsdPtrs[m] = (numPairs > 0) ? devRmsdVecs.back().data() : nullptr;
      }

      nvMolKit::AsyncDeviceVector<double*> devRmsdPtrs(numMols, stream);
      hostRmsdPtrs.copyToDevice(devRmsdPtrs, stream);

      // Launch a single kernel covering all pairs from all molecules.
      if (totalPairs > 0) {
        nvMolKit::conformerRmsdBatchMatrixGpu(
            toSpan(devCoords),
            toSpan(devRmsdPtrs),
            toSpan(devPairOffsets),
            toSpan(devCoordOffsets),
            toSpan(devNumConfs),
            toSpan(devNumAtoms),
            numMols,
            totalPairs,
            prealigned,
            stream);
      }

      // Return a Python list of per-molecule PyArray objects.
      boost::python::list results;
      for (int m = 0; m < numMols; ++m) {
        const int numPairs = pairOffsetsArr[m + 1] - pairOffsetsArr[m];
        results.append(toOwnedPyArray(
            nvMolKit::makePyArray(devRmsdVecs[m], boost::python::make_tuple(numPairs))));
      }
      return results;
    },
    (boost::python::arg("mols"),
     boost::python::arg("prealigned") = false,
     boost::python::arg("stream")     = 0));

  boost::python::def(
    "GetConformerRMSMatrix",
    +[](RDKit::ROMol& mol,
        const bool     prealigned,
        std::uintptr_t streamPtr) -> boost::python::object {
      const int numConfs = mol.getNumConformers();
      if (numConfs <= 1) {
        nvMolKit::AsyncDeviceVector<double> empty(0);
        return toOwnedPyArray(nvMolKit::makePyArray(empty, boost::python::make_tuple(0)));
      }

      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      auto stream = *streamOpt;

      const int numAtoms = mol.getNumAtoms();
      if (numAtoms == 0) {
        // Intentional divergence from RDKit, which returns [nan] for exactly 2
        // zero-atom conformers and raises ZeroDivisionError for 3+. We fail fast
        // with a consistent error for all degenerate zero-atom inputs.
        throw std::invalid_argument("Molecule has no atoms");
      }

      // Extract coordinates from all conformers into a flat pinned host buffer.
      // Layout: coords[conf * numAtoms * 3 + atom * 3 + xyz]
      // Pinned memory allows the DMA engine to transfer directly to the device
      // without a staging copy, and the PinnedHostVector destructor handles
      // cleanup safely after all stream work has been submitted.
      const size_t numCoords = static_cast<size_t>(numConfs) * numAtoms * 3;
      nvMolKit::PinnedHostVector<double> hostCoords(numCoords);
      int confIdx = 0;
      for (auto it = mol.beginConformers(); it != mol.endConformers(); ++it, ++confIdx) {
        const RDKit::Conformer& conf = **it;
        for (int a = 0; a < numAtoms; ++a) {
          const auto& pos                                  = conf.getAtomPos(a);
          hostCoords[confIdx * numAtoms * 3 + a * 3 + 0] = pos.x;
          hostCoords[confIdx * numAtoms * 3 + a * 3 + 1] = pos.y;
          hostCoords[confIdx * numAtoms * 3 + a * 3 + 2] = pos.z;
        }
      }

      nvMolKit::AsyncDeviceVector<double> deviceCoords(numCoords, stream);
      hostCoords.copyToDevice(deviceCoords, stream);

      // Allocate output
      const int64_t numPairs = static_cast<int64_t>(numConfs) * (numConfs - 1) / 2;
      nvMolKit::AsyncDeviceVector<double> deviceRmsd(numPairs, stream);

      // Launch kernel
      nvMolKit::conformerRmsdMatrixGpu(
          toSpan(deviceCoords), toSpan(deviceRmsd), numConfs, numAtoms, prealigned, stream);

      return toOwnedPyArray(nvMolKit::makePyArray(deviceRmsd, boost::python::make_tuple(numPairs)));
    },
    (boost::python::arg("mol"),
     boost::python::arg("prealigned") = false,
     boost::python::arg("stream")     = 0));
}
