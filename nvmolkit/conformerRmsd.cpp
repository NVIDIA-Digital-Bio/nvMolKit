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
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/Conformer.h>

#include "array_helpers.h"
#include "conformer_rmsd.h"
#include "cuda_error_check.h"
#include "device.h"

namespace {

boost::python::object toOwnedPyArray(nvMolKit::PyArray* array) {
  using Converter = boost::python::manage_new_object::apply<nvMolKit::PyArray*>::type;
  return boost::python::object(boost::python::handle<>(Converter()(array)));
}

}  // namespace

BOOST_PYTHON_MODULE(_conformerRmsd) {
  boost::python::def(
    "GetConformerRMSMatrix",
    +[](RDKit::ROMol& mol,
        const bool     prealigned,
        std::uintptr_t streamPtr) -> boost::python::object {
      const int numConfs = mol.getNumConformers();
      if (numConfs <= 1) {
        // Return empty result — matches RDKit behaviour
        nvMolKit::AsyncDeviceVector<double> empty(0);
        return toOwnedPyArray(nvMolKit::makePyArray(empty, boost::python::make_tuple(0)));
      }

      auto streamOpt = nvMolKit::acquireExternalStream(streamPtr);
      if (!streamOpt) {
        throw std::invalid_argument("Invalid CUDA stream");
      }
      auto stream = *streamOpt;

      const int numAtoms = mol.getNumAtoms();

      // Extract coordinates from all conformers into a flat host buffer.
      // Layout: coords[conf * numAtoms * 3 + atom * 3 + xyz]
      std::vector<double> hostCoords(numConfs * numAtoms * 3);
      int confIdx = 0;
      for (auto it = mol.beginConformers(); it != mol.endConformers(); ++it, ++confIdx) {
        const RDKit::Conformer& conf = **it;
        for (int a = 0; a < numAtoms; ++a) {
          const auto& pos                                   = conf.getAtomPos(a);
          hostCoords[confIdx * numAtoms * 3 + a * 3 + 0] = pos.x;
          hostCoords[confIdx * numAtoms * 3 + a * 3 + 1] = pos.y;
          hostCoords[confIdx * numAtoms * 3 + a * 3 + 2] = pos.z;
        }
      }

      // Transfer to GPU and synchronize before hostCoords goes out of scope,
      // since copyFromHost uses cudaMemcpyAsync internally.
      nvMolKit::AsyncDeviceVector<double> deviceCoords(hostCoords.size(), stream);
      deviceCoords.copyFromHost(hostCoords);
      cudaCheckError(cudaStreamSynchronize(stream));

      // Allocate output
      const int numPairs = numConfs * (numConfs - 1) / 2;
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
