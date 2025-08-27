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

#ifndef NVMOLKIT_COORD_GEN_H
#define NVMOLKIT_COORD_GEN_H

#include <memory>
#include <vector>

namespace RDKit {
class ROMol;
namespace DGeomHelpers {
class EmbedParameters;
}  // namespace DGeomHelpers
}  // namespace RDKit

namespace ForceFields {
namespace CrystalFF {
class CrystalFFDetails;
}  // namespace CrystalFF
}  // namespace ForceFields

namespace nvMolKit {

namespace detail {

class InitialCoordinateGenerator {
 public:
  InitialCoordinateGenerator();
  ~InitialCoordinateGenerator();
  //! One-time setup of bounds matrices
  void computeBoundsMatrices(const std::vector<const RDKit::ROMol*>&                mols,
                             const RDKit::DGeomHelpers::EmbedParameters&            params,
                             std::vector<ForceFields::CrystalFF::CrystalFFDetails>& etkdgDetails);

  //! Check if the bounds matrices are set up for n molecules.
  int numSystemsPrepared();

  //! Compute initial coordinates. If active is not null, only the active systems will compute coordinates.
  void computeInitialCoordinates(double* deviceCoords, const int* deviceAtomStarts, const uint8_t* active = nullptr);
  //! Returns successful minimizations from the last call to computeInitialCoordinates. Inactive systems will show
  //! false.

  const uint8_t* getPassFail() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace detail

}  // namespace nvMolKit

#endif  // NVMOLKIT_COORD_GEN_H
