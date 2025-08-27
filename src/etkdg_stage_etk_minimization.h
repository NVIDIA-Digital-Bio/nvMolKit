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

#ifndef NVMOLKIT_ETKDG_STAGE_ETK_MINIMIZATION_H
#define NVMOLKIT_ETKDG_STAGE_ETK_MINIMIZATION_H

#include <GraphMol/DistGeomHelpers/Embedder.h>

#include "dist_geom.h"
#include "etkdg_impl.h"

using ::nvMolKit::detail::EmbedArgs;
using ::nvMolKit::detail::ETKDGContext;
using ::nvMolKit::detail::ETKDGStage;

namespace nvMolKit {
namespace detail {

class ETKMinimizationStage final : public ETKDGStage {
 public:
  ETKMinimizationStage(const std::vector<const RDKit::ROMol*>&     mols,
                       const std::vector<EmbedArgs>&               eargs,
                       const RDKit::DGeomHelpers::EmbedParameters& embedParam,
                       const ETKDGContext&                         ctx,
                       cudaStream_t                                stream = nullptr);

  void        execute(ETKDGContext& ctx) override;
  std::string name() const override { return "ETK 3D Minimization"; }

 private:
  //! Re-sets the bounds for distance constraints based on the current positions.
  void setReferenceValues(const ETKDGContext& ctx);

  nvMolKit::DistGeom::BatchedMolecular3DDeviceBuffers molSystemDevice;
  nvMolKit::DistGeom::BatchedMolecularSystem3DHost    molSystemHost;
  const RDKit::DGeomHelpers::EmbedParameters&         embedParam_;
  cudaStream_t                                        stream_;
};

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_ETKDG_STAGE_ETK_MINIMIZATION_H
