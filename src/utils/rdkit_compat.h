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

#ifndef NVMOLKIT_RDKIT_COMPAT_H
#define NVMOLKIT_RDKIT_COMPAT_H

#include <GraphMol/Atom.h>

#include <concepts>

namespace nvMolKit {
namespace compat {

// RDKit 2025.09+ deprecates getExplicitValence()/getImplicitValence()
// in favor of getValence(ValenceType). Detect at compile time via concept.
template <typename T>
concept HasValenceType = requires { typename T::ValenceType; };

template <typename T = RDKit::Atom> inline int getExplicitValence(const T* atom) {
  if constexpr (HasValenceType<T>) {
    return atom->getValence(T::ValenceType::EXPLICIT);
  } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return atom->getExplicitValence();
#pragma GCC diagnostic pop
  }
}

template <typename T = RDKit::Atom> inline int getImplicitValence(const T* atom) {
  if constexpr (HasValenceType<T>) {
    return atom->getValence(T::ValenceType::IMPLICIT);
  } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return atom->getImplicitValence();
#pragma GCC diagnostic pop
  }
}

}  // namespace compat
}  // namespace nvMolKit

#endif  // NVMOLKIT_RDKIT_COMPAT_H
