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

#ifndef NVMOLKIT_MOL_DATA_H
#define NVMOLKIT_MOL_DATA_H
#include <GraphMol/ROMol.h>
#include <stdint.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace nvMolKit::testing {

//! Load the lines of a file into a vector of strings.
//! \param filePath The path to the file.
//! \param maxLines The maximum number of lines to read.
std::vector<std::string> loadLines(const std::string& filePath, size_t maxLines);

//! Load n molecules from a list of smiles. If the number of atoms or bonds in a molecule exceeds the cutoff, it is not
//! loaded. If n > smis.size(), smiles are duplicated such than n entries are loaded with repeats as necessary. Returns
//! a pair of vectors of unique pointers to the molecules and the corresponding smiles.
std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>> loadNMols(
  const std::vector<std::string>& smiles,
  size_t                          n,
  size_t                          atomBondSizeCutoff = std::numeric_limits<size_t>::max());

//! Utility to get raw pointers from a unique pointer vec.
std::vector<const RDKit::ROMol*> makeMolsView(const std::vector<std::unique_ptr<RDKit::ROMol>>& mols);

//! Queries NVMOLKIT_CHEMBL29_SMILES_PATH environment variable for the path to the chembl29 smiles file
std::string getCheml29SmilesPath();

std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>> loadNChemblMolecules(
  size_t                n,
  std::optional<size_t> atomBondSizeCutoff = std::nullopt);

}  // namespace nvMolKit::testing

#endif  // NVMOLKIT_MOL_DATA_H
