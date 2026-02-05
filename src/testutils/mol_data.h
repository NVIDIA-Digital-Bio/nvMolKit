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

/**
 * @brief Extract the first whitespace-delimited token from a line.
 *
 * Skips leading whitespace and ignores comment-only lines that start with '#'.
 *
 * @param line Input line
 * @return First token, or empty string if none
 */
std::string extractFirstToken(const std::string& line);

//! Queries NVMOLKIT_CHEMBL29_SMILES_PATH environment variable for the path to the chembl29 smiles file
std::string getCheml29SmilesPath();

std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>> loadNChemblMolecules(
  size_t                n,
  std::optional<size_t> atomBondSizeCutoff = std::nullopt);

/**
 * @brief Read SMILES from a file, filtering by atom count.
 * @param filePath Path to SMILES file (one SMILES per line, may have trailing data after whitespace)
 * @param maxCount Maximum number of molecules to read
 * @param maxAtoms Maximum number of atoms per molecule
 * @return Vector of parsed molecules
 */
std::vector<std::unique_ptr<RDKit::ROMol>> readSmilesFile(const std::string& filePath,
                                                          size_t             maxCount,
                                                          size_t             maxAtoms);

/**
 * @brief Read SMILES from a file, filtering by atom count, returning both molecules and strings.
 * @param filePath Path to SMILES file (one SMILES per line, may have trailing data after whitespace)
 * @param maxCount Maximum number of molecules to read
 * @param maxAtoms Maximum number of atoms per molecule
 * @return Pair of (molecules, SMILES strings)
 */
std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>>
readSmilesFileWithStrings(const std::string& filePath, size_t maxCount, size_t maxAtoms);

/**
 * @brief Read SMARTS patterns from a file.
 * @param filePath Path to SMARTS file (one SMARTS per line, may have trailing data after whitespace)
 * @return Vector of parsed query molecules
 */
std::vector<std::unique_ptr<RDKit::ROMol>> readSmartsFile(const std::string& filePath);

/**
 * @brief Read SMARTS patterns from a file, returning both molecules and strings.
 * @param filePath Path to SMARTS file (one SMARTS per line, may have trailing data after whitespace)
 * @return Pair of (query molecules, SMARTS strings)
 */
std::pair<std::vector<std::unique_ptr<RDKit::ROMol>>, std::vector<std::string>> readSmartsFileWithStrings(
  const std::string& filePath);

}  // namespace nvMolKit::testing

#endif  // NVMOLKIT_MOL_DATA_H
