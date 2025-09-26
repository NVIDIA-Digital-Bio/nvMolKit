#ifndef NVMOLKIT_BOUNDS_MATRIX_H
#define NVMOLKIT_BOUNDS_MATRIX_H

#include <GraphMol/DistGeomHelpers/BoundsMatrixBuilder.h>
#include <Numerics/SymmMatrix.h>

namespace ForceFields::CrystalFF {
struct CrystalFFDetails;
}

namespace RDKit::DGeomHelpers {
struct EmbedParameters;
void initETKDG(ROMol* mol, const EmbedParameters& params, ForceFields::CrystalFF::CrystalFFDetails& etkdgDetails);

//! Normalize distance matrices as is done in the RDKit ETKDG code.
RDNumeric::SymmMatrix<double> initialCoordsNormDistances(const RDNumeric::SymmMatrix<double>& initialDistMat);

}  // namespace RDKit::DGeomHelpers

namespace nvMolKit {

std::vector<::DistGeom::BoundsMatPtr> getBoundsMatrices(
  const std::vector<const RDKit::ROMol*>&                mols,
  const RDKit::DGeomHelpers::EmbedParameters&            params,
  std::vector<ForceFields::CrystalFF::CrystalFFDetails>& etkdgDetails);

}  // namespace nvMolKit

#endif  // NVMOLKIT_BOUNDS_MATRIX_H