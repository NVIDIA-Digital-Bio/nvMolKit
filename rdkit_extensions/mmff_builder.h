#ifndef NVMOLKIT_MMFF_BUILDER_H
#define NVMOLKIT_MMFF_BUILDER_H

#include <ForceField/ForceField.h>
#include <GraphMol/ROMol.h>
#include <RDGeneral/export.h>

namespace nvMolKit {
namespace MMFF {
RDKIT_FORCEFIELDHELPERS_EXPORT ForceFields::ForceField* constructForceField(RDKit::ROMol& mol,
                                                                            double        nonBondedThresh    = 100.0,
                                                                            int           confId             = -1,
                                                                            bool ignoreInterfragInteractions = true);
}  // namespace MMFF
}  // namespace nvMolKit
#endif
