#ifndef NVMOLKIT_MMFF_FLATTENED_BUILDER_H
#define NVMOLKIT_MMFF_FLATTENED_BUILDER_H

#include <mutex>

#include "mmff.h"

namespace RDKit {
class ROMol;
namespace MMFF {
class MMFFMolProperties;

}  // namespace MMFF
}  // namespace RDKit

namespace nvMolKit::MMFF {

//! Construct flattened MMFF forcefield contribs for a molecule.
//! Uses RDKit parametrization
/*!

  \param mol       the molecule to use
  \param nonBondedThresh  the threshold to be used in adding non-bonded terms
                          to the force field. Any non-bonded contact whose current
                          distance is greater than \c nonBondedThresh * the minimum
                          value for that contact will not be included.
  \param confId    the optional conformer id, if this isn't provided, the
                   molecule's default confId will be used.
  \param ignoreInterfragInteractions if true, nonbonded terms will not be added between fragments

  \return the flattened force field
*/
EnergyForceContribsHost constructForcefieldContribs(RDKit::ROMol& mol,
                                                    double        nonBondedThresh             = 100.0,
                                                    int           confId                      = -1,
                                                    bool          ignoreInterfragInteractions = true);
//! \overload
EnergyForceContribsHost constructForcefieldContribs(const RDKit::ROMol&             mol,
                                                    RDKit::MMFF::MMFFMolProperties* mmffMolProperties,
                                                    double                          nonBondedThresh             = 100.0,
                                                    int                             confId                      = -1,
                                                    bool                            ignoreInterfragInteractions = true);

}  // namespace nvMolKit::MMFF

#endif  // NVMOLKIT_MMFF_FLATTENED_BUILDER_H
