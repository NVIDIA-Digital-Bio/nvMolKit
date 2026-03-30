#ifndef NVMOLKIT_FORCEFIELD_CONSTRAINTS_H
#define NVMOLKIT_FORCEFIELD_CONSTRAINTS_H

#include <string>
#include <vector>

#include "mmff.h"

namespace nvMolKit::ForceFieldConstraints {

struct DistanceConstraintSpec {
  int    idx1          = -1;
  int    idx2          = -1;
  bool   relative      = false;
  double minLen        = 0.0;
  double maxLen        = 0.0;
  double forceConstant = 0.0;
};

struct PositionConstraintSpec {
  int    idx           = -1;
  double maxDispl      = 0.0;
  double forceConstant = 0.0;
};

struct AngleConstraintSpec {
  int    idx1          = -1;
  int    idx2          = -1;
  int    idx3          = -1;
  bool   relative      = false;
  double minAngleDeg   = 0.0;
  double maxAngleDeg   = 0.0;
  double forceConstant = 0.0;
};

struct TorsionConstraintSpec {
  int    idx1           = -1;
  int    idx2           = -1;
  int    idx3           = -1;
  int    idx4           = -1;
  bool   relative       = false;
  double minDihedralDeg = 0.0;
  double maxDihedralDeg = 0.0;
  double forceConstant  = 0.0;
};

void   validateAtomIndex(int idx, int numAtoms, const std::string& what);
double distanceFromPositions(const std::vector<double>& positions, int idx1, int idx2);
double computeAngleDeg(const std::vector<double>& positions, int idx1, int idx2, int idx3);
double computeDihedralDeg(const std::vector<double>& positions, int idx1, int idx2, int idx3, int idx4);
double normalizeAngleDeg(double angleDeg);

void appendDistanceConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&               positions,
                              const DistanceConstraintSpec&            spec);
void appendPositionConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                              const std::vector<double>&               positions,
                              const PositionConstraintSpec&            spec);
void appendAngleConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                           const std::vector<double>&               positions,
                           const AngleConstraintSpec&               spec);
void appendTorsionConstraint(nvMolKit::MMFF::EnergyForceContribsHost& contribs,
                             const std::vector<double>&               positions,
                             const TorsionConstraintSpec&             spec);

}  // namespace nvMolKit::ForceFieldConstraints

#endif  // NVMOLKIT_FORCEFIELD_CONSTRAINTS_H
