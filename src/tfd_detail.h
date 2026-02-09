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

#ifndef NVMOLKIT_TFD_DETAIL_H
#define NVMOLKIT_TFD_DETAIL_H

#include <cmath>

#ifdef __CUDACC__
#define TFD_INLINE __host__ __device__ __forceinline__
#else
#define TFD_INLINE inline
#endif

namespace nvMolKit {
namespace detail {

constexpr float kPi       = 3.14159265358979323846f;
constexpr float kRadToDeg = 180.0f / kPi;

//! Cross product of two 3D vectors: result = a x b
TFD_INLINE void cross3(const float* a, const float* b, float* result) {
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
}

//! Dot product of two 3D vectors
TFD_INLINE float dot3(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

//! Vector subtraction: result = a - b
TFD_INLINE void sub3(const float* a, const float* b, float* result) {
  result[0] = a[0] - b[0];
  result[1] = a[1] - b[1];
  result[2] = a[2] - b[2];
}

//! Vector length
TFD_INLINE float length3(const float* v) {
  return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

//! Compute dihedral angle in degrees from four atom positions
//! @param p1, p2, p3, p4 The four atom positions (p2-p3 is the central bond)
//! @return Dihedral angle in degrees [0, 360)
TFD_INLINE float computeDihedralAngle(const float* p1, const float* p2, const float* p3, const float* p4) {
  // Vector from p2 to p3 (central bond)
  float b1[3];
  sub3(p3, p2, b1);

  // Vector from p2 to p1
  float v1[3];
  sub3(p1, p2, v1);

  // Vector from p3 to p4
  float v2[3];
  sub3(p4, p3, v2);

  // Normal to plane 1 (p1-p2-p3)
  float n1[3];
  cross3(v1, b1, n1);

  // Normal to plane 2 (p2-p3-p4)
  float n2[3];
  cross3(b1, v2, n2);

  // Lengths of normals
  float n1Len = length3(n1);
  float n2Len = length3(n2);

  if (n1Len < 1e-10f || n2Len < 1e-10f) {
    return 0.0f;  // Degenerate case
  }

  // Compute angle between normals
  float cosAngle = dot3(n1, n2) / (n1Len * n2Len);
  cosAngle       = fmaxf(-1.0f, fminf(1.0f, cosAngle));  // Clamp for numerical stability
  float angle    = acosf(cosAngle) * kRadToDeg;

  // Determine sign using cross product of normals dotted with central bond
  float crossN[3];
  cross3(n1, n2, crossN);
  if (dot3(crossN, b1) < 0) {
    angle = -angle;
  }

  // Normalize to [0, 360)
  if (angle < 0) {
    angle += 360.0f;
  }

  return angle;
}

//! Compute circular difference between two angles (both in degrees)
//! @param angle1, angle2 Angles in degrees [0, 360)
//! @return Minimum difference in degrees [0, 180]
TFD_INLINE float circularDifference(float angle1, float angle2) {
  float diff = fabsf(angle1 - angle2);
  if (360.0f - diff < diff) {
    diff = 360.0f - diff;
  }
  return diff;
}

}  // namespace detail
}  // namespace nvMolKit

#endif  // NVMOLKIT_TFD_DETAIL_H
