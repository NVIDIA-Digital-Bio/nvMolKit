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

#ifndef NVMOLKIT_MMFF_KERNELS_DEVICE_CUH
#define NVMOLKIT_MMFF_KERNELS_DEVICE_CUH

#include "kernel_utils.cuh"
#include "mmff_kernels.h"

// Constants used by MMFF energy/gradient functions
namespace {
constexpr double degreeToRadian = M_PI / 180.0;
constexpr double radianToDegree = 180.0 / M_PI;
}  // namespace

using namespace nvMolKit::FFKernelUtils;

// rdkit_ports namespace functions (header-only implementations)
namespace rdkit_ports {

static __device__ __forceinline__ void oopGrad(const double* pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const int     idx3,
                                               const int     idx4,
                                               const double  koop,
                                               double*       grad) {
  constexpr double prefactor = 143.9325 * degreeToRadian;

  float dJIx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  float dJIy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  float dJIz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  float dJKx = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  float dJKy = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  float dJKz = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  float dJLx = pos[3 * idx4 + 0] - pos[3 * idx2 + 0];
  float dJLy = pos[3 * idx4 + 1] - pos[3 * idx2 + 1];
  float dJLz = pos[3 * idx4 + 2] - pos[3 * idx2 + 2];

  const float invdJI = rsqrtf(dJIx * dJIx + dJIy * dJIy + dJIz * dJIz);
  const float invdJK = rsqrtf(dJKx * dJKx + dJKy * dJKy + dJKz * dJKz);
  const float invdJL = rsqrtf(dJLx * dJLx + dJLy * dJLy + dJLz * dJLz);

  dJIx *= invdJI;
  dJIy *= invdJI;
  dJIz *= invdJI;
  dJKx *= invdJK;
  dJKy *= invdJK;
  dJKz *= invdJK;
  dJLx *= invdJL;
  dJLy *= invdJL;
  dJLz *= invdJL;

  float normalJIKx, normalJIKy, normalJIKz;
  crossProduct(-dJIx, -dJIy, -dJIz, dJKx, dJKy, dJKz, normalJIKx, normalJIKy, normalJIKz);
  const float invNormLength = rsqrtf(normalJIKx * normalJIKx + normalJIKy * normalJIKy + normalJIKz * normalJIKz);
  normalJIKx *= invNormLength;
  normalJIKy *= invNormLength;
  normalJIKz *= invNormLength;

  const float sinChi    = clamp(dotProduct(dJLx, dJLy, dJLz, normalJIKx, normalJIKy, normalJIKz), -1.0f, 1.0f);
  const float cosChiSq  = 1.0 - sinChi * sinChi;
  const float invCosChi = cosChiSq > 0 ? rsqrtf(cosChiSq) : 1.0e8;
  const float chi       = radianToDegree * asin(sinChi);
  const float cosTheta  = clamp(dotProduct(dJIx, dJIy, dJIz, dJKx, dJKy, dJKz), -1.0f, 1.0f);

  float invSinTheta = rsqrtf(fmax(1.0 - cosTheta * cosTheta, 1.0e-8));

  float dE_dChi = prefactor * koop * chi;
  float t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
  crossProduct(dJLx, dJLy, dJLz, dJKx, dJKy, dJKz, t1x, t1y, t1z);
  crossProduct(dJIx, dJIy, dJIz, dJLx, dJLy, dJLz, t2x, t2y, t2z);
  crossProduct(dJKx, dJKy, dJKz, dJIx, dJIy, dJIz, t3x, t3y, t3z);

  float term1  = invCosChi * invSinTheta;
  float term2  = sinChi * invCosChi * (invSinTheta * invSinTheta);
  float tg1[3] = {(t1x * term1 - (dJIx - dJKx * cosTheta) * term2) * invdJI,
                  (t1y * term1 - (dJIy - dJKy * cosTheta) * term2) * invdJI,
                  (t1z * term1 - (dJIz - dJKz * cosTheta) * term2) * invdJI};
  float tg3[3] = {(t2x * term1 - (dJKx - dJIx * cosTheta) * term2) * invdJK,
                  (t2y * term1 - (dJKy - dJIy * cosTheta) * term2) * invdJK,
                  (t2z * term1 - (dJKz - dJIz * cosTheta) * term2) * invdJK};
  float tg4[3] = {(t3x * term1 - dJLx * sinChi * invCosChi) * invdJL,
                  (t3y * term1 - dJLy * sinChi * invCosChi) * invdJL,
                  (t3z * term1 - dJLz * sinChi * invCosChi) * invdJL};

  atomicAdd(&grad[3 * idx1 + 0], dE_dChi * tg1[0]);
  atomicAdd(&grad[3 * idx1 + 1], dE_dChi * tg1[1]);
  atomicAdd(&grad[3 * idx1 + 2], dE_dChi * tg1[2]);
  atomicAdd(&grad[3 * idx2 + 0], -dE_dChi * (tg1[0] + tg3[0] + tg4[0]));
  atomicAdd(&grad[3 * idx2 + 1], -dE_dChi * (tg1[1] + tg3[1] + tg4[1]));
  atomicAdd(&grad[3 * idx2 + 2], -dE_dChi * (tg1[2] + tg3[2] + tg4[2]));
  atomicAdd(&grad[3 * idx3 + 0], dE_dChi * tg3[0]);
  atomicAdd(&grad[3 * idx3 + 1], dE_dChi * tg3[1]);
  atomicAdd(&grad[3 * idx3 + 2], dE_dChi * tg3[2]);
  atomicAdd(&grad[3 * idx4 + 0], dE_dChi * tg4[0]);
  atomicAdd(&grad[3 * idx4 + 1], dE_dChi * tg4[1]);
  atomicAdd(&grad[3 * idx4 + 2], dE_dChi * tg4[2]);
}

static __device__ __forceinline__ void torsionGrad(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const int     idx3,
                                                   const int     idx4,
                                                   const float  V1,
                                                   const float  V2,
                                                   const float  V3,
                                                   double*       grad) {
  // P1 - P2
  const float dx1 = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const float dy1 = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const float dz1 = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  // P3 - P2
  const float dx2 = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const float dy2 = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const float dz2 = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  // P4 - P3
  const float dx4 = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const float dy4 = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const float dz4 = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  double cross1x, cross1y, cross1z;
  crossProduct(dx1, dy1, dz1, dx2, dy2, dz2, cross1x, cross1y, cross1z);
  const double invNorm1 = fmin(rsqrt(cross1x * cross1x + cross1y * cross1y + cross1z * cross1z), 1.0e5);
  cross1x *= invNorm1;
  cross1y *= invNorm1;
  cross1z *= invNorm1;

  double cross2x, cross2y, cross2z;
  // Use -dx2, -dy2, -dz2 directly instead of storing dx3, dy3, dz3
  crossProduct(-dx2, -dy2, -dz2, dx4, dy4, dz4, cross2x, cross2y, cross2z);
  const double invNorm2 = fmin(rsqrt(cross2x * cross2x + cross2y * cross2y + cross2z * cross2z), 1.0e5);
  cross2x *= invNorm2;
  cross2y *= invNorm2;
  cross2z *= invNorm2;

  const double cosPhi = clamp(dotProduct(cross1x, cross1y, cross1z, cross2x, cross2y, cross2z), -1.0, 1.0);

  const float sinPhiSq = 1.0f - cosPhi * cosPhi;
  float sinTerm = 0.0;
  if (sinPhiSq > 0.0) {
    const float sin2Phi  = 2.0f  * cosPhi;
    const float sin3Phi  = 3.0f  - 4.0f  * sinPhiSq;
    sinTerm  = 0.5f * (V1 - 2.0f * V2 * sin2Phi + 3.0f * V3 * sin3Phi);
  }

  float dCos_dT0 = invNorm1 * (cross2x - cosPhi * cross1x);
  float dCos_dT1 = invNorm1 * (cross2y - cosPhi * cross1y);
  float dCos_dT2 = invNorm1 * (cross2z - cosPhi * cross1z);

  atomicAdd(&grad[3 * idx1 + 0], sinTerm * (dCos_dT2 * dy2 - dCos_dT1 * dz2));
  atomicAdd(&grad[3 * idx1 + 1], sinTerm * (dCos_dT0 * dz2 - dCos_dT2 * dx2));
  atomicAdd(&grad[3 * idx1 + 2], sinTerm * (dCos_dT1 * dx2 - dCos_dT0 * dy2));

  // idx3 and idx4 gradients - reuse variables dCos_dT0-2 for dCos_dT3-5
  const float dCos_dT3 = invNorm2 * (cross1x - cosPhi * cross2x);
  const float dCos_dT4 = invNorm2 * (cross1y - cosPhi * cross2y);
  const float dCos_dT5 = invNorm2 * (cross1z - cosPhi * cross2z);

  atomicAdd(&grad[3 * idx2 + 0],
            sinTerm * (dCos_dT1 * (dz2 - dz1) + dCos_dT2 * (dy1 - dy2) + dCos_dT4 * (-dz4) + dCos_dT5 * (dy4)));
  atomicAdd(&grad[3 * idx2 + 1],
            sinTerm * (dCos_dT0 * (dz1 - dz2) + dCos_dT2 * (dx2 - dx1) + dCos_dT3 * (dz4) + dCos_dT5 * (-dx4)));
  atomicAdd(&grad[3 * idx2 + 2],
            sinTerm * (dCos_dT0 * (dy2 - dy1) + dCos_dT1 * (dx1 - dx2) + dCos_dT3 * (-dy4) + dCos_dT4 * (dx4)));

  atomicAdd(&grad[3 * idx3 + 0],
            sinTerm * (dCos_dT1 * (dz1) + dCos_dT2 * (-dy1) + dCos_dT4 * (dz4 + dz2) + dCos_dT5 * (-dy4 - dy2)));
  atomicAdd(&grad[3 * idx3 + 1],
            sinTerm * (dCos_dT0 * (-dz1) + dCos_dT2 * (dx1) + dCos_dT3 * (-dz4 - dz2) + dCos_dT5 * (dx4 + dx2)));
  atomicAdd(&grad[3 * idx3 + 2],
            sinTerm * (dCos_dT0 * (dy1) + dCos_dT1 * (-dx1) + dCos_dT3 * (dy4 + dy2) + dCos_dT4 * (-dx4 - dx2)));

  atomicAdd(&grad[3 * idx4 + 0], sinTerm * (dCos_dT4 * (-dz2) - dCos_dT5 * (-dy2)));
  atomicAdd(&grad[3 * idx4 + 1], sinTerm * (dCos_dT5 * (-dx2) - dCos_dT3 * (-dz2)));
  atomicAdd(&grad[3 * idx4 + 2], sinTerm * (dCos_dT3 * (-dy2) - dCos_dT4 * (-dx2)));
}
static __device__ __forceinline__ void vDWGrad(const double* pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const double  R_ij_star,
                                               const double  wellDepth,
                                               double*       grad) {
  constexpr float vdw1   = 1.07;
  constexpr float vdw1m1 = vdw1 - 1.0;
  constexpr float vdw2   = 1.12;
  constexpr float vdw2m1 = vdw2 - 1.0;
  constexpr float vdw2t7 = vdw2 * 7.0;

  const float invDistance = rsqrtf(distanceSquared(pos, idx1, idx2));
  const float distance = 1.0f / invDistance;

  const float invRIJStar = 1.0f / R_ij_star;

  const float q         = distance * invRIJStar;
  const float q2        = q * q;
  const float q6        = q2 * q2 * q2;
  const float q7        = q6 * q;
  const float q7pvdw2m1 = q7 + vdw2m1;
  const float invQ7Term = 1.0f / q7pvdw2m1;
  const float t         = vdw1 / (q + vdw1 - 1.0);
  const float t2        = t * t;
  const float t7        = t2 * t2 * t2 * t;
  const float dE_dr     = wellDepth * invRIJStar * t7 *
                       (-vdw2t7 * q6 * invQ7Term * invQ7Term + ((-vdw2t7 * invQ7Term + 14.0) / (q + vdw1m1)));

  float term1x, term1y, term1z;
  if (distance <= 0.0) {
    term1x = R_ij_star * 0.01f;
    term1y = R_ij_star * 0.01f;
    term1z = R_ij_star * 0.01f;
  } else {
    term1x = dE_dr * (pos[3 * idx1 + 0] - pos[3 * idx2 + 0]) * invDistance;
    term1y = dE_dr * (pos[3 * idx1 + 1] - pos[3 * idx2 + 1]) * invDistance;
    term1z = dE_dr * (pos[3 * idx1 + 2] - pos[3 * idx2 + 2]) * invDistance;
  }

  atomicAdd(&grad[3 * idx1 + 0], term1x);
  atomicAdd(&grad[3 * idx1 + 1], term1y);
  atomicAdd(&grad[3 * idx1 + 2], term1z);

  atomicAdd(&grad[3 * idx2 + 0], -term1x);
  atomicAdd(&grad[3 * idx2 + 1], -term1y);
  atomicAdd(&grad[3 * idx2 + 2], -term1z);
}

}  // namespace rdkit_ports

// Device helper functions (header-only implementations)
static __device__ __forceinline__ double bondStretchEnergy(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const double  r0,
                                                           const double  kb) {
  constexpr double prefactor           = 143.9325 / 2.0;
  constexpr double csFactorDist        = -2.0;
  constexpr double csFactorDistSquared = 7.0 / 12.0 * csFactorDist * csFactorDist;

  const double distSquared = distanceSquared(pos, idx1, idx2);
  const float distance    = sqrtf(static_cast<float>(distSquared));

  const float deltaR  = distance - r0;
  const float deltaR2 = deltaR * deltaR;
  return prefactor * kb * deltaR2 * (1.0 + csFactorDist * deltaR + csFactorDistSquared * deltaR2);
}

static __device__ __forceinline__ void bondStretchGrad(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const double  r0,
                                                       const double  kb,
                                                       double*       grad) {
  constexpr double c1                          = 143.9325;
  constexpr double cs                          = -2.0;
  constexpr double csFactorTimesSecondConstant = cs * 1.5;
  constexpr double lastFactor                  = 2.0 * 7.0 / 12.0 * cs * cs;

  double       dx, dy, dz;
  const double distanceSquared = distanceSquaredWithComponents(pos, idx1, idx2, dx, dy, dz);
  const double invDist        = rsqrt(distanceSquared);
  const double distance = 1.0 / invDist;
  const double deltaR          = distance - r0;

  const double de_dr = c1 * kb * deltaR * (1.0 + csFactorTimesSecondConstant * deltaR + lastFactor * deltaR * deltaR);

  double       dE_dx, dE_dy, dE_dz;
  if (distance > 0.0) {
    dE_dx = de_dr * dx * invDist;
    dE_dy = de_dr * dy * invDist;
    dE_dz = de_dr * dz * invDist;
  } else {
    dE_dx = kb * 0.01;
    dE_dy = kb * 0.01;
    dE_dz = kb * 0.01;
  }

  atomicAdd(&grad[3 * idx1 + 0], dE_dx);
  atomicAdd(&grad[3 * idx1 + 1], dE_dy);
  atomicAdd(&grad[3 * idx1 + 2], dE_dz);

  atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
  atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
  atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
}

static __device__ __forceinline__ double angleBendEnergy(const double* pos,
                                                         const int     idx1,
                                                         const int     idx2,
                                                         const int     idx3,
                                                         const double  theta0,
                                                         const double  ka,
                                                         const bool    isLinear) {
  constexpr double prefactor = 0.5 * 143.9325 * degreeToRadian * degreeToRadian;
  constexpr double cb        = -0.4 * degreeToRadian;

  float       dx1, dy1, dz1, dx2, dy2, dz2;
  const float dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const float dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const float dist1        = rsqrtf(dist1Squared);
  const float dist2        = rsqrtf(dist2Squared);

  const float  dot         = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta    = clamp(dot * (dist1 * dist2), -1.0f, 1.0f);
  const double theta       = radianToDegree * acos(cosTheta);
  const double deltaTheta  = theta - theta0;
  const double deltaTheta2 = deltaTheta * deltaTheta;

  if (isLinear) {
    constexpr double linearPrefactor = 143.9325;
    return linearPrefactor * ka * (1.0 + cosTheta);
  }
  return prefactor * ka * deltaTheta2 * (1.0 + cb * deltaTheta);
}

static __device__ __forceinline__ void angleBendGrad(const int     idx1,
                                                     const int     idx2,
                                                     const int     idx3,
                                                     const double  theta0,
                                                     const double  ka,
                                                     const bool    isLinear,
                                                     const double* pos,
                                                     double*       grad) {
  constexpr double c1       = 143.9325 * degreeToRadian;
  constexpr double cbFactor = -0.006981317 * 1.5;
  // These values are sensitive to double precision.
  double       dx1, dy1, dz1, dx2, dy2, dz2;
  const double dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const double dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const double invDist1        = rsqrt(dist1Squared);
  const double invDist2       = rsqrt(dist2Squared);

  const double dot         = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta    = clamp(dot *invDist1 * invDist2, -1.0, 1.0);
  const double sinThetaSq  = 1.0 - cosTheta * cosTheta;
  if (isDoubleZero(sinThetaSq) || isDoubleZero(dist1Squared) || isDoubleZero(dist2Squared)) {
    return;
  }

  const double invNegSinTheta = -rsqrt(sinThetaSq);
  const float theta       = radianToDegree * acos(cosTheta);
  const float deltaTheta  = theta - theta0;

  float de_dDeltaTheta;

  if (isLinear) {
    constexpr float linearPrefactor = 143.9325;
    de_dDeltaTheta                   = -linearPrefactor * ka * sqrtf(1.0 - (cosTheta * cosTheta));
  } else {
    de_dDeltaTheta = c1 * ka * deltaTheta * (1.0 + cbFactor * deltaTheta);
  }

  const float dxnorm1 = dx1 * invDist1;
  const float dynorm1 = dy1 * invDist1;
  const float dznorm1 = dz1 * invDist1;
  const float dxnorm2 = dx2 * invDist2;
  const float dynorm2 = dy2 * invDist2;
  const float dznorm2 = dz2 * invDist2;

  const float intermediate1 = invDist1 * (dxnorm2 - cosTheta * dxnorm1);
  const float intermediate2 = invDist1 * (dynorm2 - cosTheta * dynorm1);
  const float intermediate3 = invDist1 * (dznorm2 - cosTheta * dznorm1);
  const float intermediate4 = invDist2 * (dxnorm1 - cosTheta * dxnorm2);
  const float intermediate5 = invDist2 * (dynorm1 - cosTheta * dynorm2);
  const float intermediate6 = invDist2 * (dznorm1 - cosTheta * dznorm2);

  const float constantFactor = de_dDeltaTheta * invNegSinTheta;

  atomicAdd(&grad[3 * idx1 + 0], constantFactor * intermediate1);
  atomicAdd(&grad[3 * idx1 + 1], constantFactor * intermediate2);
  atomicAdd(&grad[3 * idx1 + 2], constantFactor * intermediate3);

  atomicAdd(&grad[3 * idx2 + 0], constantFactor * (-intermediate1 - intermediate4));
  atomicAdd(&grad[3 * idx2 + 1], constantFactor * (-intermediate2 - intermediate5));
  atomicAdd(&grad[3 * idx2 + 2], constantFactor * (-intermediate3 - intermediate6));

  atomicAdd(&grad[3 * idx3 + 0], constantFactor * intermediate4);
  atomicAdd(&grad[3 * idx3 + 1], constantFactor * intermediate5);
  atomicAdd(&grad[3 * idx3 + 2], constantFactor * intermediate6);
}

static __device__ __forceinline__ double bendStretchEnergy(const double* pos,
                                                           const int     idx1,
                                                           const int     idx2,
                                                           const int     idx3,
                                                           const double  theta0,
                                                           const double  restLen1,
                                                           const double  restLen2,
                                                           const double  forceConst1,
                                                           const double  forceConst2) {
  constexpr double prefactor = 2.51210;

  float       dx1, dy1, dz1, dx2, dy2, dz2;
  const float dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const float dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  const float dist1        = sqrtf(dist1Squared);
  const float dist2        = sqrtf(dist2Squared);

  const float  dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const double cosTheta = clamp(dot / (dist1 * dist2), -1.0f, 1.0f);
  const double theta    = 180 / M_PI * acos(cosTheta);

  const double deltaTheta = theta - theta0;
  const double deltaR1    = dist1 - restLen1;
  const double deltaR2    = dist2 - restLen2;

  return prefactor * deltaTheta * (deltaR1 * forceConst1 + deltaR2 * forceConst2);
}

static __device__ __forceinline__ void bendStretchGrad(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const int     idx3,
                                                       const double  theta0,
                                                       const double  restLen1,
                                                       const double  restLen2,
                                                       const double  forceConst1,
                                                       const double  forceConst2,
                                                       double*       grad) {
  constexpr float prefactor = 143.9325 * M_PI / 180.0;

  float       dx1, dy1, dz1, dx2, dy2, dz2;
  const float dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
  const float dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
  // Note that doing the inverse sqrt would be better here, but it causes drift in some edge case tests.
  const float dist1        = sqrtf(dist1Squared);
  const float dist2        = sqrtf(dist2Squared);
  const float invDist1 = 1.0f / dist1;
  const float invDist2 = 1.0f / dist2;
  const float dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
  const float cosTheta = clamp(dot  * invDist1 * invDist2, -1.0f, 1.0f);
  const float invSinTheta = fmin(rsqrtf(1.0f - cosTheta * cosTheta), 1.0e8f);

  constexpr float bondFactor = 180.f / M_PI;
  const float theta = bondFactor * acos(cosTheta);

  const float deltaTheta = theta - theta0;
  const float deltaR1    = dist1 - restLen1;
  const float deltaR2    = dist2 - restLen2;

  const float bondEnergyTerm = bondFactor * (forceConst1 * deltaR1 + forceConst2 * deltaR2);



  const float scaledDx1 = dx1 * invDist1;
  const float scaledDy1 = dy1 * invDist1;
  const float scaledDz1 = dz1 * invDist1;
  const float scaledDx2 = dx2 * invDist2;
  const float scaledDy2 = dy2 * invDist2;
  const float scaledDz2 = dz2 * invDist2;

  const float intermediate1 = invDist1 * (scaledDx2 - cosTheta * scaledDx1);
  const float intermediate2 = invDist1 * (scaledDy2 - cosTheta * scaledDy1);
  const float intermediate3 = invDist1 * (scaledDz2 - cosTheta * scaledDz1);
  const float intermediate4 = invDist2 * (scaledDx1 - cosTheta * scaledDx2);
  const float intermediate5 = invDist2 * (scaledDy1 - cosTheta * scaledDy2);
  const float intermediate6 = invDist2 * (scaledDz1 - cosTheta * scaledDz2);

  const float bondEnergyTimesInvSinTheta = bondEnergyTerm * invSinTheta;

  const float gradx1 = prefactor * (deltaTheta * scaledDx1 * forceConst1 - intermediate1 * bondEnergyTimesInvSinTheta);
  const float grady1 = prefactor * (deltaTheta * scaledDy1 * forceConst1 - intermediate2 * bondEnergyTimesInvSinTheta);
  const float gradz1 = prefactor * (deltaTheta * scaledDz1 * forceConst1 - intermediate3 * bondEnergyTimesInvSinTheta);

  const float gradx2 = prefactor * (-deltaTheta * (scaledDx1 * forceConst1 + scaledDx2 * forceConst2) +
                                     (intermediate1 + intermediate4) * bondEnergyTimesInvSinTheta);
  const float grady2 = prefactor * (-deltaTheta * (scaledDy1 * forceConst1 + scaledDy2 * forceConst2) +
                                     (intermediate2 + intermediate5) * bondEnergyTimesInvSinTheta);
  const float gradz2 = prefactor * (-deltaTheta * (scaledDz1 * forceConst1 + scaledDz2 * forceConst2) +
                                     (intermediate3 + intermediate6) * bondEnergyTimesInvSinTheta);

  const float gradx3 = prefactor * (deltaTheta * scaledDx2 * forceConst2 - intermediate4 * bondEnergyTimesInvSinTheta);
  const float grady3 = prefactor * (deltaTheta * scaledDy2 * forceConst2 - intermediate5 * bondEnergyTimesInvSinTheta);
  const float gradz3 = prefactor * (deltaTheta * scaledDz2 * forceConst2 - intermediate6 * bondEnergyTimesInvSinTheta);

  atomicAdd(&grad[3 * idx1 + 0], gradx1);
  atomicAdd(&grad[3 * idx1 + 1], grady1);
  atomicAdd(&grad[3 * idx1 + 2], gradz1);

  atomicAdd(&grad[3 * idx3 + 0], gradx3);
  atomicAdd(&grad[3 * idx3 + 1], grady3);
  atomicAdd(&grad[3 * idx3 + 2], gradz3);

  atomicAdd(&grad[3 * idx2 + 0], gradx2);
  atomicAdd(&grad[3 * idx2 + 1], grady2);
  atomicAdd(&grad[3 * idx2 + 2], gradz2);
}

static __device__ __forceinline__ double oopBendEnergy(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const int     idx3,
                                                       const int     idx4,
                                                       const double  koop) {
  constexpr float prefactor = 0.5 * 143.9325 * degreeToRadian * degreeToRadian;

  float       dxji, dyji, dzji, dxjk, dyjk, dzjk, dxjl, dyjl, dzjl;
  const float distSquaredJI = distanceSquaredWithComponents(pos, idx1, idx2, dxji, dyji, dzji);
  const float distSquaredJK = distanceSquaredWithComponents(pos, idx3, idx2, dxjk, dyjk, dzjk);
  const float distSquaredJL = distanceSquaredWithComponents(pos, idx4, idx2, dxjl, dyjl, dzjl);

  const float invDistJI = rsqrtf(distSquaredJI);
  const float invDistJK = rsqrtf(distSquaredJK);
  const float invDistJL = rsqrtf(distSquaredJL);

  const float scaledDxJI = dxji * invDistJI;
  const float scaledDyJI = dyji * invDistJI;
  const float scaledDzJI = dzji * invDistJI;

  const float scaledDxJK = dxjk * invDistJK;
  const float scaledDyJK = dyjk * invDistJK;
  const float scaledDzJK = dzjk * invDistJK;

  const float scaledDxJL = dxjl * invDistJL;
  const float scaledDyJL = dyjl * invDistJL;
  const float scaledDzJL = dzjl * invDistJL;

  float crossX, crossY, crossZ;
  crossProduct(scaledDxJI, scaledDyJI, scaledDzJI, scaledDxJK, scaledDyJK, scaledDzJK, crossX, crossY, crossZ);
  const float invDistCross = rsqrtf(crossX * crossX + crossY * crossY + crossZ * crossZ);

  const float scaledCrossX = crossX * invDistCross;
  const float scaledCrossY = crossY * invDistCross;
  const float scaledCrossZ = crossZ * invDistCross;

  const float dotProduct = scaledCrossX * scaledDxJL + scaledCrossY * scaledDyJL + scaledCrossZ * scaledDzJL;
  const float chi        = radianToDegree * asinf(clamp(dotProduct, -1.0f, 1.0f));

  return prefactor * koop * chi * chi;
}

static __device__ __forceinline__ double torsionEnergy(const double* pos,
                                                       const int     idx1,
                                                       const int     idx2,
                                                       const int     idx3,
                                                       const int     idx4,
                                                       const double  V1,
                                                       const double  V2,
                                                       const double  V3) {
  const float dxIJ = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  const float dyIJ = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  const float dzIJ = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  const float dxKJ = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  const float dyKJ = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  const float dzKJ = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  const float dxLK = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  const float dyLK = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  const float dzLK = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  const float crossIJKJx = dyIJ * dzKJ - dzIJ * dyKJ;
  const float crossIJKJy = dzIJ * dxKJ - dxIJ * dzKJ;
  const float crossIJKJz = dxIJ * dyKJ - dyIJ * dxKJ;

  const float crossJKLKx = -dyKJ * dzLK + dzKJ * dyLK;
  const float crossJKLKy = -dzKJ * dxLK + dxKJ * dzLK;
  const float crossJKLKz = -dxKJ * dyLK + dyKJ * dxLK;

  const float invCross1Norm = rsqrtf(crossIJKJx * crossIJKJx + crossIJKJy * crossIJKJy + crossIJKJz * crossIJKJz);
  const float invCross2Norm = rsqrtf(crossJKLKx * crossJKLKx + crossJKLKy * crossJKLKy + crossJKLKz * crossJKLKz);

  const float dotProduct = crossIJKJx * crossJKLKx + crossIJKJy * crossJKLKy + crossIJKJz * crossJKLKz;
  const float cosPhi     = dotProduct * invCross1Norm * invCross2Norm;
  const double phi        = acosf(clamp(cosPhi, -1.0f, 1.0f));

  return 0.5 * (V1 * (1.0 + cosPhi) + V2 * (1.0 - cosf(2.0 * phi)) + V3 * (1.0 + cosf(3.0 * phi)));
}

static __device__ __forceinline__ double vdwEnergy(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const double  R_ij_star,
                                                   const double  wellDepth) {
  // Note, this kernel is quite sensitive, any downcasting to fp32 causes significant drift.
  double R_ij_star2 = R_ij_star * R_ij_star;
  double R_ij_star7 = R_ij_star2 * R_ij_star2 * R_ij_star2 * R_ij_star;

  const double epsilon = wellDepth;

  const double distSquared = distanceSquared(pos, idx1, idx2);
  const double dist        = sqrtf(distSquared);
  const double dist7       = distSquared * distSquared * distSquared * dist;

  const double term1        = 1.07 * R_ij_star / (dist + 0.07 * R_ij_star);
  const double term1Squared = term1 * term1;
  const double term1_7th    = term1Squared * term1Squared * term1Squared * term1;

  const double term2Fraction = 1.12 * R_ij_star7 / (dist7 + 0.12 * R_ij_star7);

  return epsilon * term1_7th * (term2Fraction - 2.0);
}

static __device__ __forceinline__ double eleEnergy(const double* pos,
                                                   const int     idx1,
                                                   const int     idx2,
                                                   const double  chargeTerm,
                                                   const int     dielModel,
                                                   const bool    is1_4) {
  constexpr float prefactor         = 332.0716;
  constexpr float bufferingConstant = 0.05;
  const float     distSquared       = distanceSquared(pos, idx1, idx2);
  float           distTerm          = sqrtf(distSquared) + bufferingConstant;
  if (dielModel == 2) {
    distTerm *= distTerm;
  }
  float energy = prefactor * chargeTerm / (distTerm);
  if (is1_4) {
    energy *= 0.75f;
  }
  return energy;
}

static __device__ __forceinline__ void eleGrad(const double* pos,
                                               const int     idx1,
                                               const int     idx2,
                                               const float  chargeTerm,
                                               const int     dielModel,
                                               const bool    is1_4,
                                               double*       grad) {
  constexpr float prefactor         = 332.0716;
  constexpr float bufferingConstant = 0.05;

  const float distSquared = distanceSquared(pos, idx1, idx2);
  const float invDistance    = rsqrtf(distSquared);
  const float distance = 1.0f / invDistance;
  float       distTerm    = distance + bufferingConstant;
  float       numerator   = -prefactor * chargeTerm;

  if (dielModel == 2) {
    distTerm *= distTerm;
    numerator *= 2;
  }

  float dE_dr = numerator / (distTerm * distTerm);
  if (is1_4) {
    dE_dr *= 0.75;
  }

  const float dE_dx = dE_dr * (pos[3 * idx1 + 0] - pos[3 * idx2 + 0]) * invDistance;
  const float dE_dy = dE_dr * (pos[3 * idx1 + 1] - pos[3 * idx2 + 1]) * invDistance;
  const float dE_dz = dE_dr * (pos[3 * idx1 + 2] - pos[3 * idx2 + 2]) * invDistance;

  atomicAdd(&grad[3 * idx1 + 0], dE_dx);
  atomicAdd(&grad[3 * idx1 + 1], dE_dy);
  atomicAdd(&grad[3 * idx1 + 2], dE_dz);

  atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
  atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
  atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
}

#endif  // NVMOLKIT_MMFF_KERNELS_DEVICE_CUH
