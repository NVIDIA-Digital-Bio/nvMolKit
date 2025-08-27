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

#include <cub/cub.cuh>

#include "kernel_utils.cuh"
#include "mmff_kernels.h"

constexpr double degreeToRadian = M_PI / 180.0;
constexpr double radianToDegree = 180.0 / M_PI;

using namespace nvMolKit::FFKernelUtils;

// Implementations here are directly adapted from RDKit, as
// the derivations are hard and no reference equations are provided anywhere.
// TODO: implement from scratch.

namespace rdkit_ports {

__device__ __forceinline__ void oopGrad(const double* pos,
                                        const int     idx1,
                                        const int     idx2,
                                        const int     idx3,
                                        const int     idx4,
                                        const double  koop,
                                        double*       grad) {
  constexpr double prefactor = 143.9325 * degreeToRadian;

  double dJIx = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  double dJIy = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  double dJIz = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  double dJKx = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  double dJKy = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  double dJKz = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  double dJLx = pos[3 * idx4 + 0] - pos[3 * idx2 + 0];
  double dJLy = pos[3 * idx4 + 1] - pos[3 * idx2 + 1];
  double dJLz = pos[3 * idx4 + 2] - pos[3 * idx2 + 2];

  const double dJI = sqrt(dJIx * dJIx + dJIy * dJIy + dJIz * dJIz);
  const double dJK = sqrt(dJKx * dJKx + dJKy * dJKy + dJKz * dJKz);
  const double dJL = sqrt(dJLx * dJLx + dJLy * dJLy + dJLz * dJLz);

  dJIx /= dJI;
  dJIy /= dJI;
  dJIz /= dJI;
  dJKx /= dJK;
  dJKy /= dJK;
  dJKz /= dJK;
  dJLx /= dJL;
  dJLy /= dJL;
  dJLz /= dJL;

  double normalJIKx, normalJIKy, normalJIKz;
  crossProduct(-dJIx, -dJIy, -dJIz, dJKx, dJKy, dJKz, normalJIKx, normalJIKy, normalJIKz);
  const double normLength = sqrt(normalJIKx * normalJIKx + normalJIKy * normalJIKy + normalJIKz * normalJIKz);
  normalJIKx /= normLength;
  normalJIKy /= normLength;
  normalJIKz /= normLength;

  const double sinChi   = clamp(dotProduct(dJLx, dJLy, dJLz, normalJIKx, normalJIKy, normalJIKz), -1.0, 1.0);
  const double cosChiSq = 1.0 - sinChi * sinChi;
  const double cosChi   = fmax(((cosChiSq > 0.0) ? sqrt(cosChiSq) : 0.0), 1.0e-8);
  const double chi      = radianToDegree * asin(sinChi);
  const double cosTheta = clamp(dotProduct(dJIx, dJIy, dJIz, dJKx, dJKy, dJKz), -1.0, 1.0);
  ;
  double sinThetaSq = fmax(1.0 - cosTheta * cosTheta, 1.0e-8);
  double sinTheta   = fmax(((sinThetaSq > 0.0) ? sqrt(sinThetaSq) : 0.0), 1.0e-8);

  double dE_dChi = prefactor * koop * chi;
  double t1x, t1y, t1z, t2x, t2y, t2z, t3x, t3y, t3z;
  crossProduct(dJLx, dJLy, dJLz, dJKx, dJKy, dJKz, t1x, t1y, t1z);
  crossProduct(dJIx, dJIy, dJIz, dJLx, dJLy, dJLz, t2x, t2y, t2z);
  crossProduct(dJKx, dJKy, dJKz, dJIx, dJIy, dJIz, t3x, t3y, t3z);

  double term1  = cosChi * sinTheta;
  double term2  = sinChi / (cosChi * sinThetaSq);
  double tg1[3] = {(t1x / term1 - (dJIx - dJKx * cosTheta) * term2) / dJI,
                   (t1y / term1 - (dJIy - dJKy * cosTheta) * term2) / dJI,
                   (t1z / term1 - (dJIz - dJKz * cosTheta) * term2) / dJI};
  double tg3[3] = {(t2x / term1 - (dJKx - dJIx * cosTheta) * term2) / dJK,
                   (t2y / term1 - (dJKy - dJIy * cosTheta) * term2) / dJK,
                   (t2z / term1 - (dJKz - dJIz * cosTheta) * term2) / dJK};
  double tg4[3] = {(t3x / term1 - dJLx * sinChi / cosChi) / dJL,
                   (t3y / term1 - dJLy * sinChi / cosChi) / dJL,
                   (t3z / term1 - dJLz * sinChi / cosChi) / dJL};

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

__device__ __forceinline__ void torsionGrad(const double* pos,
                                            const int     idx1,
                                            const int     idx2,
                                            const int     idx3,
                                            const int     idx4,
                                            const double  V1,
                                            const double  V2,
                                            const double  V3,
                                            double*       grad) {
  double dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3, dx4, dy4, dz4;

  // P1 - P2
  dx1 = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
  dy1 = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
  dz1 = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

  // P3 - P2
  dx2 = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
  dy2 = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
  dz2 = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

  // P2 - P3
  dx3 = -dx2;
  dy3 = -dy2;
  dz3 = -dz2;

  // P4 - P3
  dx4 = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
  dy4 = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
  dz4 = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

  double cross1x, cross1y, cross1z, cross2x, cross2y, cross2z;
  crossProduct(dx1, dy1, dz1, dx2, dy2, dz2, cross1x, cross1y, cross1z);
  const double norm1 = fmax(sqrt(cross1x * cross1x + cross1y * cross1y + cross1z * cross1z), 1.0e-5);
  cross1x /= norm1;
  cross1y /= norm1;
  cross1z /= norm1;

  crossProduct(dx3, dy3, dz3, dx4, dy4, dz4, cross2x, cross2y, cross2z);
  const double norm2 = fmax(sqrt(cross2x * cross2x + cross2y * cross2y + cross2z * cross2z), 1.0e-5);
  cross2x /= norm2;
  cross2y /= norm2;
  cross2z /= norm2;

  const double dot    = dotProduct(cross1x, cross1y, cross1z, cross2x, cross2y, cross2z);
  const double cosPhi = clamp(dot, -1.0, 1.0);

  double cross3x, cross3y, cross3z;
  crossProduct(cross1x, cross1y, cross1z, dx2, dy2, dz2, cross3x, cross3y, cross3z);

  const double sinPhiSq = 1.0 - cosPhi * cosPhi;
  const double sinPhi   = ((sinPhiSq > 0.0) ? sqrt(sinPhiSq) : 0.0);
  const double sin2Phi  = 2.0 * sinPhi * cosPhi;
  const double sin3Phi  = 3.0 * sinPhi - 4.0 * sinPhi * sinPhiSq;
  const double dE_dPhi  = 0.5 * (-V1 * sinPhi + 2.0 * V2 * sin2Phi - 3.0 * V3 * sin3Phi);
  const double sinTerm  = -dE_dPhi * (isDoubleZero(sinPhi) ? (1.0 / cosPhi) : (1.0 / sinPhi));

  double dCos_dT[6] = {1.0 / norm1 * (cross2x - cosPhi * cross1x),
                       1.0 / norm1 * (cross2y - cosPhi * cross1y),
                       1.0 / norm1 * (cross2z - cosPhi * cross1z),
                       1.0 / norm2 * (cross1x - cosPhi * cross2x),
                       1.0 / norm2 * (cross1y - cosPhi * cross2y),
                       1.0 / norm2 * (cross1z - cosPhi * cross2z)};

  atomicAdd(&grad[3 * idx1 + 0], sinTerm * (dCos_dT[2] * dy2 - dCos_dT[1] * dz2));
  atomicAdd(&grad[3 * idx1 + 1], sinTerm * (dCos_dT[0] * dz2 - dCos_dT[2] * dx2));
  atomicAdd(&grad[3 * idx1 + 2], sinTerm * (dCos_dT[1] * dx2 - dCos_dT[0] * dy2));

  atomicAdd(&grad[3 * idx2 + 0],
            sinTerm * (dCos_dT[1] * (dz2 - dz1) + dCos_dT[2] * (dy1 - dy2) + dCos_dT[4] * (-dz4) + dCos_dT[5] * (dy4)));
  atomicAdd(&grad[3 * idx2 + 1],
            sinTerm * (dCos_dT[0] * (dz1 - dz2) + dCos_dT[2] * (dx2 - dx1) + dCos_dT[3] * (dz4) + dCos_dT[5] * (-dx4)));
  atomicAdd(&grad[3 * idx2 + 2],
            sinTerm * (dCos_dT[0] * (dy2 - dy1) + dCos_dT[1] * (dx1 - dx2) + dCos_dT[3] * (-dy4) + dCos_dT[4] * (dx4)));

  atomicAdd(&grad[3 * idx3 + 0],
            sinTerm * (dCos_dT[1] * (dz1) + dCos_dT[2] * (-dy1) + dCos_dT[4] * (dz4 - dz3) + dCos_dT[5] * (dy3 - dy4)));
  atomicAdd(&grad[3 * idx3 + 1],
            sinTerm * (dCos_dT[0] * (-dz1) + dCos_dT[2] * (dx1) + dCos_dT[3] * (dz3 - dz4) + dCos_dT[5] * (dx4 - dx3)));
  atomicAdd(&grad[3 * idx3 + 2],
            sinTerm * (dCos_dT[0] * (dy1) + dCos_dT[1] * (-dx1) + dCos_dT[3] * (dy4 - dy3) + dCos_dT[4] * (dx3 - dx4)));

  atomicAdd(&grad[3 * idx4 + 0], sinTerm * (dCos_dT[4] * dz3 - dCos_dT[5] * dy3));
  atomicAdd(&grad[3 * idx4 + 1], sinTerm * (dCos_dT[5] * dx3 - dCos_dT[3] * dz3));
  atomicAdd(&grad[3 * idx4 + 2], sinTerm * (dCos_dT[3] * dy3 - dCos_dT[4] * dx3));
}

__device__ __forceinline__ void vDWGrad(const double* pos,
                                        const int     idx1,
                                        const int     idx2,
                                        const double  R_ij_star,
                                        const double  wellDepth,
                                        double*       grad) {
  constexpr double vdw1   = 1.07;
  constexpr double vdw1m1 = vdw1 - 1.0;
  constexpr double vdw2   = 1.12;
  constexpr double vdw2m1 = vdw2 - 1.0;
  constexpr double vdw2t7 = vdw2 * 7.0;

  const double distance = sqrt(distanceSquared(pos, idx1, idx2));

  const double q         = distance / R_ij_star;
  const double q2        = q * q;
  const double q6        = q2 * q2 * q2;
  const double q7        = q6 * q;
  const double q7pvdw2m1 = q7 + vdw2m1;
  const double t         = vdw1 / (q + vdw1 - 1.0);
  const double t2        = t * t;
  const double t7        = t2 * t2 * t2 * t;
  const double dE_dr     = wellDepth / R_ij_star * t7 *
                       (-vdw2t7 * q6 / (q7pvdw2m1 * q7pvdw2m1) + ((-vdw2t7 / q7pvdw2m1 + 14.0) / (q + vdw1m1)));

  double term1x, term1y, term1z;
  if (distance <= 0.0) {
    term1x = R_ij_star * 0.01;
    term1y = R_ij_star * 0.01;
    term1z = R_ij_star * 0.01;
  } else {
    term1x = dE_dr * (pos[3 * idx1 + 0] - pos[3 * idx2 + 0]) / distance;
    term1y = dE_dr * (pos[3 * idx1 + 1] - pos[3 * idx2 + 1]) / distance;
    term1z = dE_dr * (pos[3 * idx1 + 2] - pos[3 * idx2 + 2]) / distance;
  }

  atomicAdd(&grad[3 * idx1 + 0], term1x);
  atomicAdd(&grad[3 * idx1 + 1], term1y);
  atomicAdd(&grad[3 * idx1 + 2], term1z);

  atomicAdd(&grad[3 * idx2 + 0], -term1x);
  atomicAdd(&grad[3 * idx2 + 1], -term1y);
  atomicAdd(&grad[3 * idx2 + 2], -term1z);
}

}  // namespace rdkit_ports

__global__ void bondStretchEnergyKernel(const int     numBonds,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const double* r0,
                                        const double* kb,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // The bond stretch potential is not simple harmonic for MMFF. See
  // https://docs.eyesopen.com/toolkits/python/oefftk/fftheory.html#bond-stretching

  constexpr double prefactor           = 143.9325 / 2.0;
  constexpr double csFactorDist        = -2.0;
  constexpr double csFactorDistSquared = 7.0 / 12.0 * csFactorDist * csFactorDist;
  if (idx < numBonds) {
    // First compute distance between atoms
    const double distSquared = distanceSquared(pos, idx1[idx], idx2[idx]);
    const double distance    = sqrt(distSquared);

    const double deltaR    = distance - r0[idx];
    const double deltaR2   = deltaR * deltaR;
    const int    batchIdx  = atomBatchMap[idx1[idx]];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] +=
      prefactor * kb[idx] * deltaR2 * (1.0 + csFactorDist * deltaR + csFactorDistSquared * deltaR2);
  }
}

__global__ void bondStretchGradKernel(const int     numBonds,
                                      const int*    idx1s,
                                      const int*    idx2s,
                                      const double* r0,
                                      const double* kb,
                                      const double* pos,
                                      double*       grad) {
  const int bondIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // Functional form of potential here:
  // https://docs.eyesopen.com/toolkits/python/oefftk/fftheory.html#bond-stretching
  // Gradient equation:
  // DE/DdeltaR = c1 * kb * deltaR * (1 + cs * deltaR + 2.0 * 7.0 / 12.0 * cs * cs * deltaR^2)

  constexpr double c1                          = 143.9325;
  constexpr double cs                          = -2.0;
  constexpr double csFactorTimesSecondConstant = cs * 1.5;
  constexpr double lastFactor                  = 2.0 * 7.0 / 12.0 * cs * cs;  // 7/12 * cs * cs

  if (bondIdx < numBonds) {
    const int idx1 = idx1s[bondIdx];
    const int idx2 = idx2s[bondIdx];

    double       dx, dy, dz;
    const double distanceSquared = distanceSquaredWithComponents(pos, idx1, idx2, dx, dy, dz);
    const double distance        = sqrt(distanceSquared);
    const double deltaR          = distance - r0[bondIdx];

    const double de_dr =
      c1 * kb[bondIdx] * deltaR * (1.0 + csFactorTimesSecondConstant * deltaR + lastFactor * deltaR * deltaR);

    // Compute dx gradients;
    const double invDist = 1.0 / distance;
    double       dE_dx, dE_dy, dE_dz;
    if (distance > 0.0) {
      dE_dx = de_dr * dx * invDist;
      dE_dy = de_dr * dy * invDist;
      dE_dz = de_dr * dz * invDist;
    } else {
      // Taken from RDKit implementation for 1:1 parity
      dE_dx = kb[bondIdx] * 0.01;
      dE_dy = kb[bondIdx] * 0.01;
      dE_dz = kb[bondIdx] * 0.01;
    }

    atomicAdd(&grad[3 * idx1 + 0], dE_dx);
    atomicAdd(&grad[3 * idx1 + 1], dE_dy);
    atomicAdd(&grad[3 * idx1 + 2], dE_dz);

    atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
    atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
    atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
  }
}

__global__ void angleBendEnergyKernel(const int      numAngles,
                                      const int*     idx1s,
                                      const int*     idx2s,
                                      const int*     idx3s,
                                      const double*  theta0,
                                      const double*  ka,
                                      const uint8_t* isLinear,
                                      const double*  pos,
                                      double*        energyBuffer,
                                      const int*     energyBufferStarts,
                                      const int*     atomBatchMap,
                                      const int*     termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Angle terms are not simple harmonic
  // https://docs.eyesopen.com/toolkits/python/oefftk/fftheory.html#angle-bending

  // Prefactor converted to inv radian^2
  constexpr double prefactor = 0.5 * 143.9325 * degreeToRadian * degreeToRadian;
  constexpr double cb        = -0.4 * degreeToRadian;
  if (idx < numAngles) {
    const int idx1 = idx1s[idx];
    const int idx2 = idx2s[idx];
    const int idx3 = idx3s[idx];

    // Calculate angle between two points
    double       dx1, dy1, dz1, dx2, dy2, dz2;
    const double dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
    const double dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
    const double dist1        = sqrt(dist1Squared);
    const double dist2        = sqrt(dist2Squared);

    const double dot         = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
    const double cosTheta    = clamp(dot / (dist1 * dist2), -1.0, 1.0);
    const double theta       = radianToDegree * acos(cosTheta);
    const double deltaTheta  = theta - theta0[idx];
    const double deltaTheta2 = deltaTheta * deltaTheta;

    const int batchIdx  = atomBatchMap[idx1];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);

    if (isLinear[idx]) {
      constexpr double linearPrefactor = 143.9325;
      energyBuffer[outputIdx] += linearPrefactor * ka[idx] * (1.0 + cosTheta);
    } else {
      energyBuffer[outputIdx] += prefactor * ka[idx] * deltaTheta2 * (1.0 + cb * deltaTheta);
    }
  }
}

__global__ void angleBendGradientKernel(const int      numAngles,
                                        const int*     idx1s,
                                        const int*     idx2s,
                                        const int*     idx3s,
                                        const double*  theta0,
                                        const double*  ka,
                                        const uint8_t* isLinear,
                                        const double*  pos,
                                        double*        grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Angle terms are not simple harmonic
  // https://docs.eyesopen.com/toolkits/python/oefftk/fftheory.html#angle-bending
  // Grad equation:
  // DE/DdeltaTheta = c1 * ka * deltaTheta( 1 + 3/2 cb * deltaTheata)
  // Derivatives can be found from https://grigoryanlab.org/docs/dynamics_derivatives.pdf

  // Prefactor converted to inv radian^2
  constexpr double c1       = 143.9325 * degreeToRadian;
  constexpr double cbFactor = -0.006981317 * 1.5;
  if (idx < numAngles) {
    const int idx1 = idx1s[idx];
    const int idx2 = idx2s[idx];
    const int idx3 = idx3s[idx];

    // Calculate angle between two points
    double       dx1, dy1, dz1, dx2, dy2, dz2;
    const double dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
    const double dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
    const double dist1        = sqrt(dist1Squared);
    const double dist2        = sqrt(dist2Squared);

    const double dot         = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
    const double cosTheta    = clamp(dot / (dist1 * dist2), -1.0, 1.0);
    const double sinThetaSq  = 1.0 - cosTheta * cosTheta;
    const double negSinTheta = -(fmax(((sinThetaSq > 0.0) ? sqrt(sinThetaSq) : 0.0), 1.0e-8));
    const double theta       = radianToDegree * acos(cosTheta);
    const double deltaTheta  = theta - theta0[idx];

    double de_dDeltaTheta;

    if (isLinear[idx]) {
      // Linear term is c1 * k * sin(theta), which we get from the cosine
      constexpr double linearPrefactor = 143.9325;
      de_dDeltaTheta                   = -linearPrefactor * ka[idx] * sqrt(1.0 - cosTheta * cosTheta);
    } else {
      de_dDeltaTheta = c1 * ka[idx] * deltaTheta * (1.0 + cbFactor * deltaTheta);
    }

    // Now do dDeltaTheta/dx for all 3 atoms. Taken from RDKit;
    if (isDoubleZero(dist1) || isDoubleZero(dist2)) {
      return;
    }

    const double invDist1 = 1.0 / dist1;
    const double invDist2 = 1.0 / dist2;

    const double dxnorm1 = dx1 * invDist1;  // From 1 to 2
    const double dynorm1 = dy1 * invDist1;
    const double dznorm1 = dz1 * invDist1;
    const double dxnorm2 = dx2 * invDist2;  // From 3 to 2
    const double dynorm2 = dy2 * invDist2;
    const double dznorm2 = dz2 * invDist2;

    const double intermediate1 = invDist1 * (dxnorm2 - cosTheta * dxnorm1);
    const double intermediate2 = invDist1 * (dynorm2 - cosTheta * dynorm1);
    const double intermediate3 = invDist1 * (dznorm2 - cosTheta * dznorm1);
    const double intermediate4 = invDist2 * (dxnorm1 - cosTheta * dxnorm2);
    const double intermediate5 = invDist2 * (dynorm1 - cosTheta * dynorm2);
    const double intermediate6 = invDist2 * (dznorm1 - cosTheta * dznorm2);

    if (isDoubleZero(negSinTheta)) {
      return;
    }
    const double constantFactor = de_dDeltaTheta / negSinTheta;

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
}

__global__ void bendStretchEnergyKernel(const int     numAngles,
                                        const int*    idx1s,
                                        const int*    idx2s,
                                        const int*    idx3s,
                                        const double* theta0,
                                        const double* restLen1,
                                        const double* restLen2,
                                        const double* forceConst1,
                                        const double* forceConst2,
                                        const double* pos,
                                        double*       energyBuffer,
                                        const int*    energyBufferStarts,
                                        const int*    atomBatchMap,
                                        const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr double prefactor = 2.51210;
  // Functional form:
  // https://docs.eyesopen.com/toolkits/python/oefftk/fftheory.html#stretch-bend-interaction

  if (idx < numAngles) {
    const int idx1 = idx1s[idx];
    const int idx2 = idx2s[idx];
    const int idx3 = idx3s[idx];

    // Calculate angle between two points
    double       dx1, dy1, dz1, dx2, dy2, dz2;
    const double dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
    const double dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
    const double dist1        = sqrt(dist1Squared);
    const double dist2        = sqrt(dist2Squared);

    const double dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
    const double cosTheta = clamp(dot / (dist1 * dist2), -1.0, 1.0);
    const double theta    = 180 / M_PI * acos(cosTheta);

    const double deltaTheta = theta - theta0[idx];
    const double deltaR1    = dist1 - restLen1[idx];
    const double deltaR2    = dist2 - restLen2[idx];

    const double energy    = prefactor * deltaTheta * (deltaR1 * forceConst1[idx] + deltaR2 * forceConst2[idx]);
    const int    batchIdx  = atomBatchMap[idx1];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void bendStretchGradKernel(const int     numAngles,
                                      const int*    idx1s,
                                      const int*    idx2s,
                                      const int*    idx3s,
                                      const double* theta0,
                                      const double* restLen1,
                                      const double* restLen2,
                                      const double* forceConst1,
                                      const double* forceConst2,
                                      const double* pos,
                                      double*       grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr double prefactor = 143.9325 * M_PI / 180.0;
  // Functional form:
  // https://docs.eyesopen.com/toolkits/python/oefftk/fftheory.html#stretch-bend-interaction

  if (idx < numAngles) {
    const int idx1 = idx1s[idx];
    const int idx2 = idx2s[idx];
    const int idx3 = idx3s[idx];

    // Calculate angle between two points
    double       dx1, dy1, dz1, dx2, dy2, dz2;
    const double dist1Squared = distanceSquaredWithComponents(pos, idx1, idx2, dx1, dy1, dz1);
    const double dist2Squared = distanceSquaredWithComponents(pos, idx3, idx2, dx2, dy2, dz2);
    const double dist1        = sqrt(dist1Squared);
    const double dist2        = sqrt(dist2Squared);

    const double dot      = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
    const double cosTheta = clamp(dot / (dist1 * dist2), -1.0, 1.0);
    const double sinTheta = fmax(sqrt(1.0 - cosTheta * cosTheta), 1.0e-8);

    const double theta = 180 / M_PI * acos(cosTheta);

    const double deltaTheta = theta - theta0[idx];
    const double deltaR1    = dist1 - restLen1[idx];
    const double deltaR2    = dist2 - restLen2[idx];

    const double bondEnergyTerm = 180.0 / M_PI * (forceConst1[idx] * deltaR1 + forceConst2[idx] * deltaR2);

    const double invDist1 = 1.0 / dist1;
    const double invDist2 = 1.0 / dist2;

    const double scaledDx1 = dx1 * invDist1;
    const double scaledDy1 = dy1 * invDist1;
    const double scaledDz1 = dz1 * invDist1;
    const double scaledDx2 = dx2 * invDist2;
    const double scaledDy2 = dy2 * invDist2;
    const double scaledDz2 = dz2 * invDist2;

    const double intermediate1 = invDist1 * (scaledDx2 - cosTheta * scaledDx1);
    const double intermediate2 = invDist1 * (scaledDy2 - cosTheta * scaledDy1);
    const double intermediate3 = invDist1 * (scaledDz2 - cosTheta * scaledDz1);
    const double intermediate4 = invDist2 * (scaledDx1 - cosTheta * scaledDx2);
    const double intermediate5 = invDist2 * (scaledDy1 - cosTheta * scaledDy2);
    const double intermediate6 = invDist2 * (scaledDz1 - cosTheta * scaledDz2);

    const double gradx1 =
      prefactor * (deltaTheta * scaledDx1 * forceConst1[idx] - intermediate1 * bondEnergyTerm / sinTheta);
    const double grady1 =
      prefactor * (deltaTheta * scaledDy1 * forceConst1[idx] - intermediate2 * bondEnergyTerm / sinTheta);
    const double gradz1 =
      prefactor * (deltaTheta * scaledDz1 * forceConst1[idx] - intermediate3 * bondEnergyTerm / sinTheta);

    const double gradx2 = prefactor * (-deltaTheta * (scaledDx1 * forceConst1[idx] + scaledDx2 * forceConst2[idx]) +
                                       (intermediate1 + intermediate4) * bondEnergyTerm / sinTheta);
    const double grady2 = prefactor * (-deltaTheta * (scaledDy1 * forceConst1[idx] + scaledDy2 * forceConst2[idx]) +
                                       (intermediate2 + intermediate5) * bondEnergyTerm / sinTheta);
    const double gradz2 = prefactor * (-deltaTheta * (scaledDz1 * forceConst1[idx] + scaledDz2 * forceConst2[idx]) +
                                       (intermediate3 + intermediate6) * bondEnergyTerm / sinTheta);

    const double gradx3 =
      prefactor * (deltaTheta * scaledDx2 * forceConst2[idx] - intermediate4 * bondEnergyTerm / sinTheta);
    const double grady3 =
      prefactor * (deltaTheta * scaledDy2 * forceConst2[idx] - intermediate5 * bondEnergyTerm / sinTheta);
    const double gradz3 =
      prefactor * (deltaTheta * scaledDz2 * forceConst2[idx] - intermediate6 * bondEnergyTerm / sinTheta);

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
}

__global__ void oopBendEnergyKernel(const int     numOopBends,
                                    const int*    idx1s,
                                    const int*    idx2s,
                                    const int*    idx3s,
                                    const int*    idx4s,
                                    const double* koop,
                                    const double* pos,
                                    double*       energyBuffer,
                                    const int*    energyBufferStarts,
                                    const int*    atomBatchMap,
                                    const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  constexpr double prefactor = 0.5 * 143.9325 * degreeToRadian * degreeToRadian;
  if (idx < numOopBends) {
    // Using I, J, K, L notation

    double       dxji, dyji, dzji, dxjk, dyjk, dzjk, dxjl, dyjl, dzjl;
    const double distSquaredJI = distanceSquaredWithComponents(pos, idx1s[idx], idx2s[idx], dxji, dyji, dzji);
    const double distSquaredJK = distanceSquaredWithComponents(pos, idx3s[idx], idx2s[idx], dxjk, dyjk, dzjk);
    const double distSquaredJL = distanceSquaredWithComponents(pos, idx4s[idx], idx2s[idx], dxjl, dyjl, dzjl);

    const double distJI = sqrt(distSquaredJI);
    const double distJK = sqrt(distSquaredJK);
    const double distJL = sqrt(distSquaredJL);

    const double scaledDxJI = dxji / distJI;
    const double scaledDyJI = dyji / distJI;
    const double scaledDzJI = dzji / distJI;

    const double scaledDxJK = dxjk / distJK;
    const double scaledDyJK = dyjk / distJK;
    const double scaledDzJK = dzjk / distJK;

    const double scaledDxJL = dxjl / distJL;
    const double scaledDyJL = dyjl / distJL;
    const double scaledDzJL = dzjl / distJL;

    // Cross product between JI and JK
    double crossX, crossY, crossZ;
    crossProduct(scaledDxJI, scaledDyJI, scaledDzJI, scaledDxJK, scaledDyJK, scaledDzJK, crossX, crossY, crossZ);
    const double distCross = sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);

    const double scaledCrossX = crossX / distCross;
    const double scaledCrossY = crossY / distCross;
    const double scaledCrossZ = crossZ / distCross;

    // Dot product between cross product and JL
    const double dotProduct = scaledCrossX * scaledDxJL + scaledCrossY * scaledDyJL + scaledCrossZ * scaledDzJL;
    const double chi        = radianToDegree * asin(clamp(dotProduct, -1.0, 1.0));

    const int batchIdx  = atomBatchMap[idx1s[idx]];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += prefactor * koop[idx] * chi * chi;
  }
}
__global__ void oopBendGradKernel(const int     numOopBends,
                                  const int*    idx1s,
                                  const int*    idx2s,
                                  const int*    idx3s,
                                  const int*    idx4s,
                                  const double* koop,
                                  const double* pos,
                                  double*       grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numOopBends) {
    rdkit_ports::oopGrad(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], koop[idx], grad);
  }
}

__global__ void torsionEnergyKernel(const int     numTorsions,
                                    const int*    idx1s,
                                    const int*    idx2s,
                                    const int*    idx3s,
                                    const int*    idx4s,
                                    const double* V1s,
                                    const double* V2s,
                                    const double* V3s,
                                    const double* pos,
                                    double*       energyBuffer,
                                    const int*    energyBufferStarts,
                                    const int*    atomBatchMap,
                                    const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numTorsions) {
    const int idx1 = idx1s[idx];
    const int idx2 = idx2s[idx];
    const int idx3 = idx3s[idx];
    const int idx4 = idx4s[idx];

    const double V1 = V1s[idx];
    const double V2 = V2s[idx];
    const double V3 = V3s[idx];

    // Compute dihedral angle.
    const double dxIJ = pos[3 * idx1 + 0] - pos[3 * idx2 + 0];
    const double dyIJ = pos[3 * idx1 + 1] - pos[3 * idx2 + 1];
    const double dzIJ = pos[3 * idx1 + 2] - pos[3 * idx2 + 2];

    const double dxKJ = pos[3 * idx3 + 0] - pos[3 * idx2 + 0];
    const double dyKJ = pos[3 * idx3 + 1] - pos[3 * idx2 + 1];
    const double dzKJ = pos[3 * idx3 + 2] - pos[3 * idx2 + 2];

    const double dxLK = pos[3 * idx4 + 0] - pos[3 * idx3 + 0];
    const double dyLK = pos[3 * idx4 + 1] - pos[3 * idx3 + 1];
    const double dzLK = pos[3 * idx4 + 2] - pos[3 * idx3 + 2];

    const double crossIJKJx = dyIJ * dzKJ - dzIJ * dyKJ;
    const double crossIJKJy = dzIJ * dxKJ - dxIJ * dzKJ;
    const double crossIJKJz = dxIJ * dyKJ - dyIJ * dxKJ;

    // Second product JK -> LK. Note the inversion of KJ, we switch the negatives
    const double crossJKLKx = -dyKJ * dzLK + dzKJ * dyLK;
    const double crossJKLKy = -dzKJ * dxLK + dxKJ * dzLK;
    const double crossJKLKz = -dxKJ * dyLK + dyKJ * dxLK;

    const double cross1Norm = sqrt(crossIJKJx * crossIJKJx + crossIJKJy * crossIJKJy + crossIJKJz * crossIJKJz);
    const double cross2Norm = sqrt(crossJKLKx * crossJKLKx + crossJKLKy * crossJKLKy + crossJKLKz * crossJKLKz);

    const double dotProduct = crossIJKJx * crossJKLKx + crossIJKJy * crossJKLKy + crossIJKJz * crossJKLKz;
    const double cosPhi     = dotProduct / (cross1Norm * cross2Norm);
    const double phi        = acos(clamp(cosPhi, -1.0, 1.0));

    const double energy    = 0.5 * (V1 * (1.0 + cosPhi) + V2 * (1.0 - cos(2.0 * phi)) + V3 * (1.0 + cos(3.0 * phi)));
    const int    batchIdx  = atomBatchMap[idx1];
    const int    outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void torsionGradKernel(const int     numTorsions,
                                  const int*    idx1s,
                                  const int*    idx2s,
                                  const int*    idx3s,
                                  const int*    idx4s,
                                  const double* V1s,
                                  const double* V2s,
                                  const double* V3s,
                                  const double* pos,
                                  double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numTorsions) {
    rdkit_ports::torsionGrad(pos, idx1s[idx], idx2s[idx], idx3s[idx], idx4s[idx], V1s[idx], V2s[idx], V3s[idx], grad);
  }
}

__global__ void vdwEnergyKernel(const int     numVdws,
                                const int*    idxs1,
                                const int*    idx2s,
                                const double* R_ij_stars,
                                const double* wellDepths,
                                const double* pos,
                                double*       energyBuffer,
                                const int*    energyBufferStarts,
                                const int*    atomBatchMap,
                                const int*    termBatchStarts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < numVdws) {
    const int    idx1       = idxs1[idx];
    const int    idx2       = idx2s[idx];
    const double R_ij_star  = R_ij_stars[idx];
    double       R_ij_star2 = R_ij_star * R_ij_star;
    double       R_ij_star7 = R_ij_star2 * R_ij_star2 * R_ij_star2 * R_ij_star;

    const double epsilon = wellDepths[idx];

    const double distSquared = distanceSquared(pos, idx1, idx2);
    const double dist        = sqrt(distSquared);
    const double dist7       = distSquared * distSquared * distSquared * dist;

    const double term1        = 1.07 * R_ij_star / (dist + 0.07 * R_ij_star);
    const double term1Squared = term1 * term1;
    const double term1_7th    = term1Squared * term1Squared * term1Squared * term1;

    const double term2Fraction = 1.12 * R_ij_star7 / (dist7 + 0.12 * R_ij_star7);

    const int batchIdx  = atomBatchMap[idx1];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += epsilon * term1_7th * (term2Fraction - 2.0);
  }
}
__global__ void vdwGradKernel(const int     numVdws,
                              const int*    idx1,
                              const int*    idx2,
                              const double* R_ij_star,
                              const double* wellDepth,
                              const double* pos,
                              double*       grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numVdws) {
    rdkit_ports::vDWGrad(pos, idx1[idx], idx2[idx], R_ij_star[idx], wellDepth[idx], grad);
  }
}

__global__ void eleEnergyKernel(const int      numEles,
                                const int*     idx1s,
                                const int*     idx2s,
                                const double*  chargeTerms,
                                const uint8_t* dielModels,
                                const uint8_t* is1_4s,
                                const double*  pos,
                                double*        energyBuffer,
                                const int*     energyBufferStarts,
                                const int*     atomBatchMap,
                                const int*     termBatchStarts) {
  int              idx               = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr double prefactor         = 332.0716;
  constexpr double bufferingConstant = 0.05;
  if (idx < numEles) {
    const int     idx1       = idx1s[idx];
    const int     idx2       = idx2s[idx];
    const double  chargeTerm = chargeTerms[idx];
    const uint8_t dielModel  = dielModels[idx];
    const bool    is1_4      = is1_4s[idx];

    const double distSquared = distanceSquared(pos, idx1, idx2);
    double       distTerm    = sqrt(distSquared) + bufferingConstant;
    if (dielModel == 2) {
      distTerm *= distTerm;
    }
    double energy = prefactor * chargeTerm / (distTerm);
    if (is1_4) {
      energy *= 0.75;
    }

    const int batchIdx  = atomBatchMap[idx1];
    const int outputIdx = getEnergyAccumulatorIndex(idx, batchIdx, energyBufferStarts, termBatchStarts);
    energyBuffer[outputIdx] += energy;
  }
}
__global__ void eleGradKernel(const int      numEles,
                              const int*     idx1s,
                              const int*     idx2s,
                              const double*  chargeTerms,
                              const uint8_t* dielModels,
                              const uint8_t* is1_4s,
                              const double*  pos,
                              double*        grad) {
  int              idx               = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr double prefactor         = 332.0716;
  constexpr double bufferingConstant = 0.05;
  if (idx < numEles) {
    const int     idx1       = idx1s[idx];
    const int     idx2       = idx2s[idx];
    const double  chargeTerm = chargeTerms[idx];
    const uint8_t dielModel  = dielModels[idx];
    const bool    is1_4      = is1_4s[idx];

    const double distSquared = distanceSquared(pos, idx1, idx2);
    const double distance    = sqrt(distSquared);
    double       distTerm    = distance + bufferingConstant;
    double       numerator   = -prefactor * chargeTerm;

    // If it's diel model 2, the distance term is squared in the energy, so the derivative has another factor of 2.
    // Note we're further squaring the distance term regardless. If it's diel model = 1, it's a typical 1/r E -> 1/r^2
    // F.
    if (dielModel == 2) {
      distTerm *= distTerm;
      numerator *= 2;
    }

    double dE_dr = numerator / (distTerm * distTerm);
    if (is1_4) {
      dE_dr *= 0.75;
    }

    // Be careful here to use the actual distance, not the offset one.
    const double dE_dx = dE_dr * (pos[3 * idx1 + 0] - pos[3 * idx2 + 0]) / distance;
    const double dE_dy = dE_dr * (pos[3 * idx1 + 1] - pos[3 * idx2 + 1]) / distance;
    const double dE_dz = dE_dr * (pos[3 * idx1 + 2] - pos[3 * idx2 + 2]) / distance;

    atomicAdd(&grad[3 * idx1 + 0], dE_dx);
    atomicAdd(&grad[3 * idx1 + 1], dE_dy);
    atomicAdd(&grad[3 * idx1 + 2], dE_dz);

    atomicAdd(&grad[3 * idx2 + 0], -dE_dx);
    atomicAdd(&grad[3 * idx2 + 1], -dE_dy);
    atomicAdd(&grad[3 * idx2 + 2], -dE_dz);
  }
}

namespace nvMolKit {
namespace MMFF {

cudaError_t launchBondStretchEnergyKernel(const int     numBonds,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const double* r0,
                                          const double* kb,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream) {
  assert(numBonds > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds,
                                                               idx1,
                                                               idx2,
                                                               r0,
                                                               kb,
                                                               pos,
                                                               energyBuffer,
                                                               energyBufferStarts,
                                                               atomBatchMap,
                                                               termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchBondStretchGradientKernel(const int     numBonds,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const double* r0,
                                            const double* kb,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream) {
  assert(numBonds > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numBonds + blockSize - 1) / blockSize;
  bondStretchGradKernel<<<numBlocks, blockSize, 0, stream>>>(numBonds, idx1, idx2, r0, kb, pos, grad);

  return cudaGetLastError();
}

cudaError_t launchAngleBendEnergyKernel(const int      numAngles,
                                        const int*     idx1,
                                        const int*     idx2,
                                        const int*     idx3,
                                        const double*  theta0,
                                        const double*  ka,
                                        const uint8_t* isLinear,
                                        const double*  pos,
                                        double*        energyBuffer,
                                        const int*     energyBufferStarts,
                                        const int*     atomBatchMap,
                                        const int*     termBatchStarts,
                                        cudaStream_t   stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             theta0,
                                                             ka,
                                                             isLinear,
                                                             pos,
                                                             energyBuffer,
                                                             energyBufferStarts,
                                                             atomBatchMap,
                                                             termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchAngleBendGradientKernel(const int      numAngles,
                                          const int*     idx1,
                                          const int*     idx2,
                                          const int*     idx3,
                                          const double*  theta0,
                                          const double*  ka,
                                          const uint8_t* isLinear,
                                          const double*  pos,
                                          double*        grad,
                                          cudaStream_t   stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  angleBendGradientKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                               idx1,
                                                               idx2,
                                                               idx3,
                                                               theta0,
                                                               ka,
                                                               isLinear,
                                                               pos,
                                                               grad);

  return cudaGetLastError();
}

cudaError_t launchBendStretchEnergyKernel(const int     numAngles,
                                          const int*    idx1,
                                          const int*    idx2,
                                          const int*    idx3,
                                          const double* theta0,
                                          const double* restLen1,
                                          const double* restLen2,
                                          const double* forceConst1,
                                          const double* forceConst2,
                                          const double* pos,
                                          double*       energyBuffer,
                                          const int*    energyBufferStarts,
                                          const int*    atomBatchMap,
                                          const int*    termBatchStarts,
                                          cudaStream_t  stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  bendStretchEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                               idx1,
                                                               idx2,
                                                               idx3,
                                                               theta0,
                                                               restLen1,
                                                               restLen2,
                                                               forceConst1,
                                                               forceConst2,
                                                               pos,
                                                               energyBuffer,
                                                               energyBufferStarts,
                                                               atomBatchMap,
                                                               termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchBendStretchGradientKernel(const int     numAngles,
                                            const int*    idx1,
                                            const int*    idx2,
                                            const int*    idx3,
                                            const double* theta0,
                                            const double* restLen1,
                                            const double* restLen2,
                                            const double* forceConst1,
                                            const double* forceConst2,
                                            const double* pos,
                                            double*       grad,
                                            cudaStream_t  stream) {
  assert(numAngles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numAngles + blockSize - 1) / blockSize;
  bendStretchGradKernel<<<numBlocks, blockSize, 0, stream>>>(numAngles,
                                                             idx1,
                                                             idx2,
                                                             idx3,
                                                             theta0,
                                                             restLen1,
                                                             restLen2,
                                                             forceConst1,
                                                             forceConst2,
                                                             pos,
                                                             grad);
  return cudaGetLastError();
}

cudaError_t launchOopBendEnergyKernel(const int     numOopBends,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const double* koop,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts,
                                      cudaStream_t  stream) {
  assert(numOopBends > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numOopBends + blockSize - 1) / blockSize;
  oopBendEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numOopBends,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           koop,
                                                           pos,
                                                           energyBuffer,
                                                           energyBufferStarts,
                                                           atomBatchMap,
                                                           termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchOopBendGradientKernel(const int     numOopBends,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const double* koop,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream) {
  assert(numOopBends > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numOopBends + blockSize - 1) / blockSize;
  oopBendGradKernel<<<numBlocks, blockSize, 0, stream>>>(numOopBends, idx1, idx2, idx3, idx4, koop, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchTorsionEnergyKernel(const int     numTorsions,
                                      const int*    idx1,
                                      const int*    idx2,
                                      const int*    idx3,
                                      const int*    idx4,
                                      const double* V1,
                                      const double* V2,
                                      const double* V3,
                                      const double* pos,
                                      double*       energyBuffer,
                                      const int*    energyBufferStarts,
                                      const int*    atomBatchMap,
                                      const int*    termBatchStarts,
                                      cudaStream_t  stream) {
  assert(numTorsions > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions,
                                                           idx1,
                                                           idx2,
                                                           idx3,
                                                           idx4,
                                                           V1,
                                                           V2,
                                                           V3,
                                                           pos,
                                                           energyBuffer,
                                                           energyBufferStarts,
                                                           atomBatchMap,
                                                           termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchTorsionGradientKernel(const int     numTorsions,
                                        const int*    idx1,
                                        const int*    idx2,
                                        const int*    idx3,
                                        const int*    idx4,
                                        const double* V1,
                                        const double* V2,
                                        const double* V3,
                                        const double* pos,
                                        double*       grad,
                                        cudaStream_t  stream) {
  assert(numTorsions > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numTorsions + blockSize - 1) / blockSize;
  torsionGradKernel<<<numBlocks, blockSize, 0, stream>>>(numTorsions, idx1, idx2, idx3, idx4, V1, V2, V3, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchVdwEnergyKernel(const int     numVdws,
                                  const int*    idx1,
                                  const int*    idx2,
                                  const double* R_ij_star,
                                  const double* wellDepth,
                                  const double* pos,
                                  double*       energyBuffer,
                                  const int*    energyBufferStarts,
                                  const int*    atomBatchMap,
                                  const int*    termBatchStarts,
                                  cudaStream_t  stream) {
  assert(numVdws > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws,
                                                       idx1,
                                                       idx2,
                                                       R_ij_star,
                                                       wellDepth,
                                                       pos,
                                                       energyBuffer,
                                                       energyBufferStarts,
                                                       atomBatchMap,
                                                       termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchVdwGradientKernel(const int     numVdws,
                                    const int*    idx1,
                                    const int*    idx2,
                                    const double* R_ij_star,
                                    const double* wellDepth,
                                    const double* pos,
                                    double*       grad,
                                    cudaStream_t  stream) {
  assert(numVdws > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numVdws + blockSize - 1) / blockSize;
  vdwGradKernel<<<numBlocks, blockSize, 0, stream>>>(numVdws, idx1, idx2, R_ij_star, wellDepth, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchEleEnergyKernel(const int      numEles,
                                  const int*     idx1,
                                  const int*     idx2,
                                  const double*  chargeTerm,
                                  const uint8_t* dielModel,
                                  const uint8_t* is1_4,
                                  const double*  pos,
                                  double*        energyBuffer,
                                  const int*     energyBufferStarts,
                                  const int*     atomBatchMap,
                                  const int*     termBatchStarts,
                                  cudaStream_t   stream) {
  assert(numEles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numEles + blockSize - 1) / blockSize;
  eleEnergyKernel<<<numBlocks, blockSize, 0, stream>>>(numEles,
                                                       idx1,
                                                       idx2,
                                                       chargeTerm,
                                                       dielModel,
                                                       is1_4,
                                                       pos,
                                                       energyBuffer,
                                                       energyBufferStarts,
                                                       atomBatchMap,
                                                       termBatchStarts);
  return cudaGetLastError();
}

cudaError_t launchEleGradientKernel(const int      numEles,
                                    const int*     idx1,
                                    const int*     idx2,
                                    const double*  chargeTerm,
                                    const uint8_t* dielModel,
                                    const uint8_t* is1_4,
                                    const double*  pos,
                                    double*        grad,
                                    cudaStream_t   stream) {
  assert(numEles > 0);
  constexpr int blockSize = 256;
  const int     numBlocks = (numEles + blockSize - 1) / blockSize;
  eleGradKernel<<<numBlocks, blockSize, 0, stream>>>(numEles, idx1, idx2, chargeTerm, dielModel, is1_4, pos, grad);
  return cudaGetLastError();
}

cudaError_t launchReduceEnergiesKernel(const int     numBlocks,
                                       const double* energyBuffer,
                                       const int*    energyBufferBlockIdxToBatchIdx,
                                       double*       outs,
                                       cudaStream_t  stream) {
  reduceEnergiesKernel<<<numBlocks, nvMolKit::FFKernelUtils::blockSizeEnergyReduction, 0, stream>>>(
    energyBuffer,
    energyBufferBlockIdxToBatchIdx,
    outs);
  return cudaGetLastError();
}
}  // namespace MMFF
}  // namespace nvMolKit
