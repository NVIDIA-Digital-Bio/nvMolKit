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

#ifndef NVMOLKIT_MACRO_CHECK_H
#define NVMOLKIT_MACRO_CHECK_H

#include <cuda_runtime.h>
#define NVMOLKIT_UNROLL _Pragma("unroll")

namespace nvMolKit {

/**
 * @file macros_ptx.cuh
 * @brief Function to retrieve the lane ID of the current thread in a warp.
 *
 * This function uses inline PTX assembly to directly access the special register `laneid`
 * which holds the lane ID of the current thread in its warp. The lane ID is a unique identifier
 * for each thread within a warp, ranging from 0 to 31 in a warp of 32 threads.
 *
 * @section Input and Output Parameter ranges
 * - Input: None
 * - Output: lane_id (int) - The lane ID of the current thread. Range: [0, 31]
 *
 * @section Hardware-Software Interfaces
 * This function interfaces directly with the NVIDIA GPU hardware, specifically utilizing
 * the PTX assembly language to interact with the GPU's special registers. It requires
 * a CUDA-capable GPU and appropriate CUDA software environment to compile and execute.
 *
 **/
__forceinline__ __device__ void get_lane(int& lane_id) {
  asm("mov.s32 %0, %%laneid;" : "=r"(lane_id));
}

/**
 * @file macros_ptx.cuh
 * @brief Asynchronous copy from global to shared memory in cooperative array.
 *
 * This function provides an efficient way to copy a 32-bit unsigned integer
 * from global to shared memory asynchronously, utilizing the cp.async.ca.shared.global
 * instruction available in CUDA architectures starting from 8.0. For older architectures,
 * a simple memory copy is performed.
 *
 * @param dst Pointer to the destination address in shared memory.
 * @param src Pointer to the source address in global memory.
 *
 * @section Input and Output Parameter ranges
 * - Input:
 *   - dst: A valid pointer to shared memory where data will be copied to.
 *   - src: A valid pointer to global memory where data will be copied from.
 * - Output:
 *   - dst: Contains the value copied from src after the function execution.
 *
 * @section Hardware-Software Interfaces
 * - Requires CUDA capable hardware with architecture 8.0 or higher for asynchronous copy.
 * - For architectures lower than 8.0, a standard memory copy operation is performed.
 *
 */
__forceinline__ __device__ void CP_ASYNC_CA(uint32_t* dst, const uint32_t* src) {
#if __CUDA__ARCH__ >= 800
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(((unsigned)__cvta_generic_to_shared(dst))),
               "l"(src),
               "n"(4));
#else
  // Alternative implementation for older architectures
  *dst = *src;
#endif
}

/**
 * @brief Commits and waits for a group of asynchronous copy operations.
 *
 * This function is designed to be used on NVIDIA GPUs with compute capability 8.0 or higher.
 * It commits all outstanding asynchronous copy operations in the current thread group and then waits for all those
 *operations to complete.
 *
 * @note This function uses inline PTX assembly to directly interact with the GPU's control hardware.
 *
 * ## Input and Output Parameter ranges
 * - No input or output parameters.
 *
 * ## Hardware-Software Interfaces
 * - Requires NVIDIA GPU with compute capability 8.0 or higher.
 * - Directly interfaces with the GPU's asynchronous copy engine using PTX assembly instructions.
 *
 **/
__forceinline__ __device__ void commit_wait_group() {
#if __CUDA__ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

/**
 * @brief Performs a bitwise XOR followed by population count on matrices using BMMA instruction.
 *
 * This function utilizes the BMMA (Binary Matrix Multiply Accumulate) instruction to perform
 * a bitwise XOR operation between two input matrices (RA and RB), and then performs a population count
 * (popcount) on the result. The final results are accumulated into the output matrix RD with initial
 * values from matrix RC.
 *
 * @param[out] RD0 Output matrix element at row 0
 * @param[out] RD1 Output matrix element at row 1
 * @param[out] RD2 Output matrix element at row 2
 * @param[out] RD3 Output matrix element at row 3
 * @param[in] RA0 Input matrix A element at row 0
 * @param[in] RA1 Input matrix A element at row 1
 * @param[in] RA2 Input matrix A element at row 2
 * @param[in] RA3 Input matrix A element at row 3
 * @param[in] RB0 Input matrix B element at row 0
 * @param[in] RB1 Input matrix B element at row 1
 * @param[in,out] RC0 Accumulator/input matrix C element at row 0
 * @param[in,out] RC1 Accumulator/input matrix C element at row 1
 * @param[in,out] RC2 Accumulator/input matrix C element at row 2
 * @param[in,out] RC3 Accumulator/input matrix C element at row 3
 *
 * ## Input and Output Parameter Ranges
 * - RA0, RA1, RA2, RA3, RB0, RB1: 32-bit unsigned integers representing binary matrices.
 * - RD0, RD1, RD2, RD3, RC0, RC1, RC2, RC3: 32-bit unsigned integers, where RDx are outputs and RCx are both inputs and
 *outputs.
 *
 * ## Hardware-Software Interfaces
 * This function is designed to be run on NVIDIA GPUs that support the BMMA instruction set, specifically tailored for
 *binary operations.
 **/
__forceinline__ __device__ void bmma_xor_m16n8k256(uint32_t& RD0,
                                                   uint32_t& RD1,
                                                   uint32_t& RD2,
                                                   uint32_t& RD3,
                                                   uint32_t  RA0,
                                                   uint32_t  RA1,
                                                   uint32_t  RA2,
                                                   uint32_t  RA3,
                                                   uint32_t  RB0,
                                                   uint32_t  RB1,
                                                   uint32_t& RC0,
                                                   uint32_t& RC1,
                                                   uint32_t& RC2,
                                                   uint32_t& RC3

) {
  asm volatile(
    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0,%1,%2,%3}, "
    "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)
    : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1), "r"(RC2), "r"(RC3));
}

/**
 * @brief Performs a bitwise AND followed by population count on matrices using BMMA instruction.
 *
 * This function utilizes the BMMA (Binary Matrix Multiply Accumulate) instruction to perform
 * a bitwise AND operation between two input matrices (RA and RB), and then performs a population count
 * (popcount) on the result. The final results are accumulated into the output matrix RD with initial
 * values from matrix RC.
 *
 * @param[out] RD0 Output matrix element at row 0
 * @param[out] RD1 Output matrix element at row 1
 * @param[out] RD2 Output matrix element at row 2
 * @param[out] RD3 Output matrix element at row 3
 * @param[in] RA0 Input matrix A element at row 0
 * @param[in] RA1 Input matrix A element at row 1
 * @param[in] RA2 Input matrix A element at row 2
 * @param[in] RA3 Input matrix A element at row 3
 * @param[in] RB0 Input matrix B element at row 0
 * @param[in] RB1 Input matrix B element at row 1
 * @param[in,out] RC0 Accumulator/input matrix C element at row 0
 * @param[in,out] RC1 Accumulator/input matrix C element at row 1
 * @param[in,out] RC2 Accumulator/input matrix C element at row 2
 * @param[in,out] RC3 Accumulator/input matrix C element at row 3
 *
 * ## Input and Output Parameter Ranges
 * - RA0, RA1, RA2, RA3, RB0, RB1: 32-bit unsigned integers representing binary matrices.
 * - RD0, RD1, RD2, RD3, RC0, RC1, RC2, RC3: 32-bit unsigned integers, where RDx are outputs and RCx are both inputs and
 *outputs.
 *
 * ## Hardware-Software Interfaces
 * This function is designed to be run on NVIDIA GPUs that support the BMMA instruction set, specifically tailored for
 *binary operations.
 **/
__forceinline__ __device__ void bmma_and_m16n8k256(uint32_t& RD0,
                                                   uint32_t& RD1,
                                                   uint32_t& RD2,
                                                   uint32_t& RD3,
                                                   uint32_t  RA0,
                                                   uint32_t  RA1,
                                                   uint32_t  RA2,
                                                   uint32_t  RA3,
                                                   uint32_t  RB0,
                                                   uint32_t  RB1,
                                                   uint32_t& RC0,
                                                   uint32_t& RC1,
                                                   uint32_t& RC2,
                                                   uint32_t& RC3) {
  asm volatile(
    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
    "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)
    : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1), "r"(RC2), "r"(RC3));
}

}  // namespace nvMolKit

#endif  // NVMOLKIT_MACRO_CHECK_H
