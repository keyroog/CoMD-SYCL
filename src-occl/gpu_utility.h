/*************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
 * SYCL port (c) 2024
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#ifndef __GPU_UTILITY_H_
#define __GPU_UTILITY_H_

#include "CoMDTypes.h"
#include "gpu_types.h"
#include <memory.h>
#include <stdlib.h>

// Only include SYCL headers in C++ code
#ifdef __cplusplus
#include <sycl/sycl.hpp>
// Global SYCL queue declaration (only visible in C++)
extern sycl::queue* g_sycl_queue;
#endif

#if defined(_WIN32) || defined(_WIN64) 
#include <winsock2.h>
#else
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif
#include <strings.h>
#include <unistd.h>
#endif

#ifdef DO_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct LinkCellsGpuSt;

void SetupGpu(int deviceId);
void AllocateGpu(SimFlat *flat, int do_eam, real_t skinDistance);
void SetBoundaryCells(SimFlat *flat, HaloExchange* hh);		// for communication latency hiding
void CopyDataToGpu(SimFlat *flat, int do_eam);
void GetDataFromGpu(SimFlat *flat);
void GetLocalAtomsFromGpu(SimFlat *flat);
void DestroyGpu(SimFlat *flat);
void initLinkCellsGpu(SimFlat *sim, struct LinkCellGpuSt* boxes);
void updateGpuHalo(SimFlat *sim);
void updateNAtomsCpu(SimFlat* sim);
void updateNAtomsGpu(SimFlat* sim);
void emptyHaloCellsGpu(SimFlat* sim);
void syclCopyDtH(void* dst, const void* src, int size);
void initSplineCoefficients(real_t* gpu_coefficients, int n, real_t* values, real_t x0, real_t invDx);

int compactHaloCells(SimFlat* sim, char* h_compactAtoms, int* h_cellOffset);

#ifdef __cplusplus
}  // end extern "C"

// SYCL error checking macro - only available in C++
#define SYCL_CHECK(command)                                                     \
{                                                                               \
  try {                                                                         \
    command;                                                                    \
  } catch (sycl::exception const& e) {                                          \
    fprintf(stderr, "Error in file %s at line %d\n", __FILE__, __LINE__);       \
    fprintf(stderr, "SYCL error: %s\n", e.what());                              \
    exit(-1);                                                                   \
  }                                                                             \
}

#ifdef DEBUG
#ifdef DO_MPI
#define SYCL_GET_LAST_ERROR                                                     \
{                                                                               \
  try {                                                                         \
    g_sycl_queue->wait_and_throw();                                             \
  } catch (sycl::exception const& e) {                                          \
    int rank;                                                                   \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                       \
    fprintf(stderr, "rank %d: Error in file %s at line %d\n", rank, __FILE__, __LINE__); \
    fprintf(stderr, "SYCL error: %s\n", e.what());                              \
    exit(-1);                                                                   \
  }                                                                             \
}
#else
#define SYCL_GET_LAST_ERROR                                                     \
{                                                                               \
  try {                                                                         \
    g_sycl_queue->wait_and_throw();                                             \
  } catch (sycl::exception const& e) {                                          \
    fprintf(stderr, "Error in file %s at line %d\n", __FILE__, __LINE__);       \
    fprintf(stderr, "SYCL error: %s\n", e.what());                              \
    exit(-1);                                                                   \
  }                                                                             \
}
#endif
#else
#define SYCL_GET_LAST_ERROR 
#endif

#endif  // __cplusplus

#endif  // __GPU_UTILITY_H_
