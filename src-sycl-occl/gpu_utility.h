/*************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "CoMDTypes.h"
#include "gpu_types.h"
#include <memory.h>

#include <stdlib.h>

#if defined(_WIN32) || defined(_WIN64) 
#include <winsock2.h>
#else
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif
#include <strings.h>
#include <unistd.h>
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
void cudaCopyDtH(void* dst, const void* src, int size);

int compactHaloCells(SimFlat* sim, char* h_compactAtoms, int* h_cellOffset);

/*
DPCT1009:183: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define CUDA_CHECK(command)                                                    \
   {                                                                           \
      dpct::get_current_device().queues_wait_and_throw();                      \
      dpct::err0 status = (command);                                           \
   }

#ifdef DEBUG
#define CUDA_GET_LAST_ERROR                                             \
{                                                                       \
  dpct::get_current_device().queues_wait_and_throw();                   \
}
#else
#define CUDA_GET_LAST_ERROR
#endif





#endif
