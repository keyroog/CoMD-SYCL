/*************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
 * SYCL port: 2024
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

/// \file
/// Functions to maintain neighbor list of each atom. 

#include <sycl/sycl.hpp>

extern "C" {
#include "gpu_neighborList.h"
#include "linkCells.h"
#include "initAtoms.h"
#include "memUtils.h"
}

#include "defines.h"
#include "CoMDTypes.h"
#include "parallel.h"
#include "gpu_types.h"
#include "gpu_kernels.h"

#include <assert.h>

// Access to the global SYCL queue
extern sycl::queue* g_sycl_queue;

/// Initialize Neighborlist. Allocates all required data structures and initializes all
/// variables. Requires atoms to be initialized and nLocal needs to be set.
/// \param [in] nLocalBoxes  The index with box iBox of the atom to be moved.
/// \param [in] skinDistance Skin distance used by buildNeighborList.
void initNeighborListGpu(SimGpu * sim, NeighborListGpu* neighborList, const int nLocalBoxes, const real_t skinDistance)
{

   neighborList->nMaxLocal = MAXATOMS*nLocalBoxes; // make this list a little larger to make room for migrated particles
   neighborList->nMaxNeighbors = MAXNEIGHBORLISTSIZE;
   neighborList->skinDistance = skinDistance;
   neighborList->skinDistance2 = skinDistance*skinDistance;
   neighborList->skinDistanceHalf2 = (skinDistance/2.0)*(skinDistance/2.0);
   neighborList->nStepsSinceLastBuild = 0;
   neighborList->updateNeighborListRequired = 1;
   neighborList->updateLinkCellsRequired = 0;
   neighborList->forceRebuildFlag = 1; 

   // SYCL device memory allocation
   neighborList->list = sycl::malloc_device<int>(neighborList->nMaxLocal * neighborList->nMaxNeighbors, *g_sycl_queue);
   neighborList->nNeighbors = sycl::malloc_device<int>(neighborList->nMaxLocal, *g_sycl_queue);

   neighborList->lastR.x = sycl::malloc_device<real_t>(neighborList->nMaxLocal, *g_sycl_queue);
   neighborList->lastR.y = sycl::malloc_device<real_t>(neighborList->nMaxLocal, *g_sycl_queue);
   neighborList->lastR.z = sycl::malloc_device<real_t>(neighborList->nMaxLocal, *g_sycl_queue);

   emptyNeighborListGpu(sim, BOTH);

} 

/// Free all the memory associated with Neighborlist
void destroyNeighborListGpu(NeighborListGpu** neighborList)
{
   if (! neighborList) return;
   if (! *neighborList) return;

   comdFree((*neighborList)->list);
   comdFree((*neighborList)->nNeighbors);
   // SYCL device memory deallocation
   sycl::free((*neighborList)->lastR.x, *g_sycl_queue);
   sycl::free((*neighborList)->lastR.y, *g_sycl_queue);
   sycl::free((*neighborList)->lastR.z, *g_sycl_queue);
   comdFree((*neighborList));
   *neighborList = NULL;

   return;
}

void neighborListForceRebuildGpu(struct NeighborListGpuSt* neighborList)
{
   neighborList->forceRebuildFlag = 1; 
}



