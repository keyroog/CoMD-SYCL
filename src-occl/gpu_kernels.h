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

#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include "CoMDTypes.h"
#include "gpu_types.h"

#ifdef __cplusplus
#include <sycl/sycl.hpp>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void ljForceGpu(SimGpu * sim, int interpolation, int num_cells, int * cells_list, real_t plcutoff, int method);

void updateNeighborsGpu(SimGpu sim, int * temp);
void updateNeighborsGpuAsync(SimGpu sim, int * temp, int nCells, int * cellList);
void eamForce1Gpu(SimGpu sim, int method, int spline);
void eamForce2Gpu(SimGpu sim, int method, int spline);
void eamForce3Gpu(SimGpu sim, int method, int spline);

// latency hiding opt
void eamForce1GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, int spline);
void eamForce2GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, int spline);
void eamForce3GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, int spline);

void emptyNeighborListGpu(SimGpu * sim, int boundaryFlag);

int compactCellsGpu(char* work_d, int nCells, int *d_cellList, SimGpu sim_gpu, int* d_cellOffsets, int * d_workScan, real3_old shift);
void unloadAtomsBufferToGpu(char *buf, int nBuf, SimFlat *s, char *gpu_buf);
void loadForceBufferFromGpu(char *buf, int *nbuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf);
void unloadForceBufferToGpu(char *buf, int nBuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf);

void initHashTableGpu(HashTableGpu* hashTable, int nMaxEntries);

void getAtomMsgSoAPtr(char* const buffer, AtomMsgSoA *atomMsg, int n);

void buildNeighborListGpu(SimGpu* sim, int method, int boundaryFlag);
int neighborListUpdateRequiredGpu(SimGpu* sim);
int pairlistUpdateRequiredGpu(SimGpu* sim);

// computes local potential and kinetic energies
void computeEnergy(SimFlat *sim, real_t *eLocal);

void advanceVelocityGpu(SimGpu sim, real_t dt);
void advancePositionGpu(SimGpu* sim, real_t dt);

void buildAtomListGpu(SimFlat *sim);
void updateLinkCellsGpu(SimFlat *sim);
void sortAtomsGpu(SimFlat *sim);

void emptyHashTableGpu(HashTableGpu* hashTable);

#ifdef __cplusplus
}
#endif

#endif
