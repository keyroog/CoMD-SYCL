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

#include <stdio.h>
#include <assert.h>
#include <vector>
#include <sycl/sycl.hpp>

#include "CoMDTypes.h"
#include "haloExchange.h"

#include "gpu_types.h"
#include "defines.h"
#include "gpu_utility.h"
#include "sycl_common.h"
#include "gpu_kernels.h"
#include "parallel.h"

// External SYCL queue from gpu_utility.cpp
extern sycl::queue* g_sycl_queue;

//=============================================================================
// Timestep kernels
//=============================================================================

void advanceVelocityGpu(SimGpu sim, real_t dt)
{
    if (sim.a_list.n == 0) return;

    int n = sim.a_list.n;
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;

            int iAtom = sim.a_list.atoms[tid];
            int iBox = sim.a_list.cells[tid];
            int iOff = iBox * MAXATOMS + iAtom;

            sim.atoms.p.x[iOff] += dt * sim.atoms.f.x[iOff]; 
            sim.atoms.p.y[iOff] += dt * sim.atoms.f.y[iOff]; 
            sim.atoms.p.z[iOff] += dt * sim.atoms.f.z[iOff]; 
        }
    );
    
    SYCL_GET_LAST_ERROR
}

void advancePositionGpu(SimGpu* sim, real_t dt)
{
    if (sim->a_list.n == 0) return;

    int n = sim->a_list.n;
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    SimGpu sim_copy = *sim;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;

            int iAtom = sim_copy.a_list.atoms[tid];
            int iBox = sim_copy.a_list.cells[tid];
            int iOff = iBox * MAXATOMS + iAtom;
            
            int iSpecies = sim_copy.atoms.iSpecies[iOff];
            real_t invMass = 1.0/sim_copy.species_mass[iSpecies];

            sim_copy.atoms.r.x[iOff] += dt * sim_copy.atoms.p.x[iOff] * invMass;
            sim_copy.atoms.r.y[iOff] += dt * sim_copy.atoms.p.y[iOff] * invMass;
            sim_copy.atoms.r.z[iOff] += dt * sim_copy.atoms.p.z[iOff] * invMass;
        }
    );

    sim->atoms.neighborList.updateNeighborListRequired = -1;
    
    SYCL_GET_LAST_ERROR
}

//=============================================================================
// LJ Force kernel (thread-per-atom version)
//=============================================================================

void ljForceGpu(SimGpu * sim, int interpolation, int num_cells, int * cells_list, real_t plcutoff, int method)
{
    if (sim->a_list.n == 0) return;
    
    int n = sim->a_list.n;
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    SimGpu sim_copy = *sim;
    
    // Thread-atom method
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;

            // compute box ID and local atom ID
            int iAtom = sim_copy.a_list.atoms[tid];
            int iBox = sim_copy.a_list.cells[tid]; 

            // common constants for LJ potential
            real_t sigma = sim_copy.lj_pot.sigma;
            real_t epsilon = sim_copy.lj_pot.epsilon;
            real_t rCut = sim_copy.lj_pot.cutoff;
            real_t rCut2 = rCut*rCut;

            real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

            real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
            real_t eShift = rCut6 * (rCut6 - 1.0f);

            // zero out forces and energy
            real_t ifx = 0;
            real_t ify = 0;
            real_t ifz = 0;
            real_t ie = 0;

            // fetch position
            int iOff = iBox * MAXATOMS + iAtom;
            real_t irx = sim_copy.atoms.r.x[iOff];
            real_t iry = sim_copy.atoms.r.y[iOff];
            real_t irz = sim_copy.atoms.r.z[iOff];

            // loop over my cell first
            {
                const int jBox = iBox;
                int jOff = jBox * MAXATOMS;
                for (int jAtom = 0; jAtom < sim_copy.boxes.nAtoms[jBox]; jAtom++) {
                    real_t dx = irx - sim_copy.atoms.r.x[jOff];
                    real_t dy = iry - sim_copy.atoms.r.y[jOff];
                    real_t dz = irz - sim_copy.atoms.r.z[jOff];

                    real_t r2 = dx*dx + dy*dy + dz*dz;

                    if (r2 <= rCut2 && r2 > 0.0f) {
                        r2 = 1.0f/r2;
                        real_t r6 = s6 * (r2*r2*r2);
                        real_t eLocal = r6 * (r6 - 1.0f) - eShift;

                        ie += 0.5f * eLocal;
                        real_t fr = r6*r2*(48.0f*r6 - 24.0f);

                        ifx += fr * dx;
                        ify += fr * dy;
                        ifz += fr * dz;
                    }
                    ++jOff;
                }
            }

            // loop over neighbor cells
            for (int j = 1; j < N_MAX_NEIGHBORS; j++) {
                const int jBox = sim_copy.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
                int jOff = jBox * MAXATOMS;
                
                for (int jAtom = 0; jAtom < sim_copy.boxes.nAtoms[jBox]; jAtom++) {
                    real_t dx = irx - sim_copy.atoms.r.x[jOff];
                    real_t dy = iry - sim_copy.atoms.r.y[jOff];
                    real_t dz = irz - sim_copy.atoms.r.z[jOff];

                    real_t r2 = dx*dx + dy*dy + dz*dz;

                    if (r2 <= rCut2) {
                        r2 = 1.0f/r2;
                        real_t r6 = s6 * (r2*r2*r2);
                        real_t eLocal = r6 * (r6 - 1.0f) - eShift;

                        ie += 0.5f * eLocal;
                        real_t fr = r6*r2*(48.0f*r6 - 24.0f);

                        ifx += fr * dx;
                        ify += fr * dy;
                        ifz += fr * dz;
                    }
                    ++jOff;
                }
            }

            sim_copy.atoms.f.x[iOff] = ifx * epsilon;
            sim_copy.atoms.f.y[iOff] = ify * epsilon;
            sim_copy.atoms.f.z[iOff] = ifz * epsilon;
            sim_copy.atoms.e[iOff] = ie * 4 * epsilon;
        }
    );
    
    SYCL_GET_LAST_ERROR
}

//=============================================================================
// Energy reduction kernel
//=============================================================================

void computeEnergy(SimFlat *flat, real_t *eLocal)
{
    int n = flat->gpu.a_list.n;
    if (n == 0) {
        eLocal[0] = 0.0;
        eLocal[1] = 0.0;
        return;
    }
    
    // Allocate device memory for partial sums
    real_t* e_gpu = sycl::malloc_device<real_t>(2, *g_sycl_queue);
    g_sycl_queue->memset(e_gpu, 0, 2 * sizeof(real_t)).wait();
    
    SimGpu sim_copy = flat->gpu;
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    g_sycl_queue->submit([&](sycl::handler& cgh) {
        // Local memory for reduction
        sycl::local_accessor<real_t, 1> sp(THREAD_ATOM_CTA, cgh);
        sycl::local_accessor<real_t, 1> sk(THREAD_ATOM_CTA, cgh);
        
        cgh.parallel_for(
            sycl::nd_range<1>(num_wg * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int tid = item.get_global_id(0);
                int lid = item.get_local_id(0);

                real_t ep = 0;
                real_t ek = 0;
                
                if (tid < n) {
                    int iAtom = sim_copy.a_list.atoms[tid];
                    int iBox = sim_copy.a_list.cells[tid];
                    int iOff = iBox * MAXATOMS + iAtom;

                    int iSpecies = sim_copy.atoms.iSpecies[iOff];
                    real_t invMass = 0.5/sim_copy.species_mass[iSpecies];
                    ep = sim_copy.atoms.e[iOff]; 
                    ek = (sim_copy.atoms.p.x[iOff] * sim_copy.atoms.p.x[iOff] + 
                          sim_copy.atoms.p.y[iOff] * sim_copy.atoms.p.y[iOff] + 
                          sim_copy.atoms.p.z[iOff] * sim_copy.atoms.p.z[iOff]) * invMass;
                }
                
                // Store in local memory
                sp[lid] = ep;
                sk[lid] = ek;
                item.barrier(sycl::access::fence_space::local_space);
                
                // Reduction in shared memory
                for (int i = wg_size / 2; i > 0; i /= 2) {
                    if (lid < i) {
                        sp[lid] += sp[lid + i];
                        sk[lid] += sk[lid + i];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                
                // One thread adds to global memory
                if (lid == 0) {
                    sycl::atomic_ref<real_t, sycl::memory_order::relaxed, 
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> ref_ep(e_gpu[0]);
                    sycl::atomic_ref<real_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> ref_ek(e_gpu[1]);
                    ref_ep.fetch_add(sp[0]);
                    ref_ek.fetch_add(sk[0]);
                }
            }
        );
    });
    
    g_sycl_queue->memcpy(eLocal, e_gpu, 2 * sizeof(real_t)).wait();
    sycl::free(e_gpu, *g_sycl_queue);
    
    SYCL_GET_LAST_ERROR
}

//=============================================================================
// Update neighbors kernel
//=============================================================================

void updateNeighborsGpu(SimGpu sim, int *temp)
{
    int nLocalBoxes = sim.boxes.nLocalBoxes;
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (nLocalBoxes + wg_size - 1) / wg_size;
    
    // Update # of neighbor atoms per cell - 1 thread per cell
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= nLocalBoxes) return;
            
            int count = 0;
            for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
                int jBox = sim.neighbor_cells[tid * N_MAX_NEIGHBORS + j];
                count += sim.boxes.nAtoms[jBox];
            }
            sim.num_neigh_atoms[tid] = count;
        }
    );
    
    SYCL_GET_LAST_ERROR
}

//=============================================================================
// Build atom list kernel
//=============================================================================

void buildAtomListGpu(SimFlat *sim)
{
    int nLocalBoxes = sim->boxes->nLocalBoxes;
    SimGpu* gpu = &sim->gpu;
    
    // Count total atoms
    int n = 0;
    for (int iBox = 0; iBox < nLocalBoxes; iBox++)
        n += sim->boxes->nAtoms[iBox];
    
    gpu->a_list.n = n;
    
    // Build list on host
    int idx = 0;
    for (int iBox = 0; iBox < nLocalBoxes; iBox++) {
        for (int iAtom = 0; iAtom < sim->boxes->nAtoms[iBox]; iAtom++) {
            sim->host.a_list.atoms[idx] = iAtom;
            sim->host.a_list.cells[idx] = iBox;
            idx++;
        }
    }
    
    // Copy to device
    g_sycl_queue->memcpy(gpu->a_list.atoms, sim->host.a_list.atoms, n * sizeof(int));
    g_sycl_queue->memcpy(gpu->a_list.cells, sim->host.a_list.cells, n * sizeof(int));
    g_sycl_queue->wait();
}

//=============================================================================
// Neighbor list operations
//=============================================================================

void emptyNeighborListGpu(SimGpu *sim, int boundaryFlag)
{
    int n = sim->a_list.n;
    if (n == 0) return;
    
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    SimGpu sim_copy = *sim;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;

            int iBox = sim_copy.a_list.cells[tid];
            if (boundaryFlag == INTERIOR && sim_copy.cell_type[iBox] != 0) return;
            if (boundaryFlag == BOUNDARY && sim_copy.cell_type[iBox] != 1) return;
            sim_copy.atoms.neighborList.nNeighbors[tid] = 0;
        }
    );
    
    SYCL_GET_LAST_ERROR
}

int neighborListUpdateRequiredGpu(SimGpu* sim)
{
    if (sim->atoms.neighborList.forceRebuildFlag == 1) {
        sim->atoms.neighborList.updateNeighborListRequired = 1;
    } else if (sim->atoms.neighborList.updateNeighborListRequired == -1) {
        int n = sim->a_list.n;
        int wg_size = THREAD_ATOM_CTA;
        int num_wg = (n + wg_size - 1) / wg_size;
        
        int* d_updateNeighborListRequired = sycl::malloc_device<int>(1, *g_sycl_queue);
        g_sycl_queue->memset(d_updateNeighborListRequired, 0, sizeof(int)).wait();
        
        SimGpu sim_copy = *sim;
        real_t skinDistanceHalf2 = sim->atoms.neighborList.skinDistanceHalf2;
        
        g_sycl_queue->parallel_for(
            sycl::nd_range<1>(num_wg * wg_size, wg_size),
            [=](sycl::nd_item<1> item) {
                int tid = item.get_global_id(0);
                if (tid >= n) return;

                int iAtom = sim_copy.a_list.atoms[tid];
                int iBox = sim_copy.a_list.cells[tid];
                int iOff = iBox * MAXATOMS + iAtom;

                real_t dx = sim_copy.atoms.r.x[iOff] - sim_copy.atoms.neighborList.lastR.x[tid];
                real_t dy = sim_copy.atoms.r.y[iOff] - sim_copy.atoms.neighborList.lastR.y[tid];
                real_t dz = sim_copy.atoms.r.z[iOff] - sim_copy.atoms.neighborList.lastR.z[tid];

                if ((dx*dx + dy*dy + dz*dz) > skinDistanceHalf2) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> ref(*d_updateNeighborListRequired);
                    ref.store(1);
                }
            }
        );
        
        int h_updateNeighborListRequired;
        g_sycl_queue->memcpy(&h_updateNeighborListRequired, d_updateNeighborListRequired, sizeof(int)).wait();
        sycl::free(d_updateNeighborListRequired, *g_sycl_queue);
        
        int tmpUpdateNeighborListRequired = 0;
        addIntParallel(&h_updateNeighborListRequired, &tmpUpdateNeighborListRequired, 1);
        
        if (tmpUpdateNeighborListRequired > 0)
            sim->atoms.neighborList.updateNeighborListRequired = 1;
        else
            sim->atoms.neighborList.updateNeighborListRequired = 0;
    }
    
    SYCL_GET_LAST_ERROR
    return sim->atoms.neighborList.updateNeighborListRequired;
}

void buildNeighborListGpu(SimGpu* sim, int method, int boundaryFlag)
{
    // Implementation depends on method - simplified version
    int n = sim->a_list.n;
    if (n == 0) return;
    
    // For now, just update the last positions
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    SimGpu sim_copy = *sim;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;

            int iAtom = sim_copy.a_list.atoms[tid];
            int iBox = sim_copy.a_list.cells[tid];
            int iOff = iBox * MAXATOMS + iAtom;

            sim_copy.atoms.neighborList.lastR.x[tid] = sim_copy.atoms.r.x[iOff];
            sim_copy.atoms.neighborList.lastR.y[tid] = sim_copy.atoms.r.y[iOff];
            sim_copy.atoms.neighborList.lastR.z[tid] = sim_copy.atoms.r.z[iOff];
        }
    );
    
    sim->atoms.neighborList.forceRebuildFlag = 0;
    
    SYCL_GET_LAST_ERROR
}

//=============================================================================
// Hash table operations
//=============================================================================

void initHashTableGpu(HashTableGpu* hashTable, int nMaxEntries)
{
    hashTable->nMaxEntries = nMaxEntries;
    hashTable->nEntriesPut = 0;
    hashTable->nEntriesGet = 0;
    hashTable->offset = sycl::malloc_device<int>(nMaxEntries, *g_sycl_queue);
    g_sycl_queue->memset(hashTable->offset, HASHTABLE_FREE, nMaxEntries * sizeof(int)).wait();
}

void emptyHashTableGpu(HashTableGpu* hashTable)
{
    hashTable->nEntriesPut = 0;
    hashTable->nEntriesGet = 0;
    g_sycl_queue->memset(hashTable->offset, HASHTABLE_FREE, hashTable->nMaxEntries * sizeof(int)).wait();
}

//=============================================================================
// EAM Force kernels - Full Implementation
//=============================================================================

void eamForce1Gpu(SimGpu sim, int method, int spline)
{
    int n = sim.a_list.n;
    if (n == 0) return;
    
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;
            
            int iAtom = sim.a_list.atoms[tid];
            int iBox = sim.a_list.cells[tid];
            int iOff = iBox * MAXATOMS + iAtom;
            
            real_t rCut = sim.eam_pot.cutoff;
            real_t rCut2 = rCut * rCut;
            
            real_t ifx = 0, ify = 0, ifz = 0;
            real_t ie = 0, irho = 0;
            
            real_t irx = sim.atoms.r.x[iOff];
            real_t iry = sim.atoms.r.y[iOff];
            real_t irz = sim.atoms.r.z[iOff];
            
            // Loop over neighbor cells
            for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
                int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
                
                for (int jAtom = 0; jAtom < sim.boxes.nAtoms[jBox]; jAtom++) {
                    int jOff = jBox * MAXATOMS + jAtom;
                    
                    real_t dx = irx - sim.atoms.r.x[jOff];
                    real_t dy = iry - sim.atoms.r.y[jOff];
                    real_t dz = irz - sim.atoms.r.z[jOff];
                    
                    real_t r2 = dx*dx + dy*dy + dz*dz;
                    
                    if (r2 <= rCut2 && r2 > 0.0) {
                        real_t r = sycl::sqrt(r2);
                        real_t phiTmp, dPhi, rhoTmp, dRho;
                        
                        if (!spline) {
                            interpolate(sim.eam_pot.phi, r, phiTmp, dPhi);
                            interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
                            dPhi /= r;
                        } else {
                            interpolateSpline(sim.eam_pot.phiS, r2, phiTmp, dPhi);
                            interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
                        }
                        
                        ifx -= dPhi * dx;
                        ify -= dPhi * dy;
                        ifz -= dPhi * dz;
                        ie += phiTmp;
                        irho += rhoTmp;
                    }
                }
            }
            
            sim.atoms.f.x[iOff] = ifx;
            sim.atoms.f.y[iOff] = ify;
            sim.atoms.f.z[iOff] = ifz;
            sim.atoms.e[iOff] = 0.5 * ie;
            sim.eam_pot.rhobar[iOff] = irho;
        }
    );
    
    SYCL_GET_LAST_ERROR
}

void eamForce2Gpu(SimGpu sim, int method, int spline)
{
    int n = sim.a_list.n;
    if (n == 0) return;
    
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;
            
            int iAtom = sim.a_list.atoms[tid];
            int iBox = sim.a_list.cells[tid];
            int iOff = iBox * MAXATOMS + iAtom;
            
            real_t fEmbed, dfEmbed;
            interpolate(sim.eam_pot.f, sim.eam_pot.rhobar[iOff], fEmbed, dfEmbed);
            sim.eam_pot.dfEmbed[iOff] = dfEmbed;
            sim.atoms.e[iOff] += fEmbed;
        }
    );
    
    SYCL_GET_LAST_ERROR
}

void eamForce3Gpu(SimGpu sim, int method, int spline)
{
    int n = sim.a_list.n;
    if (n == 0) return;
    
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;
            
            int iAtom = sim.a_list.atoms[tid];
            int iBox = sim.a_list.cells[tid];
            int iOff = iBox * MAXATOMS + iAtom;
            
            real_t rCut = sim.eam_pot.cutoff;
            real_t rCut2 = rCut * rCut;
            
            real_t ifx = sim.atoms.f.x[iOff];
            real_t ify = sim.atoms.f.y[iOff];
            real_t ifz = sim.atoms.f.z[iOff];
            
            real_t irx = sim.atoms.r.x[iOff];
            real_t iry = sim.atoms.r.y[iOff];
            real_t irz = sim.atoms.r.z[iOff];
            
            // Loop over neighbor cells
            for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
                int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
                
                for (int jAtom = 0; jAtom < sim.boxes.nAtoms[jBox]; jAtom++) {
                    int jOff = jBox * MAXATOMS + jAtom;
                    
                    real_t dx = irx - sim.atoms.r.x[jOff];
                    real_t dy = iry - sim.atoms.r.y[jOff];
                    real_t dz = irz - sim.atoms.r.z[jOff];
                    
                    real_t r2 = dx*dx + dy*dy + dz*dz;
                    
                    if (r2 <= rCut2 && r2 > 0.0) {
                        real_t rhoTmp, dRho, dPhi;
                        
                        if (!spline) {
                            real_t r = sycl::sqrt(r2);
                            interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
                            dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
                            dPhi /= r;
                        } else {
                            interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
                            dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
                        }
                        
                        ifx -= dPhi * dx;
                        ify -= dPhi * dy;
                        ifz -= dPhi * dz;
                    }
                }
            }
            
            sim.atoms.f.x[iOff] = ifx;
            sim.atoms.f.y[iOff] = ify;
            sim.atoms.f.z[iOff] = ifz;
        }
    );
    
    SYCL_GET_LAST_ERROR
}

void eamForce1GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, int spline)
{
    if (atoms_list.n == 0) return;
    
    int n = atoms_list.n;
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;
            
            int iAtom = atoms_list.atoms[tid];
            int iBox = atoms_list.cells[tid];
            int iOff = iBox * MAXATOMS + iAtom;
            
            real_t rCut = sim.eam_pot.cutoff;
            real_t rCut2 = rCut * rCut;
            
            real_t ifx = 0, ify = 0, ifz = 0;
            real_t ie = 0, irho = 0;
            
            real_t irx = sim.atoms.r.x[iOff];
            real_t iry = sim.atoms.r.y[iOff];
            real_t irz = sim.atoms.r.z[iOff];
            
            for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
                int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
                
                for (int jAtom = 0; jAtom < sim.boxes.nAtoms[jBox]; jAtom++) {
                    int jOff = jBox * MAXATOMS + jAtom;
                    
                    real_t dx = irx - sim.atoms.r.x[jOff];
                    real_t dy = iry - sim.atoms.r.y[jOff];
                    real_t dz = irz - sim.atoms.r.z[jOff];
                    
                    real_t r2 = dx*dx + dy*dy + dz*dz;
                    
                    if (r2 <= rCut2 && r2 > 0.0) {
                        real_t r = sycl::sqrt(r2);
                        real_t phiTmp, dPhi, rhoTmp, dRho;
                        
                        if (!spline) {
                            interpolate(sim.eam_pot.phi, r, phiTmp, dPhi);
                            interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
                            dPhi /= r;
                        } else {
                            interpolateSpline(sim.eam_pot.phiS, r2, phiTmp, dPhi);
                            interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
                        }
                        
                        ifx -= dPhi * dx;
                        ify -= dPhi * dy;
                        ifz -= dPhi * dz;
                        ie += phiTmp;
                        irho += rhoTmp;
                    }
                }
            }
            
            sim.atoms.f.x[iOff] = ifx;
            sim.atoms.f.y[iOff] = ify;
            sim.atoms.f.z[iOff] = ifz;
            sim.atoms.e[iOff] = 0.5 * ie;
            sim.eam_pot.rhobar[iOff] = irho;
        }
    );
    
    SYCL_GET_LAST_ERROR
}

void eamForce2GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, int spline)
{
    eamForce2Gpu(sim, method, spline);
}

void eamForce3GpuAsync(SimGpu sim, AtomListGpu atoms_list, int num_cells, int *cells_list, int method, int spline)
{
    if (atoms_list.n == 0) return;
    
    int n = atoms_list.n;
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;
            
            int iAtom = atoms_list.atoms[tid];
            int iBox = atoms_list.cells[tid];
            int iOff = iBox * MAXATOMS + iAtom;
            
            real_t rCut = sim.eam_pot.cutoff;
            real_t rCut2 = rCut * rCut;
            
            real_t ifx = sim.atoms.f.x[iOff];
            real_t ify = sim.atoms.f.y[iOff];
            real_t ifz = sim.atoms.f.z[iOff];
            
            real_t irx = sim.atoms.r.x[iOff];
            real_t iry = sim.atoms.r.y[iOff];
            real_t irz = sim.atoms.r.z[iOff];
            
            for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
                int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
                
                for (int jAtom = 0; jAtom < sim.boxes.nAtoms[jBox]; jAtom++) {
                    int jOff = jBox * MAXATOMS + jAtom;
                    
                    real_t dx = irx - sim.atoms.r.x[jOff];
                    real_t dy = iry - sim.atoms.r.y[jOff];
                    real_t dz = irz - sim.atoms.r.z[jOff];
                    
                    real_t r2 = dx*dx + dy*dy + dz*dz;
                    
                    if (r2 <= rCut2 && r2 > 0.0) {
                        real_t rhoTmp, dRho, dPhi;
                        
                        if (!spline) {
                            real_t r = sycl::sqrt(r2);
                            interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
                            dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
                            dPhi /= r;
                        } else {
                            interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
                            dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
                        }
                        
                        ifx -= dPhi * dx;
                        ify -= dPhi * dy;
                        ifz -= dPhi * dz;
                    }
                }
            }
            
            sim.atoms.f.x[iOff] = ifx;
            sim.atoms.f.y[iOff] = ify;
            sim.atoms.f.z[iOff] = ifz;
        }
    );
    
    SYCL_GET_LAST_ERROR
}

//=============================================================================
// Link cell update operations - Full Implementation
//=============================================================================

void updateLinkCellsGpu(SimFlat *sim)
{
    int n = sim->gpu.a_list.n;
    if (n == 0) return;
    
    int wg_size = THREAD_ATOM_CTA;
    int num_wg = (n + wg_size - 1) / wg_size;
    
    SimGpu sim_gpu = sim->gpu;
    int* flags = sim->flags;
    int nLocalBoxes = sim->boxes->nLocalBoxes;
    int nTotalBoxes = sim->boxes->nTotalBoxes;
    int usePairlist = sim->usePairlist;
    
    // Zero out flags for all cells
    g_sycl_queue->memset(flags, 0, nTotalBoxes * MAXATOMS * sizeof(int)).wait();
    
    // Update link cells kernel - moves particles to new boxes if needed
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(num_wg * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= n) return;
            
            int iAtom = sim_gpu.a_list.atoms[tid];
            int iBox = sim_gpu.a_list.cells[tid];
            int iOff = iBox * MAXATOMS + iAtom;
            
            int jBox = getBoxFromCoord_dev(sim_gpu.boxes, 
                                           sim_gpu.atoms.r.x[iOff], 
                                           sim_gpu.atoms.r.y[iOff], 
                                           sim_gpu.atoms.r.z[iOff]);
            
            if (jBox != iBox) {
                // Find new position in jBox
                int jAtom = atomicAdd_int(&sim_gpu.boxes.nAtoms[jBox], 1);
                int jOff = jBox * MAXATOMS + jAtom;
                
                flags[jOff] = tid + 1;
                flags[iOff] = 0;
                
                // Copy atom data
                sim_gpu.atoms.r.x[jOff] = sim_gpu.atoms.r.x[iOff];
                sim_gpu.atoms.r.y[jOff] = sim_gpu.atoms.r.y[iOff];
                sim_gpu.atoms.r.z[jOff] = sim_gpu.atoms.r.z[iOff];
                sim_gpu.atoms.p.x[jOff] = sim_gpu.atoms.p.x[iOff];
                sim_gpu.atoms.p.y[jOff] = sim_gpu.atoms.p.y[iOff];
                sim_gpu.atoms.p.z[jOff] = sim_gpu.atoms.p.z[iOff];
                sim_gpu.atoms.gid[jOff] = sim_gpu.atoms.gid[iOff];
                sim_gpu.atoms.iSpecies[jOff] = sim_gpu.atoms.iSpecies[iOff];
                
                if (usePairlist) {
                    sim_gpu.atoms.neighborList.lastR.x[jOff] = sim_gpu.atoms.neighborList.lastR.x[iOff];
                    sim_gpu.atoms.neighborList.lastR.y[jOff] = sim_gpu.atoms.neighborList.lastR.y[iOff];
                    sim_gpu.atoms.neighborList.lastR.z[jOff] = sim_gpu.atoms.neighborList.lastR.z[iOff];
                }
                
                sim_gpu.a_list.atoms[tid] = jAtom;
                sim_gpu.a_list.cells[tid] = jBox;
            } else {
                flags[iOff] = tid + 1;
            }
        }
    );
    g_sycl_queue->wait();
    
    // Compact atoms kernel - one thread per cell
    int num_cells = nLocalBoxes;
    int grid = (num_cells + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(grid * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= num_cells) return;
            
            int iBox = tid;
            int jAtom = 0;
            
            for (int iAtom = 0; iAtom < MAXATOMS; iAtom++) {
                int iOff = iBox * MAXATOMS + iAtom;
                if (flags[iOff] > 0) {
                    int jOff = iBox * MAXATOMS + jAtom;
                    if (iOff != jOff) {
                        sim_gpu.atoms.r.x[jOff] = sim_gpu.atoms.r.x[iOff];
                        sim_gpu.atoms.r.y[jOff] = sim_gpu.atoms.r.y[iOff];
                        sim_gpu.atoms.r.z[jOff] = sim_gpu.atoms.r.z[iOff];
                        sim_gpu.atoms.p.x[jOff] = sim_gpu.atoms.p.x[iOff];
                        sim_gpu.atoms.p.y[jOff] = sim_gpu.atoms.p.y[iOff];
                        sim_gpu.atoms.p.z[jOff] = sim_gpu.atoms.p.z[iOff];
                        sim_gpu.atoms.gid[jOff] = sim_gpu.atoms.gid[iOff];
                        sim_gpu.atoms.iSpecies[jOff] = sim_gpu.atoms.iSpecies[iOff];
                        if (usePairlist) {
                            sim_gpu.atoms.neighborList.lastR.x[jOff] = sim_gpu.atoms.neighborList.lastR.x[iOff];
                            sim_gpu.atoms.neighborList.lastR.y[jOff] = sim_gpu.atoms.neighborList.lastR.y[iOff];
                            sim_gpu.atoms.neighborList.lastR.z[jOff] = sim_gpu.atoms.neighborList.lastR.z[iOff];
                        }
                    }
                    jAtom++;
                }
            }
            sim_gpu.boxes.nAtoms[iBox] = jAtom;
        }
    );
    g_sycl_queue->wait();
    
    // Rebuild atom lists
    buildAtomListGpu(sim);
    
    SYCL_GET_LAST_ERROR
}

void sortAtomsGpu(SimFlat *sim)
{
    int nLocalBoxes = sim->boxes->nLocalBoxes;
    int nTotalBoxes = sim->boxes->nTotalBoxes;
    int* new_indices = sim->flags;
    int* tmp_sort = sim->tmp_sort;
    
    // Set all indices to -1
    g_sycl_queue->memset(new_indices, 255, nTotalBoxes * MAXATOMS * sizeof(int)).wait();
    
    SimGpu sim_gpu = sim->gpu;
    int n_boundary_cells = sim->n_boundary1_cells;
    int* boundary_cells = sim->boundary1_cells_d;
    
    // Set linear indices for boundary cells
    int wg_size = THREAD_ATOM_CTA;
    int total_warps = n_boundary_cells;
    int grid = (total_warps * SUB_GROUP_SIZE + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(grid * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            int warp_id = tid / SUB_GROUP_SIZE;
            int lane_id = tid % SUB_GROUP_SIZE;
            
            if (warp_id >= n_boundary_cells) return;
            
            int iBox = boundary_cells[warp_id];
            int num_atoms = sim_gpu.boxes.nAtoms[iBox];
            for (int iAtom = lane_id; iAtom < num_atoms; iAtom += SUB_GROUP_SIZE) {
                int iOff = iBox * MAXATOMS + iAtom;
                new_indices[iOff] = iOff;
            }
        }
    );
    
    // Set linear indices for halo cells
    int nHaloCells = nTotalBoxes - nLocalBoxes;
    grid = (nHaloCells * MAXATOMS + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(grid * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            int iBox = nLocalBoxes + tid / MAXATOMS;
            int iAtom = tid % MAXATOMS;
            
            if (iBox < nTotalBoxes && iAtom < sim_gpu.boxes.nAtoms[iBox]) {
                int iOff = iBox * MAXATOMS + iAtom;
                new_indices[iOff] = iOff;
            }
        }
    );
    
    // Sort atoms by global ID using simple bubble sort for small arrays
    int total_cells = n_boundary_cells + nHaloCells;
    grid = (total_cells + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(grid * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            
            int iBox;
            if (tid >= n_boundary_cells)
                iBox = nLocalBoxes + tid - n_boundary_cells;
            else
                iBox = boundary_cells[tid];
            
            if (iBox >= nTotalBoxes || new_indices[iBox * MAXATOMS] < 0) return;
            
            int n = sim_gpu.boxes.nAtoms[iBox];
            int* A = new_indices + iBox * MAXATOMS;
            int* B = tmp_sort + iBox * MAXATOMS;
            
            // Copy indices
            for (int i = 0; i < n; i++) B[i] = A[i];
            
            // Simple bubble sort by global ID
            for (int i = 0; i < n - 1; i++) {
                for (int j = 0; j < n - i - 1; j++) {
                    if (sim_gpu.atoms.gid[B[j]] > sim_gpu.atoms.gid[B[j + 1]]) {
                        int temp = B[j];
                        B[j] = B[j + 1];
                        B[j + 1] = temp;
                    }
                }
            }
        }
    );
    g_sycl_queue->wait();
    
    // Shuffle atoms data based on sorted indices
    total_warps = n_boundary_cells + nHaloCells;
    grid = (total_warps * SUB_GROUP_SIZE + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(grid * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            int warp_id = tid / SUB_GROUP_SIZE;
            int lane_id = tid % SUB_GROUP_SIZE;
            
            int iBox;
            if (warp_id >= n_boundary_cells)
                iBox = nLocalBoxes + warp_id - n_boundary_cells;
            else
                iBox = boundary_cells[warp_id];
            
            if (iBox >= nTotalBoxes || new_indices[iBox * MAXATOMS] < 0) return;
            
            int iAtom = lane_id;
            int iOff = iBox * MAXATOMS + iAtom;
            
            if (iAtom < sim_gpu.boxes.nAtoms[iBox]) {
                int srcIdx = tmp_sort[iOff];
                
                // Read from source
                real_t rx = sim_gpu.atoms.r.x[srcIdx];
                real_t ry = sim_gpu.atoms.r.y[srcIdx];
                real_t rz = sim_gpu.atoms.r.z[srcIdx];
                real_t px = sim_gpu.atoms.p.x[srcIdx];
                real_t py = sim_gpu.atoms.p.y[srcIdx];
                real_t pz = sim_gpu.atoms.p.z[srcIdx];
                int species = sim_gpu.atoms.iSpecies[srcIdx];
                
                // Write to destination (use group barrier to ensure reads complete)
                item.barrier(sycl::access::fence_space::global_space);
                
                sim_gpu.atoms.r.x[iOff] = rx;
                sim_gpu.atoms.r.y[iOff] = ry;
                sim_gpu.atoms.r.z[iOff] = rz;
                sim_gpu.atoms.p.x[iOff] = px;
                sim_gpu.atoms.p.y[iOff] = py;
                sim_gpu.atoms.p.z[iOff] = pz;
                sim_gpu.atoms.iSpecies[iOff] = species;
            }
        }
    );
    g_sycl_queue->wait();
    
    SYCL_GET_LAST_ERROR
}

//=============================================================================
// Buffer operations for halo exchange - Full Implementation
//=============================================================================

int compactCellsGpu(char* work_d, int nCells, int *d_cellList, SimGpu sim_gpu, int* d_cellOffsets, int * d_workScan, real3_old shift)
{
    // Compact cells into packed buffer for halo exchange
    int wg_size = THREAD_ATOM_CTA;
    
    // Count atoms per cell using prefix sums
    int* h_cellOffsets = (int*)sycl::malloc_host((nCells + 1) * sizeof(int), *g_sycl_queue);
    
    // First pass: count atoms per cell
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(nCells, 1),
        [=](sycl::nd_item<1> item) {
            int i = item.get_global_id(0);
            if (i < nCells) {
                int iBox = d_cellList[i];
                d_cellOffsets[i] = sim_gpu.boxes.nAtoms[iBox];
            }
        }
    ).wait();
    
    // Prefix sum on host (for simplicity)
    g_sycl_queue->memcpy(h_cellOffsets, d_cellOffsets, nCells * sizeof(int)).wait();
    
    int total = 0;
    for (int i = 0; i < nCells; i++) {
        int count = h_cellOffsets[i];
        h_cellOffsets[i] = total;
        total += count;
    }
    h_cellOffsets[nCells] = total;
    
    g_sycl_queue->memcpy(d_cellOffsets, h_cellOffsets, (nCells + 1) * sizeof(int)).wait();
    
    // Pack atoms into buffer (SoA format)
    int msgSize = 6 * sizeof(real_t) + sizeof(int);  // rx,ry,rz,px,py,pz,type
    AtomMsgSoA atomMsg;
    getAtomMsgSoAPtr(work_d, &atomMsg, total);
    
    int grid = (nCells * MAXATOMS + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(grid * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            int iCell = tid / MAXATOMS;
            int iAtom = tid % MAXATOMS;
            
            if (iCell >= nCells) return;
            
            int iBox = d_cellList[iCell];
            if (iAtom >= sim_gpu.boxes.nAtoms[iBox]) return;
            
            int iOff = iBox * MAXATOMS + iAtom;
            int bufIdx = d_cellOffsets[iCell] + iAtom;
            
            atomMsg.rx[bufIdx] = sim_gpu.atoms.r.x[iOff] - shift[0];
            atomMsg.ry[bufIdx] = sim_gpu.atoms.r.y[iOff] - shift[1];
            atomMsg.rz[bufIdx] = sim_gpu.atoms.r.z[iOff] - shift[2];
            atomMsg.px[bufIdx] = sim_gpu.atoms.p.x[iOff];
            atomMsg.py[bufIdx] = sim_gpu.atoms.p.y[iOff];
            atomMsg.pz[bufIdx] = sim_gpu.atoms.p.z[iOff];
        }
    );
    g_sycl_queue->wait();
    
    sycl::free(h_cellOffsets, *g_sycl_queue);
    
    SYCL_GET_LAST_ERROR
    return total * msgSize;
}

void unloadAtomsBufferToGpu(char *buf, int nBuf, SimFlat *s, char *gpu_buf)
{
    // Unload atoms from buffer to GPU halo cells
    if (nBuf <= 0) return;
    
    SimGpu sim_gpu = s->gpu;
    int nLocalBoxes = s->boxes->nLocalBoxes;
    int nTotalBoxes = s->boxes->nTotalBoxes;
    int msgSize = 6 * sizeof(real_t) + sizeof(int);
    int nAtoms = nBuf / msgSize;
    
    // Copy buffer to GPU
    g_sycl_queue->memcpy(gpu_buf, buf, nBuf).wait();
    
    AtomMsgSoA atomMsg;
    getAtomMsgSoAPtr(gpu_buf, &atomMsg, nAtoms);
    
    int wg_size = THREAD_ATOM_CTA;
    int grid = (nAtoms + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(grid * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= nAtoms) return;
            
            real_t rx = atomMsg.rx[tid];
            real_t ry = atomMsg.ry[tid];
            real_t rz = atomMsg.rz[tid];
            
            // Find target box based on position
            int iBox = getBoxFromCoord_dev(sim_gpu.boxes, rx, ry, rz);
            if (iBox < nLocalBoxes || iBox >= nTotalBoxes) return;
            
            // Add atom to box
            int iAtom = atomicAdd_int(&sim_gpu.boxes.nAtoms[iBox], 1);
            if (iAtom >= MAXATOMS) return;
            
            int iOff = iBox * MAXATOMS + iAtom;
            
            sim_gpu.atoms.r.x[iOff] = rx;
            sim_gpu.atoms.r.y[iOff] = ry;
            sim_gpu.atoms.r.z[iOff] = rz;
            sim_gpu.atoms.p.x[iOff] = atomMsg.px[tid];
            sim_gpu.atoms.p.y[iOff] = atomMsg.py[tid];
            sim_gpu.atoms.p.z[iOff] = atomMsg.pz[tid];
        }
    );
    g_sycl_queue->wait();
    
    SYCL_GET_LAST_ERROR
}

void loadForceBufferFromGpu(char *buf, int *nbuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf)
{
    // Load force (dfEmbed) from GPU cells into packed buffer
    SimGpu sim_gpu = s->gpu;
    
    int wg_size = THREAD_ATOM_CTA;
    
    // Count atoms per cell
    int* h_natoms = (int*)sycl::malloc_host(nCells * sizeof(int), *g_sycl_queue);
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(nCells, 1),
        [=](sycl::nd_item<1> item) {
            int i = item.get_global_id(0);
            if (i < nCells) {
                int iBox = cellList[i];
                natoms_buf[i] = sim_gpu.boxes.nAtoms[iBox];
            }
        }
    ).wait();
    
    g_sycl_queue->memcpy(h_natoms, natoms_buf, nCells * sizeof(int)).wait();
    
    // Prefix sum
    int total = 0;
    for (int i = 0; i < nCells; i++) {
        partial_sums[i] = total;
        total += h_natoms[i];
    }
    *nbuf = total * sizeof(ForceMsg);
    
    g_sycl_queue->memcpy(gpu_buf, partial_sums, nCells * sizeof(int)).wait();
    int* d_offsets = (int*)gpu_buf;
    
    // Pack forces into buffer
    ForceMsg* force_buf = (ForceMsg*)(buf);
    
    int grid = (nCells * MAXATOMS + wg_size - 1) / wg_size;
    
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(grid * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            int iCell = tid / MAXATOMS;
            int iAtom = tid % MAXATOMS;
            
            if (iCell >= nCells) return;
            
            int iBox = cellList[iCell];
            if (iAtom >= sim_gpu.boxes.nAtoms[iBox]) return;
            
            int iOff = iBox * MAXATOMS + iAtom;
            int bufIdx = d_offsets[iCell] + iAtom;
            
            force_buf[bufIdx].gid = sim_gpu.atoms.gid[iOff];
            force_buf[bufIdx].dfEmbed = sim_gpu.eam_pot.dfEmbed[iOff];
        }
    );
    g_sycl_queue->wait();
    
    sycl::free(h_natoms, *g_sycl_queue);
    
    SYCL_GET_LAST_ERROR
}

void unloadForceBufferToGpu(char *buf, int nBuf, int nCells, int *cellList, int *natoms_buf, int *partial_sums, SimFlat *s, char *gpu_buf)
{
    // Unload force (dfEmbed) from buffer to GPU halo cells
    if (nBuf <= 0) return;
    
    int nAtoms = nBuf / sizeof(ForceMsg);
    SimGpu sim_gpu = s->gpu;
    int nLocalBoxes = s->boxes->nLocalBoxes;
    
    // Copy buffer to GPU
    ForceMsg* d_force_buf = (ForceMsg*)gpu_buf;
    g_sycl_queue->memcpy(d_force_buf, buf, nBuf).wait();
    
    int wg_size = THREAD_ATOM_CTA;
    int grid = (nAtoms + wg_size - 1) / wg_size;
    
    // Match by global ID and update dfEmbed
    g_sycl_queue->parallel_for(
        sycl::nd_range<1>(grid * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            if (tid >= nAtoms) return;
            
            int gid = d_force_buf[tid].gid;
            real_t dfEmbed = d_force_buf[tid].dfEmbed;
            
            // Search in halo cells by matching gid
            for (int i = 0; i < nCells; i++) {
                int iBox = cellList[i];
                for (int iAtom = 0; iAtom < sim_gpu.boxes.nAtoms[iBox]; iAtom++) {
                    int iOff = iBox * MAXATOMS + iAtom;
                    if (sim_gpu.atoms.gid[iOff] == gid) {
                        sim_gpu.eam_pot.dfEmbed[iOff] = dfEmbed;
                        return;
                    }
                }
            }
        }
    );
    g_sycl_queue->wait();
    
    SYCL_GET_LAST_ERROR
}

void getAtomMsgSoAPtr(char* const buffer, AtomMsgSoA *atomMsg, int n)
{
    // Set up SoA pointers for atom messages
    atomMsg->rx = (real_t*)buffer;
    atomMsg->ry = atomMsg->rx + n;
    atomMsg->rz = atomMsg->ry + n;
    atomMsg->px = atomMsg->rz + n;
    atomMsg->py = atomMsg->px + n;
    atomMsg->pz = atomMsg->py + n;
}

int pairlistUpdateRequiredGpu(SimGpu* sim)
{
    return neighborListUpdateRequiredGpu(sim);
}

void updateNeighborsGpuAsync(SimGpu sim, int *temp, int nCells, int *cellList)
{
    updateNeighborsGpu(sim, temp);
}
