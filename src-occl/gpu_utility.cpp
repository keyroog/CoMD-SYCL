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

#include <sycl/sycl.hpp>
#include <assert.h>

#include "defines.h"
#include "gpu_utility.h"
#include "gpu_neighborList.h"
#include "gpu_kernels.h"

// Global SYCL queue
sycl::queue* g_sycl_queue = nullptr;

void syclCopyDtH(void* dst, const void* src, int size)
{
   g_sycl_queue->memcpy(dst, src, size).wait();
}

void SetupGpu(int deviceId)
{
    // Get all GPU devices
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    
    if (devices.empty()) {
        fprintf(stderr, "No GPU devices found, falling back to default selector\n");
        g_sycl_queue = new sycl::queue(sycl::default_selector_v);
    } else {
        // Select device by ID
        int selectedDevice = (deviceId < (int)devices.size()) ? deviceId : 0;
        g_sycl_queue = new sycl::queue(devices[selectedDevice]);
    }
    
    auto device = g_sycl_queue->get_device();
    auto deviceName = device.get_info<sycl::info::device::name>();
    
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    printf("Host %s using GPU %i: %s\n\n", hostname, deviceId, deviceName.c_str());
    
    // Print additional device info
    printf("  Max compute units: %u\n", 
           device.get_info<sycl::info::device::max_compute_units>());
    printf("  Max work group size: %lu\n", 
           device.get_info<sycl::info::device::max_work_group_size>());
    printf("  Global memory size: %lu MB\n", 
           device.get_info<sycl::info::device::global_mem_size>() / (1024*1024));
    printf("  Local memory size: %lu KB\n", 
           device.get_info<sycl::info::device::local_mem_size>() / 1024);
}

// Helper function to allocate USM device memory
template<typename T>
T* sycl_malloc_device(size_t count) {
    return sycl::malloc_device<T>(count, *g_sycl_queue);
}

// Helper function to allocate USM shared memory
template<typename T>
T* sycl_malloc_shared(size_t count) {
    return sycl::malloc_shared<T>(count, *g_sycl_queue);
}

// Helper function to free USM memory
template<typename T>
void sycl_free(T* ptr) {
    if (ptr) sycl::free(ptr, *g_sycl_queue);
}

// input is haloExchange structure for forces
// this function sets the following static GPU arrays:
//   gpu.cell_type - 0 if interior, 1 if boundary (assuming 2-rings: corresponding to boundary/interior)
//   n_boundary_cells - number of 2-ring boundary cells
//   n_boundary1_cells - number of immediate boundary cells (1 ring)
//   boundary_cells - list of boundary cells ids (2 rings)
//   interior_cells - list of interior cells ids (w/o 2 rings)
//   boundary1_cells - list of immediate boundary cells ids (1 ring)
void SetBoundaryCells(SimFlat *flat, HaloExchange *hh)
{
    int nLocalBoxes = flat->boxes->nLocalBoxes;
    flat->boundary1_cells_h = (int*)malloc(nLocalBoxes * sizeof(int)); 
    int *h_boundary_cells = (int*)malloc(nLocalBoxes * sizeof(int)); 
    int *h_cell_type = (int*)malloc(nLocalBoxes * sizeof(int));
    memset(h_cell_type, 0, nLocalBoxes * sizeof(int));

    // gather data to a single list, set cell type
    int n = 0;
    ForceExchangeParms *parms = (ForceExchangeParms*)hh->parms;
    for (int ii=0; ii<6; ++ii) {
        int *cellList = parms->sendCells[ii];               
        for (int j = 0; j < parms->nCells[ii]; j++) 
            if (cellList[j] < nLocalBoxes && h_cell_type[cellList[j]] == 0) {
                flat->boundary1_cells_h[n] = cellList[j];
                h_boundary_cells[n] = cellList[j];
                h_cell_type[cellList[j]] = 1;
                n++;
            }
    }

    flat->n_boundary1_cells = n;
    int n_boundary1_cells = n;

    // find 2nd ring
    int neighbor_cells[N_MAX_NEIGHBORS];
    for (int i = 0; i < nLocalBoxes; i++)
        if (h_cell_type[i] == 0) {
            getNeighborBoxes(flat->boxes, i, neighbor_cells);
            for (int j = 0; j < N_MAX_NEIGHBORS; j++)
                if (h_cell_type[neighbor_cells[j]] == 1) {  
                    // found connection to the boundary node - add to the list
                    h_boundary_cells[n] = i;
                    h_cell_type[i] = 2;
                    n++;
                    break;
                }
        }

    flat->n_boundary_cells = n;
    int n_boundary_cells = n;

    int n_interior_cells = flat->boxes->nLocalBoxes - n;

    // find interior cells
    int *h_interior_cells = (int*)malloc(n_interior_cells * sizeof(int));
    n = 0;
    for (int i = 0; i < nLocalBoxes; i++) {
        if (h_cell_type[i] == 0) {
            h_interior_cells[n] = i;
            n++;
        }
        else if (h_cell_type[i] == 2) {
            h_cell_type[i] = 1;
        }
    }

    // allocate on GPU using USM
    flat->boundary1_cells_d = sycl::malloc_device<int>(n_boundary1_cells, *g_sycl_queue);
    flat->boundary_cells = sycl::malloc_device<int>(n_boundary_cells, *g_sycl_queue);
    flat->interior_cells = sycl::malloc_device<int>(n_interior_cells, *g_sycl_queue);

    // copy to GPU  
    g_sycl_queue->memcpy(flat->boundary1_cells_d, flat->boundary1_cells_h, n_boundary1_cells * sizeof(int));
    g_sycl_queue->memcpy(flat->boundary_cells, h_boundary_cells, n_boundary_cells * sizeof(int));
    g_sycl_queue->memcpy(flat->interior_cells, h_interior_cells, n_interior_cells * sizeof(int));

    // set cell types
    flat->gpu.cell_type = sycl::malloc_device<int>(nLocalBoxes, *g_sycl_queue);
    g_sycl_queue->memcpy(flat->gpu.cell_type, h_cell_type, nLocalBoxes * sizeof(int));
    
    g_sycl_queue->wait();

    free(h_boundary_cells);
    free(h_cell_type);
}

void AllocateGpu(SimFlat *sim, int do_eam, real_t skinDistance)
{
    SimGpu *gpu = &sim->gpu;

    int total_boxes = sim->boxes->nTotalBoxes;
    int nLocalBoxes = sim->boxes->nLocalBoxes;
    int num_species = 1;

    // allocate positions, momentum, forces & energies using USM
    int r_size = total_boxes * MAXATOMS;
    int f_size = nLocalBoxes * MAXATOMS;

    gpu->atoms.r.x = sycl::malloc_device<real_t>(r_size, *g_sycl_queue);
    gpu->atoms.r.y = sycl::malloc_device<real_t>(r_size, *g_sycl_queue);
    gpu->atoms.r.z = sycl::malloc_device<real_t>(r_size, *g_sycl_queue);

    gpu->atoms.p.x = sycl::malloc_device<real_t>(r_size, *g_sycl_queue);
    gpu->atoms.p.y = sycl::malloc_device<real_t>(r_size, *g_sycl_queue);
    gpu->atoms.p.z = sycl::malloc_device<real_t>(r_size, *g_sycl_queue);

    gpu->atoms.f.x = sycl::malloc_device<real_t>(f_size, *g_sycl_queue);
    gpu->atoms.f.y = sycl::malloc_device<real_t>(f_size, *g_sycl_queue);
    gpu->atoms.f.z = sycl::malloc_device<real_t>(f_size, *g_sycl_queue);

    gpu->atoms.e = sycl::malloc_device<real_t>(f_size, *g_sycl_queue);
    gpu->d_updateLinkCellsRequired = sycl::malloc_device<int>(1, *g_sycl_queue);
    g_sycl_queue->memset(gpu->d_updateLinkCellsRequired, 0, sizeof(int));

    gpu->atoms.gid = sycl::malloc_device<int>(total_boxes * MAXATOMS, *g_sycl_queue);

    // species data
    gpu->atoms.iSpecies = sycl::malloc_device<int>(total_boxes * MAXATOMS, *g_sycl_queue);
    gpu->species_mass = sycl::malloc_device<real_t>(num_species, *g_sycl_queue);

    // allocate indices, neighbors, etc.
    gpu->neighbor_cells = sycl::malloc_device<int>(nLocalBoxes * N_MAX_NEIGHBORS, *g_sycl_queue);
    gpu->neighbor_atoms = sycl::malloc_device<int>(nLocalBoxes * N_MAX_NEIGHBORS * MAXATOMS, *g_sycl_queue);
    gpu->num_neigh_atoms = sycl::malloc_device<int>(nLocalBoxes, *g_sycl_queue);

    // total # of atoms in local boxes
    int n = 0;
    for (int iBox=0; iBox < sim->boxes->nLocalBoxes; iBox++)
        n += sim->boxes->nAtoms[iBox];
    gpu->a_list.n = n;
    gpu->a_list.atoms = sycl::malloc_device<int>(n, *g_sycl_queue);
    gpu->a_list.cells = sycl::malloc_device<int>(n, *g_sycl_queue);

    // allocate other lists as well
    gpu->i_list.atoms = sycl::malloc_device<int>(n, *g_sycl_queue);
    gpu->i_list.cells = sycl::malloc_device<int>(n, *g_sycl_queue);
    gpu->b_list.atoms = sycl::malloc_device<int>(n, *g_sycl_queue);
    gpu->b_list.cells = sycl::malloc_device<int>(n, *g_sycl_queue);

    initNeighborListGpu(gpu, &(gpu->atoms.neighborList), nLocalBoxes, skinDistance);
    initLinkCellsGpu(sim, &(gpu->boxes));

    int nMaxHaloParticles = (sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes)*MAXATOMS;
    initHashTableGpu(&(gpu->d_hashTable), 2*nMaxHaloParticles);

    // Allocate pairlist
    if(sim->usePairlist) { 
        gpu->pairlist = sycl::malloc_device<int>(
            nLocalBoxes * MAXATOMS/WARP_SIZE*N_MAX_NEIGHBORS * 
            (MAXATOMS + PAIRLIST_ATOMS_PER_INT-1)/PAIRLIST_ATOMS_PER_INT, *g_sycl_queue);
    }

    // init EAM arrays
    if (do_eam) {
        EamPotential* pot = (EamPotential*) sim->pot;
        
        gpu->eam_pot.f.values = sycl::malloc_device<real_t>(pot->f->n+3, *g_sycl_queue);
        if(!sim->spline) {
            gpu->eam_pot.rho.values = sycl::malloc_device<real_t>(pot->rho->n+3, *g_sycl_queue);
            gpu->eam_pot.phi.values = sycl::malloc_device<real_t>(pot->phi->n+3, *g_sycl_queue);
        } else {
            gpu->eam_pot.fS.coefficients = sycl::malloc_device<real_t>(4*pot->f->n, *g_sycl_queue);
            gpu->eam_pot.rhoS.coefficients = sycl::malloc_device<real_t>(4*pot->rho->n, *g_sycl_queue);
            gpu->eam_pot.phiS.coefficients = sycl::malloc_device<real_t>(4*pot->phi->n, *g_sycl_queue);
        }
        gpu->eam_pot.dfEmbed = sycl::malloc_device<real_t>(r_size, *g_sycl_queue);
        gpu->eam_pot.rhobar = sycl::malloc_device<real_t>(r_size, *g_sycl_queue);
    } else {
        // init LJ interpolation table
        LjPotential * pot = (LjPotential*) sim->pot;
        gpu->lj_pot.lj_interpolation.values = sycl::malloc_device<real_t>(1003, *g_sycl_queue);
    }

    // initialize host data as well
    SimGpu *host = &sim->host;
    
    host->atoms.r.x=NULL; host->atoms.r.y=NULL; host->atoms.r.z=NULL;
    host->atoms.f.x=NULL; host->atoms.f.y=NULL; host->atoms.f.z=NULL;
    host->atoms.p.x=NULL; host->atoms.p.y=NULL; host->atoms.p.z=NULL;
    host->atoms.e=NULL;

    host->neighbor_cells = (int*)malloc(nLocalBoxes * N_MAX_NEIGHBORS * sizeof(int));
    host->neighbor_atoms = (int*)malloc(nLocalBoxes * N_MAX_NEIGHBORS * MAXATOMS * sizeof(int));
    host->num_neigh_atoms = (int*)malloc(nLocalBoxes * sizeof(int));

    // on host allocate list of all local atoms only
    host->a_list.atoms = (int*)malloc(n * sizeof(int));
    host->a_list.cells = (int*)malloc(n * sizeof(int));

    // temp arrays
    sim->flags = sycl::malloc_device<int>(sim->boxes->nTotalBoxes * MAXATOMS, *g_sycl_queue);
    sim->tmp_sort = sycl::malloc_device<int>(sim->boxes->nTotalBoxes * MAXATOMS, *g_sycl_queue);
    sim->gpu_atoms_buf = (char*)sycl::malloc_device<char>(sim->boxes->nTotalBoxes * MAXATOMS * sizeof(AtomMsg), *g_sycl_queue);
    sim->gpu_force_buf = (char*)sycl::malloc_device<char>(sim->boxes->nTotalBoxes * MAXATOMS * sizeof(ForceMsg), *g_sycl_queue);
    
    g_sycl_queue->wait();
}

void DestroyGpu(SimFlat *flat)
{
    SimGpu *gpu = &flat->gpu;
    SimGpu *host = &flat->host;

    sycl::free(gpu->d_updateLinkCellsRequired, *g_sycl_queue);
    sycl::free(gpu->atoms.r.x, *g_sycl_queue);
    sycl::free(gpu->atoms.r.y, *g_sycl_queue);
    sycl::free(gpu->atoms.r.z, *g_sycl_queue);

    sycl::free(gpu->atoms.p.x, *g_sycl_queue);
    sycl::free(gpu->atoms.p.y, *g_sycl_queue);
    sycl::free(gpu->atoms.p.z, *g_sycl_queue);

    sycl::free(gpu->atoms.f.x, *g_sycl_queue);
    sycl::free(gpu->atoms.f.y, *g_sycl_queue);
    sycl::free(gpu->atoms.f.z, *g_sycl_queue);

    sycl::free(gpu->atoms.e, *g_sycl_queue);
    sycl::free(gpu->atoms.gid, *g_sycl_queue);

    sycl::free(gpu->atoms.iSpecies, *g_sycl_queue);
    sycl::free(gpu->species_mass, *g_sycl_queue);

    sycl::free(gpu->neighbor_cells, *g_sycl_queue);
    sycl::free(gpu->neighbor_atoms, *g_sycl_queue);
    sycl::free(gpu->num_neigh_atoms, *g_sycl_queue);
    sycl::free(gpu->boxes.nAtoms, *g_sycl_queue);

    sycl::free(gpu->a_list.atoms, *g_sycl_queue);
    sycl::free(gpu->a_list.cells, *g_sycl_queue);

    sycl::free(gpu->i_list.atoms, *g_sycl_queue);
    sycl::free(gpu->i_list.cells, *g_sycl_queue);

    sycl::free(gpu->b_list.atoms, *g_sycl_queue);
    sycl::free(gpu->b_list.cells, *g_sycl_queue);

    sycl::free(flat->flags, *g_sycl_queue);
    sycl::free(flat->tmp_sort, *g_sycl_queue);
    sycl::free(flat->gpu_atoms_buf, *g_sycl_queue);
    sycl::free(flat->gpu_force_buf, *g_sycl_queue);

    if (gpu->eam_pot.f.values) sycl::free(gpu->eam_pot.f.values, *g_sycl_queue);
    if (gpu->eam_pot.rho.values) sycl::free(gpu->eam_pot.rho.values, *g_sycl_queue);
    if (gpu->eam_pot.phi.values) sycl::free(gpu->eam_pot.phi.values, *g_sycl_queue);

    if (gpu->eam_pot.fS.coefficients) sycl::free(gpu->eam_pot.fS.coefficients, *g_sycl_queue);
    if (gpu->eam_pot.rhoS.coefficients) sycl::free(gpu->eam_pot.rhoS.coefficients, *g_sycl_queue);
    if (gpu->eam_pot.phiS.coefficients) sycl::free(gpu->eam_pot.phiS.coefficients, *g_sycl_queue);

    if (gpu->eam_pot.dfEmbed) sycl::free(gpu->eam_pot.dfEmbed, *g_sycl_queue);
    if (gpu->eam_pot.rhobar) sycl::free(gpu->eam_pot.rhobar, *g_sycl_queue);

    free(host->species_mass);
    free(host->neighbor_cells);
    free(host->neighbor_atoms);
    free(host->num_neigh_atoms);
    free(host->a_list.atoms);
    free(host->a_list.cells);
    
    // Destroy SYCL queue
    if (g_sycl_queue) {
        delete g_sycl_queue;
        g_sycl_queue = nullptr;
    }
}

void initLJinterpolation(LjPotentialGpu * pot)
{
    pot->lj_interpolation.x0 = 0.5 * pot->sigma;
    pot->lj_interpolation.n = 1000;
    pot->lj_interpolation.invDx = pot->lj_interpolation.n/(pot->cutoff - pot->lj_interpolation.x0);
    pot->lj_interpolation.invDxHalf = pot->lj_interpolation.invDx * 0.5;
    pot->lj_interpolation.invDxXx0 = pot->lj_interpolation.invDx * pot->lj_interpolation.x0;
    pot->lj_interpolation.xn = pot->lj_interpolation.x0 + pot->lj_interpolation.n / pot->lj_interpolation.invDx;
    
    real_t * temp = (real_t *) malloc((pot->lj_interpolation.n+3) * sizeof(real_t));
    real_t sigma = pot->sigma;
    real_t epsilon = pot->epsilon;
    real_t rCut2 = pot->cutoff * pot->cutoff;
    real_t s6 = sigma * sigma * sigma * sigma * sigma * sigma;
    real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
    real_t eShift = rCut6 * (rCut6 - 1.0);
    
    for(int i = 0; i < pot->lj_interpolation.n+3; ++i) {        
        real_t x = pot->lj_interpolation.x0 + (i-1)/pot->lj_interpolation.invDx;
        real_t r2 = 1.0/(x*x);
        real_t r6 = s6 * r2*r2*r2;
        temp[i] = 4 * epsilon * (r6 * (r6 - 1.0) - eShift);
    }
    
    g_sycl_queue->memcpy(pot->lj_interpolation.values, temp, 
                         (pot->lj_interpolation.n+3)*sizeof(real_t)).wait();

    free(temp);
}

void initSplineCoefficients(real_t* gpu_coefficients, int n, real_t* values, real_t x0, real_t invDx)
{
    real_t *u = (real_t*) malloc(n * sizeof(real_t));
    real_t *y2 = (real_t*) malloc((n+1)*sizeof(real_t));

    // Second derivative is 0 at the beginning of the interval
    y2[0] = 0;
    u[0] = 0;

    for(int i = 1; i < n; ++i)
    {
        real_t xi = (x0 + i/invDx)*(x0+i/invDx);
        real_t xp = (x0 + (i-1)/invDx)*(x0 + (i-1)/invDx);
        real_t xn = (x0 + (i+1)/invDx)*(x0 + (i+1)/invDx);

        real_t sig = (xi - xp)/(xn-xp);
        real_t p = sig*y2[i-1]+2.0;
        y2[i] = (sig-1.0)/p;
        u[i] = (values[i+1]-values[i])/(xn-xi) - (values[i]-values[i-1])/(xi-xp);
        u[i] = (6.0 * u[i]/(xn-xp)-sig*u[i-1])/p;
    }

    real_t xn = (x0 + n/invDx)*(x0 + n/invDx);
    real_t xnp = (x0 + (n-1)/invDx)*(x0 + (n-1)/invDx);
    // First derivative is 0 at the end of the interval
    real_t qn = 0.5;
    real_t un = (-3.0/(xn-xnp))*(values[n]-values[n-1])/(xn-xnp);
    y2[n] = (un-qn*u[n-1])/(qn*y2[n-1]+1.0);

    for(int i = n-1; i >= 0; --i)
    {
        y2[i] = y2[i]*y2[i+1] + u[i];
    }
    
    real_t* coefficients = (real_t*) malloc(4*n*sizeof(real_t));
    for(int i = 0; i < n; i++)
    {
        real_t x1 = (x0 + i/invDx)*(x0+i/invDx);
        real_t x2 = (x0 + (i+1)/invDx)*(x0+(i+1)/invDx);
        real_t d2y1 = y2[i];
        real_t d2y2 = y2[i+1];
        real_t y1_val = values[i];
        real_t y2_val = values[i+1];
        
        coefficients[i*4] = 1.0/(6.0*(x2-x1))*(d2y2-d2y1);
        coefficients[i*4+1] = 1.0/(2.0*(x2-x1))*(x2*d2y1-x1*d2y2);
        coefficients[i*4+2] = 1.0/(x2-x1) * (1.0/6.0*(-3*x2*x2+(x2-x1)*(x2-x1))*d2y1+1.0/6.0*(3*x1*x1-(x2-x1)*(x2-x1))*d2y2-y1_val+y2_val);
        coefficients[i*4+3] = 1/(x2-x1)*(x2*y1_val-x1*y2_val+1.0/6.0*d2y1*(x2*x2*x2-x2*(x2-x1)*(x2-x1)) + 1.0/6.0*d2y2*(-x1*x1*x1+x1*(x2-x1)*(x2-x1)));
    }
    
    g_sycl_queue->memcpy(gpu_coefficients, coefficients, 4 * n * sizeof(real_t)).wait();

    free(y2);
    free(u);
    free(coefficients);
}

int compactHaloCells(SimFlat* sim, char* h_compactAtoms, int* h_cellOffset)
{
    int nHaloCells = sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes;
    
    h_cellOffset[0] = 0;
    for(int i = 1, iBox = sim->boxes->nLocalBoxes; i <= nHaloCells; ++i, ++iBox)
    {
        h_cellOffset[i] = sim->boxes->nAtoms[iBox] + h_cellOffset[i-1];
    }
    int nTotalAtomsInHaloCells = h_cellOffset[nHaloCells];

    AtomMsgSoA msg_h;
    getAtomMsgSoAPtr(h_compactAtoms, &msg_h, nTotalAtomsInHaloCells);

    // Compact atoms from atoms struct to msg_h
    for (int ii = 0; ii < nHaloCells; ++ii)
    {
        int iOff = (sim->boxes->nLocalBoxes + ii) * MAXATOMS;
        for(int i = h_cellOffset[ii]; i < h_cellOffset[ii+1]; ++i, ++iOff)
        {
            msg_h.rx[i] = sim->atoms->r[iOff];
            msg_h.ry[i] = sim->atoms->r[sim->boxes->nTotalBoxes*MAXATOMS + iOff];
            msg_h.rz[i] = sim->atoms->r[2*sim->boxes->nTotalBoxes*MAXATOMS + iOff];

            msg_h.px[i] = sim->atoms->p[iOff];
            msg_h.py[i] = sim->atoms->p[sim->boxes->nTotalBoxes*MAXATOMS + iOff];
            msg_h.pz[i] = sim->atoms->p[2*sim->boxes->nTotalBoxes*MAXATOMS + iOff];

            msg_h.type[i] = sim->atoms->iSpecies[iOff];
            msg_h.gid[i] = sim->atoms->gid[iOff];
        }
    }
    return nTotalAtomsInHaloCells;
}

void initLinkCellsGpu(SimFlat *sim, LinkCellGpu* boxes_gpu)
{
    LinkCell *boxes = sim->boxes;
    
    boxes_gpu->nLocalBoxes = boxes->nLocalBoxes;
    boxes_gpu->nTotalBoxes = boxes->nTotalBoxes;
    
    boxes_gpu->gridSize.x = boxes->gridSize[0];
    boxes_gpu->gridSize.y = boxes->gridSize[1];
    boxes_gpu->gridSize.z = boxes->gridSize[2];
    
    boxes_gpu->localMin.x = boxes->localMin[0];
    boxes_gpu->localMin.y = boxes->localMin[1];
    boxes_gpu->localMin.z = boxes->localMin[2];
    
    boxes_gpu->localMax.x = boxes->localMax[0];
    boxes_gpu->localMax.y = boxes->localMax[1];
    boxes_gpu->localMax.z = boxes->localMax[2];
    
    boxes_gpu->invBoxSize.x = boxes->invBoxSize[0];
    boxes_gpu->invBoxSize.y = boxes->invBoxSize[1];
    boxes_gpu->invBoxSize.z = boxes->invBoxSize[2];
    
    // Allocate and copy nAtoms array
    boxes_gpu->nAtoms = sycl::malloc_device<int>(boxes->nTotalBoxes, *g_sycl_queue);
    g_sycl_queue->memcpy(boxes_gpu->nAtoms, boxes->nAtoms, 
                         boxes->nTotalBoxes * sizeof(int)).wait();
    
    // Allocate boxIDLookUp
    int lookupSize = boxes->gridSize[0] * boxes->gridSize[1] * boxes->gridSize[2];
    boxes_gpu->boxIDLookUp = sycl::malloc_device<int>(lookupSize, *g_sycl_queue);
    
    // Build lookup table on host
    int *h_lookup = (int*)malloc(lookupSize * sizeof(int));
    for (int iz = 0; iz < boxes->gridSize[2]; iz++)
        for (int iy = 0; iy < boxes->gridSize[1]; iy++)
            for (int ix = 0; ix < boxes->gridSize[0]; ix++) {
                int idx = IDX3D(ix, iy, iz, boxes->gridSize[0], boxes->gridSize[1]);
                h_lookup[idx] = getBoxFromTuple(boxes, ix, iy, iz);
            }
    
    g_sycl_queue->memcpy(boxes_gpu->boxIDLookUp, h_lookup, lookupSize * sizeof(int)).wait();
    free(h_lookup);
}

void CopyDataToGpu(SimFlat *flat, int do_eam)
{
    SimGpu *gpu = &flat->gpu;
    Atoms *atoms = flat->atoms;
    LinkCell *boxes = flat->boxes;
    
    int total_atoms = boxes->nTotalBoxes * MAXATOMS;
    int local_atoms = boxes->nLocalBoxes * MAXATOMS;
    
    // Set up EAM potential data
    if (do_eam) {
        EamPotential* pot = (EamPotential*) flat->pot;
        gpu->eam_pot.cutoff = pot->cutoff;
        
        // f is needed for second phase of EAM
        gpu->eam_pot.f.n = pot->f->n;
        gpu->eam_pot.f.x0 = pot->f->x0;
        gpu->eam_pot.f.xn = pot->f->x0 + pot->f->n / pot->f->invDx;
        gpu->eam_pot.f.invDx = pot->f->invDx;
        gpu->eam_pot.f.invDxHalf = pot->f->invDx * 0.5;
        gpu->eam_pot.f.invDxXx0 = pot->f->invDxXx0;
        g_sycl_queue->memcpy(gpu->eam_pot.f.values, pot->f->values-1, (pot->f->n+3) * sizeof(real_t));
        
        if (!flat->spline) {
            gpu->eam_pot.rho.n = pot->rho->n;
            gpu->eam_pot.phi.n = pot->phi->n;

            gpu->eam_pot.rho.x0 = pot->rho->x0;
            gpu->eam_pot.phi.x0 = pot->phi->x0;

            gpu->eam_pot.rho.xn = pot->rho->x0 + pot->rho->n / pot->rho->invDx;
            gpu->eam_pot.phi.xn = pot->phi->x0 + pot->phi->n / pot->phi->invDx;

            gpu->eam_pot.rho.invDx = pot->rho->invDx;
            gpu->eam_pot.phi.invDx = pot->phi->invDx;

            gpu->eam_pot.rho.invDxHalf = pot->rho->invDx * 0.5;
            gpu->eam_pot.phi.invDxHalf = pot->phi->invDx * 0.5;

            gpu->eam_pot.rho.invDxXx0 = pot->rho->invDxXx0;
            gpu->eam_pot.phi.invDxXx0 = pot->phi->invDxXx0;

            g_sycl_queue->memcpy(gpu->eam_pot.rho.values, pot->rho->values-1, (pot->rho->n+3) * sizeof(real_t));
            g_sycl_queue->memcpy(gpu->eam_pot.phi.values, pot->phi->values-1, (pot->phi->n+3) * sizeof(real_t));
        } else {
            gpu->eam_pot.fS.n = pot->f->n;
            gpu->eam_pot.rhoS.n = pot->rho->n;
            gpu->eam_pot.phiS.n = pot->phi->n;

            gpu->eam_pot.fS.x0 = pot->f->x0;
            gpu->eam_pot.rhoS.x0 = pot->rho->x0;
            gpu->eam_pot.phiS.x0 = pot->phi->x0;

            gpu->eam_pot.fS.xn = pot->f->x0 + pot->f->n / pot->f->invDx;
            gpu->eam_pot.rhoS.xn = pot->rho->x0 + pot->rho->n / pot->rho->invDx;
            gpu->eam_pot.phiS.xn = pot->phi->x0 + pot->phi->n / pot->phi->invDx;

            gpu->eam_pot.fS.invDx = pot->f->invDx;
            gpu->eam_pot.rhoS.invDx = pot->rho->invDx;
            gpu->eam_pot.phiS.invDx = pot->phi->invDx;

            gpu->eam_pot.fS.invDxXx0 = pot->f->invDxXx0;
            gpu->eam_pot.rhoS.invDxXx0 = pot->rho->invDxXx0;
            gpu->eam_pot.phiS.invDxXx0 = pot->phi->invDxXx0;

            initSplineCoefficients(gpu->eam_pot.fS.coefficients, pot->f->n, pot->f->values, pot->f->x0, pot->f->invDx);
            initSplineCoefficients(gpu->eam_pot.rhoS.coefficients, pot->rho->n, pot->rho->values, pot->rho->x0, pot->rho->invDx);
            initSplineCoefficients(gpu->eam_pot.phiS.coefficients, pot->phi->n, pot->phi->values, pot->phi->x0, pot->phi->invDx);
        }
    } else {
        LjPotential* pot = (LjPotential*) flat->pot;
        gpu->lj_pot.sigma = pot->sigma;
        gpu->lj_pot.epsilon = pot->epsilon;
        gpu->lj_pot.cutoff = pot->cutoff;
        initLJinterpolation(&gpu->lj_pot);
    }
    
    // Copy positions
    g_sycl_queue->memcpy(gpu->atoms.r.x, atoms->r, total_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(gpu->atoms.r.y, atoms->r + total_atoms, total_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(gpu->atoms.r.z, atoms->r + 2*total_atoms, total_atoms * sizeof(real_t));
    
    // Copy momenta
    g_sycl_queue->memcpy(gpu->atoms.p.x, atoms->p, total_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(gpu->atoms.p.y, atoms->p + total_atoms, total_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(gpu->atoms.p.z, atoms->p + 2*total_atoms, total_atoms * sizeof(real_t));
    
    // Copy species and gid
    g_sycl_queue->memcpy(gpu->atoms.iSpecies, atoms->iSpecies, total_atoms * sizeof(int));
    g_sycl_queue->memcpy(gpu->atoms.gid, atoms->gid, total_atoms * sizeof(int));
    
    // Copy species mass
    g_sycl_queue->memcpy(gpu->species_mass, flat->species->mass, sizeof(real_t));
    
    // Copy nAtoms
    g_sycl_queue->memcpy(gpu->boxes.nAtoms, boxes->nAtoms, boxes->nTotalBoxes * sizeof(int));
    
    g_sycl_queue->wait();
}

void GetDataFromGpu(SimFlat *flat)
{
    SimGpu *gpu = &flat->gpu;
    Atoms *atoms = flat->atoms;
    LinkCell *boxes = flat->boxes;
    
    int total_atoms = boxes->nTotalBoxes * MAXATOMS;
    
    // Copy back positions
    g_sycl_queue->memcpy(atoms->r, gpu->atoms.r.x, total_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(atoms->r + total_atoms, gpu->atoms.r.y, total_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(atoms->r + 2*total_atoms, gpu->atoms.r.z, total_atoms * sizeof(real_t));
    
    // Copy back momenta
    g_sycl_queue->memcpy(atoms->p, gpu->atoms.p.x, total_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(atoms->p + total_atoms, gpu->atoms.p.y, total_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(atoms->p + 2*total_atoms, gpu->atoms.p.z, total_atoms * sizeof(real_t));
    
    g_sycl_queue->wait();
}

void GetLocalAtomsFromGpu(SimFlat *flat)
{
    SimGpu *gpu = &flat->gpu;
    Atoms *atoms = flat->atoms;
    LinkCell *boxes = flat->boxes;
    
    int local_atoms = boxes->nLocalBoxes * MAXATOMS;
    
    // Copy back local positions only
    g_sycl_queue->memcpy(atoms->r, gpu->atoms.r.x, local_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(atoms->r + boxes->nTotalBoxes * MAXATOMS, gpu->atoms.r.y, local_atoms * sizeof(real_t));
    g_sycl_queue->memcpy(atoms->r + 2*boxes->nTotalBoxes * MAXATOMS, gpu->atoms.r.z, local_atoms * sizeof(real_t));
    
    g_sycl_queue->wait();
}

void updateNAtomsCpu(SimFlat* sim)
{
    g_sycl_queue->memcpy(sim->boxes->nAtoms, sim->gpu.boxes.nAtoms, 
                         sim->boxes->nTotalBoxes * sizeof(int)).wait();
}

void updateNAtomsGpu(SimFlat* sim)
{
    g_sycl_queue->memcpy(sim->gpu.boxes.nAtoms, sim->boxes->nAtoms, 
                         sim->boxes->nTotalBoxes * sizeof(int)).wait();
}

void emptyHaloCellsGpu(SimFlat* sim)
{
    int nHaloCells = sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes;
    g_sycl_queue->memset(sim->gpu.boxes.nAtoms + sim->boxes->nLocalBoxes, 
                         0, nHaloCells * sizeof(int)).wait();
}

void updateGpuHalo(SimFlat *sim)
{
    // Update halo cell data on GPU
    int nLocalBoxes = sim->boxes->nLocalBoxes;
    int nTotalBoxes = sim->boxes->nTotalBoxes;
    int nHaloBoxes = nTotalBoxes - nLocalBoxes;
    
    // Copy halo positions
    int offset = nLocalBoxes * MAXATOMS;
    int size = nHaloBoxes * MAXATOMS;
    
    g_sycl_queue->memcpy(sim->gpu.atoms.r.x + offset, sim->atoms->r + offset, size * sizeof(real_t));
    g_sycl_queue->memcpy(sim->gpu.atoms.r.y + offset, sim->atoms->r + nTotalBoxes*MAXATOMS + offset, size * sizeof(real_t));
    g_sycl_queue->memcpy(sim->gpu.atoms.r.z + offset, sim->atoms->r + 2*nTotalBoxes*MAXATOMS + offset, size * sizeof(real_t));
    
    // Update nAtoms for halo cells
    g_sycl_queue->memcpy(sim->gpu.boxes.nAtoms + nLocalBoxes, sim->boxes->nAtoms + nLocalBoxes, nHaloBoxes * sizeof(int));
    
    g_sycl_queue->wait();
}
