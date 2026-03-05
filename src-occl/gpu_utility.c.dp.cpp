#define DPCT_COMPAT_RT_VERSION 12080
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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>

#include "defines.h"
#include "gpu_utility.h"
#include "gpu_neighborList.h"

#include "gpu_kernels.h"

// fallback for old CUDA runtimes removed (not applicable to SYCL)

void cudaCopyDtH(void* dst, const void* src, int size)
{
   dpct::get_in_order_queue().memcpy(dst, src, size).wait();
}

void SetupGpu(int deviceId)
{
  /*
  DPCT1093:182: The "deviceId" device may be not the one intended for use.
  Adjust the selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(deviceId)));

  dpct::device_info props;
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_device(deviceId).get_device_info(props)));

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  printf("Host %s using GPU %i: %s\n\n", hostname, deviceId, props.get_name());
}

// input is haloExchange structure for forces
// this function sets the following static GPU arrays:
//   gpu.cell_type - 0 if interior, 1 if boundary (assuming 2-rings: corresponding to boundary/interior)
//   n_boundary_cells - number of 2-ring boundary cells
//   n_boundary1_cells - number of immediate boundary cells (1 ring)
//   boundary_cells - list of boundary cells ids (2 rings)
//   interior_cells - list of interior cells ids (w/o 2 rings)
//   boundary1_cells - list of immediate boundary cells ids (1 ring)
// also it creates necessary streams
void SetBoundaryCells(SimFlat *flat, HaloExchange *hh) try {
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

  // allocate on GPU
  CUDA_CHECK(
      DPCT_CHECK_ERROR(flat->boundary1_cells_d = sycl::malloc_device<int>(
                           n_boundary1_cells, dpct::get_in_order_queue())));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(flat->boundary_cells = sycl::malloc_device<int>(
                           n_boundary_cells, dpct::get_in_order_queue())));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(flat->interior_cells = sycl::malloc_device<int>(
                           n_interior_cells, dpct::get_in_order_queue())));

  // copy to GPU
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                  .memcpy(flat->boundary1_cells_d,
                                          flat->boundary1_cells_h,
                                          n_boundary1_cells * sizeof(int))
                                  .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(flat->boundary_cells, h_boundary_cells,
                                   n_boundary_cells * sizeof(int))
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(flat->interior_cells, h_interior_cells,
                                   n_interior_cells * sizeof(int))
                           .wait()));

  // set cell types
  CUDA_CHECK(DPCT_CHECK_ERROR(flat->gpu.cell_type = sycl::malloc_device<int>(
                                  nLocalBoxes, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(flat->gpu.cell_type, h_cell_type, nLocalBoxes * sizeof(int))
          .wait()));

  if (flat->gpuAsync) {
    // create priority & normal streams
    /*
    DPCT1025:184: The SYCL queue is created ignoring the flag and priority
    options.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(
        flat->boundary_stream =
            dpct::get_current_device().create_queue())); // set higher priority
    CUDA_CHECK(DPCT_CHECK_ERROR(flat->interior_stream =
                                    dpct::get_current_device().create_queue()));
  }
  else {
    // set streams to NULL
    flat->interior_stream = &dpct::get_in_order_queue();
    flat->boundary_stream = &dpct::get_in_order_queue();
  }

  free(h_boundary_cells);
  free(h_cell_type);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void AllocateGpu(SimFlat *sim, int do_eam, real_t skinDistance) try {
  int deviceId;
  dpct::device_info props;
  CUDA_CHECK(DPCT_CHECK_ERROR(deviceId = dpct::get_current_device_id()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_device(deviceId).get_device_info(props)));

  SimGpu *gpu = &sim->gpu;

  int total_boxes = sim->boxes->nTotalBoxes;
  int nLocalBoxes = sim->boxes->nLocalBoxes;
  int num_species = 1;

  // allocate positions, momentum, forces & energies
  int r_size = total_boxes * MAXATOMS * sizeof(real_t);
  int f_size = nLocalBoxes * MAXATOMS * sizeof(real_t);

  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.r.x = (real_t *)sycl::malloc_device(
                                  r_size, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.r.y = (real_t *)sycl::malloc_device(
                                  r_size, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.r.z = (real_t *)sycl::malloc_device(
                                  r_size, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.p.x = (real_t *)sycl::malloc_device(
                                  r_size, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.p.y = (real_t *)sycl::malloc_device(
                                  r_size, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.p.z = (real_t *)sycl::malloc_device(
                                  r_size, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.f.x = (real_t *)sycl::malloc_device(
                                  f_size, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.f.y = (real_t *)sycl::malloc_device(
                                  f_size, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.f.z = (real_t *)sycl::malloc_device(
                                  f_size, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->atoms.e = (real_t *)sycl::malloc_device(
                                  f_size, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      gpu->d_updateLinkCellsRequired =
          sycl::malloc_device<int>(1, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memset(gpu->d_updateLinkCellsRequired, 0, sizeof(int))
          .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      gpu->d_updateLinkCellsResult =
          sycl::malloc_device<int>(1, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      gpu->d_eLocal =
          sycl::malloc_device<real_t>(2, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      gpu->d_eSum =
          sycl::malloc_device<real_t>(2, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      gpu->atoms.gid = sycl::malloc_device<int>(total_boxes * MAXATOMS,
                                                dpct::get_in_order_queue())));

  // species data
  CUDA_CHECK(DPCT_CHECK_ERROR(
      gpu->atoms.iSpecies = sycl::malloc_device<int>(
          total_boxes * MAXATOMS, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->species_mass = sycl::malloc_device<real_t>(
                                  num_species, dpct::get_in_order_queue())));

  // allocate indices, neighbors, etc.
  CUDA_CHECK(DPCT_CHECK_ERROR(
      gpu->neighbor_cells = sycl::malloc_device<int>(
          nLocalBoxes * N_MAX_NEIGHBORS, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->neighbor_atoms = sycl::malloc_device<int>(
                                  nLocalBoxes * N_MAX_NEIGHBORS * MAXATOMS,
                                  dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->num_neigh_atoms = sycl::malloc_device<int>(
                                  nLocalBoxes, dpct::get_in_order_queue())));

  // total # of atoms in local boxes
  int n = 0;
  for (int iBox=0; iBox < sim->boxes->nLocalBoxes; iBox++)
    n += sim->boxes->nAtoms[iBox];
  gpu->a_list.n = n;
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->a_list.atoms = sycl::malloc_device<int>(
                                  n, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->a_list.cells = sycl::malloc_device<int>(
                                  n, dpct::get_in_order_queue())));

  // allocate other lists as well
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->i_list.atoms = sycl::malloc_device<int>(
                                  n, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->i_list.cells = sycl::malloc_device<int>(
                                  n, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->b_list.atoms = sycl::malloc_device<int>(
                                  n, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(gpu->b_list.cells = sycl::malloc_device<int>(
                                  n, dpct::get_in_order_queue())));

  initNeighborListGpu(gpu, &(gpu->atoms.neighborList),nLocalBoxes, skinDistance);
  initLinkCellsGpu(sim, &(gpu->boxes));

  int nMaxHaloParticles = (sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes)*MAXATOMS;
  initHashTableGpu(&(gpu->d_hashTable), 2*nMaxHaloParticles);

  //Allocate pairlist
  if(sim->usePairlist)
  {
      gpu->pairlist = sycl::malloc_device<int>(
          nLocalBoxes * MAXATOMS / WARP_SIZE * N_MAX_NEIGHBORS *
              (MAXATOMS + PAIRLIST_ATOMS_PER_INT - 1) / PAIRLIST_ATOMS_PER_INT,
          dpct::get_in_order_queue());
  }
  // init EAM arrays
  if (do_eam)  
  {
    EamPotential* pot = (EamPotential*) sim->pot;

    gpu->eam_pot.f.values = sycl::malloc_device<real_t>(
        (pot->f->n + 3), dpct::get_in_order_queue());
    if(!sim->spline)
    {
        gpu->eam_pot.rho.values = sycl::malloc_device<real_t>(
            (pot->rho->n + 3), dpct::get_in_order_queue());
        gpu->eam_pot.phi.values = sycl::malloc_device<real_t>(
            (pot->phi->n + 3), dpct::get_in_order_queue());
    }
    else
    {
        gpu->eam_pot.fS.coefficients = sycl::malloc_device<real_t>(
            (4 * pot->f->n), dpct::get_in_order_queue());
        gpu->eam_pot.rhoS.coefficients = sycl::malloc_device<real_t>(
            (4 * pot->rho->n), dpct::get_in_order_queue());
        gpu->eam_pot.phiS.coefficients = sycl::malloc_device<real_t>(
            (4 * pot->phi->n), dpct::get_in_order_queue());
    }
    gpu->eam_pot.dfEmbed =
        (real_t *)sycl::malloc_device(r_size, dpct::get_in_order_queue());
    gpu->eam_pot.rhobar =
        (real_t *)sycl::malloc_device(r_size, dpct::get_in_order_queue());
  }
  else //init LJ iterpolation table
  {
    LjPotential * pot = (LjPotential*) sim->pot;
//TODO: configurable length
   CUDA_CHECK(DPCT_CHECK_ERROR(
       gpu->lj_pot.lj_interpolation.values =
           sycl::malloc_device<real_t>(1003, dpct::get_in_order_queue())));
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
  CUDA_CHECK(DPCT_CHECK_ERROR(
      sim->flags = sycl::malloc_device<int>(sim->boxes->nTotalBoxes * MAXATOMS,
                                            dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      sim->tmp_sort = sycl::malloc_device<int>(
          sim->boxes->nTotalBoxes * MAXATOMS, dpct::get_in_order_queue())));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(sim->gpu_atoms_buf = (char *)sycl::malloc_device(
                           sim->boxes->nTotalBoxes * MAXATOMS * sizeof(AtomMsg),
                           dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      sim->gpu_force_buf = (char *)sycl::malloc_device(
          sim->boxes->nTotalBoxes * MAXATOMS * sizeof(ForceMsg),
          dpct::get_in_order_queue())));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void DestroyGpu(SimFlat *flat) try {
  SimGpu *gpu = &flat->gpu;
  SimGpu *host = &flat->host;

  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(gpu->d_updateLinkCellsRequired,
                                              dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(gpu->d_updateLinkCellsResult,
                                              dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->d_eLocal, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->d_eSum, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.r.x, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.r.y, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.r.z, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.p.x, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.p.y, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.p.z, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.f.x, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.f.y, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.f.z, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.e, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.gid, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->atoms.iSpecies, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->species_mass, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->neighbor_cells, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->neighbor_atoms, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->num_neigh_atoms, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->boxes.nAtoms, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->a_list.atoms, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->a_list.cells, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->i_list.atoms, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->i_list.cells, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->b_list.atoms, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->b_list.cells, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(flat->flags, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(flat->tmp_sort, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(flat->gpu_atoms_buf, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(flat->gpu_force_buf, dpct::get_in_order_queue())));

  if (gpu->eam_pot.f.values) CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->eam_pot.f.values, dpct::get_in_order_queue())));
  if (gpu->eam_pot.rho.values) CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->eam_pot.rho.values, dpct::get_in_order_queue())));
  if (gpu->eam_pot.phi.values) CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(gpu->eam_pot.phi.values, dpct::get_in_order_queue())));

  if (gpu->eam_pot.fS.coefficients)
      dpct::dpct_free(gpu->eam_pot.fS.coefficients, dpct::get_in_order_queue());
  if (gpu->eam_pot.rhoS.coefficients) dpct::dpct_free(
      gpu->eam_pot.rhoS.coefficients, dpct::get_in_order_queue());
  if (gpu->eam_pot.phiS.coefficients) dpct::dpct_free(
      gpu->eam_pot.phiS.coefficients, dpct::get_in_order_queue());

  if (gpu->eam_pot.dfEmbed)
      dpct::dpct_free(gpu->eam_pot.dfEmbed, dpct::get_in_order_queue());
  if (gpu->eam_pot.rhobar)
      dpct::dpct_free(gpu->eam_pot.rhobar, dpct::get_in_order_queue());

  free(host->species_mass);

  free(host->neighbor_cells);
  free(host->neighbor_atoms);
  free(host->num_neigh_atoms);

  free(host->a_list.atoms);
  free(host->a_list.cells);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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
   for(int i = 0; i < pot->lj_interpolation.n+3; ++i)
   {        
       real_t x = pot->lj_interpolation.x0 + (i-1)/pot->lj_interpolation.invDx;
       real_t r2 = 1.0/(x*x);
       real_t r6 = s6 * r2*r2*r2;
       temp[i] = 4 * epsilon * (r6 * (r6 - 1.0) - eShift);
  }
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(pot->lj_interpolation.values, temp,
                  (pot->lj_interpolation.n + 3) * sizeof(real_t))
          .wait()));

  free(temp);
}

//Algorithm for computing spline coefficients from Numerical Recipes in C, chapter 3.3
void initSplineCoefficients(real_t * gpu_coefficients, int n, real_t * values, real_t x0, real_t invDx)
{
    real_t *u = (real_t*) malloc(n * sizeof(real_t));
    real_t *y2 = (real_t*) malloc((n+1)*sizeof(real_t));

    //Second derivative is 0 at the beginning of the interval
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
    //First derivative is 0 at the end of the interval
    real_t qn = 0.5;
    real_t un = (-3.0/(xn-xnp))*(values[n]-values[n-1])/(xn-xnp);
    y2[n] = (un-qn*u[n-1])/(qn*y2[n-1]+1.0);

    for(int i = n-1; i >= 0; --i)
    {
        y2[i] = y2[i]*y2[i+1] + u[i];
    }
    real_t * coefficients = (real_t *) malloc(4*n* sizeof(real_t));
    for(int i = 0; i < n; i++)
    {
        real_t x1 = (x0 + i/invDx)*(x0+i/invDx);
        real_t x2 = (x0 + (i+1)/invDx)*(x0+(i+1)/invDx);
        real_t d2y1 = y2[i];
        real_t d2y2 = y2[i+1];
        real_t y1 = values[i];
        real_t y2 = values[i+1];
        
        coefficients[i*4] = 1.0/(6.0*(x2-x1))*(d2y2-d2y1);
        coefficients[i*4+1] = 1.0/(2.0*(x2-x1))*(x2*d2y1-x1*d2y2);
        coefficients[i*4+2] = 1.0/(x2-x1) * (1.0/6.0*(-3*x2*x2+(x2-x1)*(x2-x1))*d2y1+1.0/6.0*(3*x1*x1-(x2-x1)*(x2-x1))*d2y2-y1+y2);
        coefficients[i*4+3] = 1/(x2-x1)*(x2*y1-x1*y2+1.0/6.0*d2y1*(x2*x2*x2-x2*(x2-x1)*(x2-x1)) + 1.0/6.0*d2y2*(-x1*x1*x1+x1*(x2-x1)*(x2-x1)));
    }
    dpct::get_in_order_queue()
        .memcpy(gpu_coefficients, coefficients, 4 * n * sizeof(real_t))
        .wait();

    free(y2);
    free(u);
    free(coefficients);
}

void CopyDataToGpu(SimFlat *sim, int do_eam) try {
  SimGpu *gpu = &sim->gpu;
  SimGpu *host = &sim->host;

  // set potential
  if (do_eam) 
  {
      EamPotential* pot = (EamPotential*) sim->pot;
      gpu->eam_pot.cutoff = pot->cutoff;
      
      //f is needed for second phase of EAM, not yet changed to spline 
      gpu->eam_pot.f.n = pot->f->n;
      gpu->eam_pot.f.x0 = pot->f->x0;
      gpu->eam_pot.f.xn = pot->f->x0 + pot->f->n / pot->f->invDx;
      gpu->eam_pot.f.invDx = pot->f->invDx;
      gpu->eam_pot.f.invDxHalf = pot->f->invDx * 0.5;
      gpu->eam_pot.f.invDxXx0 = pot->f->invDxXx0;
      dpct::get_in_order_queue()
          .memcpy(gpu->eam_pot.f.values, pot->f->values - 1,
                  (pot->f->n + 3) * sizeof(real_t))
          .wait();

      if(!sim->spline)
      {
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

    CUDA_CHECK(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(gpu->eam_pot.f.values, pot->f->values - 1,
                                     (pot->f->n + 3) * sizeof(real_t))
                             .wait()));
        dpct::get_in_order_queue().memcpy(gpu->eam_pot.rho.values,
                                          pot->rho->values - 1,
                                          (pot->rho->n + 3) * sizeof(real_t));
        dpct::get_in_order_queue()
            .memcpy(gpu->eam_pot.phi.values, pot->phi->values - 1,
                    (pot->phi->n + 3) * sizeof(real_t))
            .wait();
    }
    else
    {
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
  }
  else
  {
      LjPotential* pot = (LjPotential*)sim->pot;
      gpu->lj_pot.sigma = pot->sigma;
      gpu->lj_pot.cutoff = pot->cutoff;
      gpu->lj_pot.epsilon = pot->epsilon;
      if(sim->ljInterpolation)
          initLJinterpolation(&(gpu->lj_pot));
  }

  int total_boxes = sim->boxes->nTotalBoxes;
  int nLocalBoxes = sim->boxes->nLocalBoxes;
  int r_size = total_boxes * MAXATOMS * sizeof(real_t);
  int f_size = nLocalBoxes * MAXATOMS * sizeof(real_t);
  int num_species = 1;


  for (int iBox=0; iBox < nLocalBoxes; iBox++) {
    getNeighborBoxes(sim->boxes, iBox, host->neighbor_cells + iBox * N_MAX_NEIGHBORS);

    // find itself and put first
    for (int j = 0; j < N_MAX_NEIGHBORS; j++)
      if (host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j] == iBox) {
        int q = host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
	host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j] = host->neighbor_cells[iBox * N_MAX_NEIGHBORS + 0];
        host->neighbor_cells[iBox * N_MAX_NEIGHBORS + 0] = q;
        break;
      }
  }

  // prepare neighbor list
  for (int iBox=0; iBox < nLocalBoxes; iBox++) {
    int num_neigh_atoms = 0;
    for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
      int jBox = host->neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
      for (int k = 0; k < sim->boxes->nAtoms[jBox]; k++) {
        host->neighbor_atoms[iBox * N_MAX_NEIGHBORS * MAXATOMS + num_neigh_atoms] = jBox * MAXATOMS + k;
        num_neigh_atoms++;
      }
    }
    host->num_neigh_atoms[iBox] = num_neigh_atoms;
  }

  // compute total # of atoms in local boxes
  int n_total = 0;
  gpu->max_atoms_cell = 0;
  for (int iBox=0; iBox < sim->boxes->nLocalBoxes; iBox++) {
    n_total += sim->boxes->nAtoms[iBox];
    if (sim->boxes->nAtoms[iBox] > gpu->max_atoms_cell)
      gpu->max_atoms_cell = sim->boxes->nAtoms[iBox];
  }
  gpu->a_list.n = n_total;
  gpu->boxes.nLocalBoxes = sim->boxes->nLocalBoxes;

  // compute and copy compact list of all atoms/cells
  int cur = 0;
  for (int iBox=0; iBox < sim->boxes->nLocalBoxes; iBox++) {
    int nIBox = sim->boxes->nAtoms[iBox];
    if (nIBox == 0) continue;
    // loop over atoms in iBox
    for (int iOff = iBox * MAXATOMS, ii=0; ii<nIBox; ii++, iOff++) {
      host->a_list.atoms[cur] = ii;
      host->a_list.cells[cur] = iBox;
      cur++;
    }
  }

  // initialize species
  host->species_mass = (real_t*)malloc(num_species * sizeof(real_t));
  for (int i = 0; i < num_species; i++)
    host->species_mass[i] = sim->species[i].mass;

  // copy all data to gpus
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(gpu->atoms.r.x, sim->atoms->r.x, r_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(gpu->atoms.r.y, sim->atoms->r.y, r_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(gpu->atoms.r.z, sim->atoms->r.z, r_size)
                           .wait()));

  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(gpu->atoms.p.x, sim->atoms->p.x, r_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(gpu->atoms.p.y, sim->atoms->p.y, r_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(gpu->atoms.p.z, sim->atoms->p.z, r_size)
                           .wait()));

  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(gpu->atoms.iSpecies, sim->atoms->iSpecies,
                                   nLocalBoxes * MAXATOMS * sizeof(int))
                           .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                  .memcpy(gpu->atoms.gid, sim->atoms->gid,
                                          total_boxes * MAXATOMS * sizeof(int))
                                  .wait()));

  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                  .memcpy(gpu->species_mass, host->species_mass,
                                          num_species * sizeof(real_t))
                                  .wait()));

  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(gpu->neighbor_cells, host->neighbor_cells,
                                   nLocalBoxes * N_MAX_NEIGHBORS * sizeof(int))
                           .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->neighbor_atoms, host->neighbor_atoms,
                  nLocalBoxes * N_MAX_NEIGHBORS * MAXATOMS * sizeof(int))
          .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(gpu->num_neigh_atoms, host->num_neigh_atoms,
                                   nLocalBoxes * sizeof(int))
                           .wait()));

  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                  .memcpy(gpu->boxes.nAtoms, sim->boxes->nAtoms,
                                          total_boxes * sizeof(int))
                                  .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                  .memcpy(gpu->boxes.boxIDLookUp,
                                          sim->boxes->boxIDLookUp,
                                          nLocalBoxes * sizeof(int))
                                  .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                  .memcpy(gpu->boxes.boxIDLookUpReverse,
                                          sim->boxes->boxIDLookUpReverse,
                                          nLocalBoxes * sizeof(int3_t))
                                  .wait()));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->a_list.atoms, host->a_list.atoms, n_total * sizeof(int))
          .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->a_list.cells, host->a_list.cells, n_total * sizeof(int))
          .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void updateNAtomsGpu(SimFlat* sim)
{
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->gpu.boxes.nAtoms, sim->boxes->nAtoms,
                                   sim->boxes->nTotalBoxes * sizeof(int))
                           .wait()));
}

void updateNAtomsCpu(SimFlat* sim)
{
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->boxes->nAtoms, sim->gpu.boxes.nAtoms,
                                   sim->boxes->nTotalBoxes * sizeof(int))
                           .wait()));
}

void emptyHaloCellsGpu(SimFlat* sim)
{
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memset(sim->gpu.boxes.nAtoms + sim->boxes->nLocalBoxes, 0,
                  (sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes) *
                      sizeof(int))
          .wait()));
}

void GetDataFromGpu(SimFlat *sim)
{
  SimGpu *gpu = &sim->gpu;
  SimGpu *host = &sim->host;

  // copy back forces & energies
  int f_size = sim->boxes->nLocalBoxes * MAXATOMS * sizeof(real_t);

  // update num atoms
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                  .memcpy(sim->boxes->nAtoms, gpu->boxes.nAtoms,
                                          sim->boxes->nTotalBoxes * sizeof(int))
                                  .wait()));

  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->p.x, gpu->atoms.p.x, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->p.y, gpu->atoms.p.y, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->p.z, gpu->atoms.p.z, f_size)
                           .wait()));

  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->r.x, gpu->atoms.r.x, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->r.y, gpu->atoms.r.y, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->r.z, gpu->atoms.r.z, f_size)
                           .wait()));

  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->f.x, gpu->atoms.f.x, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->f.y, gpu->atoms.f.y, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->f.z, gpu->atoms.f.z, f_size)
                           .wait()));

  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                  .memcpy(sim->atoms->U, gpu->atoms.e, f_size)
                                  .wait()));

  // assign energy and forces
  // compute total energy
  sim->ePotential = 0.0;
  for (int iBox=0; iBox < sim->boxes->nLocalBoxes; iBox++) {
    int nIBox = sim->boxes->nAtoms[iBox];
    // loop over atoms in iBox
    for (int iOff = iBox * MAXATOMS, ii=0; ii<nIBox; ii++, iOff++) {

      sim->ePotential += sim->atoms->U[iOff];
    }
  }
}

/// Copies positions and momentum of local particles to CPU
void GetLocalAtomsFromGpu(SimFlat *sim) 
{
  SimGpu *gpu = &sim->gpu;
  SimGpu *host = &sim->host;

  // copy back forces & energies
  int f_size = sim->boxes->nLocalBoxes * MAXATOMS * sizeof(real_t);

  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->p.x, gpu->atoms.p.x, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->p.y, gpu->atoms.p.y, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->p.z, gpu->atoms.p.z, f_size)
                           .wait()));

  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->r.x, gpu->atoms.r.x, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->r.y, gpu->atoms.r.y, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->r.z, gpu->atoms.r.z, f_size)
                           .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(sim->atoms->gid, gpu->atoms.gid,
                  sim->boxes->nLocalBoxes * MAXATOMS * sizeof(int))
          .wait())); // only req. if nlforced TODO
}

/// Compacts all atoms within the halo cells into h_compactAtoms (stored in SoA data-layout).
/// @param [in] sim
/// @param [out] h_compactAtoms stores the compacted atoms in SoA format
/// @param [out] h_cellOffset Array of at least (nHaloBoxes+1) elements will store the scan of nHaloAtoms (e.g. nAtoms(haloCell_0)=2, nAtoms(haloCell_1)=3 => h_cellOffset(0)=0,h_cellOffset(1)=3,h_cellOffset(2)=5)
int compactHaloCells(SimFlat* sim, char* h_compactAtoms, int* h_cellOffset)
{
      int nHaloCells = sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes;

      
      h_cellOffset[sim->boxes->nLocalBoxes] = 0;
      for(int i = 1, iBox = sim->boxes->nLocalBoxes; i <= nHaloCells; ++i, ++iBox)
      {
         h_cellOffset[i] = sim->boxes->nAtoms[iBox] + h_cellOffset[i-1];
      }
      int nTotalAtomsInHaloCells = h_cellOffset[nHaloCells];

      AtomMsgSoA msg_h;
      getAtomMsgSoAPtr(h_compactAtoms, &msg_h, nTotalAtomsInHaloCells);

      //compact atoms from atoms struct to msg_h
      for (int ii = 0; ii < nHaloCells; ++ii)
      {
              int iOff = (sim->boxes->nLocalBoxes + ii) * MAXATOMS;
              for(int i = h_cellOffset[ii]; i < h_cellOffset[ii+1]; ++i, ++iOff)
              {
                 msg_h.rx[i] = sim->atoms->r.x[iOff];
                 msg_h.ry[i] = sim->atoms->r.y[iOff];
                 msg_h.rz[i] = sim->atoms->r.z[iOff];

                 msg_h.px[i] = sim->atoms->p.x[iOff];
                 msg_h.py[i] = sim->atoms->p.y[iOff];
                 msg_h.pz[i] = sim->atoms->p.z[iOff];

                 msg_h.type[i] = sim->atoms->iSpecies[iOff];
                 msg_h.gid[i] = sim->atoms->gid[iOff];
              }
      }
      return nTotalAtomsInHaloCells;
}

void updateGpuHalo(SimFlat *sim)
{
  //Optimization: implement version using compactHaloCells()
  SimGpu *gpu = &sim->gpu;
  SimGpu *host = &sim->host;

  int nHaloCells = sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes;
//  char* h_compactAtoms = sim->atomExchange->recvBufM;
//  int*  h_cellOffset = ((AtomExchangeParms*)sim->atomExchange->parms)->h_natoms_buf;
//
//  int nTotalAtomsInHaloCells= compactHaloCells( sim, h_compactAtoms, h_cellOffset); 
//    
//  //copy compacted atoms to gpu
//  char* d_compactAtoms = sim->gpu_atoms_buf;
//  CUDA_CHECK(cudaMemcpy((void*)(d_compactAtoms), h_compactAtoms, nTotalAtomsInHaloCells * sizeof(AtomMsg), cudaMemcpyHostToDevice));
//
//  //alias host and device buffers with AtomMsgSoA
//  AtomMsgSoA msg_d;
//  getAtomMsgSoAPtr(d_compactAtoms, &msg_d, nTotalAtomsInHaloCells);
//
//  //copy cellOffset to cpu
//  int* d_cellOffset = ((AtomExchangeParms*)sim->atomExchange->parms)->d_natoms_buf;
//  CUDA_CHECK(cudaMemcpy(d_cellOffsets, h_cellOffset, (nHaloCells+1) * sizeof(int), cudaMemcpyHostToDevice));
//
//  const int blockDim = 256;
//  int grid = nTotalAtomsInHaloCells + (blockDim - 1) / blockDim;
//  unpackHaloCells<<<grid, block>>>(d_cellOffset, d_compactAtoms, sim->gpu); //TODO implement this function

  int f_size = nHaloCells * MAXATOMS * sizeof(real_t);
  int i_size = nHaloCells * MAXATOMS * sizeof(int);

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->atoms.p.x + (sim->boxes->nLocalBoxes * MAXATOMS),
                  sim->atoms->p.x + (sim->boxes->nLocalBoxes * MAXATOMS),
                  f_size)
          .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->atoms.p.y + (sim->boxes->nLocalBoxes * MAXATOMS),
                  sim->atoms->p.y + (sim->boxes->nLocalBoxes * MAXATOMS),
                  f_size)
          .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->atoms.p.z + (sim->boxes->nLocalBoxes * MAXATOMS),
                  sim->atoms->p.z + (sim->boxes->nLocalBoxes * MAXATOMS),
                  f_size)
          .wait()));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->atoms.r.x + (sim->boxes->nLocalBoxes * MAXATOMS),
                  sim->atoms->r.x + (sim->boxes->nLocalBoxes * MAXATOMS),
                  f_size)
          .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->atoms.r.y + (sim->boxes->nLocalBoxes * MAXATOMS),
                  sim->atoms->r.y + (sim->boxes->nLocalBoxes * MAXATOMS),
                  f_size)
          .wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->atoms.r.z + (sim->boxes->nLocalBoxes * MAXATOMS),
                  sim->atoms->r.z + (sim->boxes->nLocalBoxes * MAXATOMS),
                  f_size)
          .wait()));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->atoms.gid + (sim->boxes->nLocalBoxes * MAXATOMS),
                  sim->atoms->gid + (sim->boxes->nLocalBoxes * MAXATOMS),
                  i_size)
          .wait())); // TODO REmove?
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memcpy(gpu->atoms.iSpecies + (sim->boxes->nLocalBoxes * MAXATOMS),
                  sim->atoms->iSpecies + (sim->boxes->nLocalBoxes * MAXATOMS),
                  i_size)
          .wait())); // TODO remove?
}

void initLinkCellsGpu(SimFlat *sim, LinkCellGpu* boxes)
{

  boxes->nTotalBoxes = sim->boxes->nTotalBoxes;
  boxes->nLocalBoxes = sim->boxes->nLocalBoxes;

  boxes->gridSize.x = sim->boxes->gridSize[0];
  boxes->gridSize.y = sim->boxes->gridSize[1];
  boxes->gridSize.z = sim->boxes->gridSize[2];

  boxes->localMin.x = sim->boxes->localMin[0];
  boxes->localMin.y = sim->boxes->localMin[1];
  boxes->localMin.z = sim->boxes->localMin[2];

  boxes->localMax.x = sim->boxes->localMax[0];
  boxes->localMax.y = sim->boxes->localMax[1];
  boxes->localMax.z = sim->boxes->localMax[2];

  boxes->invBoxSize.x = sim->boxes->invBoxSize[0];
  boxes->invBoxSize.y = sim->boxes->invBoxSize[1];
  boxes->invBoxSize.z = sim->boxes->invBoxSize[2];

  assert (sim->boxes->nLocalBoxes == sim->boxes->gridSize[0] * sim->boxes->gridSize[1] * sim->boxes->gridSize[2]);
  CUDA_CHECK(DPCT_CHECK_ERROR(
      boxes->nAtoms = sycl::malloc_device<int>(sim->boxes->nTotalBoxes,
                                               dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      boxes->boxIDLookUpReverse = sycl::malloc_device<int3_t>(
          sim->boxes->nLocalBoxes, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      boxes->boxIDLookUp = sycl::malloc_device<int>(
          sim->boxes->nLocalBoxes, dpct::get_in_order_queue())));
}

void AnalyzeInput(SimFlat *sim, int step)
{
  // copy positions data to host for all cells (including halos)
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->boxes->nAtoms, sim->gpu.boxes.nAtoms,
                                   sim->boxes->nTotalBoxes * sizeof(int))
                           .wait()));
  int f_size = sim->boxes->nTotalBoxes * MAXATOMS * sizeof(real_t);
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->r.x, sim->gpu.atoms.r.x, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->r.y, sim->gpu.atoms.r.y, f_size)
                           .wait()));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->atoms->r.z, sim->gpu.atoms.r.z, f_size)
                           .wait()));
  sim->host.atoms.neighborList.nNeighbors = (int*)malloc(sim->atoms->nLocal * sizeof(int));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                           .memcpy(sim->host.atoms.neighborList.nNeighbors,
                                   sim->gpu.atoms.neighborList.nNeighbors,
                                   sim->atoms->nLocal * sizeof(int))
                           .wait()));

  GetDataFromGpu(sim);

  int size = MAXATOMS * N_MAX_NEIGHBORS;

  // atoms per cell
  int atoms_per_cell_hist[size];      
  memset(atoms_per_cell_hist, 0, size * sizeof(int));
  for (int iBox = 0; iBox < sim->boxes->nLocalBoxes; iBox++) 
    atoms_per_cell_hist[sim->boxes->nAtoms[iBox]]++;

  // cell neighbors 
  int cell_neigh_hist[size];
  memset(cell_neigh_hist, 0, size * sizeof(int));
  for (int iBox = 0; iBox < sim->boxes->nLocalBoxes; iBox++) {
    int neighbor_atoms = 0;
    for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
      int jBox = sim->host.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
      neighbor_atoms += sim->boxes->nAtoms[jBox];
    }
    cell_neigh_hist[neighbor_atoms] += sim->boxes->nAtoms[iBox];
  }

  // find # of atoms in neighbor lists (cut-off + skin distance)
  int neigh_lists_hist[size];
  memset(neigh_lists_hist, 0, size * sizeof(int));
#if 0
  int id = 0;
  for (int i = 0; i < sim->atoms->nLocal; i++) 
      neigh_lists_hist[sim->host.atoms.neighborList.nNeighbors[i]]++;
#endif

  // find # of neighbors strictly under cut-off
  int passed_cutoff_hist[size];      
  memset(passed_cutoff_hist, 0, size * sizeof(int));
  for (int iBox = 0; iBox < sim->boxes->nLocalBoxes; iBox++) {
    for (int iAtom = 0; iAtom < sim->boxes->nAtoms[iBox]; iAtom++) {
      int passed_atoms = 0;
      for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
        int jBox = sim->host.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
        for (int jAtom = 0; jAtom < sim->boxes->nAtoms[jBox]; jAtom++) 
        {
          int i_particle = iBox * MAXATOMS + iAtom;
          int j_particle = jBox * MAXATOMS + jAtom;

          real_t dx = sim->atoms->r.x[i_particle] - sim->atoms->r.x[j_particle];
          real_t dy = sim->atoms->r.y[i_particle] - sim->atoms->r.y[j_particle];
          real_t dz = sim->atoms->r.z[i_particle] - sim->atoms->r.z[j_particle];

          real_t r2 = dx*dx + dy*dy + dz*dz;

          // TODO: this is only works for EAM potential
          if (r2 < sim->gpu.eam_pot.cutoff * sim->gpu.eam_pot.cutoff)
            passed_atoms++;
        }
      }
      passed_cutoff_hist[passed_atoms]++;
    }
  }

  char fileName[100];
  sprintf(fileName, "histogram_%i.csv", step);
  FILE *file = fopen(fileName, "w");
  fprintf(file,"# of atoms,cell size = cutoff,neighbor cells,cutoff = 4.95 + 10%%,cutoff = 4.95,\n");
  for(int i = 0; i < size; i++)
    fprintf(file,"%i,%i,%i,%i,%i,\n", i,atoms_per_cell_hist[i],cell_neigh_hist[i],neigh_lists_hist[i],passed_cutoff_hist[i]);
  fclose(file);
}
