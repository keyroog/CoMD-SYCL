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

#ifndef __GPU_REDISTRIBUTE_H_
#define __GPU_REDISTRIBUTE_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "mytype.h"
/// Calculates the link cell index from the grid coords.  The valid
/// coordinate range in direction ii is [-1, gridSize[ii]].  Any
/// coordinate that involves a -1 or gridSize[ii] is a halo link cell.
/// Because of the order in which the local and halo link cells are
/// stored the indices of the halo cells are special cases.
/// \see initLinkCells for an explanation of storage order.

int getBoxFromTuple(LinkCellGpu boxes, int ix, int iy, int iz)
{
   int iBox = 0;

   // Halo in Z+
   if (iz == boxes.gridSize.z)
   {
      iBox = boxes.nLocalBoxes + 2 * boxes.gridSize.z * boxes.gridSize.y + 2 * boxes.gridSize.z * (boxes.gridSize.x + 2) +
         (boxes.gridSize.x + 2) * (boxes.gridSize.y + 2) + (boxes.gridSize.x + 2) * (iy + 1) + (ix + 1);
   }
   // Halo in Z-
   else if (iz == -1)
   {
      iBox = boxes.nLocalBoxes + 2 * boxes.gridSize.z * boxes.gridSize.y + 2 * boxes.gridSize.z * (boxes.gridSize.x + 2) +
         (boxes.gridSize.x + 2) * (iy + 1) + (ix + 1);
   }
   // Halo in Y+
   else if (iy == boxes.gridSize.y)
   {
      iBox = boxes.nLocalBoxes + 2 * boxes.gridSize.z * boxes.gridSize.y + boxes.gridSize.z * (boxes.gridSize.x + 2) +
         (boxes.gridSize.x + 2) * iz + (ix + 1);
   }
   // Halo in Y-
   else if (iy == -1)
   {
      iBox = boxes.nLocalBoxes + 2 * boxes.gridSize.z * boxes.gridSize.y + iz * (boxes.gridSize.x + 2) + (ix + 1);
   }
   // Halo in X+
   else if (ix == boxes.gridSize.x)
   {
      iBox = boxes.nLocalBoxes + boxes.gridSize.y * boxes.gridSize.z + iz * boxes.gridSize.y + iy;
   }
   // Halo in X-
   else if (ix == -1)
   {
      iBox = boxes.nLocalBoxes + iz * boxes.gridSize.y + iy;
   }
   // local link celll.
   else
   {
      iBox = boxes.boxIDLookUp[IDX3D(ix,iy,iz, boxes.gridSize.x, boxes.gridSize.y)];
   }

   return iBox;
}

/// Get the index of the link cell that contains the specified
/// coordinate.  This can be either a halo or a local link cell.
///
/// Because the rank ownership of an atom is strictly determined by the
/// atom's position, we need to take care that all ranks will agree which
/// rank owns an atom.  The conditionals at the end of this function are
/// special care to ensure that all ranks make compatible link cell
/// assignments for atoms that are near a link cell boundaries.  If no
/// ranks claim an atom in a local cell it will be lost.  If multiple
/// ranks claim an atom it will be duplicated.

int getBoxFromCoord(LinkCellGpu cells, real_t rx, real_t ry, real_t rz)
{
   int ix = (int)(sycl::floor((rx - cells.localMin.x) * cells.invBoxSize.x));
   int iy = (int)(sycl::floor((ry - cells.localMin.y) * cells.invBoxSize.y));
   int iz = (int)(sycl::floor((rz - cells.localMin.z) * cells.invBoxSize.z));

   // For each axis, if we are inside the local domain, make sure we get
   // a local link cell.  Otherwise, make sure we get a halo link cell.
   if (rx < cells.localMax.x)
   {
      if (ix == cells.gridSize.x) ix = cells.gridSize.x - 1;
   }
   else
      ix = cells.gridSize.x; // assign to halo cell
   if (ry < cells.localMax.y)
   {
      if (iy == cells.gridSize.y) iy = cells.gridSize.y - 1;
   }
   else
      iy = cells.gridSize.y;
   if (rz < cells.localMax.z)
   {
      if (iz == cells.gridSize.z) iz = cells.gridSize.z - 1;
   }
   else
      iz = cells.gridSize.z;

   return getBoxFromTuple(cells, ix, iy, iz);
}

/**
 * Moves particles from one box into another box (if required).
 *
 * Particles that have left the box will be flagged with 0.
 *
 * \param[out] flags flags[ibox * MAXATOMS + i] corresponds to the ith atom in box ibox.
 *                   The flag will be set to zero iff the particle has just left the box.
 */
template<int usePairlist>
void UpdateLinkCells(SimGpu sim, LinkCellGpu cells, int *flags)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (tid >= sim.a_list.n) return;

  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  int iOff = iBox * MAXATOMS + iAtom;
  assert(iOff < sim.boxes.nLocalBoxes * MAXATOMS && iOff >=0 );

  int jBox = getBoxFromCoord(cells, sim.atoms.r.x[iOff], sim.atoms.r.y[iOff], sim.atoms.r.z[iOff]);

  if (jBox != iBox) {
    // find new position in jBox list
    int jAtom =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &sim.boxes.nAtoms[jBox], 1);
    assert(jAtom < MAXATOMS);
    int jOff = jBox * MAXATOMS + jAtom;
    assert(jOff < sim.boxes.nTotalBoxes * MAXATOMS && jOff >=0 );

    // flag set/unset
    flags[jOff] = tid+1; //TODO: Why do we set this value to tid+1, wouldn't '1' suffice? (ask Nikolai)
    flags[iOff] = 0;

    // copy over the atoms data
    sim.atoms.r.x[jOff] = sim.atoms.r.x[iOff];
    sim.atoms.r.y[jOff] = sim.atoms.r.y[iOff];
    sim.atoms.r.z[jOff] = sim.atoms.r.z[iOff];
    sim.atoms.p.x[jOff] = sim.atoms.p.x[iOff];
    sim.atoms.p.y[jOff] = sim.atoms.p.y[iOff];
    sim.atoms.p.z[jOff] = sim.atoms.p.z[iOff];
    sim.atoms.gid[jOff] = sim.atoms.gid[iOff];
    sim.atoms.iSpecies[jOff] = sim.atoms.iSpecies[iOff];
    if(usePairlist)
    {
        sim.atoms.neighborList.lastR.x[jOff] = sim.atoms.neighborList.lastR.x[iOff];
        sim.atoms.neighborList.lastR.y[jOff] = sim.atoms.neighborList.lastR.y[iOff];
        sim.atoms.neighborList.lastR.z[jOff] = sim.atoms.neighborList.lastR.z[iOff];
    }
    sim.a_list.atoms[tid] = jAtom;
    sim.a_list.cells[tid] = jBox;
  }
  else
    flags[iOff] = tid+1;
}

/**
 * Compacts the Atoms per cell such that the position of atoms which have left the box
 * will be overwritten by new atoms which have just entered the same cell.
 *
 * E.g.: Let 'x' denote a particle which remained in the same box,
 * '0' denotes a particle that has left the box and 'y' represents a new particle of this
 *  box.
 * A box (xx0xxxy) would be compacted to: (xxxxy)
 */
// TODO: improve parallelism, currently it's one thread per cell!
template <int usePairlist>
void CompactAtoms(SimGpu sim, int num_cells, int *flags, int *natoms)
{
  // only process local cells
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  natoms[item_ct1.get_local_id(2)] = 0;

  int jAtom = 0;
  if (tid < num_cells) {
    int iBox = tid;
    int jBox = tid;

    for (int iAtom = 0, iOff = iBox * MAXATOMS; iAtom < MAXATOMS; iAtom++, iOff++)
      if (flags[iBox * MAXATOMS + iAtom] > 0) {
        int jOff = jBox * MAXATOMS + jAtom;
        if (iOff != jOff) {
          sim.atoms.r.x[jOff] = sim.atoms.r.x[iOff];
          sim.atoms.r.y[jOff] = sim.atoms.r.y[iOff];
          sim.atoms.r.z[jOff] = sim.atoms.r.z[iOff];
          sim.atoms.p.x[jOff] = sim.atoms.p.x[iOff];
          sim.atoms.p.y[jOff] = sim.atoms.p.y[iOff];
          sim.atoms.p.z[jOff] = sim.atoms.p.z[iOff];
          sim.atoms.gid[jOff] = sim.atoms.gid[iOff];
          sim.atoms.iSpecies[jOff] = sim.atoms.iSpecies[iOff];
          if(usePairlist)
          {
              sim.atoms.neighborList.lastR.x[jOff] = sim.atoms.neighborList.lastR.x[iOff];
              sim.atoms.neighborList.lastR.y[jOff] = sim.atoms.neighborList.lastR.y[iOff];
              sim.atoms.neighborList.lastR.z[jOff] = sim.atoms.neighborList.lastR.z[iOff];
          }

        }
        jAtom++;
      }

    // update # of atoms in the box
    sim.boxes.nAtoms[jBox] = jAtom;

    // compute global max and store in flags[0]
    natoms[item_ct1.get_local_id(2)] = jAtom;
  }

  /*
  DPCT1065:124: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  for (int i = THREAD_ATOM_CTA / 2; i >= 32; i /= 2) {
    if (item_ct1.get_local_id(2) < i) {
      /*
      DPCT1064:126: Migrated max call is used in a macro/template definition and
      may not be valid for all macro/template uses. Adjust the code.
      */
      natoms[item_ct1.get_local_id(2)] =
          sycl::max((int)(natoms[item_ct1.get_local_id(2)]),
                    (int)(natoms[item_ct1.get_local_id(2) + i]));
    }
    /*
    DPCT1118:6: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:125: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

    // reduce in warp
  if (item_ct1.get_local_id(2) < 32) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 300
    jAtom = natoms[item_ct1.get_local_id(2)];
    /*
    DPCT1086:7: __activemask() is migrated to 0xffffffff. You may need to adjust
    the code.
    */
    const unsigned int active_mask = 0xffffffff;
    for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
      /*
      DPCT1023:8: The SYCL sub-group does not support mask options for
      dpct::permute_sub_group_by_xor. You can specify
      "--use-experimental-features=masked-sub-group-operation" to use the
      experimental helper function to migrate __shfl_xor_sync.
      */
      /*
      DPCT1096:213: The right-most dimension of the work-group used in the SYCL
      kernel that calls this function may be less than "32". The function
      "dpct::permute_sub_group_by_xor" may return an unexpected result on the
      CPU device. Modify the size of the work-group to ensure that the value of
      the right-most dimension is a multiple of "32".
      */
      jAtom = sycl::max(
          jAtom,
          (int)(dpct::permute_sub_group_by_xor(
              sycl::ext::oneapi::this_work_item::get_sub_group(), jAtom, i)));
    }
#else
    if (threadIdx.x < 16) natoms[threadIdx.x] = max(natoms[threadIdx.x], natoms[threadIdx.x+16]);
    if (threadIdx.x < 8) natoms[threadIdx.x] = max(natoms[threadIdx.x], natoms[threadIdx.x+8]);
    if (threadIdx.x < 4) natoms[threadIdx.x] = max(natoms[threadIdx.x], natoms[threadIdx.x+4]);
    if (threadIdx.x < 2) natoms[threadIdx.x] = max(natoms[threadIdx.x], natoms[threadIdx.x+2]);
    if (threadIdx.x < 1) natoms[threadIdx.x] = max(natoms[threadIdx.x], natoms[threadIdx.x+1]);

    if (threadIdx.x == 0) {
      jAtom = natoms[threadIdx.x];
    }
#endif
  }

  // one thread adds to gmem
  if (item_ct1.get_local_id(2) == 0) {
    dpct::atomic_fetch_max<sycl::access::address_space::generic_space>(
        &flags[num_cells * MAXATOMS], jAtom);
  }
}

// 1 warp per neighbor cell, 1 CTA per cell
void UpdateNeighborAtomIndices(SimGpu sim, int num_cells, int *cell_list, int *scan,
                               real_t *ncell, real_t *natoms, real_t *npos)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int iBox = item_ct1.get_group(2);
  if (cell_list != NULL)
    iBox = cell_list[item_ct1.get_group(2)];

  // load num atoms into smem

  if (item_ct1.get_local_id(2) < N_MAX_NEIGHBORS) {
    int j = item_ct1.get_local_id(2);
    int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
    ncell[j] = jBox;
    natoms[j] = sim.boxes.nAtoms[jBox];
    npos[j] = scan[iBox * N_MAX_NEIGHBORS + j];
  }

  /*
  DPCT1065:127: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // each thread finds its box index
  int local_index = item_ct1.get_local_id(2);
  int j = 0;
  while (j < N_MAX_NEIGHBORS) {
    while (j < N_MAX_NEIGHBORS && natoms[j] <= local_index) { local_index -= natoms[j]; j++; }
    if (j < N_MAX_NEIGHBORS) {
      int pos = iBox * N_MAX_NEIGHBORS * MAXATOMS + npos[j] + local_index;
      sim.neighbor_atoms[pos] = ncell[j] * MAXATOMS + local_index;
      local_index += item_ct1.get_local_range(2);
    }
  }
/*
  int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
  int pos = scan[iBox * N_MAX_NEIGHBORS + j];
  int num = sim.num_atoms[jBox];

  if (lane_id < num)
    sim.neighbor_atoms[iBox * N_MAX_NEIGHBORS * MAXATOMS + pos + lane_id] = jBox * MAXATOMS + lane_id;
*/
}

void UpdateNeighborNumAtoms(SimGpu sim, int num_cells, int *cell_list, int *scan)
{
  // only process local cells
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (tid >= num_cells) return;

  int iBox = tid;
  if (cell_list != NULL)
    iBox = cell_list[tid];

  int num_neigh = 0;
  for (int j = 0; j < N_MAX_NEIGHBORS; j++) {
    int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
    scan[iBox * N_MAX_NEIGHBORS + j] = num_neigh;
    num_neigh += sim.boxes.nAtoms[jBox];
  }

  sim.num_neigh_atoms[iBox] = num_neigh;
}

// 1 warp per cell
void UpdateAtomList(SimGpu sim, AtomListGpu list, int nCells, int *cell_offsets)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

  int iBox = tid / WARP_SIZE;
  if (iBox >= nCells) return;

  int nAtoms = sim.boxes.nAtoms[iBox];
  for (int i = lane_id; i < nAtoms; i += WARP_SIZE) {
    int off = cell_offsets[iBox] + i;
    list.atoms[off] = i;
    list.cells[off] = iBox;
  }
}

// 1 warp per cell
void UpdateBoundaryList(SimGpu sim, AtomListGpu list, int nCells, int *cell_offsets, int *cellList)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

  int iListBox = tid / WARP_SIZE;
  if (iListBox >= nCells) return;

  int iBox = cellList[iListBox];
  int nAtoms = sim.boxes.nAtoms[iBox];
  for (int i = lane_id; i < nAtoms; i += WARP_SIZE) {
    int off = cell_offsets[iListBox] + i;
    list.atoms[off] = i;
    list.cells[off] = iBox;
  }
}

/// Packs all atoms of all cells provided by gpu_cells into a single buffer (compact_atoms) in a compact manner.
/// @param [in] cellIDs stores the cell IDs of all the cell we want to extract the particles from
/// @param [in] nCells
/// @param [in] sim
/// @param [in] cellOffset
/// @param [in] shift_x
/// @param [in] shift_y
/// @param [in] shift_z
/// @param [out] compact_atoms data structure to store the compacted atoms
void LoadAtomsBufferPacked(AtomMsgSoA compactAtoms, int *cellIDs, SimGpu sim_gpu, int *cellOffset, real_t shift_x, real_t shift_y, real_t shift_z)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int iCell = tid / MAXATOMS;
  int iAtom = tid % MAXATOMS;
  assert(iCell < sim_gpu.boxes.nLocalBoxes + 1 && iCell >= 0);

  int iBox = cellIDs[iCell];
  int ii = iBox * MAXATOMS + iAtom;

  if (iAtom < sim_gpu.boxes.nAtoms[iBox])
  {
    int iBuf = cellOffset[iCell] + iAtom;

    // coalescing writes: structure of arrays
    compactAtoms.gid[iBuf] = sim_gpu.atoms.gid[ii];
    compactAtoms.type[iBuf] = sim_gpu.atoms.iSpecies[ii];
    compactAtoms.rx[iBuf] = sim_gpu.atoms.r.x[ii] + shift_x;
    compactAtoms.ry[iBuf] = sim_gpu.atoms.r.y[ii] + shift_y;
    compactAtoms.rz[iBuf] = sim_gpu.atoms.r.z[ii] + shift_z;
    compactAtoms.px[iBuf] = sim_gpu.atoms.p.x[ii];
    compactAtoms.py[iBuf] = sim_gpu.atoms.p.y[ii];
    compactAtoms.pz[iBuf] = sim_gpu.atoms.p.z[ii];
  }
}


/// @param [out] boxId Stores the boxId for each received particle
/// @param [in] nBuf number of received particles
void computeBoxIds(LinkCellGpu cells, vec_t r,
                                                 int *boxId, int nBuf)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int tid = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if(tid >= nBuf) return;
        real_t x = r.x[tid];
        real_t y = r.y[tid];
        real_t z = r.z[tid];
        int iBox = getBoxFromCoord(cells, x, y, z);
        boxId[tid] = iBox ;
}

/// @param [in] hashTable reads iOff from hashTable
/// @param [out] iOffsets iOffsets will store the offsets for each received particle
/// @param [inout] nAtoms the number of particles per Atom will be updated
void computeOffsetsNoUpdateReq(int* iOffsets, int nBuf, int* nAtoms, HashTableGpu hashTable)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int tid = item_ct1.get_local_id(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if(tid >= nBuf) return;

        int iOff = hashTable.offset[tid + hashTable.nEntriesGet];
        int iBox = iOff/MAXATOMS;
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &nAtoms[iBox], 1);

        iOffsets[tid] = iOff;
}

/// @param [out] iOffset iOffsets will store the offsets for each received particle
/// @param [inout] nAtoms the number of particles per Atom will be updated
/// @param [inout] sim Stores iOff to hashTable
template<int neighborlist>
void computeOffsetsUpdateReq(int* iOffsets, int nBuf, int *const nAtoms, int * const boxId, SimGpu sim)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int tid = item_ct1.get_local_id(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if(tid >= nBuf) return;

        int iBox = boxId[tid];
        int iOff = iBox * MAXATOMS + nAtoms[iBox];
        int index = tid - 1;
        //search for the start of the box
        while(index >= 0 && boxId[index] == iBox)
                --index;

        int offsetInBox = (tid - index - 1);
        assert(offsetInBox +nAtoms[iBox] < MAXATOMS);

        iOff += offsetInBox;
        iOffsets[tid] = iOff;

        if(neighborlist)
                if(iBox >= sim.boxes.nLocalBoxes){//remember iOff only for particles which are mapped to haloCells
                        sim.d_hashTable.offset[tid + sim.d_hashTable.nEntriesPut] = iOff;
                }else{ //particle has mapped local cell
                        *(sim.d_updateLinkCellsRequired) = 1;
                }
}

void updateNAtoms(int *nAtoms, int *boxId, int nBuf)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int tid = item_ct1.get_local_id(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if(tid >= nBuf) return;

        int iBox = boxId[tid];
        int index = tid - 1;
        //search for the start of the box
        if(tid+1 == nBuf || boxId[tid+1] != iBox) //last thread in the box updates nAtoms
        {
           while(index >= 0 && boxId[index] == iBox)
                --index;

           int nAddedParticles = (tid - index);

           nAtoms[iBox] += nAddedParticles;
        }
}

void computeOffsets(int nlUpdateRequired, SimFlat *sim, vec_t r, int *d_iOffset,
                    int *d_boxId, int nBuf, dpct::queue_ptr stream)
{
  if (nBuf == 0) return;
  int grid = (nBuf + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  // computeBoxIds(), compute, ... are required to assure sequential ordering!
  if(nlUpdateRequired){

     /*
     DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
     limit. To get the device limit, query info::device::max_work_group_size.
     Adjust the work-group size if needed.
     */
    stream->submit([&](sycl::handler &cgh) {
      LinkCellGpu sim_gpu_boxes_ct0 = sim->gpu.boxes;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         computeBoxIds(sim_gpu_boxes_ct0, r, d_boxId, nBuf);
                       });
    }); // fill d_boxId with iBox for each atom
     if(sim->method == THREAD_ATOM_NL || sim->method == WARP_ATOM_NL || sim->usePairlist){
        //compute iOff for each particle and store the result to iOffsetOut
        /*
        DPCT1049:11: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      stream->submit([&](sycl::handler &cgh) {
        auto sim_gpu_boxes_nAtoms_ct2 = sim->gpu.boxes.nAtoms;
        SimGpu sim_gpu_ct4 = sim->gpu;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                               sycl::range<3>(1, 1, block),
                                           sycl::range<3>(1, 1, block)),
                         [=](sycl::nd_item<3> item_ct1) {
                           computeOffsetsUpdateReq<1>(d_iOffset, nBuf,
                                                      sim_gpu_boxes_nAtoms_ct2,
                                                      d_boxId, sim_gpu_ct4);
                         });
      });
        sim->gpu.d_hashTable.nEntriesPut += nBuf;
     }else{
        //compute iOff for each particle and store the result to iOffsetOut
        /*
        DPCT1049:12: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
      stream->submit([&](sycl::handler &cgh) {
        auto sim_gpu_boxes_nAtoms_ct2 = sim->gpu.boxes.nAtoms;
        SimGpu sim_gpu_ct4 = sim->gpu;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                               sycl::range<3>(1, 1, block),
                                           sycl::range<3>(1, 1, block)),
                         [=](sycl::nd_item<3> item_ct1) {
                           computeOffsetsUpdateReq<0>(d_iOffset, nBuf,
                                                      sim_gpu_boxes_nAtoms_ct2,
                                                      d_boxId, sim_gpu_ct4);
                         });
      });
     }
     //update nAtoms
     /*
     DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
     limit. To get the device limit, query info::device::max_work_group_size.
     Adjust the work-group size if needed.
     */
    stream->submit([&](sycl::handler &cgh) {
      auto sim_gpu_boxes_nAtoms_ct0 = sim->gpu.boxes.nAtoms;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         updateNAtoms(sim_gpu_boxes_nAtoms_ct0, d_boxId, nBuf);
                       });
    });
  }else{
     //updates nAtoms and filles d_iOffset with the proper iOffs (read from hashTable)
     /*
     DPCT1049:13: The work-group size passed to the SYCL kernel may exceed the
     limit. To get the device limit, query info::device::max_work_group_size.
     Adjust the work-group size if needed.
     */
    stream->submit([&](sycl::handler &cgh) {
      auto sim_gpu_boxes_nAtoms_ct2 = sim->gpu.boxes.nAtoms;
      HashTableGpu sim_gpu_d_hashTable_ct3 = sim->gpu.d_hashTable;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         computeOffsetsNoUpdateReq(d_iOffset, nBuf,
                                                   sim_gpu_boxes_nAtoms_ct2,
                                                   sim_gpu_d_hashTable_ct3);
                       });
    });
     sim->gpu.d_hashTable.nEntriesGet += nBuf;
  }
  CUDA_GET_LAST_ERROR
}

void UnloadAtomsBufferPacked(vec_t r, vec_t p, int* type, int* gid, int nBuf, AtomsGpu atoms, int* iOffsets)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int tid = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
        if(tid >= nBuf) return;

        int iOff = iOffsets[tid];

        atoms.r.x[iOff] = r.x[tid];
        atoms.r.y[iOff] = r.y[tid];
        atoms.r.z[iOff] = r.z[tid];
        atoms.p.x[iOff] = p.x[tid];
        atoms.p.y[iOff] = p.y[tid];
        atoms.p.z[iOff] = p.z[tid];
        atoms.iSpecies[iOff] = type[tid];
        atoms.gid[iOff] = gid[tid];
}

/// @return sim.a_list.atoms and sim.a_list.cells are compacted
void UpdateCompactIndices(int *cell_indices, int nLocalBoxes, SimGpu sim)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int iBox = tid / MAXATOMS;
  int iAtom = tid % MAXATOMS;

  if (iBox < nLocalBoxes && iAtom < sim.boxes.nAtoms[iBox])
  {
    int iAtom = tid % MAXATOMS;
    int id = cell_indices[iBox] + iAtom;
    sim.a_list.atoms[id] = iAtom; // local offset within box iBox
    sim.a_list.cells[id] = iBox;
  }
}

void LoadForceBuffer(ForceMsg *buf, int nCells, int *gpu_cells, SimGpu sim, int *cell_indices)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int iCell = tid / MAXATOMS;
  int iAtom = tid % MAXATOMS;

  if (iCell < nCells) {
    int iBox = gpu_cells[iCell];
    int ii = iBox * MAXATOMS + iAtom;

    if (iAtom < sim.boxes.nAtoms[iBox])
    {
      int nBuf = cell_indices[iCell] + iAtom;
      buf[nBuf].dfEmbed = sim.eam_pot.dfEmbed[ii];
    }
  }
}

void UnloadForceBuffer(ForceMsg *buf, int nCells, int *gpu_cells, SimGpu sim, int *cell_indices)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  int iCell = tid / MAXATOMS;
  int iAtom = tid % MAXATOMS;

  if (iCell < nCells) {
    int iBox = gpu_cells[iCell];
    int ii = iBox * MAXATOMS + iAtom;

    if (iAtom < sim.boxes.nAtoms[iBox])
    {
      int nBuf = cell_indices[iCell] + iAtom;
      sim.eam_pot.dfEmbed[ii] = buf[nBuf].dfEmbed;
    }
  }
}

template<typename T>
void swap(T &a, T &b)
{
  T c = a;
  a = b;
  b = c;
}

void SetLinearIndices(SimGpu sim, int num_cells, int *cell_list, int *new_indices)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  if (warp_id >= num_cells) return;

  int iBox = cell_list[warp_id];
  int num_atoms = sim.boxes.nAtoms[iBox];
  for (int iAtom = lane_id; iAtom < num_atoms; iAtom += WARP_SIZE) {
    int iOff = iBox * MAXATOMS + iAtom;
    new_indices[iOff] = iOff;
  }
}

void SetLinearIndices(SimGpu sim, int nLocalBoxes, int nTotalBoxes, int *new_indices)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int iBox = nLocalBoxes + tid / MAXATOMS;
  int iAtom = tid % MAXATOMS;

  if (iBox < nTotalBoxes) {
    if (iAtom < sim.boxes.nAtoms[iBox]) {
      int iOff = iBox * MAXATOMS + iAtom;
      new_indices[iOff] = iOff;
    }
  }
}

#if 0
// bubble sort
__global__ void SortAtomsByGlobalId(SimGpu sim, int nTotalBoxes, int *new_indices)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int iBox = tid;

  if (iBox < nTotalBoxes) {
    int natoms = sim.num_atoms[iBox];
    for (int i = 0; i < natoms; i++) {
      int iOff = iBox * MAXATOMS + i;
      for (int j = i+1; j < natoms; j++) {
	int jOff = iBox * MAXATOMS + j;
        if (sim.atoms.gid[new_indices[iOff]] > sim.atoms.gid[new_indices[jOff]]) {
	  swap(new_indices[iOff], new_indices[jOff]);
        }
      }
    }
  }
}
#else
void bottomUpMerge(int *gid, int *A, int iLeft, int iRight, int iEnd, int *B)
{
  int i0 = iLeft;
  int i1 = iRight;

  for (int j = iLeft; j < iEnd; j++) {
    if (i0 < iRight && (i1 >= iEnd || gid[A[i0]] <= gid[A[i1]])) {
      B[j] = A[i0];
      i0++;
    }
    else {
      B[j] = A[i1];
      i1++;
    }
  }
}

// merge sort
void SortAtomsByGlobalId(SimGpu sim, int nLocalBoxes, int nTotalBoxes, int *boundary_cells, int nBoundaryCells, int *new_indices, int *tmp_sort)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int iBox;
  if (tid >= nBoundaryCells) iBox = nLocalBoxes + tid - nBoundaryCells;
    else iBox = boundary_cells[tid];

  if (iBox < nTotalBoxes && new_indices[iBox * MAXATOMS] >= 0)
  {
    int n = sim.boxes.nAtoms[iBox];
    int *A = new_indices + iBox * MAXATOMS;
    int *B = tmp_sort + iBox * MAXATOMS;
    // each 1-element run in A is already "sorted"
    // make succcessively longer sorted runs of length 2, 4, 8, etc.
    for (int width = 1; width < n; width *= 2) {
      // full or runs of length width
      for (int i = 0; i < n; i = i + 2 * width) {
        // merge two runs: A[i:i+width-1] and A[i+width:i+2*width-1] to B[]
        bottomUpMerge(sim.atoms.gid, A, i, sycl::min(i + width, n),
                      sycl::min(i + 2 * width, n), B);
      }
      // swap A and B for the next iteration
      swap(A, B);
      // now A is full of runs of length 2*width
    }

    // copy to B just in case it is new_indices array
    // TODO: avoid this copy?
    for (int i = 0; i < n; i++)
      B[i] = A[i];
  }
}
#endif

void ShuffleAtomsData(SimGpu sim, int nLocalBoxes, int nTotalBoxes, int *boundary_cells, int nBoundaryCells, int *new_indices)
{
#if !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP < 300
  __shared__ volatile real_t shfl_mem[THREAD_ATOM_CTA];		// assuming block size = THREAD_ATOM_CTA
#endif

  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int iBox;
  if (warp_id >= nBoundaryCells) iBox = nLocalBoxes + warp_id - nBoundaryCells;
    else iBox = boundary_cells[warp_id];
  int iAtom = lane_id;

  const int nAtomsInBox = (iBox < nTotalBoxes) ? sim.boxes.nAtoms[iBox] : 0;
  const unsigned int active_mask = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(),
      (0xFFFFFFFF & (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                .get_local_linear_id())) &&
              iAtom < nAtomsInBox
          ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                        .get_local_linear_id())
          : 0,
      sycl::ext::oneapi::plus<>());
  if (iBox >= nTotalBoxes || active_mask == 0 || new_indices[iBox * MAXATOMS] < 0) return;

  int iOff = iBox * MAXATOMS + iAtom;

  int id = 0;
  real_t rx = 0, ry = 0, rz = 0, px = 0, py = 0, pz = 0;

  // load into regs
  if (iAtom < nAtomsInBox) {
    id = sim.atoms.iSpecies[iOff];
    rx = sim.atoms.r.x[iOff];
    ry = sim.atoms.r.y[iOff];
    rz = sim.atoms.r.z[iOff];
    px = sim.atoms.p.x[iOff];
    py = sim.atoms.p.y[iOff];
    pz = sim.atoms.p.z[iOff];
  }

  int idx;
  if (iAtom < nAtomsInBox)
    idx = new_indices[iOff] - iBox * MAXATOMS;

#if !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP < 300
  if (iAtom < sim.boxes.nAtoms[iBox])
    sim.atoms.iSpecies[iOff] = __shfl(id, idx, (int*)shfl_mem);
  __syncthreads();
  if (iAtom < sim.boxes.nAtoms[iBox]) {
    sim.atoms.r.x[iOff] = __shfl(rx, idx, shfl_mem);
    sim.atoms.r.y[iOff] = __shfl(ry, idx, shfl_mem);
    sim.atoms.r.z[iOff] = __shfl(rz, idx, shfl_mem);
    sim.atoms.p.x[iOff] = __shfl(px, idx, shfl_mem);
    sim.atoms.p.y[iOff] = __shfl(py, idx, shfl_mem);
    sim.atoms.p.z[iOff] = __shfl(pz, idx, shfl_mem);
  }
#else
  if (iAtom < nAtomsInBox) {
    /*
    DPCT1023:14: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_sync.
    */
    /*
    DPCT1096:215: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sim.atoms.iSpecies[iOff] = dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), id, idx);
    /*
    DPCT1023:15: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_sync.
    */
    /*
    DPCT1096:216: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sim.atoms.r.x[iOff] = dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), rx, idx);
    /*
    DPCT1023:16: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_sync.
    */
    /*
    DPCT1096:217: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sim.atoms.r.y[iOff] = dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), ry, idx);
    /*
    DPCT1023:17: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_sync.
    */
    /*
    DPCT1096:218: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sim.atoms.r.z[iOff] = dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), rz, idx);
    /*
    DPCT1023:18: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_sync.
    */
    /*
    DPCT1096:219: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sim.atoms.p.x[iOff] = dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), px, idx);
    /*
    DPCT1023:19: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_sync.
    */
    /*
    DPCT1096:220: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sim.atoms.p.y[iOff] = dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), py, idx);
    /*
    DPCT1023:20: The SYCL sub-group does not support mask options for
    dpct::select_from_sub_group. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_sync.
    */
    /*
    DPCT1096:221: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::select_from_sub_group" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    sim.atoms.p.z[iOff] = dpct::select_from_sub_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), pz, idx);
  }
#endif
}

#endif
