#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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

// templated for the 1st and 3rd EAM passes
template <int step, bool spline>
/*
DPCT1110:30: The total declared local variable size in device function
EAM_Force_warp_atom exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

void EAM_Force_warp_atom(SimGpu sim, AtomListGpu list, int *smem_nl_off)
{
  // warp & lane ids
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
  int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

  int tid = item_ct1.get_group(2) * (WARP_ATOM_CTA / WARP_SIZE) + warp_id;
  if (tid >= list.n) return;

  // compute box ID and local atom ID
  int iAtom = list.atoms[tid];
  int iBox = list.cells[tid];

  // per-warp neighbor offsets

  int *nl_off = smem_nl_off + warp_id * 64;

  real_t rCut = sim.eam_pot.cutoff;
  real_t rCut2 = rCut*rCut;
  
  int iOff = iBox * MAXATOMS + iAtom;

  // init forces and energy
  real_t ifx = 0;
  real_t ify = 0;
  real_t ifz = 0;
  real_t ie = 0;
  real_t irho = 0;

  // fetch position
  real_t irx = sim.atoms.r.x[iOff];
  real_t iry = sim.atoms.r.y[iOff];
  real_t irz = sim.atoms.r.z[iOff];

  // create neighbor list
  int j = lane_id;
  int numNeigh = sim.num_neigh_atoms[iBox];
  int numSteps = (numNeigh + (WARP_SIZE-1)) / WARP_SIZE;
  int warpTotal = 0;
  for (int it = 0; it < numSteps; it++) 
  {
    int jOff;
    real_t dx, dy, dz, r2;

    // check for out of bounds
    if (j < numNeigh) {
      // index
      jOff = sim.neighbor_atoms[iBox * N_MAX_NEIGHBORS * MAXATOMS + j];

      dx = irx - sim.atoms.r.x[jOff];
      dy = iry - sim.atoms.r.y[jOff];
      dz = irz - sim.atoms.r.z[jOff];

      // distance^2
      r2 = dx*dx + dy*dy + dz*dz;
    }

    // aggregate neighbors that passes cut-off check
    // warp-scan using ballot/popc 
    uint flag = (j < numNeigh && r2 <= rCut2 && r2 > 0);  // flag(lane id)
    /*
    DPCT1086:31: __activemask() is migrated to 0xffffffff. You may need to
    adjust the code.
    */
    uint bits = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(),
        (0xffffffff & (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())) &&
                flag
            ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_id())
            : 0,
        sycl::ext::oneapi::plus<>()); // 0 1 0 1  1 1 0 0 = flag(0) flag(1) ..
                                      // flag(31)
    uint mask = bfi(0, 0xffffffff, 0, lane_id);           // bits < lane id = 1, bits > lane id = 0
    uint exc = sycl::popcount(mask & bits);             // exclusive scan

    if (flag) 
      nl_off[warpTotal + exc] = jOff;     		  // fill nl array - compacted

    warpTotal += sycl::popcount(bits); // total 1s per warp

    // move on to the next neighbor atom
    j += WARP_SIZE;
  }

  int neighbor_id = lane_id;
  for (int iters = 0; iters < 64 / WARP_SIZE; iters++) 
  {
    if (neighbor_id >= warpTotal) break;
    int jOff = nl_off[neighbor_id];

    real_t dx = irx - sim.atoms.r.x[jOff];
    real_t dy = iry - sim.atoms.r.y[jOff];
    real_t dz = irz - sim.atoms.r.z[jOff];

    real_t r2 = dx*dx + dy*dy + dz*dz;
    real_t phiTmp, dPhi, rhoTmp, dRho;
    if(!spline)
    {
        real_t r = sycl::sqrt(r2);

        if (step == 1) {
            interpolate(sim.eam_pot.phi, r, phiTmp, dPhi);
            interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
        }
        else {
            // step = 3
            interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
            dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
        }

        dPhi /= r;
    }
    else
    {
        if(step == 1) {
            interpolateSpline(sim.eam_pot.phiS, r2, phiTmp, dPhi);
            interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
        }
        else
        {
            //step 3
            interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp,dRho);
            dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
        }

    }
    // update forces
    ifx -= dPhi * dx;
    ify -= dPhi * dy;
    ifz -= dPhi * dz;

    // update energy & accumulate rhobar
    if (step == 1) {
      ie += phiTmp;
      irho += rhoTmp;
    }

    neighbor_id += WARP_SIZE;
  }

  warp_reduce<step>(ifx, ify, ifz, ie, irho);

  // single thread writes the final result
  if (lane_id == 0) {
    if (step == 1)
    {
      sim.atoms.f.x[iOff] = ifx;
      sim.atoms.f.y[iOff] = ify;
      sim.atoms.f.z[iOff] = ifz;
      sim.atoms.e[iOff] = 0.5 * ie;
      sim.eam_pot.rhobar[iOff] = irho;
    }
    else {
      // step 3
      sim.atoms.f.x[iOff] += ifx;
      sim.atoms.f.y[iOff] += ify;
      sim.atoms.f.z[iOff] += ifz;
    }
  }
}


/// templated for the 1st and 3rd EAM passes using the neighborlist
template <int step, int packSize, int maxNeighbors, bool spline>
/*
DPCT1110:32: The total declared local variable size in device function
EAM_Force_warp_atom_NL exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

void EAM_Force_warp_atom_NL(SimGpu sim, AtomListGpu list, real_t rCut2)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int tid = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2)) /
              packSize;
    if (tid >= list.n) return;
    // compute box ID and local atom ID
    const int iAtom = list.atoms[tid];
    const int iBox = list.cells[tid]; 
    const int iOff = iBox * MAXATOMS + iAtom;

    //Index in pack
    const int id = item_ct1.get_local_id(2) % packSize;

    // init forces and energy
    real_t ifx = 0;
    real_t ify = 0;
    real_t ifz = 0;
    real_t ie = 0;
    real_t irho = 0;

    if (step == 3 && id == 0) {
        ifx = sim.atoms.f.x[iOff];
        ify = sim.atoms.f.y[iOff];
        ifz = sim.atoms.f.z[iOff];
    }

    real_t *const __restrict__ rx = sim.atoms.r.x;
    real_t *const __restrict__ ry = sim.atoms.r.y;
    real_t *const __restrict__ rz = sim.atoms.r.z;

    // fetch position
    const real_t irx = rx[iOff];
    const real_t iry = ry[iOff];
    const real_t irz = rz[iOff];

    const int iLid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    const int ldNeighborList = sim.atoms.neighborList.nMaxLocal*packSize; //leading dimension

    int* neighborList = sim.atoms.neighborList.list; 
    int nNeighbors = sim.atoms.neighborList.nNeighbors[tid];

    int current = id;
    // loop over my neighboring particles within the neighbor-list
    int jOff_prefetch = neighborList[iLid];

#pragma unroll
    for (int j = 0; j < maxNeighbors/packSize; ++j) 
    { 
        real_t dx, dy, dz, r2;

        int jOff = jOff_prefetch;
        if(j + 1 < maxNeighbors/packSize)
            jOff_prefetch = neighborList[(j+1) * ldNeighborList + iLid ];

        if(current < nNeighbors)
        {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 350
            /*
            DPCT1098:136: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            dx = irx - rx[jOff];
            /*
            DPCT1098:137: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            dy = iry - ry[jOff];
            /*
            DPCT1098:138: The '*' expression is used instead of the __ldg call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            dz = irz - rz[jOff];
#else
            dx = irx - rx[jOff];
            dy = iry - ry[jOff];
            dz = irz - rz[jOff];
#endif
            // distance^2
            r2 = dx*dx + dy*dy + dz*dz;
        }
        else
            r2 = 0.0;

        current += packSize;
        // no divide by zero
        if (r2 <= rCut2 && r2 > 0.0) 
        {

            real_t phiTmp, dPhi, rhoTmp, dRho;
            if(!spline)
            {
                real_t r = sycl::sqrt(r2);

                if (step == 1) {
                    interpolate(sim.eam_pot.phi, r, phiTmp, dPhi);
                    interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
                }
                else {
                    // step = 3
                    interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
                    dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
                }

                dPhi /= r;
            }
            else
            {
                if(step == 1) {
                    interpolateSpline(sim.eam_pot.phiS, r2, phiTmp, dPhi);
                    interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
                }
                else
                {
                    //step 3
                    interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp,dRho);
                    dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
                }

            }
            // update forces
            ifx -= dPhi * dx;
            ify -= dPhi * dy;
            ifz -= dPhi * dz;

            // update energy & accumulate rhobar
            if (step == 1) {
                ie += phiTmp;
                irho += rhoTmp;
            }
        } 
    } // loop over neighbor-list

    //Reduction inside warp
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 300
    /*
    DPCT1086:33: __activemask() is migrated to 0xffffffff. You may need to
    adjust the code.
    */
    const unsigned int active_mask = 0xffffffff;
#pragma unroll
    for(int j = 1; j < 32; j *= 2)
    {
        if(packSize > j)
        {
            /*
            DPCT1023:34: The SYCL sub-group does not support mask options for
            dpct::shift_sub_group_left. You can specify
            "--use-experimental-features=masked-sub-group-operation" to use the
            experimental helper function to migrate __shfl_down_sync.
            */
            /*
            DPCT1096:200: The right-most dimension of the work-group used in the
            SYCL kernel that calls this function may be less than "32". The
            function "dpct::shift_sub_group_left" may return an unexpected
            result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of
            "32".
            */
            const real_t tmpx = dpct::shift_sub_group_left(
                sycl::ext::oneapi::this_work_item::get_sub_group(), ifx, j,
                packSize);
            /*
            DPCT1023:35: The SYCL sub-group does not support mask options for
            dpct::shift_sub_group_left. You can specify
            "--use-experimental-features=masked-sub-group-operation" to use the
            experimental helper function to migrate __shfl_down_sync.
            */
            /*
            DPCT1096:201: The right-most dimension of the work-group used in the
            SYCL kernel that calls this function may be less than "32". The
            function "dpct::shift_sub_group_left" may return an unexpected
            result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of
            "32".
            */
            const real_t tmpy = dpct::shift_sub_group_left(
                sycl::ext::oneapi::this_work_item::get_sub_group(), ify, j,
                packSize);
            /*
            DPCT1023:36: The SYCL sub-group does not support mask options for
            dpct::shift_sub_group_left. You can specify
            "--use-experimental-features=masked-sub-group-operation" to use the
            experimental helper function to migrate __shfl_down_sync.
            */
            /*
            DPCT1096:202: The right-most dimension of the work-group used in the
            SYCL kernel that calls this function may be less than "32". The
            function "dpct::shift_sub_group_left" may return an unexpected
            result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of
            "32".
            */
            const real_t tmpz = dpct::shift_sub_group_left(
                sycl::ext::oneapi::this_work_item::get_sub_group(), ifz, j,
                packSize);
            if(step == 1)
            {
                /*
                DPCT1023:37: The SYCL sub-group does not support mask options
                for dpct::shift_sub_group_left. You can specify
                "--use-experimental-features=masked-sub-group-operation" to use
                the experimental helper function to migrate __shfl_down_sync.
                */
                /*
                DPCT1096:203: The right-most dimension of the work-group used in
                the SYCL kernel that calls this function may be less than "32".
                The function "dpct::shift_sub_group_left" may return an
                unexpected result on the CPU device. Modify the size of the
                work-group to ensure that the value of the right-most dimension
                is a multiple of "32".
                */
                const real_t tmpe = dpct::shift_sub_group_left(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), ie, j,
                    packSize);
                /*
                DPCT1023:38: The SYCL sub-group does not support mask options
                for dpct::shift_sub_group_left. You can specify
                "--use-experimental-features=masked-sub-group-operation" to use
                the experimental helper function to migrate __shfl_down_sync.
                */
                /*
                DPCT1096:204: The right-most dimension of the work-group used in
                the SYCL kernel that calls this function may be less than "32".
                The function "dpct::shift_sub_group_left" may return an
                unexpected result on the CPU device. Modify the size of the
                work-group to ensure that the value of the right-most dimension
                is a multiple of "32".
                */
                const real_t tmprho = dpct::shift_sub_group_left(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), irho, j,
                    packSize);
                ie += tmpe;
                irho += tmprho;
            }
            ifx += tmpx;
            ify += tmpy;
            ifz += tmpz;
        }
    }
#else
    __shared__ real_t smem[THREAD_ATOM_CTA];
    for(int j = 1; j < 32; j *= 2)
    {
        if(packSize > j)
        {
            const real_t tmpx = __shfl_down(ifx, j, packSize, smem);
            const real_t tmpy = __shfl_down(ify, j, packSize, smem);
            const real_t tmpz = __shfl_down(ifz, j, packSize, smem);
            if(step == 1)
            {
                const real_t tmpe = __shfl_down(ie, j, packSize, smem);
                const real_t tmprho = __shfl_down(irho, j, packSize, smem);
                ie += tmpe;
                irho += tmprho;
            }
            ifx += tmpx;
            ify += tmpy;
            ifz += tmpz;
        }
    }
#endif

    if(id == 0)
    {
        sim.atoms.f.x[iOff] = ifx;
        sim.atoms.f.y[iOff] = ify;
        sim.atoms.f.z[iOff] = ifz;

        if (step == 1) {
            sim.atoms.e[iOff] = 0.5 * ie;
            sim.eam_pot.rhobar[iOff] = irho;
        }
    }
}

