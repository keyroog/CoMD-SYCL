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

#ifndef __GPU_REDUCE_H_
#define __GPU_REDUCE_H_

void ReduceEnergy(SimGpu sim, real_t *e_pot, real_t *e_kin, real_t *sp,
                  real_t *sk)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  int iOff = iBox * MAXATOMS + iAtom;

  real_t ep = 0;
  real_t ek = 0; 
  if (tid < sim.a_list.n) {
    int iSpecies = sim.atoms.iSpecies[iOff];
    real_t invMass = 0.5/sim.species_mass[iSpecies];
    ep = sim.atoms.e[iOff]; 
    ek = (sim.atoms.p.x[iOff] * sim.atoms.p.x[iOff] + sim.atoms.p.y[iOff] * sim.atoms.p.y[iOff] + sim.atoms.p.z[iOff] * sim.atoms.p.z[iOff]) * invMass;
  }
  
  // reduce in smem

  sp[item_ct1.get_local_id(2)] = ep;
  sk[item_ct1.get_local_id(2)] = ek;
  /*
  DPCT1065:143: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  for (int i = THREAD_ATOM_CTA / 2; i >= 32; i /= 2) {
    if (item_ct1.get_local_id(2) < i) {
      sp[item_ct1.get_local_id(2)] += sp[item_ct1.get_local_id(2) + i];
      sk[item_ct1.get_local_id(2)] += sk[item_ct1.get_local_id(2) + i];
    }
    /*
    DPCT1118:44: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:144: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
  
  // reduce in warp
  if (item_ct1.get_local_id(2) < 32) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 300
    ep = sp[item_ct1.get_local_id(2)];
    ek = sk[item_ct1.get_local_id(2)];
    /*
    DPCT1086:45: __activemask() is migrated to 0xffffffff. You may need to
    adjust the code.
    */
    const unsigned int active_mask = 0xffffffff;
    for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
      /*
      DPCT1023:46: The SYCL sub-group does not support mask options for
      dpct::permute_sub_group_by_xor. You can specify
      "--use-experimental-features=masked-sub-group-operation" to use the
      experimental helper function to migrate __shfl_xor_sync.
      */
      /*
      DPCT1096:224: The right-most dimension of the work-group used in the SYCL
      kernel that calls this function may be less than "32". The function
      "dpct::permute_sub_group_by_xor" may return an unexpected result on the
      CPU device. Modify the size of the work-group to ensure that the value of
      the right-most dimension is a multiple of "32".
      */
      ep += dpct::permute_sub_group_by_xor(
          sycl::ext::oneapi::this_work_item::get_sub_group(), ep, i);
      /*
      DPCT1023:47: The SYCL sub-group does not support mask options for
      dpct::permute_sub_group_by_xor. You can specify
      "--use-experimental-features=masked-sub-group-operation" to use the
      experimental helper function to migrate __shfl_xor_sync.
      */
      /*
      DPCT1096:225: The right-most dimension of the work-group used in the SYCL
      kernel that calls this function may be less than "32". The function
      "dpct::permute_sub_group_by_xor" may return an unexpected result on the
      CPU device. Modify the size of the work-group to ensure that the value of
      the right-most dimension is a multiple of "32".
      */
      ek += dpct::permute_sub_group_by_xor(
          sycl::ext::oneapi::this_work_item::get_sub_group(), ek, i);
    }
#else
    if (threadIdx.x < 16) sp[threadIdx.x] += sp[threadIdx.x+16];
    if (threadIdx.x < 8) sp[threadIdx.x] += sp[threadIdx.x+8];
    if (threadIdx.x < 4) sp[threadIdx.x] += sp[threadIdx.x+4];
    if (threadIdx.x < 2) sp[threadIdx.x] += sp[threadIdx.x+2];
    if (threadIdx.x < 1) sp[threadIdx.x] += sp[threadIdx.x+1];

    if (threadIdx.x < 16) sk[threadIdx.x] += sk[threadIdx.x+16];
    if (threadIdx.x < 8) sk[threadIdx.x] += sk[threadIdx.x+8];
    if (threadIdx.x < 4) sk[threadIdx.x] += sk[threadIdx.x+4];
    if (threadIdx.x < 2) sk[threadIdx.x] += sk[threadIdx.x+2];
    if (threadIdx.x < 1) sk[threadIdx.x] += sk[threadIdx.x+1];

    if (threadIdx.x == 0) {
      ep = sp[threadIdx.x];
      ek = sk[threadIdx.x];
    }
#endif
  }

  // one thread adds to gmem
  if (item_ct1.get_local_id(2) == 0) {
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(e_pot,
                                                                       ep);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(e_kin,
                                                                       ek);
  }
}
#endif
