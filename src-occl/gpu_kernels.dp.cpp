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
#include <stdio.h>
#include <assert.h>

#include "CoMDTypes.h"
#include "haloExchange.h"

#include "gpu_types.h"
#include "gpu_timestep.h"
#include "defines.h"

#include "gpu_utility.h"

#include "gpu_common.h"
#include "gpu_redistribute.h"
#include "gpu_neighborList.h"

#include "gpu_lj_thread_atom.h"
#include "gpu_lj_cta_cell.h"
#include "gpu_eam_thread_atom.h"
#include "gpu_eam_warp_atom.h"
#include "gpu_eam_cta_cell.h"

#include "gpu_scan.h"
#include "gpu_reduce.h"

#include "hashTable.h"

#undef EXTERN_C
#define EXTERN_C extern "C"
#include "gpu_kernels.h"
#undef EXTERN_C
extern "C"
{
#include "parallel.h"
}

extern "C"
void ljForceGpu(SimGpu * sim, int interpolation, int num_cells, int * cells_list, real_t plcutoff, int method)
{
    if(method != CTA_CELL)
        /*
        DPCT1026:145: The call to cudaDeviceSetCacheConfig was removed because
        SYCL currently does not support setting cache config on devices.
        */
        ;
    else
        /*
        DPCT1026:146: The call to cudaDeviceSetCacheConfig was removed because
        SYCL currently does not support setting cache config on devices.
        */
        ;
  if(method == THREAD_ATOM)
  {
      int grid = (sim->a_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
      int block = THREAD_ATOM_CTA;
      if(interpolation == 0)
          /*
          DPCT1049:48: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        SimGpu sim_ct0 = *sim;
        AtomListGpu sim_a_list_ct1 = (*sim).a_list;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                               sycl::range<3>(1, 1, block),
                                           sycl::range<3>(1, 1, block)),
                         [=](sycl::nd_item<3> item_ct1) {
                           LJ_Force_thread_atom(sim_ct0, sim_a_list_ct1);
                         });
      });
      else
          /*
          DPCT1049:49: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        SimGpu sim_ct0 = *sim;
        AtomListGpu sim_a_list_ct1 = (*sim).a_list;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                               sycl::range<3>(1, 1, block),
                                           sycl::range<3>(1, 1, block)),
                         [=](sycl::nd_item<3> item_ct1) {
                           LJ_Force_thread_atom_interpolation(sim_ct0,
                                                              sim_a_list_ct1);
                         });
      });
  }
  else if(method == CTA_CELL)
  {
      if(!sim->usePairlist)
      {
          int grid = num_cells;
          int block = CTA_CELL_CTA;

          real_t sigma = sim->lj_pot.sigma;

          const real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

          /*
          DPCT1049:50: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
      dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:185: 'SHARED_SIZE_CTA_CELL' expression was replaced with
        a value. Modify the code to use the original expression, provided
        in comments, if it is correct.
        */
        sycl::local_accessor<real_t, 1> otherX_acc_ct1(
            sycl::range<1>(128 /*SHARED_SIZE_CTA_CELL*/), cgh);
        /*
        DPCT1101:186: 'SHARED_SIZE_CTA_CELL' expression was replaced with
        a value. Modify the code to use the original expression, provided
        in comments, if it is correct.
        */
        sycl::local_accessor<real_t, 1> otherY_acc_ct1(
            sycl::range<1>(128 /*SHARED_SIZE_CTA_CELL*/), cgh);
        /*
        DPCT1101:187: 'SHARED_SIZE_CTA_CELL' expression was replaced with
        a value. Modify the code to use the original expression, provided
        in comments, if it is correct.
        */
        sycl::local_accessor<real_t, 1> otherZ_acc_ct1(
            sycl::range<1>(128 /*SHARED_SIZE_CTA_CELL*/), cgh);

        SimGpu sim_ct0 = *sim;
        auto sim_lj_pot_cutoff_sim_lj_pot_cutoff_ct2 =
            sim->lj_pot.cutoff * sim->lj_pot.cutoff;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                  sycl::range<3>(1, 1, block),
                              sycl::range<3>(1, 1, block)),
            [=](sycl::nd_item<3> item_ct1) {
              LJ_Force_cta_cell(
                  sim_ct0, cells_list, sim_lj_pot_cutoff_sim_lj_pot_cutoff_ct2,
                  s6,
                  otherX_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  otherY_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  otherZ_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
      }
      else
      {
          int grid = num_cells;
          int block = CTA_CELL_CTA;

          real_t sigma = sim->lj_pot.sigma;

          const real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

          if(sim->genPairlist)
          {
              /*
              DPCT1049:51: The work-group size passed to the SYCL kernel may
              exceed the limit. To get the device limit, query
              info::device::max_work_group_size. Adjust the work-group size if
              needed.
              */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          /*
          DPCT1101:188: 'CTA_CELL_CTA' expression was replaced with a
          value. Modify the code to use the original expression,
          provided in comments, if it is correct.
          */
          sycl::local_accessor<real_t, 1> otherX_acc_ct1(
              sycl::range<1>(128 /*CTA_CELL_CTA*/), cgh);
          /*
          DPCT1101:189: 'CTA_CELL_CTA' expression was replaced with a
          value. Modify the code to use the original expression,
          provided in comments, if it is correct.
          */
          sycl::local_accessor<real_t, 1> otherY_acc_ct1(
              sycl::range<1>(128 /*CTA_CELL_CTA*/), cgh);
          /*
          DPCT1101:190: 'CTA_CELL_CTA' expression was replaced with a
          value. Modify the code to use the original expression,
          provided in comments, if it is correct.
          */
          sycl::local_accessor<real_t, 1> otherZ_acc_ct1(
              sycl::range<1>(128 /*CTA_CELL_CTA*/), cgh);

          SimGpu sim_ct0 = *sim;
          auto sim_lj_pot_cutoff_sim_lj_pot_cutoff_ct2 =
              sim->lj_pot.cutoff * sim->lj_pot.cutoff;

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                LJ_Force_cta_cell_pairlist<true, PAIRLIST_ATOMS_PER_INT>(
                    sim_ct0, cells_list,
                    sim_lj_pot_cutoff_sim_lj_pot_cutoff_ct2, s6, plcutoff,
                    otherX_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    otherY_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    otherZ_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
              sim->genPairlist = 0;
              sim->atoms.neighborList.forceRebuildFlag = 0;
          }
          else
          {
              /*
              DPCT1049:52: The work-group size passed to the SYCL kernel may
              exceed the limit. To get the device limit, query
              info::device::max_work_group_size. Adjust the work-group size if
              needed.
              */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          /*
          DPCT1101:191: 'CTA_CELL_CTA' expression was replaced with a
          value. Modify the code to use the original expression,
          provided in comments, if it is correct.
          */
          sycl::local_accessor<real_t, 1> otherX_acc_ct1(
              sycl::range<1>(128 /*CTA_CELL_CTA*/), cgh);
          /*
          DPCT1101:192: 'CTA_CELL_CTA' expression was replaced with a
          value. Modify the code to use the original expression,
          provided in comments, if it is correct.
          */
          sycl::local_accessor<real_t, 1> otherY_acc_ct1(
              sycl::range<1>(128 /*CTA_CELL_CTA*/), cgh);
          /*
          DPCT1101:193: 'CTA_CELL_CTA' expression was replaced with a
          value. Modify the code to use the original expression,
          provided in comments, if it is correct.
          */
          sycl::local_accessor<real_t, 1> otherZ_acc_ct1(
              sycl::range<1>(128 /*CTA_CELL_CTA*/), cgh);

          SimGpu sim_ct0 = *sim;
          auto sim_lj_pot_cutoff_sim_lj_pot_cutoff_ct2 =
              sim->lj_pot.cutoff * sim->lj_pot.cutoff;

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                LJ_Force_cta_cell_pairlist<false, PAIRLIST_ATOMS_PER_INT>(
                    sim_ct0, cells_list,
                    sim_lj_pot_cutoff_sim_lj_pot_cutoff_ct2, s6, plcutoff,
                    otherX_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    otherY_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    otherZ_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
          }

      }
  }
  if(method == CTA_CELL)
      /*
      DPCT1026:147: The call to cudaDeviceSetCacheConfig was removed because
      SYCL currently does not support setting cache config on devices.
      */
      ;
}

template<int step>
int compute_eam_smem_size(SimGpu sim)
{
  int smem = 0;

  // neighbors data
  // positions
  smem += 3 * sizeof(real_t) * CTA_CELL_CTA;

  // embed force
  if (step == 3)
    smem += sizeof(real_t) * CTA_CELL_CTA;

  // local data
  // forces
  smem += 3 * sim.max_atoms_cell * sizeof(real_t);

  // positions
  smem += 3 * sim.max_atoms_cell * sizeof(real_t);

  // ie, irho
  if (step == 1)
    smem += 2 * sim.max_atoms_cell * sizeof(real_t);

  // local neighbor list
  smem += (CTA_CELL_CTA / WARP_SIZE) * 64 * sizeof(char);

  return smem;
}

template <int step>
void eamForce(SimGpu sim, AtomListGpu atoms_list, int num_cells,
              int *cells_list, int method, int spline,
              dpct::queue_ptr stream = &dpct::get_in_order_queue())
{
    assert(method < CPU_NL);
    if (method == CTA_CELL) {
      if (num_cells == 0) return;
    }
    else
      if (atoms_list.n == 0) return;

    if(spline)
    {
        if (method == THREAD_ATOM) {

            int grid = (atoms_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
            int block = THREAD_ATOM_CTA;
            /*
            DPCT1049:53: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                 sycl::range<3>(1, 1, block),
                                             sycl::range<3>(1, 1, block)),
                           [=](sycl::nd_item<3> item_ct1) {
                             EAM_Force_thread_atom<step, true>(sim, atoms_list);
                           });
        }
        else if (method == WARP_ATOM) {
            int block = WARP_ATOM_CTA;
            int grid = (atoms_list.n + (WARP_ATOM_CTA/WARP_SIZE)-1)/ (WARP_ATOM_CTA/WARP_SIZE);
            /*
            DPCT1049:54: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:194: '(WARP_ATOM_CTA / WARP_SIZE) * 64' expression was
        replaced with a value. Modify the code to use the original
        expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<int, 1> smem_nl_off_acc_ct1(
            sycl::range<1>(256 /*(WARP_ATOM_CTA / WARP_SIZE) * 64*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                  sycl::range<3>(1, 1, block),
                              sycl::range<3>(1, 1, block)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              EAM_Force_warp_atom<step, true>(
                  sim, atoms_list,
                  smem_nl_off_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        }
        else if (method == CTA_CELL) {
            /*
            DPCT1026:148: The call to cudaDeviceSetCacheConfig was removed
            because SYCL currently does not support setting cache config on
            devices.
            */
; // necessary for good occupancy
            int block = CTA_CELL_CTA;
            int grid = num_cells;
            int smem = compute_eam_smem_size<step>(sim);
            /*
            DPCT1049:55: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smem), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                EAM_Force_cta_cell<step, true>(
                    sim, cells_list,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
      }
            /*
            DPCT1026:149: The call to cudaDeviceSetCacheConfig was removed
            because SYCL currently does not support setting cache config on
            devices.
            */
        } else if (method == THREAD_ATOM_NL) {
            int grid = (atoms_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
            int block = THREAD_ATOM_CTA;
            /*
            DPCT1049:56: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                 sycl::range<3>(1, 1, block),
                                             sycl::range<3>(1, 1, block)),
                           [=](sycl::nd_item<3> item_ct1) {
                             EAM_Force_thread_atom_NL<step, true>(sim,
                                                                  atoms_list);
                           });
        }else if (method == WARP_ATOM_NL) {
            int grid = (atoms_list.n * KERNEL_PACKSIZE + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
            int block = THREAD_ATOM_CTA;
            /*
            DPCT1049:57: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      stream->submit([&](sycl::handler &cgh) {
        auto sim_eam_pot_cutoff_sim_eam_pot_cutoff_ct2 =
            sim.eam_pot.cutoff * sim.eam_pot.cutoff;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                  sycl::range<3>(1, 1, block),
                              sycl::range<3>(1, 1, block)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              EAM_Force_warp_atom_NL<step, KERNEL_PACKSIZE, MAXNEIGHBORLISTSIZE,
                                     true>(
                  sim, atoms_list, sim_eam_pot_cutoff_sim_eam_pot_cutoff_ct2);
            });
      });
        }

    }
    else
    {
        if (method == THREAD_ATOM) {

            int grid = (atoms_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
            int block = THREAD_ATOM_CTA;
            /*
            DPCT1049:58: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                 sycl::range<3>(1, 1, block),
                                             sycl::range<3>(1, 1, block)),
                           [=](sycl::nd_item<3> item_ct1) {
                             EAM_Force_thread_atom<step, false>(sim,
                                                                atoms_list);
                           });
        }
        else if (method == WARP_ATOM) {
            int block = WARP_ATOM_CTA;
            int grid = (atoms_list.n + (WARP_ATOM_CTA/WARP_SIZE)-1)/ (WARP_ATOM_CTA/WARP_SIZE);
            /*
            DPCT1049:59: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:205: '(WARP_ATOM_CTA / WARP_SIZE) * 64' expression was
        replaced with a value. Modify the code to use the original
        expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<int, 1> smem_nl_off_acc_ct1(
            sycl::range<1>(256 /*(WARP_ATOM_CTA / WARP_SIZE) * 64*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                  sycl::range<3>(1, 1, block),
                              sycl::range<3>(1, 1, block)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              EAM_Force_warp_atom<step, false>(
                  sim, atoms_list,
                  smem_nl_off_acc_ct1
                      .get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
        }
        else if (method == CTA_CELL) {
            /*
            DPCT1026:150: The call to cudaDeviceSetCacheConfig was removed
            because SYCL currently does not support setting cache config on
            devices.
            */
; // necessary for good occupancy
            int block = CTA_CELL_CTA;
            int grid = num_cells;
            int smem = compute_eam_smem_size<step>(sim);
            /*
            DPCT1049:60: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(smem), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                EAM_Force_cta_cell<step, false>(
                    sim, cells_list,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
      }
            /*
            DPCT1026:151: The call to cudaDeviceSetCacheConfig was removed
            because SYCL currently does not support setting cache config on
            devices.
            */
        } else if (method == THREAD_ATOM_NL) {
            int grid = (atoms_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
            int block = THREAD_ATOM_CTA;
            /*
            DPCT1049:61: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                                 sycl::range<3>(1, 1, block),
                                             sycl::range<3>(1, 1, block)),
                           [=](sycl::nd_item<3> item_ct1) {
                             EAM_Force_thread_atom_NL<step, false>(sim,
                                                                   atoms_list);
                           });
        }else if (method == WARP_ATOM_NL) {
            int grid = (atoms_list.n * KERNEL_PACKSIZE + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
            int block = THREAD_ATOM_CTA;
            /*
            DPCT1049:62: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
      stream->submit([&](sycl::handler &cgh) {
        auto sim_eam_pot_cutoff_sim_eam_pot_cutoff_ct2 =
            sim.eam_pot.cutoff * sim.eam_pot.cutoff;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                  sycl::range<3>(1, 1, block),
                              sycl::range<3>(1, 1, block)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
              EAM_Force_warp_atom_NL<step, KERNEL_PACKSIZE, MAXNEIGHBORLISTSIZE,
                                     false>(
                  sim, atoms_list, sim_eam_pot_cutoff_sim_eam_pot_cutoff_ct2);
            });
      });
        }
    }
    CUDA_GET_LAST_ERROR
}

template <>
void eamForce<2>(SimGpu sim, AtomListGpu atoms_list, int num_cells,
                 int *cells_list, int method, int spline,
                 dpct::queue_ptr stream)
{
  assert(method < CPU_NL);
  if (method == CTA_CELL) {
    if (num_cells == 0) return;
  }
  else
    if (atoms_list.n == 0) return;

  if (method == THREAD_ATOM || method == WARP_ATOM || method == THREAD_ATOM_NL || method == WARP_ATOM_NL) {
    int grid = (atoms_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
    int block = THREAD_ATOM_CTA;
    /*
    DPCT1049:63: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                               sycl::range<3>(1, 1, block),
                                           sycl::range<3>(1, 1, block)),
                         [=](sycl::nd_item<3> item_ct1) {
                           EAM_Force_thread_atom2(sim, atoms_list);
                         });
  }
  else if (method == CTA_CELL) {
    int grid = num_cells;
    int block = CTA_CELL_CTA;
    /*
    DPCT1049:64: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                               sycl::range<3>(1, 1, block),
                                           sycl::range<3>(1, 1, block)),
                         [=](sycl::nd_item<3> item_ct1) {
                           EAM_Force_cta_cell2(sim, cells_list);
                         });
  }
  CUDA_GET_LAST_ERROR
}

extern "C" void updateNeighborsGpuAsync(SimGpu sim, int *temp, int nCells,
                                        int *cellList, dpct::queue_ptr stream)
{
  // update # of neighbor atoms per cell - 1 thread per cell
  int block = THREAD_ATOM_CTA;
  int grid = (nCells + (block-1))/ block;
  /*
  DPCT1049:65: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         UpdateNeighborNumAtoms(sim, nCells, cellList, temp);
                       });

  // update atom indices - 1 CTA per cell
  grid = nCells;
  /*
  DPCT1049:66: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:206: 'N_MAX_NEIGHBORS' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<real_t, 1> ncell_acc_ct1(
        sycl::range<1>(27 /*N_MAX_NEIGHBORS*/), cgh);
    /*
    DPCT1101:207: 'N_MAX_NEIGHBORS' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<real_t, 1> natoms_acc_ct1(
        sycl::range<1>(27 /*N_MAX_NEIGHBORS*/), cgh);
    /*
    DPCT1101:208: 'N_MAX_NEIGHBORS' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<real_t, 1> npos_acc_ct1(
        sycl::range<1>(27 /*N_MAX_NEIGHBORS*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                              sycl::range<3>(1, 1, block),
                          sycl::range<3>(1, 1, block)),
        [=](sycl::nd_item<3> item_ct1) {
          UpdateNeighborAtomIndices(
              sim, nCells, cellList, temp,
              ncell_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
              natoms_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
              npos_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
        });
  });

  CUDA_GET_LAST_ERROR
}

extern "C"
extern "C" void updateNeighborsGpu(SimGpu sim, int *temp)
{
  // update # of neighbor atoms per cell - 1 thread per cell
  int block = THREAD_ATOM_CTA;
  int grid = (sim.boxes.nLocalBoxes + (block-1))/ block;
  /*
  DPCT1049:67: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                            sycl::range<3>(1, 1, block),
                        sycl::range<3>(1, 1, block)),
      [=](sycl::nd_item<3> item_ct1) {
        UpdateNeighborNumAtoms(sim, sim.boxes.nLocalBoxes, NULL, temp);
      });

  // update atom indices - 1 CTA per cell
  grid = sim.boxes.nLocalBoxes;
  /*
  DPCT1049:68: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:209: 'N_MAX_NEIGHBORS' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<real_t, 1> ncell_acc_ct1(
        sycl::range<1>(27 /*N_MAX_NEIGHBORS*/), cgh);
    /*
    DPCT1101:210: 'N_MAX_NEIGHBORS' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<real_t, 1> natoms_acc_ct1(
        sycl::range<1>(27 /*N_MAX_NEIGHBORS*/), cgh);
    /*
    DPCT1101:211: 'N_MAX_NEIGHBORS' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<real_t, 1> npos_acc_ct1(
        sycl::range<1>(27 /*N_MAX_NEIGHBORS*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                              sycl::range<3>(1, 1, block),
                          sycl::range<3>(1, 1, block)),
        [=](sycl::nd_item<3> item_ct1) {
          UpdateNeighborAtomIndices(
              sim, sim.boxes.nLocalBoxes, NULL, temp,
              ncell_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
              natoms_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
              npos_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
        });
  });

  CUDA_GET_LAST_ERROR
}

extern "C"
extern "C" void eamForce1Gpu(SimGpu sim, int method, int spline)
{
  /*
  DPCT1026:152: The call to cudaDeviceSetCacheConfig was removed because SYCL
  currently does not support setting cache config on devices.
  */
  eamForce<1>(sim, sim.a_list, sim.boxes.nLocalBoxes, NULL, method, spline);
  CUDA_GET_LAST_ERROR
}

// async launch, latency hiding opt
extern "C" void eamForce1GpuAsync(SimGpu sim, AtomListGpu atoms_list,
                                  int num_cells, int *cells_list, int method,
                                  dpct::queue_ptr stream, int spline)
{
  /*
  DPCT1026:153: The call to cudaDeviceSetCacheConfig was removed because SYCL
  currently does not support setting cache config on devices.
  */
  eamForce<1>(sim, atoms_list, num_cells, cells_list, method, spline, stream);
  CUDA_GET_LAST_ERROR
}

extern "C"
extern "C" void eamForce2Gpu(SimGpu sim, int method, int spline)
{
  eamForce<2>(sim, sim.a_list, sim.boxes.nLocalBoxes, NULL, method, spline);
  CUDA_GET_LAST_ERROR
}

extern "C" void eamForce2GpuAsync(SimGpu sim, AtomListGpu atoms_list,
                                  int num_cells, int *cells_list, int method,
                                  dpct::queue_ptr stream, int spline)
{
  eamForce<2>(sim, atoms_list, num_cells, cells_list, method, spline, stream);
  CUDA_GET_LAST_ERROR
}

extern "C"
extern "C" void eamForce3Gpu(SimGpu sim, int method, int spline)
{
  eamForce<3>(sim, sim.a_list, sim.boxes.nLocalBoxes, NULL, method, spline);
  CUDA_GET_LAST_ERROR
}

extern "C" void eamForce3GpuAsync(SimGpu sim, AtomListGpu atoms_list,
                                  int num_cells, int *cells_list, int method,
                                  dpct::queue_ptr stream, int spline)
{
  eamForce<3>(sim, atoms_list, num_cells, cells_list, method, spline, stream);
  CUDA_GET_LAST_ERROR
}

extern "C"
extern "C" void advanceVelocityGpu(SimGpu sim, real_t dt)
{
  if (sim.a_list.n == 0) return;

  int grid = (sim.a_list.n + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  /*
  DPCT1049:69: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                            sycl::range<3>(1, 1, block),
                        sycl::range<3>(1, 1, block)),
      [=](sycl::nd_item<3> item_ct1) {
        AdvanceVelocity(sim, dt);
      });

  CUDA_GET_LAST_ERROR
}

extern "C"
extern "C" void advancePositionGpu(SimGpu* sim, real_t dt)
{
  if (sim->a_list.n == 0) return;

  int grid = (sim->a_list.n + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  /*
  DPCT1049:70: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    SimGpu sim_ct0 = *sim;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       AdvancePosition(sim_ct0, dt);
                     });
  });

  //TODO: this functionality should not be here. It seems like a nasty side-effect. REFACTORING!
  sim->atoms.neighborList.updateNeighborListRequired = -1; //next call to neighborListUpdateRequired() will loop over all particles

  CUDA_GET_LAST_ERROR
}


/// Launch one thread per cell and fill cellOffsets with the number of atoms of each cell (used for scan).
/// @param [out] cellOffsets
/// @param [in] nCells
/// @param [in] cellList ID of every cell
/// @param [in] num_atoms number of atoms for each cell (of size nTotalBoxes)
void fill(int *cellOffsets, int nCells, int *cellList, int *num_atoms)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (tid < nCells)
    cellOffsets[tid] = num_atoms[cellList[tid]];
  else if (tid == nCells)
    cellOffsets[tid] = 0;
}

void fill(int *cellOffsets, int nCells, int *num_atoms)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (tid < nCells)
    cellOffsets[tid] = num_atoms[tid];
  else if (tid == nCells)
    cellOffsets[tid] = 0;
}

/// Computes the scan of number of atoms of the speciefied cell IDs (specified by cellList).
/// @param [in] nCell number of cells
/// @param [in] cellList ID of every cell
/// @param [in] num_atoms number of atoms for each cell (of size nTotalBoxes)
/// @param [out] nAtomsOffset result of the scan.
/// @param [out] work Temporary array with minimum size of ceil((nCell+1)/256)
void scanCells(int *d_cellOffsets, int nCells, int *cellList, int *num_atoms,
               int *work, dpct::queue_ptr stream = &dpct::get_in_order_queue())
{
  // natoms[i] = num_atoms[cellList[i]]
  int block = THREAD_ATOM_CTA;
  int grid = (nCells + 1 + block-1) / block;
  /*
  DPCT1049:71: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         fill(d_cellOffsets, nCells, cellList, num_atoms);
                       });
  CUDA_GET_LAST_ERROR

  // scan to compute linear index
  scan(d_cellOffsets, nCells + 1, work, stream);

  CUDA_GET_LAST_ERROR
}

void scanCells(int *natoms_buf, int nCells, int *num_atoms, int *work,
               dpct::queue_ptr stream = &dpct::get_in_order_queue())
{
  // natoms[i] = num_atoms[i]
  int block = THREAD_ATOM_CTA;
  int grid = (nCells + 1 + block-1) / block;
  /*
  DPCT1049:72: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         fill(natoms_buf, nCells, num_atoms);
                       });
  CUDA_GET_LAST_ERROR

  // scan to compute linear index
  scan(natoms_buf, nCells + 1, work, stream);

  CUDA_GET_LAST_ERROR
}

void BuildAtomLists(SimFlat *s)
{
  int nCells = s->boxes->nLocalBoxes;
  int n_interior_cells = s->boxes->nLocalBoxes - s->n_boundary_cells;

  int size = nCells+1;
  if (size % 256 != 0) size = ((size + 255)/256)*256;

  CUDA_GET_LAST_ERROR
  int *cell_offsets1;
  int *cell_offsets2;
  cell_offsets1 = sycl::malloc_device<int>(size, dpct::get_in_order_queue());
  cell_offsets2 = sycl::malloc_device<int>(size, dpct::get_in_order_queue());
  int *partial_sums;
  partial_sums = sycl::malloc_device<int>(size, dpct::get_in_order_queue());
  CUDA_GET_LAST_ERROR

  scanCells(cell_offsets1, nCells, s->gpu.boxes.nAtoms, partial_sums);
  CUDA_GET_LAST_ERROR

  int block = THREAD_ATOM_CTA;
  int grid = (nCells + (block/WARP_SIZE)-1)/(block/WARP_SIZE);
  /*
  DPCT1049:73: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    SimGpu s_gpu_ct0 = s->gpu;
    AtomListGpu s_gpu_a_list_ct1 = s->gpu.a_list;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       UpdateAtomList(s_gpu_ct0, s_gpu_a_list_ct1, nCells,
                                      cell_offsets1);
                     });
  });
  CUDA_GET_LAST_ERROR

  // build interior & boundary lists
  scanCells(cell_offsets1, s->n_boundary_cells, s->boundary_cells, s->gpu.boxes.nAtoms, partial_sums);
  scanCells(cell_offsets2, n_interior_cells, s->interior_cells, s->gpu.boxes.nAtoms, partial_sums);

  grid = (s->n_boundary_cells + (block/WARP_SIZE)-1)/(block/WARP_SIZE);
  /*
  DPCT1049:74: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    SimGpu s_gpu_ct0 = s->gpu;
    AtomListGpu s_gpu_b_list_ct1 = s->gpu.b_list;
    auto s_n_boundary_cells_ct2 = s->n_boundary_cells;
    auto s_boundary_cells_ct4 = s->boundary_cells;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       UpdateBoundaryList(s_gpu_ct0, s_gpu_b_list_ct1,
                                          s_n_boundary_cells_ct2, cell_offsets1,
                                          s_boundary_cells_ct4);
                     });
  });
  CUDA_GET_LAST_ERROR

  grid = (n_interior_cells + (block/WARP_SIZE)-1)/(block/WARP_SIZE);
  /*
  DPCT1049:75: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    SimGpu s_gpu_ct0 = s->gpu;
    AtomListGpu s_gpu_i_list_ct1 = s->gpu.i_list;
    auto s_interior_cells_ct4 = s->interior_cells;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       UpdateBoundaryList(s_gpu_ct0, s_gpu_i_list_ct1,
                                          n_interior_cells, cell_offsets2,
                                          s_interior_cells_ct4);
                     });
  });
  CUDA_GET_LAST_ERROR

  dpct::get_in_order_queue().memcpy(
      &s->gpu.b_list.n, cell_offsets1 + s->n_boundary_cells, sizeof(int));
  dpct::get_in_order_queue()
      .memcpy(&s->gpu.i_list.n, cell_offsets2 + n_interior_cells, sizeof(int))
      .wait();

  dpct::dpct_free(partial_sums, dpct::get_in_order_queue());
  dpct::dpct_free(cell_offsets1, dpct::get_in_order_queue());
  dpct::dpct_free(cell_offsets2, dpct::get_in_order_queue());

  CUDA_GET_LAST_ERROR
}

/// \details
/// This is the first step in returning data structures to a consistent
/// state after the atoms move each time step.  First we discard all
/// atoms in the halo link cells.  These are all atoms that are
/// currently stored on other ranks and so any information we have about
/// them is stale.  Next, we move any atoms that have crossed link cell
/// boundaries into their new link cells.  It is likely that some atoms
/// will be moved into halo link cells.  Since we have deleted halo
/// atoms from other tasks, it is clear that any atoms that are in halo
/// cells at the end of this routine have just transitioned from local
/// to halo atoms.  Such atom must be sent to other tasks by a halo
/// exchange to avoid being lost.
/// \see redistributeAtoms
extern "C"
extern "C" void updateLinkCellsGpu(SimFlat *sim)
{
  if (sim->gpu.a_list.n == 0) return;

  int *flags = sim->flags;
  //empty haloCells
  dpct::get_in_order_queue()
      .memset(sim->gpu.boxes.nAtoms + sim->boxes->nLocalBoxes, 0,
              (sim->boxes->nTotalBoxes - sim->boxes->nLocalBoxes) * sizeof(int))
      .wait();

  // set all flags to 0
  dpct::get_in_order_queue()
      .memset(flags, 0, sim->boxes->nTotalBoxes * MAXATOMS * sizeof(int))
      .wait();

  // 1 thread updates 1 atom
  int grid = (sim->gpu.a_list.n + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  if(sim->usePairlist)
      /*
      DPCT1049:76: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      SimGpu sim_gpu_ct0 = sim->gpu;
      LinkCellGpu sim_gpu_boxes_ct1 = sim->gpu.boxes;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         UpdateLinkCells<true>(sim_gpu_ct0, sim_gpu_boxes_ct1,
                                               flags);
                       });
    });
  else
      /*
      DPCT1049:77: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      SimGpu sim_gpu_ct0 = sim->gpu;
      LinkCellGpu sim_gpu_boxes_ct1 = sim->gpu.boxes;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         UpdateLinkCells<false>(sim_gpu_ct0, sim_gpu_boxes_ct1,
                                                flags);
                       });
    });
  CUDA_GET_LAST_ERROR
  // 1 thread updates 1 cell
  grid = (sim->boxes->nLocalBoxes + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  block = THREAD_ATOM_CTA;
  if(sim->usePairlist)
      /*
      DPCT1049:78: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:212: 'THREAD_ATOM_CTA' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments,
      if it is correct.
      */
      sycl::local_accessor<int, 1> natoms_acc_ct1(
          sycl::range<1>(128 /*THREAD_ATOM_CTA*/), cgh);

      SimGpu sim_gpu_ct0 = sim->gpu;
      auto sim_boxes_nLocalBoxes_ct1 = sim->boxes->nLocalBoxes;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                sycl::range<3>(1, 1, block),
                            sycl::range<3>(1, 1, block)),
          [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            CompactAtoms<true>(
                sim_gpu_ct0, sim_boxes_nLocalBoxes_ct1, flags,
                natoms_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
  else
      /*
      DPCT1049:79: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:214: 'THREAD_ATOM_CTA' expression was replaced with a value.
      Modify the code to use the original expression, provided in comments,
      if it is correct.
      */
      sycl::local_accessor<int, 1> natoms_acc_ct1(
          sycl::range<1>(128 /*THREAD_ATOM_CTA*/), cgh);

      SimGpu sim_gpu_ct0 = sim->gpu;
      auto sim_boxes_nLocalBoxes_ct1 = sim->boxes->nLocalBoxes;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                sycl::range<3>(1, 1, block),
                            sycl::range<3>(1, 1, block)),
          [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            CompactAtoms<false>(
                sim_gpu_ct0, sim_boxes_nLocalBoxes_ct1, flags,
                natoms_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
  CUDA_GET_LAST_ERROR

  // update max # of atoms per cell
  dpct::get_in_order_queue()
      .memcpy(&sim->gpu.max_atoms_cell,
              &flags[sim->boxes->nLocalBoxes * MAXATOMS], sizeof(int))
      .wait();

  // build new atom lists: only for thread/atom or warp/atom approaches
  if (sim->method == THREAD_ATOM || sim->method == WARP_ATOM || sim->method == THREAD_ATOM_NL || sim->method == WARP_ATOM_NL)
    BuildAtomLists(sim);

  CUDA_GET_LAST_ERROR
}

extern "C"
extern "C" void getAtomMsgSoAPtr(char* const buffer, AtomMsgSoA *atomMsg, int n)
{
        atomMsg->gid  =  (int*) buffer;
        atomMsg->type = atomMsg->gid + n;
        atomMsg->rx = (real_t*)(atomMsg->type + n);
        atomMsg->ry = atomMsg->rx + n;
        atomMsg->rz = atomMsg->rx + 2*n;
        atomMsg->px = atomMsg->rx + 3*n;
        atomMsg->py = atomMsg->rx + 4*n;
        atomMsg->pz = atomMsg->rx + 5*n;
}

/// compacts all particles within all the cells specified by cellList into compactAtoms (see AtomMsgSoA for data layout)
/// @param [out] d_compactAtoms Device-pointer, On-exit: stores the compacted atoms in SoA format.
/// @param [in] nCells number of cells in cellList
/// @param [in] cellList Device-pointer. Holds the cell id of the cells of interest
/// @param [in] nAtomsCell Device-pointer. Holds the number of cells of each cell (most likely sim->boxes->nAtoms)
/// @param [out] d_cellOffsets Device-pointer. On-exit: Contains the starting offsets for each cell within d_compactAtoms. (e.g.: numAtoms(0)=3, numAtoms(1)=2 => cellOffsets(0)=0,cellOffsets(1)=3,cellOffsets(2)=5)
extern "C" int compactCellsGpu(char *d_compactAtoms, int nCells,
                               int *d_cellList, SimGpu sim_gpu,
                               int *d_cellOffsets, int *d_workScan,
                               real3_old shift, dpct::queue_ptr stream)
{

    // compute starting offsets for each cell within the compacted array
    scanCells(d_cellOffsets, nCells, d_cellList, sim_gpu.boxes.nAtoms, d_workScan, stream);

    int nTotalAtomsCellList;
    // the last entry of d_nAtomsOffset will store the total number of atoms within all specified cells
    /*
    DPCT1124:154: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    stream->memcpy(&nTotalAtomsCellList, d_cellOffsets + nCells, sizeof(int));
    stream->wait();

    //alias host and device buffers with AtomMsgSoA
    AtomMsgSoA msg_d;
    getAtomMsgSoAPtr(d_compactAtoms, &msg_d, nTotalAtomsCellList);

    //assemble compact array of particles
    int block = MAXATOMS;
    int grid = nCells;
    /*
    DPCT1049:80: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  stream->submit([&](sycl::handler &cgh) {
    auto shift_ct4 = shift[0];
    auto shift_ct5 = shift[1];
    auto shift_ct6 = shift[2];

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       LoadAtomsBufferPacked(msg_d, d_cellList, sim_gpu,
                                             d_cellOffsets, shift_ct4,
                                             shift_ct5, shift_ct6);
                     });
  });
    CUDA_GET_LAST_ERROR

    return nTotalAtomsCellList;
}

/// builds sim->gpu.a_list
/// @param [out] natoms_buf (temporary)
/// @param [out] partial_sums (temporary)
extern "C" void buildAtomListGpu(SimFlat *sim, dpct::queue_ptr stream)
{
  int* natoms_buf = ((AtomExchangeParms*)(sim->atomExchange->parms))->d_natoms_buf;
  int *partial_sums = ((AtomExchangeParms*)(sim->atomExchange->parms))->d_partial_sums;
  int nCells = sim->boxes->nLocalBoxes;
  scanCells(natoms_buf, nCells, sim->gpu.boxes.nAtoms, partial_sums, stream);

  // rebuild compact list of atoms & cells
  int grid = (nCells * MAXATOMS + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  /*
  DPCT1049:81: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    SimGpu sim_gpu_ct2 = sim->gpu;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       UpdateCompactIndices(natoms_buf, nCells, sim_gpu_ct2);
                     });
  });

  // new number of local atoms
  /*
  DPCT1124:155: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
  the origin API might be synchronous, it depends on the type of operand memory,
  so you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  stream->memcpy(&(sim->gpu.a_list.n), natoms_buf + nCells, sizeof(int));
  // a_list.n is consumed on host right after this function returns.
  // Ensure the async D2H copy is visible before the next force launch computes grid size.
  stream->wait();
  if (sim->gpu.a_list.n < 0 || sim->gpu.a_list.n > sim->boxes->nTotalBoxes * MAXATOMS) {
    fprintf(stderr, "Invalid a_list.n on host after buildAtomListGpu: %d\n", sim->gpu.a_list.n);
    exit(-1);
  }

  CUDA_GET_LAST_ERROR
}

/// The unloadBuffer function for a halo exchange of atom data.
/// Iterates the receive buffer and places each atom that was received
/// into the link cell that corresponds to the atom coordinate.  Note
/// that this naturally accomplishes transfer of ownership of atoms that
/// have moved from one spatial domain to another.  Atoms with
/// coordinates in local link cells automatically become local
/// particles.  Atoms that are owned by other ranks are automatically
/// placed in halo kink cells.
/// @param bBuf [in] Total number of received atoms
/// @param buf [in] Pointer to the received data
/// @param sim [inout] The gpu field of sim will be updated
/// @param gpu_buf [out] Already allocated gpu buffer (temporary)
extern "C" void unloadAtomsBufferToGpu(char *buf, int nBuf, SimFlat *sim,
                                       char *gpu_buf, dpct::queue_ptr stream)
{
  if (nBuf == 0) return;
  /*
  DPCT1124:156: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
  the origin API might be synchronous, it depends on the type of operand memory,
  so you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  stream->memcpy(gpu_buf, buf, nBuf * sizeof(AtomMsg));

  // TODO: don't need to check if we're running cell-based approach
  int nlUpdateRequired = neighborListUpdateRequiredGpu(&(sim->gpu));

  int grid = (nBuf + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;

  vec_t r,p;
  int *gid = (int*)gpu_buf;
  int *type = gid + nBuf;
  r.x = (real_t*)(type + nBuf);
  r.y = r.x + nBuf;
  r.z = r.y + nBuf;
  p.x = r.z + nBuf;
  p.y = p.x + nBuf;
  p.z = p.y + nBuf;

  // use temp arrays
  int *d_iOffset = sim->flags;
  int *d_boxId = sim->tmp_sort;

  computeOffsets(nlUpdateRequired, sim, r, d_iOffset, d_boxId, nBuf, stream);

  // map received particles to cells
  /*
  DPCT1049:82: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    AtomsGpu sim_gpu_atoms_ct5 = sim->gpu.atoms;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       UnloadAtomsBufferPacked(r, p, type, gid, nBuf,
                                               sim_gpu_atoms_ct5, d_iOffset);
                     });
  });

  CUDA_GET_LAST_ERROR
}

/// The loadBuffer function for a force exchange.
/// Iterate the send list and load the derivative of the embedding
/// energy with respect to the local density into the send buffer.
extern "C" void loadForceBufferFromGpu(char *buf, int *nbuf, int nCells,
                                       int *cellList, int *natoms_buf,
                                       int *partial_sums, SimFlat *s,
                                       char *gpu_buf, dpct::queue_ptr stream)
{
  CUDA_GET_LAST_ERROR
  scanCells(natoms_buf, nCells, cellList, s->gpu.boxes.nAtoms, partial_sums, stream);

  // copy data to compacted array
  int grid = (nCells * MAXATOMS + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  /*
  DPCT1049:83: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    SimGpu s_gpu_ct3 = s->gpu;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       LoadForceBuffer((ForceMsg *)gpu_buf, nCells, cellList,
                                       s_gpu_ct3, natoms_buf);
                     });
  });
  CUDA_GET_LAST_ERROR

  int nBuf;
  /*
  DPCT1124:157: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
  the origin API might be synchronous, it depends on the type of operand memory,
  so you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  stream->memcpy(&nBuf, natoms_buf + nCells, sizeof(int));
  CUDA_GET_LAST_ERROR
  /*
  DPCT1124:158: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
  the origin API might be synchronous, it depends on the type of operand memory,
  so you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  stream->memcpy(buf, gpu_buf, nBuf * sizeof(ForceMsg));
  CUDA_GET_LAST_ERROR

  stream->wait();
  *nbuf = nBuf;
  CUDA_GET_LAST_ERROR
}

/// The unloadBuffer function for a force exchange.
/// Data is received in an order that naturally aligns with the atom
/// storage so it is simple to put the data where it belongs.
extern "C" void unloadForceBufferToGpu(char *buf, int nBuf, int nCells,
                                       int *cellList, int *natoms_buf,
                                       int *partial_sums, SimFlat *s,
                                       char *gpu_buf, dpct::queue_ptr stream)
{
  // copy raw data to gpu
  /*
  DPCT1124:159: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
  the origin API might be synchronous, it depends on the type of operand memory,
  so you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  stream->memcpy(gpu_buf, buf, nBuf * sizeof(ForceMsg));

  scanCells(natoms_buf, nCells, cellList, s->gpu.boxes.nAtoms, partial_sums, stream);

  // copy data for the list of cells
  int grid = (nCells * MAXATOMS + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  /*
  DPCT1049:84: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    SimGpu s_gpu_ct3 = s->gpu;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       UnloadForceBuffer((ForceMsg *)gpu_buf, nCells, cellList,
                                         s_gpu_ct3, natoms_buf);
                     });
  });

  CUDA_GET_LAST_ERROR
}

extern "C" void sortAtomsGpu(SimFlat *s, dpct::queue_ptr stream)
{
  int *new_indices = s->flags;
  // set all indices to -1
  stream->memset(new_indices, 255,
                 s->boxes->nTotalBoxes * MAXATOMS * sizeof(int));

  // one thread per atom, only update boundary cells
  int block = MAXATOMS;
  int grid = (s->n_boundary1_cells * WARP_SIZE + block-1)/block;
  /*
  DPCT1049:85: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    SimGpu s_gpu_ct0 = s->gpu;
    auto s_n_boundary1_cells_ct1 = s->n_boundary1_cells;
    auto s_boundary1_cells_d_ct2 = s->boundary1_cells_d;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       SetLinearIndices(s_gpu_ct0, s_n_boundary1_cells_ct1,
                                        s_boundary1_cells_d_ct2, new_indices);
                     });
  });
  CUDA_GET_LAST_ERROR

  // update halo cells
  grid = ((s->boxes->nTotalBoxes - s->boxes->nLocalBoxes) * MAXATOMS + block-1)/block;
  /*
  DPCT1049:86: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    SimGpu s_gpu_ct0 = s->gpu;
    auto s_boxes_nLocalBoxes_ct1 = s->boxes->nLocalBoxes;
    auto s_boxes_nTotalBoxes_ct2 = s->boxes->nTotalBoxes;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       SetLinearIndices(s_gpu_ct0, s_boxes_nLocalBoxes_ct1,
                                        s_boxes_nTotalBoxes_ct2, new_indices);
                     });
  });
  CUDA_GET_LAST_ERROR

  // one thread per cell: process halo & boundary cells only
  int block2 = MAXATOMS;
  int grid2 = (s->n_boundary1_cells + (s->boxes->nTotalBoxes - s->boxes->nLocalBoxes) + block2-1) / block2;
  /*
  DPCT1049:87: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    SimGpu s_gpu_ct0 = s->gpu;
    auto s_boxes_nLocalBoxes_ct1 = s->boxes->nLocalBoxes;
    auto s_boxes_nTotalBoxes_ct2 = s->boxes->nTotalBoxes;
    auto s_boundary1_cells_d_ct3 = s->boundary1_cells_d;
    auto s_n_boundary1_cells_ct4 = s->n_boundary1_cells;
    auto s_tmp_sort_ct6 = s->tmp_sort;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid2) *
                                           sycl::range<3>(1, 1, block2),
                                       sycl::range<3>(1, 1, block2)),
                     [=](sycl::nd_item<3> item_ct1) {
                       SortAtomsByGlobalId(s_gpu_ct0, s_boxes_nLocalBoxes_ct1,
                                           s_boxes_nTotalBoxes_ct2,
                                           s_boundary1_cells_d_ct3,
                                           s_n_boundary1_cells_ct4, new_indices,
                                           s_tmp_sort_ct6);
                     });
  });
  CUDA_GET_LAST_ERROR

  // one warp per cell
  int block3 = THREAD_ATOM_CTA;
  int grid3 = ((s->n_boundary1_cells + (s->boxes->nTotalBoxes - s->boxes->nLocalBoxes)) * WARP_SIZE + block3-1) / block3;
  /*
  DPCT1049:88: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler &cgh) {
    SimGpu s_gpu_ct0 = s->gpu;
    auto s_boxes_nLocalBoxes_ct1 = s->boxes->nLocalBoxes;
    auto s_boxes_nTotalBoxes_ct2 = s->boxes->nTotalBoxes;
    auto s_boundary1_cells_d_ct3 = s->boundary1_cells_d;
    auto s_n_boundary1_cells_ct4 = s->n_boundary1_cells;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid3) *
                              sycl::range<3>(1, 1, block3),
                          sycl::range<3>(1, 1, block3)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
          ShuffleAtomsData(s_gpu_ct0, s_boxes_nLocalBoxes_ct1,
                           s_boxes_nTotalBoxes_ct2, s_boundary1_cells_d_ct3,
                           s_n_boundary1_cells_ct4, new_indices);
        });
  });

  CUDA_GET_LAST_ERROR
}

extern "C"
extern "C" void computeEnergy(SimFlat *flat, real_t *eLocal)
{
  if (flat->gpu.a_list.n == 0) return;

  real_t *e_gpu;
  e_gpu = sycl::malloc_device<real_t>(2, dpct::get_in_order_queue());
  dpct::get_in_order_queue().memset(e_gpu, 0, 2 * sizeof(real_t)).wait();

  int grid = (flat->gpu.a_list.n + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;
  /*
  DPCT1049:89: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:222: 'THREAD_ATOM_CTA' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<real_t, 1> sp_acc_ct1(
        sycl::range<1>(128 /*THREAD_ATOM_CTA*/), cgh);
    /*
    DPCT1101:223: 'THREAD_ATOM_CTA' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments, if
    it is correct.
    */
    sycl::local_accessor<real_t, 1> sk_acc_ct1(
        sycl::range<1>(128 /*THREAD_ATOM_CTA*/), cgh);

    SimGpu flat_gpu_ct0 = flat->gpu;
    auto e_gpu_ct1 = &e_gpu[0];
    auto e_gpu_ct2 = &e_gpu[1];

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                              sycl::range<3>(1, 1, block),
                          sycl::range<3>(1, 1, block)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
          ReduceEnergy(
              flat_gpu_ct0, e_gpu_ct1, e_gpu_ct2,
              sp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
              sk_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
        });
  });

  dpct::get_in_order_queue().memcpy(eLocal, e_gpu, 2 * sizeof(real_t)).wait();

  CUDA_GET_LAST_ERROR
}

/// Variant of computeEnergy that writes the result into a caller-supplied
/// device buffer (d_eOut[0]=ePotential, d_eOut[1]=eKinetic) without doing
/// a D2H copy. The caller is responsible for the D2H transfer after any
/// cross-rank reduction on d_eOut.
extern "C"
void computeEnergyDevice(SimFlat *flat, real_t *d_eOut)
{
  if (flat->gpu.a_list.n == 0) return;

  dpct::get_in_order_queue().memset(d_eOut, 0, 2 * sizeof(real_t)).wait();

  int grid = (flat->gpu.a_list.n + (THREAD_ATOM_CTA-1)) / THREAD_ATOM_CTA;
  int block = THREAD_ATOM_CTA;

  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    sycl::local_accessor<real_t, 1> sp_acc_ct1(
        sycl::range<1>(128 /*THREAD_ATOM_CTA*/), cgh);
    sycl::local_accessor<real_t, 1> sk_acc_ct1(
        sycl::range<1>(128 /*THREAD_ATOM_CTA*/), cgh);

    SimGpu flat_gpu_ct0 = flat->gpu;
    auto e_gpu_ct1 = &d_eOut[0];
    auto e_gpu_ct2 = &d_eOut[1];

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                              sycl::range<3>(1, 1, block),
                          sycl::range<3>(1, 1, block)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
          ReduceEnergy(
              flat_gpu_ct0, e_gpu_ct1, e_gpu_ct2,
              sp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
              sk_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
        });
  });

  CUDA_GET_LAST_ERROR
}


void emptyNeighborListGpuKernel(SimGpu sim, int boundaryFlag)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (tid >= sim.a_list.n) return;

  // compute box ID and local atom ID
  int iBox = sim.a_list.cells[tid];
  if (boundaryFlag == INTERIOR && sim.cell_type[iBox] != 0) return;
  if (boundaryFlag == BOUNDARY && sim.cell_type[iBox] != 1) return;
  sim.atoms.neighborList.nNeighbors[tid] = 0;
}

/// Sets all neighbor counts to zero
extern "C"
extern "C" void emptyNeighborListGpu(SimGpu *sim, int boundaryFlag)
{

    int grid = (sim->a_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
    int block = THREAD_ATOM_CTA;
    /*
    DPCT1049:90: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
    SimGpu sim_ct0 = *sim;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       emptyNeighborListGpuKernel(sim_ct0, boundaryFlag);
                     });
  });

  CUDA_GET_LAST_ERROR
}

/**
  * Checks if any atom has moved more than half of the skin distance
  */


void updateNeighborListRequiredKernel(SimGpu sim, int* updateNeighborListRequired)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (tid >= sim.a_list.n) return;

  // compute box ID and local atom ID
  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];
  int iOff = iBox * MAXATOMS + iAtom;

  // fetch position
  real_t dx = sim.atoms.r.x[iOff] - sim.atoms.neighborList.lastR.x[tid];
  real_t dy = sim.atoms.r.y[iOff] - sim.atoms.neighborList.lastR.y[tid];
  real_t dz = sim.atoms.r.z[iOff] - sim.atoms.neighborList.lastR.z[tid];

  if( (dx*dx + dy*dy + dz*dz) > sim.atoms.neighborList.skinDistanceHalf2 )
          *updateNeighborListRequired = 1;
}



void updatePairlistRequiredKernel(SimGpu sim, int * updatePairlistRequired)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int iBox = item_ct1.get_group(2);
    const int nAtoms = sim.boxes.nAtoms[iBox];

    for (int iAtom = item_ct1.get_local_id(2); iAtom < nAtoms;
         iAtom += item_ct1.get_local_range(2))
    {
        int iOff = iBox * MAXATOMS + iAtom;

        real_t dx = sim.atoms.r.x[iOff] - sim.atoms.neighborList.lastR.x[iOff];
        real_t dy = sim.atoms.r.y[iOff] - sim.atoms.neighborList.lastR.y[iOff];
        real_t dz = sim.atoms.r.z[iOff] - sim.atoms.neighborList.lastR.z[iOff];

        if( (dx*dx + dy*dy + dz*dz) > sim.atoms.neighborList.skinDistanceHalf2 )
        {
            *updatePairlistRequired = 1;
            return;
        }

    }
}

// Function checks
extern "C"
extern "C" int pairlistUpdateRequiredGpu(SimGpu * sim)
{
    if(sim->atoms.neighborList.forceRebuildFlag == 1)
    {
        sim->atoms.neighborList.updateNeighborListRequired = 1;
    }
    else if(sim->atoms.neighborList.updateNeighborListRequired == -1)
    {
        int grid = sim->boxes.nLocalBoxes;
        int block = CTA_CELL_CTA;
        int *d_updatePairlistRequired;
        int h_updatePairlistRequired;
        d_updatePairlistRequired =
            sycl::malloc_device<int>(1, dpct::get_in_order_queue());
        dpct::get_in_order_queue()
            .memset(d_updatePairlistRequired, 0, sizeof(int))
            .wait();

        /*
        DPCT1049:91: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      SimGpu sim_ct0 = *sim;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         updatePairlistRequiredKernel(sim_ct0,
                                                      d_updatePairlistRequired);
                       });
    });
        CUDA_GET_LAST_ERROR

        dpct::get_in_order_queue()
            .memcpy(&h_updatePairlistRequired, d_updatePairlistRequired,
                    sizeof(int))
            .wait();

        int tmpUpdatePairlistRequired = h_updatePairlistRequired;

        sim->atoms.neighborList.updateNeighborListRequired = tmpUpdatePairlistRequired;
    }

  CUDA_GET_LAST_ERROR

    return sim->atoms.neighborList.updateNeighborListRequired;
}

/// \param [inout] neighborList NeighborList (the only value that might be changed is updateNeighborListRequired
/// \return 1 iff neighborlist update is required in this step
extern "C"
extern "C" int neighborListUpdateRequiredGpu(SimGpu* sim)
{
        if(sim->atoms.neighborList.forceRebuildFlag== 1){
                sim->atoms.neighborList.updateNeighborListRequired = 1;
        }else if(sim->atoms.neighborList.updateNeighborListRequired == -1){
//        }else {
                //only do a real neighborlistupdate check if the particles have moved (indicated by updateNeighborListRequired == -1)
                int grid = (sim->a_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
                int block = THREAD_ATOM_CTA;
                int *d_updateNeighborListRequired;
                int h_updateNeighborListRequired;
                d_updateNeighborListRequired =
                    sycl::malloc_device<int>(1, dpct::get_in_order_queue());

                dpct::get_in_order_queue()
                    .memset(d_updateNeighborListRequired, 0, sizeof(int))
                    .wait();
                /*
                DPCT1049:92: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      SimGpu sim_ct0 = *sim;

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                             sycl::range<3>(1, 1, block),
                                         sycl::range<3>(1, 1, block)),
                       [=](sycl::nd_item<3> item_ct1) {
                         updateNeighborListRequiredKernel(
                             sim_ct0, d_updateNeighborListRequired);
                       });
    });

                // Allreduce directly on the device buffer — no D2H before the collective.
                // One D2H after to read the result on CPU.
                int* d_result = sycl::malloc_device<int>(1, dpct::get_in_order_queue());
                addIntParallelDevice(d_updateNeighborListRequired, d_result, 1);
                int tmpUpdateNeighborListRequired = 0;
                dpct::get_in_order_queue()
                    .memcpy(&tmpUpdateNeighborListRequired, d_result, sizeof(int))
                    .wait();
                dpct::dpct_free(d_result, dpct::get_in_order_queue());
                dpct::dpct_free(d_updateNeighborListRequired, dpct::get_in_order_queue());

                if(tmpUpdateNeighborListRequired > 0)
                        sim->atoms.neighborList.updateNeighborListRequired = 1;
                else
                        sim->atoms.neighborList.updateNeighborListRequired = 0;
        }

  CUDA_GET_LAST_ERROR

        return  sim->atoms.neighborList.updateNeighborListRequired;
}


//Neighborlist generation for warp_atoms_NL only
//> maxNeighbors is the maximum number of entries in neighbor list
//> packSize is the number of threads cooperating to compute neighbor list for single atom
//> logPackSize is log_2(packSize)
//> memoryPackSize is the number of threads in the compute force kernel cooperating on a single atom
//    and number of entries in a single pack of neighbor list that is written to memory,
//    memoryPackSize must be <= packSize
//> boundaryFlag is BOUNDARY, INTERNAL or BOTH
template <int maxNeighbors, int packSize, int logPackSize, int memoryPackSize,
          int boundaryFlag>
/*
DPCT1110:93: The total declared local variable size in device function
buildNeighborListKernel_warp exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/

void buildNeighborListKernel_warp(SimGpu sim, real_t rCut2, int *temp)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int atid = item_ct1.get_local_id(2) / packSize;
    const int laneid = item_ct1.get_local_id(2) & 31;
    const int tid = item_ct1.get_group(2) * 32 + atid;
    int * __restrict__ neighborList;

    const unsigned int tmpmask = (1 << laneid) - 1;

    const unsigned int tmpmask2 = ~((1 << (laneid - (laneid & (packSize-1)))) - 1);
    const unsigned int mask = tmpmask2 & tmpmask;

    const unsigned int mask2 = ((1 << packSize) - 1) << ((laneid / packSize) << logPackSize);

    const int ldNeighborList = sim.atoms.neighborList.nMaxLocal*memoryPackSize; //leading dimension
    int* __restrict__ nNeighborsArray = sim.atoms.neighborList.nNeighbors;
    if (tid < sim.a_list.n)
    {
        // compute box ID and local atom ID
        int iAtom = sim.a_list.atoms[tid];
        int iBox = sim.a_list.cells[tid];

        if(boundaryFlag == BOUNDARY && sim.cell_type[iBox] == 0) return;
        if(boundaryFlag == INTERIOR && sim.cell_type[iBox] == 1) return;

        const int iOff = iBox * MAXATOMS + iAtom;
        assert(iOff < sim.boxes.nLocalBoxes * MAXATOMS && iOff >=0 );

        // fetch position
        real_t *const __restrict__ rx = sim.atoms.r.x;
        real_t *const __restrict__ ry = sim.atoms.r.y;
        real_t *const __restrict__ rz = sim.atoms.r.z;

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 350
        /*
        DPCT1098:160: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        real_t irx = *(rx + iOff);
        /*
        DPCT1098:161: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        real_t iry = *(ry + iOff);
        /*
        DPCT1098:162: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        real_t irz = *(rz + iOff);
#else
        real_t irx = rx[iOff];
        real_t iry = ry[iOff];
        real_t irz = rz[iOff];
#endif
        //get NL related data
        const int iLid = tid;
        assert( iLid < sim.atoms.neighborList.nMaxLocal );
        //assert(iLid<ldNeighborList);
        neighborList = sim.atoms.neighborList.list;
        int nNeighbors = 0;
        const int id = item_ct1.get_local_id(2) & (packSize - 1);
        if(id == 0)
        {
            sim.atoms.neighborList.lastR.x[iLid] = irx;
            sim.atoms.neighborList.lastR.y[iLid] = iry;
            sim.atoms.neighborList.lastR.z[iLid] = irz;
        }

        // loop over my neighbor cells
        const int *const __restrict__ neighbor_cells = sim.neighbor_cells;
        const int *const __restrict__ nAtoms = sim.boxes.nAtoms;
        int * mytemp = temp + atid * memoryPackSize;
//Our own cell
        {
            const int jBox = iBox;
            const int jOffset = jBox * MAXATOMS;
            const int nJBox = nAtoms[jBox] + jOffset;

            real_t * __restrict__ rx = sim.atoms.r.x + jOffset + id;
            real_t * __restrict__ ry = sim.atoms.r.y + jOffset + id;
            real_t * __restrict__ rz = sim.atoms.r.z + jOffset + id;


            // loop over all atoms in my cell
            for (int jAtom = jOffset; jAtom < nJBox; jAtom += packSize)
            {
                const int jOff = jAtom +id;
                assert(jOff < sim.boxes.nTotalBoxes * MAXATOMS && jOff >=0 );

                real_t r2;
                if(jOff < nJBox)
                {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 350
                    /*
                    DPCT1098:163: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dx = irx - *rx;
                    /*
                    DPCT1098:164: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dy = iry - *ry;
                    /*
                    DPCT1098:165: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dz = irz - *rz;
                    rx += packSize;
                    ry += packSize;
                    rz += packSize;
#else
                    real_t dx = irx - rx[jOff];
                    real_t dy = iry - ry[jOff];
                    real_t dz = irz - rz[jOff];
#endif
                // distance^2
                    r2 = dx*dx + dy*dy + dz*dz;
                }
                else
                    r2 = 0.0;

                bool flag = r2 <= rCut2 && r2 > 0.0;
                unsigned int x;
                int n;
                if (x = sycl::reduce_over_group(
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        (mask2 &
                         (0x1
                          << sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_linear_id())) &&
                                flag
                            ? (0x1 << sycl::ext::oneapi::this_work_item::
                                          get_sub_group()
                                              .get_local_linear_id())
                            : 0,
                        sycl::ext::oneapi::plus<>()))
                {
                //Scan
                    x = x & mask2;
                    n = sycl::popcount(x);
                    x = x & mask;
                    const int p = sycl::popcount(x);
                    const int place = nNeighbors + p;
                    if (flag) mytemp[(place/memoryPackSize) * 32 * memoryPackSize + (place & (memoryPackSize-1))] = jOff;
                    nNeighbors += n;
                }

            } // loop over all atoms
        }

//Other cells
#pragma unroll
        for (int j = 1; j < N_MAX_NEIGHBORS; j++)
        {
            const int jBox = neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
            const int jOffset = jBox * MAXATOMS;
            const int nJBox = nAtoms[jBox] + jOffset;

            real_t * __restrict__ rx = sim.atoms.r.x + jOffset + id;
            real_t * __restrict__ ry = sim.atoms.r.y + jOffset + id;
            real_t * __restrict__ rz = sim.atoms.r.z + jOffset + id;


            // loop over all atoms in the neighbor cell
            for (int jAtom = jOffset; jAtom < nJBox; jAtom += packSize)
            {
                const int jOff = jAtom +id;
                assert(jOff < sim.boxes.nTotalBoxes * MAXATOMS && jOff >=0 );

                real_t r2;
                if(jOff < nJBox)
                {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 350
                    /*
                    DPCT1098:166: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dx = irx - *rx;
                    /*
                    DPCT1098:167: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dy = iry - *ry;
                    /*
                    DPCT1098:168: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dz = irz - *rz;
                    rx += packSize;
                    ry += packSize;
                    rz += packSize;
#else
                    real_t dx = irx - rx[jOff];
                    real_t dy = iry - ry[jOff];
                    real_t dz = irz - rz[jOff];
#endif
                // distance^2
                    r2 = dx*dx + dy*dy + dz*dz;
                }
                else
                    r2 = 1.0e100; //Big value

                //r2 is never 0
                bool flag = r2 <= rCut2;
                unsigned int x;
                int n;
                if (x = sycl::reduce_over_group(
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        (mask2 &
                         (0x1
                          << sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_linear_id())) &&
                                flag
                            ? (0x1 << sycl::ext::oneapi::this_work_item::
                                          get_sub_group()
                                              .get_local_linear_id())
                            : 0,
                        sycl::ext::oneapi::plus<>()))
                {
                //Scan
                    x = x & mask2;
                    n = sycl::popcount(x);
                    x = x & mask;
                    const int p = sycl::popcount(x);
                    const int place = nNeighbors + p;
                    if (flag) mytemp[(place/memoryPackSize) * 32 * memoryPackSize + (place & (memoryPackSize-1))] = jOff;
                    nNeighbors += n;
                }

            } // loop over all atoms
        } // loop over neighbor cells

        if(id == 0)
        {
            nNeighborsArray[iLid] = nNeighbors;
        }
    }
    /*
    DPCT1065:169: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    const int gtid =
        item_ct1.get_group(2) * 32 * memoryPackSize + item_ct1.get_local_id(2);
    const int iLid = gtid / memoryPackSize;
    if (iLid < sim.a_list.n && item_ct1.get_local_id(2) < 32 * memoryPackSize)
    {
#pragma unroll
        for(int i = 0; i < maxNeighbors/memoryPackSize;++i)
        {
            assert(threadIdx.x + i * 32 * memoryPackSize < 32 * maxNeighbors);
            neighborList[gtid + i * ldNeighborList] =
                temp[item_ct1.get_local_id(2) + i * 32 * memoryPackSize];
        }
    }
}

template <int maxNeighbors, int packSize, int logPackSize, int boundaryFlag>
/*
DPCT1110:94: The total declared local variable size in device function
buildNeighborListKernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

void buildNeighborListKernel(SimGpu sim, int *temp)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int atid = item_ct1.get_local_id(2) / packSize;
    const int laneid = item_ct1.get_local_id(2) & 31;
    const int tid = item_ct1.get_group(2) * 32 + atid;
    int * __restrict__ neighborList;

    const unsigned int tmpmask = (1 << laneid) - 1;

    const unsigned int tmpmask2 = ~((1 << (laneid - (laneid & (packSize-1)))) - 1);
    const unsigned int mask = tmpmask2 & tmpmask;

    const unsigned int mask2 = ((1 << packSize) - 1) << ((laneid / packSize) << logPackSize);

    const int ldNeighborList = sim.atoms.neighborList.nMaxLocal; //leading dimension
    int* __restrict__ nNeighborsArray = sim.atoms.neighborList.nNeighbors;
    if (tid < sim.a_list.n)
    {


        // compute box ID and local atom ID
        int iAtom = sim.a_list.atoms[tid];
        int iBox = sim.a_list.cells[tid];

        //assert(sim.cell_type[iBox] == 0 || sim.cell_type[iBox] == 1);
        if(boundaryFlag == BOUNDARY && sim.cell_type[iBox] == 0) return;
        if(boundaryFlag == INTERIOR && sim.cell_type[iBox] == 1) return;

        const int iOff = iBox * MAXATOMS + iAtom;

        const real_t rCut = sim.eam_pot.cutoff;
        const real_t rCut2 = (rCut+sim.atoms.neighborList.skinDistance)*(rCut+sim.atoms.neighborList.skinDistance);

        // fetch position
        real_t *const __restrict__ rx = sim.atoms.r.x;
        real_t *const __restrict__ ry = sim.atoms.r.y;
        real_t *const __restrict__ rz = sim.atoms.r.z;

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 350
        /*
        DPCT1098:171: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        real_t irx = *(rx + iOff);
        /*
        DPCT1098:172: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        real_t iry = *(ry + iOff);
        /*
        DPCT1098:173: The '*' expression is used instead of the __ldg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        real_t irz = *(rz + iOff);
#else
        real_t irx = rx[iOff];
        real_t iry = ry[iOff];
        real_t irz = rz[iOff];
#endif
        //get NL related data
        const int iLid = tid;
        neighborList = sim.atoms.neighborList.list;
        int nNeighbors = 0;
        const int id = item_ct1.get_local_id(2) & (packSize - 1);
        if(id == 0)
        {
            sim.atoms.neighborList.lastR.x[iLid] = irx;
            sim.atoms.neighborList.lastR.y[iLid] = iry;
            sim.atoms.neighborList.lastR.z[iLid] = irz;
        }

        // loop over my neighbor cells
        const int *const __restrict__ neighbor_cells = sim.neighbor_cells;
        const int *const __restrict__ nAtoms = sim.boxes.nAtoms;
        int * mytemp = temp + atid;
//Our own cell
        {
            const int jBox = iBox;
            const int jOffset = jBox * MAXATOMS;
            const int nJBox = nAtoms[jBox] + jOffset;

            real_t * __restrict__ rx = sim.atoms.r.x + jOffset + id;
            real_t * __restrict__ ry = sim.atoms.r.y + jOffset + id;
            real_t * __restrict__ rz = sim.atoms.r.z + jOffset + id;


            // loop over all atoms in my cell
            for (int jAtom = jOffset; jAtom < nJBox; jAtom += packSize)
            {
                int jOff = jAtom +id;

                real_t r2;
                if(jOff < nJBox)
                {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 350
                    /*
                    DPCT1098:174: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dx = irx - *rx;
                    /*
                    DPCT1098:175: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dy = iry - *ry;
                    /*
                    DPCT1098:176: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dz = irz - *rz;
                    rx += packSize;
                    ry += packSize;
                    rz += packSize;
#else
                    real_t dx = irx - rx[jOff];
                    real_t dy = iry - ry[jOff];
                    real_t dz = irz - rz[jOff];
#endif
                // distance^2
                    r2 = dx*dx + dy*dy + dz*dz;
                }
                else
                    r2 = 0.0;

                bool flag = r2 <= rCut2 && r2 > 0.0;
                unsigned int x;
                int n;
                if (x = sycl::reduce_over_group(
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        (mask2 &
                         (0x1
                          << sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_linear_id())) &&
                                flag
                            ? (0x1 << sycl::ext::oneapi::this_work_item::
                                          get_sub_group()
                                              .get_local_linear_id())
                            : 0,
                        sycl::ext::oneapi::plus<>()))
                {
                //Scan
                    x = x & mask2;
                    n = sycl::popcount(x);
                    x = x & mask;
                    const int p = sycl::popcount(x);
                    if (flag) mytemp[(nNeighbors+p)*32] = jOff;
                    nNeighbors += n;
                }

            } // loop over all atoms
        }

//Other cells
#pragma unroll
        for (int j = 1; j < N_MAX_NEIGHBORS; j++)
        {
            const int jBox = neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
            const int jOffset = jBox * MAXATOMS;
            const int nJBox = nAtoms[jBox] + jOffset;

            real_t * __restrict__ rx = sim.atoms.r.x + jOffset + id;
            real_t * __restrict__ ry = sim.atoms.r.y + jOffset + id;
            real_t * __restrict__ rz = sim.atoms.r.z + jOffset + id;


            // loop over all atoms in the neighbor cell
            for (int jAtom = jOffset; jAtom < nJBox; jAtom += packSize)
            {
                int jOff = jAtom +id;

                real_t r2;
                if(jOff < nJBox)
                {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 350
                    /*
                    DPCT1098:177: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dx = irx - *rx;
                    /*
                    DPCT1098:178: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dy = iry - *ry;
                    /*
                    DPCT1098:179: The '*' expression is used instead of the
                    __ldg call. These two expressions do not provide the exact
                    same functionality. Check the generated code for potential
                    precision and/or performance issues.
                    */
                    real_t dz = irz - *rz;
                    rx += packSize;
                    ry += packSize;
                    rz += packSize;
#else
                    real_t dx = irx - rx[jOff];
                    real_t dy = iry - ry[jOff];
                    real_t dz = irz - rz[jOff];
#endif
                // distance^2
                    r2 = dx*dx + dy*dy + dz*dz;
                }
                else
                    r2 = 1.0e100;
//r2 never 0
                bool flag = r2 <= rCut2;
                unsigned int x;
                int n;
                if (x = sycl::reduce_over_group(
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        (mask2 &
                         (0x1
                          << sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_linear_id())) &&
                                flag
                            ? (0x1 << sycl::ext::oneapi::this_work_item::
                                          get_sub_group()
                                              .get_local_linear_id())
                            : 0,
                        sycl::ext::oneapi::plus<>()))
                {
                //Scan
                    x = x & mask2;
                    n = sycl::popcount(x);
                    x = x & mask;
                    const int p = sycl::popcount(x);
                    if (flag) mytemp[(nNeighbors+p)*32] = jOff;
                    nNeighbors += n;
                }

            } // loop over all atoms
        } // loop over neighbor cells

        if(id == 0)
            nNeighborsArray[iLid] = nNeighbors;
    }
    /*
    DPCT1065:170: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const int iLid = item_ct1.get_group(2) * 32 + laneid;
    int N;
    if(iLid < sim.a_list.n)
        N = nNeighborsArray[iLid];
    else
        N = 0;
    for (int i = item_ct1.get_local_id(2) >> 5; i < N; i += packSize)
    {
        neighborList[iLid + i * ldNeighborList] =
            temp[(i << 5) + (item_ct1.get_local_id(2) & 31)];
    }
}

template <int boundaryFlag>
/*
DPCT1110:95: The total declared local variable size in device function
buildNeighborListKernel_thread exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/

void buildNeighborListKernel_thread(SimGpu sim)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (tid >= sim.a_list.n) return;

  // compute box ID and local atom ID
  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  assert(sim.cell_type[iBox] == 0 || sim.cell_type[iBox] == 1);
  assert(iBox < sim.boxes.nLocalBoxes && iBox >= 0);
  if(boundaryFlag == BOUNDARY && sim.cell_type[iBox] == 0) return;
  if(boundaryFlag == INTERIOR && sim.cell_type[iBox] == 1) return;

  int iOff = iBox * MAXATOMS + iAtom;
  assert(iOff < sim.boxes.nLocalBoxes * MAXATOMS && iOff >=0 );

  real_t rCut = sim.eam_pot.cutoff;
  real_t rCut2 = (rCut+sim.atoms.neighborList.skinDistance)*(rCut+sim.atoms.neighborList.skinDistance);

  // fetch position
  real_t irx = sim.atoms.r.x[iOff];
  real_t iry = sim.atoms.r.y[iOff];
  real_t irz = sim.atoms.r.z[iOff];

  //get NL related data
  int iLid = tid;
  const int ldNeighborList = sim.atoms.neighborList.nMaxLocal; //leading dimension
  assert(iLid<ldNeighborList);
  int* neighborList = sim.atoms.neighborList.list;
  int nNeighbors = 0;
  sim.atoms.neighborList.lastR.x[iLid] = irx;
  sim.atoms.neighborList.lastR.y[iLid] = iry;
  sim.atoms.neighborList.lastR.z[iLid] = irz;

  real_t *const __restrict__ rx = sim.atoms.r.x;
  real_t *const __restrict__ ry = sim.atoms.r.y;
  real_t *const __restrict__ rz = sim.atoms.r.z;

  // loop over my neighbor cells
  for (int j = 0; j < N_MAX_NEIGHBORS; j++)
  {
    int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];

    // loop over all atoms in the neighbor cell
    for (int jAtom = 0; jAtom < sim.boxes.nAtoms[jBox]; jAtom++)
    {
      int jOff = jBox * MAXATOMS + jAtom;
      assert(jOff < sim.boxes.nTotalBoxes * MAXATOMS  && jOff >=0 );

      real_t dx = irx - rx[jOff];
      real_t dy = iry - ry[jOff];
      real_t dz = irz - rz[jOff];

      // distance^2
      real_t r2 = dx*dx + dy*dy + dz*dz;

      // no divide by zero
      if (r2 <= rCut2 && r2 > 0.0)
      {
         assert(nNeighbors < sim.atoms.neighborList.nMaxNeighbors); // TODO enlarge neighborlist (this should be fine for now)
         neighborList[nNeighbors * ldNeighborList + iLid ] = jOff;
         ++nNeighbors;
      }
    } // loop over all atoms
  } // loop over neighbor cells

#ifdef DEBUG
  //invalidate old neighbors
  for(int j=nNeighbors; j < sim.atoms.neighborList.nMaxNeighbors ; ++j){
     neighborList[j * ldNeighborList + iLid ] = -1;
  }
#endif

  sim.atoms.neighborList.nNeighbors[iLid] = nNeighbors;
}

/// Build the neighbor list for all boxes which are marked as dirty.
extern "C"
extern "C" void buildNeighborListGpu(SimGpu* sim, int method, int boundaryFlag)
{
   NeighborListGpu* neighborList = &(sim->atoms.neighborList);

   if(neighborListUpdateRequiredGpu(sim) == 1){
           emptyNeighborListGpu(sim, boundaryFlag);

           //int grid = (sim->a_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
           int grid = (sim->a_list.n + 31)/ 32;
           const int packSize = NEIGHLIST_PACKSIZE;
           const int logPackSize = NEIGHLIST_PACKSIZE_LOG;
           const int memoryPackSize = KERNEL_PACKSIZE;
           int block = packSize * 32;
           /*
           DPCT1026:180: The call to cudaDeviceSetCacheConfig was removed
           because SYCL currently does not support setting cache config on
           devices.
           */
           real_t rCut = sim->eam_pot.cutoff;
           real_t rCut2 = (rCut+sim->atoms.neighborList.skinDistance)*(rCut+sim->atoms.neighborList.skinDistance);
           if(method == THREAD_ATOM_NL)
           {
#if 0
              int grid = (sim->a_list.n + (THREAD_ATOM_CTA-1))/ THREAD_ATOM_CTA;
              int block = THREAD_ATOM_CTA;
              if(boundaryFlag == BOUNDARY)
                 buildNeighborListKernel_thread<BOUNDARY><<<grid, block>>>(*sim);
              else if (boundaryFlag == INTERIOR)
                 buildNeighborListKernel_thread<INTERIOR><<<grid, block >>>(*sim);
              else {
                 buildNeighborListKernel_thread<BOTH><<<grid, block>>>(*sim);
              }
#else
              if(boundaryFlag == BOUNDARY)
                 /*
                 DPCT1049:96: The work-group size passed to the SYCL kernel may
                 exceed the limit. To get the device limit, query
                 info::device::max_work_group_size. Adjust the work-group size
                 if needed.
                 */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int, 1> temp_acc_ct1(
              sycl::range<1>(32 * MAXNEIGHBORLISTSIZE), cgh);

          SimGpu sim_ct0 = *sim;

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                buildNeighborListKernel_warp<MAXNEIGHBORLISTSIZE, packSize,
                                             logPackSize, 1, BOUNDARY>(
                    sim_ct0, rCut2,
                    temp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
              else if (boundaryFlag == INTERIOR)
                 /*
                 DPCT1049:97: The work-group size passed to the SYCL kernel may
                 exceed the limit. To get the device limit, query
                 info::device::max_work_group_size. Adjust the work-group size
                 if needed.
                 */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int, 1> temp_acc_ct1(
              sycl::range<1>(32 * MAXNEIGHBORLISTSIZE), cgh);

          SimGpu sim_ct0 = *sim;

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                buildNeighborListKernel_warp<MAXNEIGHBORLISTSIZE, packSize,
                                             logPackSize, 1, INTERIOR>(
                    sim_ct0, rCut2,
                    temp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
              else {
                 /*
                 DPCT1049:98: The work-group size passed to the SYCL kernel may
                 exceed the limit. To get the device limit, query
                 info::device::max_work_group_size. Adjust the work-group size
                 if needed.
                 */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int, 1> temp_acc_ct1(
              sycl::range<1>(32 * MAXNEIGHBORLISTSIZE), cgh);

          SimGpu sim_ct0 = *sim;

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                buildNeighborListKernel_warp<MAXNEIGHBORLISTSIZE, packSize,
                                             logPackSize, 1, BOTH>(
                    sim_ct0, rCut2,
                    temp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
              }
#endif
           }
           else if(method == WARP_ATOM_NL)
           {
               if(boundaryFlag == BOUNDARY)
                   /*
                   DPCT1049:99: The work-group size passed to the SYCL kernel
                   may exceed the limit. To get the device limit, query
                   info::device::max_work_group_size. Adjust the work-group size
                   if needed.
                   */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int, 1> temp_acc_ct1(
              sycl::range<1>(32 * MAXNEIGHBORLISTSIZE), cgh);

          SimGpu sim_ct0 = *sim;

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                buildNeighborListKernel_warp<MAXNEIGHBORLISTSIZE, packSize,
                                             logPackSize, memoryPackSize,
                                             BOUNDARY>(
                    sim_ct0, rCut2,
                    temp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
               else if (boundaryFlag == INTERIOR)
                   /*
                   DPCT1049:100: The work-group size passed to the SYCL kernel
                   may exceed the limit. To get the device limit, query
                   info::device::max_work_group_size. Adjust the work-group size
                   if needed.
                   */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int, 1> temp_acc_ct1(
              sycl::range<1>(32 * MAXNEIGHBORLISTSIZE), cgh);

          SimGpu sim_ct0 = *sim;

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                buildNeighborListKernel_warp<MAXNEIGHBORLISTSIZE, packSize,
                                             logPackSize, memoryPackSize,
                                             INTERIOR>(
                    sim_ct0, rCut2,
                    temp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
               else
                   /*
                   DPCT1049:101: The work-group size passed to the SYCL kernel
                   may exceed the limit. To get the device limit, query
                   info::device::max_work_group_size. Adjust the work-group size
                   if needed.
                   */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
          sycl::local_accessor<int, 1> temp_acc_ct1(
              sycl::range<1>(32 * MAXNEIGHBORLISTSIZE), cgh);

          SimGpu sim_ct0 = *sim;

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                    sycl::range<3>(1, 1, block),
                                sycl::range<3>(1, 1, block)),
              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                buildNeighborListKernel_warp<MAXNEIGHBORLISTSIZE, packSize,
                                             logPackSize, memoryPackSize, BOTH>(
                    sim_ct0, rCut2,
                    temp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
              });
        });
           }

           /*
           DPCT1026:181: The call to cudaDeviceSetCacheConfig was removed
           because SYCL currently does not support setting cache config on
           devices.
           */
           neighborList->nStepsSinceLastBuild = 1;
           neighborList->updateNeighborListRequired = 0;
           neighborList->forceRebuildFlag = 0;
   }else
           neighborList->nStepsSinceLastBuild++;

  CUDA_GET_LAST_ERROR
}

extern "C"
extern "C" void emptyHashTableGpu(HashTableGpu* hashTable)
{
   hashTable->nEntriesPut = 0;
}

extern "C"
extern "C" void initHashTableGpu(HashTableGpu* hashTable, int nMaxEntries)
{

   hashTable->nMaxEntries = nMaxEntries;
   hashTable->nEntriesPut = 0; //allocates a 5MB hashtable. This number is prime.
   hashTable->nEntriesGet = 0; //allocates a 5MB hashtable. This number is prime.

   hashTable->offset = sycl::malloc_device<int>(hashTable->nMaxEntries,
                                                dpct::get_in_order_queue());

  CUDA_GET_LAST_ERROR

}
