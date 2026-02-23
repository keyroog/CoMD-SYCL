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

#ifndef __GPU_SCAN_H_
#define __GPU_SCAN_H_

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

void scan(int *data, int n, int *partial_sums, dpct::queue_ptr stream)
{
  size_t temp_storage_bytes = 0;
  /*
  DPCT1026:43: The call to cub::DeviceScan::ExclusiveSum was removed because
  this functionality is redundant in SYCL.
  */

  void *temp_storage = (void*)partial_sums;
  bool own_temp_storage = false;

  // Legacy call sites reserve n ints as scratch. Fallback to dynamic storage if CUB needs more.
  if (temp_storage == NULL || temp_storage_bytes > (size_t)n * sizeof(int)) {
    temp_storage = (void *)sycl::malloc_device(temp_storage_bytes,
                                               dpct::get_in_order_queue());
    own_temp_storage = true;
  }

  oneapi::dpl::exclusive_scan(
      oneapi::dpl::execution::device_policy(*stream), data, data + n, data,
      typename std::iterator_traits<decltype(data)>::value_type{});

  if (own_temp_storage) {
    dpct::dpct_free(temp_storage, dpct::get_in_order_queue());
  }
}

#endif
