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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

/// In-place exclusive scan (prefix sum) on device memory.
/// Replaces the oneDPL-based implementation to avoid the oneDPL dependency.
/// Uses a host round-trip which is acceptable for the small arrays used in
/// CoMD (nCells+1, typically a few thousand elements).
void scan(int *data, int n, int *partial_sums, dpct::queue_ptr stream)
{
  if (n <= 0) return;

  int *h_buf = sycl::malloc_host<int>(n, *stream);
  stream->memcpy(h_buf, data, n * sizeof(int)).wait();

  int sum = 0;
  for (int i = 0; i < n; i++) {
    int val = h_buf[i];
    h_buf[i] = sum;
    sum += val;
  }

  stream->memcpy(data, h_buf, n * sizeof(int)).wait();
  sycl::free(h_buf, *stream);
}

#endif
