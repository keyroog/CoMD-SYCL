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

#ifndef __GPU_COMMON_H_
#define __GPU_COMMON_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "defines.h"
#include <dpct/math.hpp>

__dpct_inline__ float sqrt_approx(float f)
{
    float ret;
  ret = sycl::sqrt(f);
    return ret;
}

/// Interpolate a table to determine f(r) and its derivative f'(r).
///
/// \param [in] table Interpolation table.
/// \param [in] r Point where function value is needed.
/// \param [out] f The interpolated value of f(r).
/// \param [out] df The interpolatedi value of df(r)/dr.
__inline__ 
void interpolate(InterpolationObjectGpu table, real_t r, real_t &f, real_t &df)
{
   const real_t* tt = table.values; // alias

   // check boundaries
   r = sycl::max(r, table.x0);
   r = sycl::min(r, table.xn);

   // compute index
   r = r * table.invDx - table.invDxXx0;

   real_t ri = sycl::floor(r);

   int ii = (int)ri;
   // GPU path reads ii..ii+3, so ii must stay within [0, n-1].
   if (ii < 0) ii = 0;
   if (ii >= table.n) ii = table.n - 1;

   // reset r to fractional distance
   r = r - (real_t)ii;
   
    // using LDG on Kepler only
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 350
    /*
    DPCT1098:111: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    real_t v0 = *(tt + ii);
    /*
    DPCT1098:112: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    real_t v1 = *(tt + ii + 1);
    /*
    DPCT1098:113: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    real_t v2 = *(tt + ii + 2);
    /*
    DPCT1098:114: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    real_t v3 = *(tt + ii + 3);
#else
    real_t v0 = tt[ii];
    real_t v1 = tt[ii + 1];
    real_t v2 = tt[ii + 2];
    real_t v3 = tt[ii + 3];
#endif
   
   real_t g1 = v2 - v0;
   real_t g2 = v3 - v1;

     f = v1 + 0.5 * r * (g1 + r * (v2 + v0 - 2.0 * v1));
     df = (g1 + r * (g2 - g1)) * table.invDxHalf;
}

/// Interpolate using spline coefficients table 
/// to determine f(r) and its derivative f'(r).
///
/// \param [in] table Table with spline coefficients.
/// \param [in] r2 Square of point where function value is needed.
/// \param [out] f The interpolated value of f(r).
/// \param [out] df The interpolated value of 1/r*df(r)/dr.
__inline__ 
void interpolateSpline(InterpolationSplineObjectGpu table, real_t r2, real_t &f, real_t &df)
{
   const real_t* tt = table.coefficients; // alias

   float r = sqrt_approx(r2);

   // check boundaries
   r = sycl::max(r, table.x0);
   r = sycl::min(r, table.xn);

   // compute index
   r = r * table.invDx - table.invDxXx0;

   real_t ri = sycl::floor(r);
   int bin = (int)ri;
   if (bin < 0) bin = 0;
   if (bin >= table.n) bin = table.n - 1;
   int ii = 4*bin;

    // using LDG on Kepler only
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 350
    /*
    DPCT1098:115: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    real_t a = *(tt + ii);
    /*
    DPCT1098:116: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    real_t b = *(tt + ii + 1);
    /*
    DPCT1098:117: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    real_t c = *(tt + ii + 2);
    /*
    DPCT1098:118: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    real_t d = *(tt + ii + 3);
#else
    real_t a = tt[ii];
    real_t b = tt[ii + 1];
    real_t c = tt[ii + 2];
    real_t d = tt[ii + 3];
#endif
   
     real_t tmp = a*r2+b;
     f = (tmp*r2+c)*r2+d;
     df =2*((3*tmp-b)*r2+c);
}

#if !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP < 300
// emulate shuffles through shared memory for old devices (SLOW)
__device__ __forceinline__
double __shfl_xor(double var, int laneMask, double *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + (lane_id ^ laneMask)] : var;
}

__device__ __forceinline__
float __shfl_xor(float var, int laneMask, float *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + (lane_id ^ laneMask)] : var;
}

__device__ __forceinline__
int __shfl_up(int var, unsigned int delta, int width, int *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / width;
  const int lane_id = threadIdx.x % width;
  return lane_id >= delta ? smem[width * warp_id + (lane_id - delta)] : var;
}

__device__ __forceinline__
double __shfl(double var, int laneMask, volatile double *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + laneMask] : var;
}

__device__ __forceinline__
float __shfl(float var, int laneMask, volatile float *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + laneMask] : var;
}

__device__ __forceinline__
int __shfl(int var, int laneMask, volatile int *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + laneMask] : var;
}

__device__ __forceinline__
int __shfl_down(real_t var, unsigned int delta, int width, real_t *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / width;
  const int lane_id = threadIdx.x % width;
  return lane_id < width - delta ? smem[width * warp_id + (lane_id + delta)] : var;
}

#else	// >= SM 3.0

#if (DPCT_COMPAT_RT_VERSION < 6050)
__device__ __forceinline__
double __shfl_xor(double var, int laneMask)
{
  int lo = __shfl_xor( __double2loint(var), laneMask );
  int hi = __shfl_xor( __double2hiint(var), laneMask );
  return __hiloint2double( hi, lo );
}

__device__ __forceinline__
double __shfl(double var, int laneMask)
{
  int lo = __shfl( __double2loint(var), laneMask );
  int hi = __shfl( __double2hiint(var), laneMask );
  return __hiloint2double( hi, lo );
}

__device__ __forceinline__
double __shfl_down(double var, unsigned int delta, int width = 32)
{
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo), "=r"(hi):"d"(var));
    lo = __shfl_down(lo, delta, width);
    hi = __shfl_down(hi, delta, width);
    return __hiloint2double(hi, lo);
}
#endif
#endif

#ifndef uint
typedef unsigned int uint;
#endif

// insert the first numBits of y into x starting at bit
uint bfi(uint x, uint y, uint bit, uint numBits) {
  uint ret;
  ret = dpct::bfi_safe<uint32_t>(y, x, bit, numBits);
  return ret;
}

__dpct_inline__ void warp_reduce(real_t &x, real_t *smem)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
  smem[item_ct1.get_local_id(2)] = x;
  // technically we also need warp sync here
  if (lane_id < 16) smem[item_ct1.get_local_id(2)] +=
      smem[item_ct1.get_local_id(2) + 16];
  if (lane_id < 8) smem[item_ct1.get_local_id(2)] +=
      smem[item_ct1.get_local_id(2) + 8];
  if (lane_id < 4) smem[item_ct1.get_local_id(2)] +=
      smem[item_ct1.get_local_id(2) + 4];
  if (lane_id < 2) smem[item_ct1.get_local_id(2)] +=
      smem[item_ct1.get_local_id(2) + 2];
  if (lane_id < 1) smem[item_ct1.get_local_id(2)] +=
      smem[item_ct1.get_local_id(2) + 1];
  x = smem[item_ct1.get_local_id(2)];
}

template <int step>
__dpct_inline__ void warp_reduce(real_t &ifx, real_t &ify, real_t &ifz,
                                 real_t &ie, real_t &irho)
{
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 300
  // warp reduction
  /*
  DPCT1086:0: __activemask() is migrated to 0xffffffff. You may need to adjust
  the code.
  */
  const unsigned int active_mask = 0xffffffff;
  for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
    /*
    DPCT1023:1: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_xor_sync.
    */
    /*
    DPCT1096:195: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    ifx += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), ifx, i);
    /*
    DPCT1023:2: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_xor_sync.
    */
    /*
    DPCT1096:196: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    ify += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), ify, i);
    /*
    DPCT1023:3: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_xor_sync.
    */
    /*
    DPCT1096:197: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    ifz += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), ifz, i);
    if (step == 1) {
      /*
      DPCT1023:4: The SYCL sub-group does not support mask options for
      dpct::permute_sub_group_by_xor. You can specify
      "--use-experimental-features=masked-sub-group-operation" to use the
      experimental helper function to migrate __shfl_xor_sync.
      */
      /*
      DPCT1096:198: The right-most dimension of the work-group used in the SYCL
      kernel that calls this function may be less than "32". The function
      "dpct::permute_sub_group_by_xor" may return an unexpected result on the
      CPU device. Modify the size of the work-group to ensure that the value of
      the right-most dimension is a multiple of "32".
      */
      ie += dpct::permute_sub_group_by_xor(
          sycl::ext::oneapi::this_work_item::get_sub_group(), ie, i);
      /*
      DPCT1023:5: The SYCL sub-group does not support mask options for
      dpct::permute_sub_group_by_xor. You can specify
      "--use-experimental-features=masked-sub-group-operation" to use the
      experimental helper function to migrate __shfl_xor_sync.
      */
      /*
      DPCT1096:199: The right-most dimension of the work-group used in the SYCL
      kernel that calls this function may be less than "32". The function
      "dpct::permute_sub_group_by_xor" may return an unexpected result on the
      CPU device. Modify the size of the work-group to ensure that the value of
      the right-most dimension is a multiple of "32".
      */
      irho += dpct::permute_sub_group_by_xor(
          sycl::ext::oneapi::this_work_item::get_sub_group(), irho, i);
    }
  }
#else
  // reduction using shared memory
  __shared__ real_t smem[WARP_ATOM_CTA];
  warp_reduce(ifx, smem);
  warp_reduce(ify, smem);
  warp_reduce(ifz, smem);
  if (step == 1) {
    warp_reduce(ie, smem);
    warp_reduce(irho, smem);
  }
#endif
}

// emulate atomic add for doubles
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 600
__device__ inline void atomicAdd(double *address, double value)
{
  unsigned long long oldval, newval, readback;
  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback = atomicCAS((unsigned long long*)address, oldval, newval)) != oldval)
  {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  }
}
#endif

static __dpct_inline__ int get_warp_id()
{
  return sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) >>
         5;
}

static __dpct_inline__ int get_lane_id()
{
  int id;
  id = sycl::ext::oneapi::this_work_item::get_nd_item<3>()
           .get_sub_group()
           .get_local_linear_id();
  return id;
}

// optimized version of DP rsqrt(a) provided by Norbert Juffa

double fast_rsqrt(double a)
{
  double x, e, t;
  float f;
  f = sycl::vec<double, 1>(a)
          .template convert<float, sycl::rounding_mode::rte>()
          .x();
  f = sycl::rsqrt(f);
  x = static_cast<double>(f);
  /*
  DPCT1013:119: The rounding mode could not be specified and the generated code
  may have different accuracy than the original code. Verify the correctness.
  SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
  standard.
  */
  t = x * x;
  /*
  DPCT1013:120: The rounding mode could not be specified and the generated code
  may have different accuracy than the original code. Verify the correctness.
  SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
  standard.
  */
  e = sycl::fma(a, -t, 1.0);
  /*
  DPCT1013:121: The rounding mode could not be specified and the generated code
  may have different accuracy than the original code. Verify the correctness.
  SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
  standard.
  */
  t = sycl::fma(0.375, e, 0.5);
  /*
  DPCT1013:122: The rounding mode could not be specified and the generated code
  may have different accuracy than the original code. Verify the correctness.
  SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
  standard.
  */
  e = e * x;
  /*
  DPCT1013:123: The rounding mode could not be specified and the generated code
  may have different accuracy than the original code. Verify the correctness.
  SYCL math built-in function rounding mode is aligned with OpenCL C 1.2
  standard.
  */
  x = sycl::fma(t, e, x);
  return x;
}


float fast_rsqrt(float a)
{
  return sycl::rsqrt(a);
}

// optimized version of sqrt(a)
template<class real>

real sqrt_opt(real a)
{
#if 1
  return a * fast_rsqrt(a);
#else
  return sqrt(a);
#endif
}

#endif
