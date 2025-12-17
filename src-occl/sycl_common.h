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

#ifndef __SYCL_COMMON_H_
#define __SYCL_COMMON_H_

#include <sycl/sycl.hpp>
#include "defines.h"

// Global SYCL queue - initialized in gpu_utility.cpp
extern sycl::queue* g_sycl_queue;

// Approximate sqrt for float (SYCL version)
inline float sqrt_approx(float f) {
    return sycl::sqrt(f);
}

/// Interpolate a table to determine f(r) and its derivative f'(r).
///
/// \param [in] table Interpolation table.
/// \param [in] r Point where function value is needed.
/// \param [out] f The interpolated value of f(r).
/// \param [out] df The interpolatedi value of df(r)/dr.
inline void interpolate(InterpolationObjectGpu table, real_t r, real_t &f, real_t &df)
{
   const real_t* tt = table.values; // alias

   // check boundaries
   r = sycl::max(r, table.x0);
   r = sycl::min(r, table.xn);
    
   // compute index
   r = r * table.invDx - table.invDxXx0;
   
   real_t ri = sycl::floor(r);
   
   int ii = (int)ri;

   // reset r to fractional distance
   r = r - ri;
   
   real_t v0 = tt[ii];
   real_t v1 = tt[ii + 1];
   real_t v2 = tt[ii + 2];
   real_t v3 = tt[ii + 3];
   
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
inline void interpolateSpline(InterpolationSplineObjectGpu table, real_t r2, real_t &f, real_t &df)
{
   const real_t* tt = table.coefficients; // alias

   float r = sycl::sqrt((float)r2);

   // check boundaries
   r = sycl::max(r, table.x0);
   r = sycl::min(r, table.xn);
    
   // compute index
   r = r * table.invDx - table.invDxXx0;
   
   real_t ri = sycl::floor(r);
   
   int ii = 4*(int)ri;

   real_t a = tt[ii];
   real_t b = tt[ii + 1];
   real_t c = tt[ii + 2];
   real_t d = tt[ii + 3];
   
   real_t tmp = a*r2+b;
   f = (tmp*r2+c)*r2+d;
   df = 2*((3*tmp-b)*r2+c);
}

// Sub-group reduction helpers (SYCL equivalent of warp shuffle)
template<typename T>
inline T sub_group_reduce_add(sycl::sub_group sg, T val) {
    return sycl::reduce_over_group(sg, val, sycl::plus<T>());
}

// Warp-level reduction using sub-groups
inline void warp_reduce(sycl::sub_group sg, real_t &x) {
    x = sycl::reduce_over_group(sg, x, sycl::plus<real_t>());
}

template<int step>
inline void warp_reduce(sycl::sub_group sg, real_t &ifx, real_t &ify, real_t &ifz, real_t &ie, real_t &irho)
{
    ifx = sycl::reduce_over_group(sg, ifx, sycl::plus<real_t>());
    ify = sycl::reduce_over_group(sg, ify, sycl::plus<real_t>());
    ifz = sycl::reduce_over_group(sg, ifz, sycl::plus<real_t>());
    if (step == 1) {
        ie = sycl::reduce_over_group(sg, ie, sycl::plus<real_t>());
        irho = sycl::reduce_over_group(sg, irho, sycl::plus<real_t>());
    }
}

// Shuffle operations using sub-groups
template<typename T>
inline T shfl_xor(sycl::sub_group sg, T var, int laneMask) {
    // XOR shuffle pattern
    unsigned int srcLane = sg.get_local_linear_id() ^ laneMask;
    return sycl::select_from_group(sg, var, srcLane);
}

template<typename T>
inline T shfl(sycl::sub_group sg, T var, int srcLane) {
    return sycl::select_from_group(sg, var, srcLane);
}

template<typename T>
inline T shfl_up(sycl::sub_group sg, T var, unsigned int delta) {
    return sycl::shift_group_right(sg, var, delta);
}

template<typename T>
inline T shfl_down(sycl::sub_group sg, T var, unsigned int delta) {
    return sycl::shift_group_left(sg, var, delta);
}

// Optimized version of DP rsqrt
inline double fast_rsqrt(double a) {
    return sycl::rsqrt(a);
}

inline float fast_rsqrt(float a) {
    return sycl::rsqrt(a);
}

// Optimized version of sqrt
template<typename real>
inline real sqrt_opt(real a) {
    return a * fast_rsqrt(a);
}

// Ballot emulation using sub-groups
inline unsigned int ballot(sycl::sub_group sg, bool pred) {
    unsigned int result = 0;
    unsigned int lane_id = sg.get_local_linear_id();
    unsigned int bit = pred ? (1u << lane_id) : 0;
    // Reduce all bits
    for (int i = 0; i < SUB_GROUP_SIZE; ++i) {
        result |= sycl::select_from_group(sg, bit, i);
    }
    return result;
}

// Population count
inline int popcount(unsigned int x) {
    return sycl::popcount(x);
}

// Atomic add for SYCL
template<typename T>
inline void atomicAdd(T* address, T val) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, 
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*address);
    ref.fetch_add(val);
}

// Atomic add for integers
inline void atomicAddInt(int* address, int val) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*address);
    ref.fetch_add(val);
}

// Warp ID and Lane ID helpers
inline int get_warp_id(sycl::nd_item<1> item) {
    return item.get_local_id(0) / SUB_GROUP_SIZE;
}

inline int get_lane_id(sycl::nd_item<1> item) {
    return item.get_local_id(0) % SUB_GROUP_SIZE;
}

//=============================================================================
// Device functions for link cell operations
//=============================================================================

/// Get box index from tuple coordinates (device function)
inline int getBoxFromTuple_dev(LinkCellGpu boxes, int ix, int iy, int iz)
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
    // local link cell
    else
    {
        iBox = boxes.boxIDLookUp[IDX3D(ix, iy, iz, boxes.gridSize.x, boxes.gridSize.y)];
    }

    return iBox;
}

/// Get box index from coordinates (device function)
inline int getBoxFromCoord_dev(LinkCellGpu cells, real_t rx, real_t ry, real_t rz)
{
    int ix = (int)(sycl::floor((rx - cells.localMin.x) * cells.invBoxSize.x));
    int iy = (int)(sycl::floor((ry - cells.localMin.y) * cells.invBoxSize.y));
    int iz = (int)(sycl::floor((rz - cells.localMin.z) * cells.invBoxSize.z));

    // For each axis, if we are inside the local domain, make sure we get
    // a local link cell. Otherwise, make sure we get a halo link cell.
    if (rx < cells.localMax.x)
    {
        if (ix == cells.gridSize.x) ix = cells.gridSize.x - 1;
    }
    else
        ix = cells.gridSize.x;
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

    return getBoxFromTuple_dev(cells, ix, iy, iz);
}

/// Atomic max for int
inline int atomicMax_int(int* address, int val) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*address);
    return ref.fetch_max(val);
}

/// Atomic add for int returning old value
inline int atomicAdd_int(int* address, int val) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space> ref(*address);
    return ref.fetch_add(val);
}

#endif
