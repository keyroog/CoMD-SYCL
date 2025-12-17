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

#ifndef __DEFINES_H_
#define __DEFINES_H_

#define HASHTABLE_FREE -1
#define BOUNDARY 1
#define INTERIOR 2
#define BOTH 0

//methods
#define THREAD_ATOM 0
#define THREAD_ATOM_NL 1
#define WARP_ATOM 2
#define WARP_ATOM_NL 3
#define CTA_CELL 4
//CPU method has to be the last
#define CPU_NL 5


#define ISPOWER2(v) ((v) && !((v) & ((v) - 1)))
            
#define IDX3D(x,y,z,X,Y) ((z)*((Y)*(X)) + ((y)*(X)) + (x))

/// The maximum number of atoms that can be stored in a link cell.
//Moved to the Makefile
//#define MAXATOMS 256 

// SYCL sub-group size (equivalent to CUDA warp size)
// Intel GPUs typically use 16 or 32, we use 32 for compatibility
#define WARP_SIZE		32
#define SUB_GROUP_SIZE  32

#define THREAD_ATOM_CTA         128
#define WARP_ATOM_CTA		128
#define CTA_CELL_CTA		128

// Work-group sizes for SYCL
#define WORK_GROUP_SIZE     128
#define THREAD_ATOM_WG      128
#define WARP_ATOM_WG        128
#define CTA_CELL_WG         128

// NOTE: the following is tuned for Intel GPUs
#ifdef COMD_DOUBLE
#define THREAD_ATOM_ACTIVE_CTAS 	10	// 62%
#define WARP_ATOM_ACTIVE_CTAS 		12	// 75%
#define CTA_CELL_ACTIVE_CTAS 		10	// 62%
#define WARP_ATOM_NL_CTAS            9  // 56%
#else
// 100% occupancy for SP
#define THREAD_ATOM_ACTIVE_CTAS 	16
#define WARP_ATOM_ACTIVE_CTAS 		16
#define CTA_CELL_ACTIVE_CTAS 		16
#define WARP_ATOM_NL_CTAS           16
#endif

//log_2(x)
#define LOG(X) _LOG( X )
#define _LOG(X) _LOG_ ## X

#define _LOG_32 5
#define _LOG_16 4
#define _LOG_8  3
#define _LOG_4  2
#define _LOG_2  1
#define _LOG_1  0

//Number of threads collaborating to make neighbor list for a single atom
#define NEIGHLIST_PACKSIZE 8
#define NEIGHLIST_PACKSIZE_LOG LOG(NEIGHLIST_PACKSIZE)
//Number of threads to compute forces of single atom in warp_atom_nl method
#define KERNEL_PACKSIZE 4

//Maximum size of neighbor list for a single atom
#define MAXNEIGHBORLISTSIZE 64

#define VECTOR_WIDTH 4

//size of shared memory used in cta_cell kernel for Lennard-Jones
//it can't be less than CTA_CELL_CTA
#define SHARED_SIZE_CTA_CELL 128 

//Number of atoms covered by a single entry of pairlist
//Cannot be bigger than 1024 (resulting in 32x32 blocks)
#define PAIRLIST_ATOMS_PER_INT 1024

#define PAIRLIST_STEP (PAIRLIST_ATOMS_PER_INT/32)

#endif
