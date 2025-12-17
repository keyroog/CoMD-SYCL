# CoMD SYCL+MPI+oneCCL Port

This directory contains a port of CoMD (Co-design Molecular Dynamics) from CUDA+MPI to SYCL+MPI+oneCCL. The port is designed to run on Intel GPUs using the Intel oneAPI toolkit.

## Overview

The original CoMD CUDA implementation has been ported to SYCL with the following key changes:

### Memory Management
- `cudaMalloc` → `sycl::malloc_device`
- `cudaMemcpy` → `queue.memcpy()` 
- `cudaFree` → `sycl::free`
- `cudaMemset` → `queue.memset()`

### Kernel Execution
- CUDA `kernel<<<grid, block>>>()` → SYCL `queue.parallel_for(nd_range<1>())`
- CUDA streams → SYCL queue (single global queue)
- `cudaDeviceSynchronize()` → `queue.wait()`

### Warp Operations (CUDA → SYCL sub-groups)
- `__shfl_down_sync()` → `sycl::shift_group_left()` via sub-group
- `__ballot_sync()` → emulated with `sycl::reduce_over_group()`
- `atomicAdd()` → `sycl::atomic_ref<T>::fetch_add()`

### Collective Communications
- `MPI_Allreduce` (for sum/max) → `ccl::allreduce` (oneCCL)
- `MPI_Allreduce` with MINLOC/MAXLOC → remains MPI (not in oneCCL)

## Directory Structure

```
src-occl/
├── CoMD.cpp              # Main program (SYCL version)
├── Makefile              # Build configuration
├── defines.h             # SYCL-adapted constants
├── gpu_types.h           # GPU data structures
├── sycl_common.h         # SYCL helper functions (interpolate, atomics, etc.)
├── gpu_utility.cpp/h     # GPU setup, allocation, data transfer
├── gpu_kernels.cpp/h     # SYCL kernels (force, velocity, position)
├── parallel.cpp/h        # MPI + oneCCL wrappers
├── timestep.cpp          # Time integration (with SYCL calls)
├── linkCells.cpp         # Link cell management
├── haloExchange.cpp      # Halo exchange for MPI
├── eam.cpp               # EAM potential (SYCL kernels)
├── gpu_neighborList.cpp  # Neighbor list management
└── [other C files]       # Common code unchanged from original
```

## Build Requirements

- Intel oneAPI Base Toolkit (2023.2+)
  - DPC++ compiler (`icpx`)
  - Intel oneCCL
- MPI implementation (Intel MPI recommended)
- Target: Intel GPUs (Data Center or client GPUs)

## Building

```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build with MPI and oneCCL
cd src-occl
make

# The executable will be in ../bin/CoMD-sycl-mpi-ccl
```

## Configuration Options (Makefile)

```makefile
DOUBLE_PRECISION = ON    # ON/OFF for double/single precision
DO_MPI = ON              # ON/OFF for MPI support
USE_ONECCL = ON          # ON/OFF for oneCCL collective operations
MAXATOMS = 256           # Maximum atoms per link cell
```

## Running

```bash
# Single GPU
./CoMD-sycl-mpi-ccl -m thread_atom -e -i 10 -j 10 -k 10

# Multiple GPUs with MPI
mpirun -n 2 ./CoMD-sycl-mpi-ccl -m thread_atom -e -i 10 -j 10 -k 10 -x 2 -y 1 -z 1
```

### Force Calculation Methods
- `thread_atom`: One thread per atom (default)
- `warp_atom`: One sub-group per atom
- `cta_cell`: One work-group per cell
- `thread_atom_nl`: Thread-atom with neighbor list
- `warp_atom_nl`: Warp-atom with neighbor list

## Key Differences from CUDA Version

1. **Single Queue Model**: SYCL uses a single global queue instead of multiple CUDA streams. Async operations are still possible but managed differently.

2. **Sub-groups**: SYCL sub-groups replace CUDA warps. The default sub-group size is 32 (like CUDA warps) but can be 16 on some Intel hardware.

3. **oneCCL Integration**: Global reductions (sum, max) use oneCCL for potentially better performance on Intel platforms. MINLOC/MAXLOC operations still use MPI.

4. **USM (Unified Shared Memory)**: Uses explicit device memory allocation (`malloc_device`) similar to CUDA's `cudaMalloc`.

## Known Limitations

1. **EAM Force Kernels**: The EAM potential kernels are simplified stubs. Full implementation requires porting the interpolation table lookups.

2. **CUB Replacement**: CUDA CUB library scan operations need SYCL equivalents (oneDPL or custom implementation).

3. **Async Operations**: The SYCL port uses synchronous operations by default. Async overlapping requires additional work.

## Performance Tuning

- Set `SUB_GROUP_SIZE` in defines.h to match your GPU (32 for Intel Data Center, may vary)
- Adjust `N_MAX_NEIGHBORS` and `MAXATOMS` based on problem size
- Use `-fsycl-targets=spir64_gen -Xs "-device pvc"` for specific GPU targeting

## License

Same as the original CoMD (BSD-3 clause license).
