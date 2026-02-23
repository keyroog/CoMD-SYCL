# CoMD-SYCL

A port of **CoMD-CUDA** to **SYCL + oneCCL**, replacing CUDA with SYCL for GPU kernels and oneCCL for MPI collectives in distributed runs.

CoMD is a reference implementation of classical molecular dynamics algorithms maintained by the [ExMatEx](http://codesign.lanl.gov/projects/exmatex) co-design center. This variant targets NVIDIA GPUs via DPC++ (Intel's LLVM-based SYCL compiler) with `nvptx64` as the SYCL target, and uses oneCCL (with NCCL backend) for GPU-aware collective communication.

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Building](#building)
  - [Environment Variables](#environment-variables)
  - [Build Options](#build-options)
  - [Compilation](#compilation)
- [Running](#running)
  - [Command-Line Options](#command-line-options)
  - [GPU Parallelization Methods](#gpu-parallelization-methods)
  - [Example Invocations](#example-invocations)
- [Potentials](#potentials)
- [Scaling Examples](#scaling-examples)
- [Output](#output)
- [Source Structure](#source-structure)

---

## Overview

CoMD-SYCL simulates the dynamics of atoms in a face-centered cubic (FCC) copper lattice using two classical interatomic potentials:

- **Lennard-Jones (LJ)** — simple pair potential, fast to evaluate.
- **Embedded Atom Model (EAM)** — more accurate many-body potential for metals; involves a three-phase GPU computation (density accumulation → embedding derivative → force update).

The simulation follows a standard MD loop:

1. Redistribute atoms across link cells.
2. Exchange halo (ghost) atoms with neighboring MPI ranks.
3. Compute interatomic forces on GPU.
4. Advance velocities and positions (leapfrog integration).
5. Accumulate global energy via oneCCL allreduce.

### What Changed from CoMD-CUDA

| Aspect | CoMD-CUDA (`src-mpi/`) | CoMD-SYCL (`src-occl/`) |
|---|---|---|
| GPU kernel language | CUDA (`.cu`) | SYCL / DPC++ (`.dp.cpp`) |
| Collective communication | MPI (GPU-unaware) | **oneCCL** with NCCL backend |
| Compiler | `nvcc` + `mpicc` | `clang++` (DPC++) + `mpicc` |
| Portability | NVIDIA only | NVIDIA (nvptx64); extensible to Intel GPUs |

Point-to-point communication (halo exchanges via `MPI_Sendrecv`) is kept as plain MPI because oneCCL does not expose a `sendrecv` primitive.

---

## Directory Structure

```
CoMD-SYCL/
├── bin/                    # Compiled binaries (created at build time)
├── pots/                   # EAM potential data files (Cu_u6.eam, Cu01.eam.alloy)
├── examples/               # Shell scripts and directories for scaling studies
│   ├── pots/               # Copies of potential files for examples
│   ├── strong-scaling/     # Output directory for strong-scaling runs
│   ├── weak-scaling/       # Output directory for weak-scaling runs
│   ├── mpi-strongScaling-sycl-occl-eam.sh
│   ├── mpi-weakScaling-sycl-occl-eam.sh
│   ├── mpi-strongScaling-cuda-mpi-eam.sh
│   └── mpi-weakScaling-cuda-mpi-eam.sh
├── src-occl/               # SYCL + oneCCL implementation  <-- primary source
├── src-mpi/                # Original CUDA + MPI implementation (reference)
├── src-openmp/             # CPU OpenMP reference implementation
└── src-serial/             # CPU serial reference implementation
```

---

## Dependencies

| Component | Purpose | Variable |
|---|---|---|
| DPC++ (Intel LLVM `clang++`) with CUDA/NVPTX support | Compile SYCL kernels for NVIDIA GPUs | `DPCPP_HOME` |
| oneCCL ≥ 2021.x (built with NCCL backend) | GPU-aware allreduce and broadcast | `ONECCL_INSTALL` |
| NCCL | Backend used by oneCCL for NVIDIA GPU collectives | `NCCL_ROOT` |
| CUDA Toolkit (≥ 12.x recommended) | `libcudart`, CUDA headers at link time | `CUDA_HOME` / `CUDA_PATH` |
| MPI (bundled with oneCCL) | Point-to-point halo exchange; bootstrap for oneCCL KVS | `MPI_HOME` |

---

## Building

### Environment Variables

Export the following before building (adapt paths to your system):

```bash
export DPCPP_HOME=$HOME/dpcpp-cuda          # DPC++ compiler root
export ONECCL_INSTALL=$HOME/oneccl-2021.17-dpcpp-install-cuda
export MPI_HOME=$ONECCL_INSTALL/opt/mpi     # MPI bundled with oneCCL
export NCCL_ROOT=$HOME/nccl/build
export CUDA_HOME=$CUDA_PATH                 # e.g. /usr/local/cuda
```

You can also `source` the oneCCL environment script if provided:

```bash
source $ONECCL_INSTALL/env/setvars.sh
```

### Build Options

Edit the top of [src-occl/Makefile](src-occl/Makefile) or pass variables on the command line:

| Variable | Default | Description |
|---|---|---|
| `DOUBLE_PRECISION` | `ON` | `ON` → `double`, `OFF` → `float` |
| `DO_MPI` | `ON` | `ON` → multi-rank (adds `-DDO_MPI`); binary named `CoMD-sycl-occl` |
| `MAXATOMS` | `64` | Maximum atoms per link cell |
| `DEBUG` | `OFF` | `ON` → disable optimisations, enable debug symbols |

### Compilation

```bash
cd src-occl
make                   # produces ../bin/CoMD-sycl-occl
make clean             # remove object files
make distclean         # remove objects + binary
```

To build a single-precision variant:

```bash
make DOUBLE_PRECISION=OFF
```

To build without MPI (single GPU):

```bash
make DO_MPI=OFF        # produces ../bin/CoMD-sycl
```

---

## Running

```bash
mpirun -np <N> ../bin/CoMD-sycl-occl [options]
```

Each MPI rank maps to one GPU. The number of ranks must equal `xproc × yproc × zproc`.

### Command-Line Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--help` | `-h` | — | Print usage and exit |
| `--potDir` | `-d` | `pots` | Directory containing potential files |
| `--potName` | `-p` | `Cu_u6.eam` / `Cu01.eam.alloy` | Potential file name |
| `--potType` | `-t` | `funcfl` | Potential format: `funcfl` or `setfl` |
| `--doeam` | `-e` | off | Use EAM potential (default: Lennard-Jones) |
| `--nx` | `-x` | `20` | Unit cells in X |
| `--ny` | `-y` | `20` | Unit cells in Y |
| `--nz` | `-z` | `20` | Unit cells in Z |
| `--xproc` | `-i` | `1` | MPI ranks along X |
| `--yproc` | `-j` | `1` | MPI ranks along Y |
| `--zproc` | `-k` | `1` | MPI ranks along Z |
| `--nSteps` | `-N` | `100` | Total number of timesteps |
| `--printRate` | `-n` | `10` | Steps between printed output |
| `--dt` | `-D` | `1.0` | Timestep in femtoseconds |
| `--lat` | `-l` | `-1.0` | Lattice constant in Å (negative = use potential default) |
| `--temp` | `-T` | `600` | Initial temperature in K |
| `--delta` | `-r` | `0` | Random displacement magnitude in Å |
| `--skinDistance` | `-S` | `0.1` | Neighbour-list skin as fraction of cutoff |
| `--method` | `-m` | `thread_atom` | GPU parallelisation method (see below) |
| `--gpuAsync` | `-a` | `0` | Overlap boundary/interior computation with MPI |
| `--gpuProfile` | `-s` | off | Profile mode: run one kernel then exit |
| `--hilbert` | `-H` | off | Hilbert-curve cell traversal order |
| `--ljInterpolation` | `-I` | off | Table interpolation for LJ potential |
| `--spline` | `-P` | off | Spline interpolation for EAM tables |
| `--usePairlist` | `-L` | off | Pairlist acceleration for CTA-cell LJ |

### GPU Parallelization Methods

Select with `--method` / `-m`:

| Method | Description |
|---|---|
| `thread_atom` | One GPU thread per atom; iterates over neighbour cells (default) |
| `warp_atom` | One warp (32 threads) collaborates on forces for a single atom |
| `warp_atom_nl` | Warp-per-atom using a pre-built neighbour list |
| `cta_cell` | One thread block (CTA) per link cell; uses shared memory |
| `thread_atom_nl` | Thread-per-atom with neighbour list |
| `cpu_nl` | CPU fallback with neighbour list (no GPU) |

### Example Invocations

```bash
# 32 000 atoms (20×20×20 unit cells), LJ, single GPU
mpirun -np 1 ../bin/CoMD-sycl-occl

# EAM potential, single GPU
mpirun -np 1 ../bin/CoMD-sycl-occl -e -d ../pots -p Cu_u6.eam -t funcfl

# 256 000 atoms (40×40×40), EAM, 8 GPUs (2×2×2)
mpirun -np 8 ../bin/CoMD-sycl-occl -e -i 2 -j 2 -k 2 -x 40 -y 40 -z 40

# EAM, warp-per-atom method, async overlap enabled
mpirun -np 1 ../bin/CoMD-sycl-occl -e -m warp_atom -a 1

# CTA-cell method with spline interpolation and pairlists
mpirun -np 1 ../bin/CoMD-sycl-occl -e -m cta_cell -P -L
```

---

## Potentials

Two interatomic potentials are supported:

### Lennard-Jones (LJ)
$$V(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]$$

Parameters are embedded in the code (copper defaults). Enable tabulated interpolation with `-I` for performance tuning.

### Embedded Atom Model (EAM)
A many-body potential for metals computed in three GPU phases:

1. **Phase 1** — accumulate pair energy φ(r) and electron density ρ(r) for each atom.
2. **Phase 2** — compute embedding energy F(ρ) and its derivative F′(ρ) (host-side table lookup).
3. **Phase 3** — apply force correction from F′(ρ) to all atom pairs.

EAM potential files (funcfl or setfl format) for copper are provided in `pots/`:

| File | Format | Description |
|---|---|---|
| `Cu_u6.eam` | funcfl | Single-element copper potential |
| `Cu01.eam.alloy` | setfl | Copper alloy potential |

---

## Scaling Examples

Ready-to-run scripts are in `examples/`:

```bash
# EAM strong scaling: fix 256 000 atoms, vary number of GPUs (1–16)
cd examples && bash mpi-strongScaling-sycl-occl-eam.sh

# EAM weak scaling: scale atoms proportionally with GPU count
cd examples && bash mpi-weakScaling-sycl-occl-eam.sh
```

Results are written to `examples/strong-scaling/` and `examples/weak-scaling/` as YAML files.

Equivalent scripts for the CUDA/MPI baseline are also provided (`*-cuda-mpi-*.sh`).

---

## Output

Each run produces:

- **Console output** every `--printRate` steps: step number, time (ps), temperature (K), potential energy (eV/atom), kinetic energy (eV/atom), total energy (eV/atom).
- **YAML file** (`CoMD-sycl-occl.<timestamp>.yaml`) with full run metadata: build info, command-line arguments, per-timer performance data (total time, number of calls, time per call).

Example console output:

```
#                                                                                         Performance
#  Loop   Time(fs)       Temp(K)   E_Pot(eV/atom)   E_Kin(eV/atom)   E_Tot(eV/atom)   atomRate(M-atom-steps/s)
      0       0.00    600.0     -3.71019        0.07765        -3.63254
     10      10.00    599.1     -3.71019        0.07754        -3.63265       123.4
    ...
```

---

## Source Structure

### `src-occl/` — SYCL + oneCCL Implementation

| File | Description |
|---|---|
| `CoMD.c.dp.cpp` | Main entry point: argument parsing, simulation init, time-step loop |
| `gpu_kernels.dp.cpp` | SYCL kernel dispatch for LJ and EAM force computations |
| `gpu_utility.c.dp.cpp` | Memory management, atom list building, GPU data transfers |
| `haloExchange.c.dp.cpp` | MPI halo exchange for ghost atoms and forces |
| `parallel.cpp` | **oneCCL wrappers**: allreduce (int, real, double), broadcast, barrier |
| `eam.c.dp.cpp` | EAM potential setup and host-side table management |
| `linkCells.c.dp.cpp` | Link-cell spatial decomposition |
| `timestep.c.dp.cpp` | Leapfrog velocity/position integration on GPU |
| `gpu_neighborList.c.dp.cpp` | GPU neighbour-list construction |
| `CoMDTypes.h` | Core data structures: `SimFlat`, `Domain`, `Atoms`, `LinkCell`, potentials |
| `gpu_types.h` | GPU-side structures: `SimGpu`, `AtomsGpu`, `NeighborListGpu` |
| `gpu_common.h` | Shared GPU utilities: interpolation, table lookup, SYCL helpers |
| `gpu_eam_thread_atom.h` | EAM kernel — one thread per atom |
| `gpu_eam_warp_atom.h` | EAM kernel — one warp per atom |
| `gpu_eam_cta_cell.h` | EAM kernel — one thread block per cell |
| `gpu_lj_thread_atom.h` | LJ kernel — one thread per atom |
| `gpu_lj_cta_cell.h` | LJ kernel — one thread block per cell, shared-memory optimised |
| `gpu_redistribute.h` | Atom redistribution among link cells after position update |
| `gpu_timestep.h` | GPU leapfrog integration kernels |
| `gpu_reduce.h` | GPU reduction (sum, max) |
| `gpu_scan.h` | GPU prefix-sum (scan) |
| `defines.h` | Compile-time constants: method enum, block/warp sizes, interpolation params |
| `mycommand.c` / `mycommand.h` | Command-line argument parsing |

### `src-mpi/` — CUDA + MPI Reference

Original CUDA implementation. Useful as a performance and correctness baseline. Build with `nvcc` + `mpicc` using the provided `Makefile.EAM` / `Makefile.NAMD`.

---

## Credits

- **Original CoMD**: ExMatEx — Exascale Co-Design Center for Materials in Extreme Environments. [https://github.com/exmatex/CoMD](https://github.com/exmatex/CoMD)
- **CoMD-CUDA**: GPU acceleration with CUDA and MPI.
- **CoMD-SYCL**: Port to SYCL (DPC++) and oneCCL.
