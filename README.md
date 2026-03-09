# CoMD-SYCL

CoMD-SYCL is a research-oriented fork of CoMD-CUDA for distributed molecular dynamics on NVIDIA GPUs. The repository preserves the original CPU and CUDA baselines and adds communication-focused variants to compare standard MPI, CUDA-aware MPI, NCCL, and SYCL + oneCCL within the same CoMD code base.

This project started from a CoMD-CUDA tree that already contained `src-serial`, `src-openmp`, and `src-mpi`. On top of that baseline, this repository adds:

- `src-mpi-gpuaware`: CUDA version with CUDA-aware MPI for device-resident collectives.
- `src-nccl`: CUDA version where device-resident collectives use NCCL.
- `src-occl`: SYCL port produced from the CUDA code with Intel SYCLomatic (`dpct`), then adapted manually and integrated with Intel oneCCL for device-resident collectives.

## Variant Summary

The main purpose of the repository is to keep the computational workload as similar as possible while changing the communication backend used in distributed runs.

| Directory | Kernel backend | Collective path | Point-to-point path | Output binary |
| --- | --- | --- | --- | --- |
| `src-mpi` | CUDA | Standard MPI on host-side buffers | MPI | `bin/CoMD-cuda-mpi` |
| `src-mpi-gpuaware` | CUDA | CUDA-aware MPI directly on device buffers | MPI | `bin/CoMD-cuda-mpi-gpuaware` |
| `src-nccl` | CUDA | NCCL for device-resident collectives, MPI for host-side control collectives | MPI | `bin/CoMD-cuda-nccl-mpi` |
| `src-occl` | SYCL/DPC++ | oneCCL for device-resident collectives, MPI for host-side control collectives | MPI | `bin/CoMD-sycl-occl` |

Notes:

- Halo exchange remains MPI-based across all distributed variants.
- In `src-nccl` and `src-occl`, only the collective communication path is replaced; MPI is still used where a point-to-point `sendrecv` style exchange is needed.
- `src-serial` and `src-openmp` are inherited CPU reference implementations kept for completeness.

## Repository Layout

- `src-mpi/`: original CUDA + MPI baseline.
- `src-mpi-gpuaware/`: CUDA + CUDA-aware MPI variant.
- `src-nccl/`: CUDA + NCCL collective variant.
- `src-occl/`: SYCL + oneCCL variant, including SYCLomatic compatibility headers under `src-occl/include/dpct/`.
- `src-openmp/`, `src-serial/`: inherited CPU reference versions.
- `pots/`: potential files used by CoMD.
- `examples/`: legacy example scripts and input assets inherited from the original code base.
- `strong-scaling-results/`, `weak-scaling-results/`: collected YAML outputs and generated plots from scaling experiments.

## Requirements

### CUDA-based variants

The CUDA implementations (`src-mpi`, `src-mpi-gpuaware`, `src-nccl`) require:

- A CUDA toolkit visible through `CUDA_HOME` or `CUDA_PATH`.
- An MPI implementation available through `mpicc`.
- For `src-mpi-gpuaware`, an MPI stack with CUDA-aware support enabled.
- For `src-nccl`, an NCCL installation available through `NCCL_HOME`.

### SYCL + oneCCL variant

The `src-occl` implementation requires:

- A DPC++ compiler with NVIDIA `nvptx64-nvidia-cuda` target support, exposed through `DPCPP_HOME`.
- An Intel oneCCL installation, exposed through `ONECCL_INSTALL`.
- The MPI bundled with oneCCL, typically exposed through `MPI_HOME`.

The variable names above match the expectations in `src-occl/Makefile`.

## Building

Each variant is built independently from its own source directory.

```bash
cd src-mpi
make

cd ../src-mpi-gpuaware
make

cd ../src-nccl
make

cd ../src-occl
make
```

The commands above produce:

- `bin/CoMD-cuda-mpi`
- `bin/CoMD-cuda-mpi-gpuaware`
- `bin/CoMD-cuda-nccl-mpi`
- `bin/CoMD-sycl-occl`

For the SYCL + oneCCL build, a typical environment setup looks like:

```bash
export DPCPP_HOME=$HOME/dpcpp-cuda
export ONECCL_INSTALL=$HOME/oneccl-install
export MPI_HOME=$ONECCL_INSTALL/opt/mpi
export NCCL_ROOT=$HOME/nccl/build
export CUDA_HOME=${CUDA_PATH:-/usr/local/cuda}

cd src-occl
make
```

The Makefiles also expose common knobs such as `DOUBLE_PRECISION`, `DO_MPI`, `MAXATOMS`, and `DEBUG`.

## Running

All distributed variants follow the same CoMD command-line structure. A typical EAM run on 8 ranks / 8 GPUs is:

```bash
mpirun -np 8 ./bin/CoMD-cuda-mpi-gpuaware -e -i 2 -j 2 -k 2 -x 40 -y 40 -z 40
mpirun -np 8 ./bin/CoMD-cuda-nccl-mpi     -e -i 2 -j 2 -k 2 -x 40 -y 40 -z 40
mpirun -np 8 ./bin/CoMD-sycl-occl         -e -i 2 -j 2 -k 2 -x 40 -y 40 -z 40
```

In practice, the benchmark scripts assume one MPI rank per GPU. The process-grid arguments `-i`, `-j`, and `-k` must multiply to the total number of ranks.

Useful flags:

- `-e` enables the EAM potential.
- `-x`, `-y`, `-z` set the problem size in unit cells.
- `-i`, `-j`, `-k` set the rank decomposition.
- `-h` prints the full command-line help for a given binary.

## Provenance

This repository builds on two layers of prior work:

- CoMD, the molecular dynamics mini-application from ExMatEx: https://github.com/exmatex/CoMD
- A CoMD-CUDA code base used here as the starting point for the distributed GPU implementations

The `src-occl` tree was generated from the CUDA version with Intel SYCLomatic (`dpct`) and then refined manually to make the SYCL/DPC++ and oneCCL version usable for the communication experiments in this repository.
