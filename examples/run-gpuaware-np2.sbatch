#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_HIGRAPH_0
#SBATCH --output=/leonardo/home/userexternal/ssirica0/CoMD-SYCL/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/ssirica0/CoMD-SYCL/logs/%x_%j.err
#SBATCH --exclusive

module purge
source /leonardo/home/userexternal/ssirica0/scripts/cuda_export_variables.sh

mkdir -p /leonardo/home/userexternal/ssirica0/CoMD-SYCL/logs

GPUAWARE=/leonardo/home/userexternal/ssirica0/CoMD-SYCL/bin/CoMD-cuda-mpi-gpuaware

export OMPI_MCA_opal_cuda_support=1

echo "=== np=2 | Strong Scaling (256k atoms) ==="
echo "--- [mpi-gpuaware] strong ---"
srun --ntasks=2 $GPUAWARE -e -i 2 -j 1 -k 1 -x 40 -y 40 -z 40

echo "=== np=2 | Weak Scaling (32k atoms/rank) ==="
echo "--- [mpi-gpuaware] weak ---"
srun --ntasks=2 $GPUAWARE -e -i 2 -j 1 -k 1 -x 40 -y 20 -z 20
