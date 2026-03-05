#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_HIGRAPH_0
#SBATCH --output=/leonardo/home/userexternal/ssirica0/CoMD-SYCL/logs/%x_%j.out
#SBATCH --error=/leonardo/home/userexternal/ssirica0/CoMD-SYCL/logs/%x_%j.err
#SBATCH --exclusive

module purge
source /leonardo/home/userexternal/ssirica0/scripts/cuda_export_variables.sh
source /leonardo/home/userexternal/ssirica0/scripts/oneccl_export_variables_multinode.sh

mkdir -p /leonardo/home/userexternal/ssirica0/CoMD-SYCL/logs

MPI=/leonardo/home/userexternal/ssirica0/CoMD-SYCL/bin/CoMD-cuda-mpi
NCCL=/leonardo/home/userexternal/ssirica0/CoMD-SYCL/bin/CoMD-cuda-nccl-mpi
OCCL=/leonardo/home/userexternal/ssirica0/CoMD-SYCL/bin/CoMD-sycl-occl

echo "=== np=16 | Strong Scaling (256k atoms) ==="

echo "--- [mpi] strong ---"
srun --ntasks=16 --ntasks-per-node=4 $MPI  -e -i 4 -j 2 -k 2 -x 40 -y 40 -z 40

echo "--- [nccl] strong ---"
srun --ntasks=16 --ntasks-per-node=4 $NCCL -e -i 4 -j 2 -k 2 -x 40 -y 40 -z 40

echo "--- [occl] strong ---"
srun --ntasks=16 --ntasks-per-node=4 $OCCL -e -i 4 -j 2 -k 2 -x 40 -y 40 -z 40

echo "=== np=16 | Weak Scaling (32k atoms/rank) ==="

echo "--- [mpi] weak ---"
srun --ntasks=16 --ntasks-per-node=4 $MPI  -e -i 4 -j 2 -k 2 -x 80 -y 40 -z 40

echo "--- [nccl] weak ---"
srun --ntasks=16 --ntasks-per-node=4 $NCCL -e -i 4 -j 2 -k 2 -x 80 -y 40 -z 40

echo "--- [occl] weak ---"
srun --ntasks=16 --ntasks-per-node=4 $OCCL -e -i 4 -j 2 -k 2 -x 80 -y 40 -z 40
