#!/bin/sh

# Simple strong scaling study with EAM potential and 256,000 atoms
mpirun -np 1  ../bin/CoMD-cuda-mpi-eam -e -i 1 -j 1 -k 1 -x 40 -y 40 -z 40
mpirun -np 2  ../bin/CoMD-cuda-mpi-eam -e -i 2 -j 1 -k 1 -x 40 -y 40 -z 40
mpirun -np 4  ../bin/CoMD-cuda-mpi-eam -e -i 2 -j 2 -k 1 -x 40 -y 40 -z 40
mpirun -np 8  ../bin/CoMD-cuda-mpi-eam -e -i 2 -j 2 -k 2 -x 40 -y 40 -z 40
mpirun -np 16 ../bin/CoMD-cuda-mpi-eam -e -i 4 -j 2 -k 2 -x 40 -y 40 -z 40
