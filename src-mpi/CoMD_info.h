#ifndef CoMD_info_hpp
#define CoMD_info_hpp

#define CoMD_VARIANT "CoMD-cuda-mpi-eam"
#define CoMD_HOSTNAME "lnode02"
#define CoMD_KERNEL_NAME "'Linux'"
#define CoMD_KERNEL_RELEASE "'6.8.0-51-generic'"
#define CoMD_PROCESSOR "'x86_64'"

#define CoMD_COMPILER "'/home/S.SIRICA3/oneccl-2021.17-dpcpp-install-cuda/opt/mpi/bin/mpicc'"
#define CoMD_COMPILER_VERSION "'gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0'"
#define CoMD_CFLAGS "'-std=c99 -Wno-unused-result -DMAXATOMS=64  -DNDEBUG -g -O5 -DCOMD_DOUBLE -DDO_MPI  -I/cm/shared/apps/cuda12.8/toolkit/12.8.0/include'"
#define CoMD_LDFLAGS "' -lm -lstdc++ -L/cm/shared/apps/cuda12.8/toolkit/12.8.0/lib64 -lcudart'"

#endif
