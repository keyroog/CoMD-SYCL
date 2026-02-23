#ifndef CoMD_info_hpp
#define CoMD_info_hpp

#define CoMD_VARIANT "CoMD-sycl-occl"
#define CoMD_HOSTNAME "gnode09"
#define CoMD_KERNEL_NAME "'Linux'"
#define CoMD_KERNEL_RELEASE "'6.8.0-51-generic'"
#define CoMD_PROCESSOR "'x86_64'"

#define CoMD_COMPILER "'/home/S.SIRICA3/dpcpp-cuda/bin/clang++'"
#define CoMD_COMPILER_VERSION "'Intel SYCL compiler development build based on:'"
#define CoMD_CFLAGS "'-fsycl -fsycl-targets=nvptx64-nvidia-cuda -g -O2 -DMAXATOMS=64 -DCOMD_DOUBLE -DDO_MPI -DNDEBUG -w'"
#define CoMD_LDFLAGS "'-fsycl -fsycl-targets=nvptx64-nvidia-cuda -L/home/S.SIRICA3/oneccl-2021.17-dpcpp-install-cuda/opt/mpi/lib -lmpi -L/home/S.SIRICA3/oneccl-2021.17-dpcpp-install-cuda/lib -lccl -lm -lstdc++'"

#endif
