#ifndef CoMD_info_hpp
#define CoMD_info_hpp

#define CoMD_VARIANT "CoMD-sycl-occl"
#define CoMD_HOSTNAME "login02.leonardo.local"
#define CoMD_KERNEL_NAME "'Linux'"
#define CoMD_KERNEL_RELEASE "'4.18.0-477.27.1.el8_8.x86_64'"
#define CoMD_PROCESSOR "'x86_64'"

#define CoMD_COMPILER "'/leonardo/home/userexternal/ssirica0/llvm/bin/clang++'"
#define CoMD_COMPILER_VERSION "'Intel SYCL compiler 6.3.0 release build based on:'"
#define CoMD_CFLAGS "'-fsycl -fsycl-targets=nvptx64-nvidia-cuda -g -O2 -DMAXATOMS=64 -DCOMD_DOUBLE -DDO_MPI -DNDEBUG -w'"
#define CoMD_LDFLAGS "'-fsycl -fsycl-targets=nvptx64-nvidia-cuda -L/leonardo/home/userexternal/ssirica0/oneCCL-cuda/opt/mpi/lib -lmpi -L/leonardo/home/userexternal/ssirica0/oneCCL-cuda/lib -lccl -lm -lstdc++'"

#endif
