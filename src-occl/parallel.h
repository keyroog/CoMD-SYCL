/// \file
/// Wrappers for distributed communication.
/// Categoria B (host data): plain MPI, no device involved.
/// Categoria A (device data): OneCCL directly on device pointers, no memcpy.

#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "mytype.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Structure for use with MPI_MINLOC and MPI_MAXLOC operations.
typedef struct RankReduceDataSt
{
   double val;
   int rank;
} RankReduceData;

/// Return total number of processors.
int getNRanks(void);

/// Return local rank.
int getMyRank(void);

/// Return non-zero if printing occurs from this rank.
int printRank(void);

/// Print a timestamp and message when all tasks arrive.
void timestampBarrier(const char* msg);

/// Wrapper for MPI_Init + ccl::init().
void initParallel(int *argc, char ***argv);

/// Initialise OneCCL communicator and stream.
/// Must be called AFTER the SYCL device has been selected (after SetupGpu).
void initCCL(void);

/// Wrapper for MPI_Finalize + OneCCL teardown.
void destroyParallel(void);

/// Wrapper for MPI_Barrier(MPI_COMM_WORLD).
void barrierParallel(void);

/// Wrapper for MPI_Sendrecv (point-to-point, kept as MPI).
int sendReceiveParallel(void* sendBuf, int sendLen, int dest,
                        void* recvBuf, int recvLen, int source);

// --- Categoria B: host-resident data, plain MPI ---------------------------

/// MPI_Allreduce integer sum on host pointers.
void addIntParallel(int* sendBuf, int* recvBuf, int count);

/// MPI_Allreduce real sum on host pointers.
void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count);

/// MPI_Allreduce double sum on host pointers.
void addDoubleParallel(double* sendBuf, double* recvBuf, int count);

/// MPI_Allreduce integer max on host pointers.
void maxIntParallel(int* sendBuf, int* recvBuf, int count);

/// MPI_Allreduce MINLOC on host pointers (MPI_DOUBLE_INT matches RankReduceData).
void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// MPI_Allreduce MAXLOC on host pointers (MPI_DOUBLE_INT matches RankReduceData).
void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// MPI_Bcast on host pointer.
void bcastParallel(void* buf, int len, int root);

// --- Categoria A: device-resident data, OneCCL on device pointers ----------

/// OneCCL allreduce integer sum directly on device pointers (no memcpy).
void addIntParallelDevice(int* d_sendBuf, int* d_recvBuf, int count);

/// OneCCL allreduce real sum directly on device pointers (no memcpy).
void addRealParallelDevice(real_t* d_sendBuf, real_t* d_recvBuf, int count);

///  Return non-zero if code was built with MPI active.
int builtWithMpi(void);

#ifdef __cplusplus
}
#endif

#endif
