/// \file
/// Wrappers for distributed communication.
/// Categoria B (host data): plain MPI, no device involved.
/// Categoria A (device data): NCCL directly on device pointers, no memcpy.

#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "mytype.h"

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

/// Wrapper for MPI_Init.
void initParallel(int *argc, char ***argv);

/// Initialise NCCL communicator and stream.
/// Must be called AFTER the CUDA device has been selected (after SetupGpu).
void initNCCL(void);

/// Wrapper for MPI_Finalize + NCCL teardown.
void destroyParallel(void);

/// Wrapper for MPI_Barrier(MPI_COMM_WORLD).
void barrierParallel(void);

/// Wrapper for MPI_Sendrecv.
int sendReceiveParallel(void* sendBuf, int sendLen, int dest,
                        void* recvBuf, int recvLen, int source);

/// Wrapper for MPI_Allreduce integer sum.
void addIntParallel(int* sendBuf, int* recvBuf, int count);

/// Wrapper for MPI_Allreduce real sum.
void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count);

/// Wrapper for MPI_Allreduce double sum.
void addDoubleParallel(double* sendBuf, double* recvBuf, int count);

/// Wrapper for MPI_Allreduce integer max.
void maxIntParallel(int* sendBuf, int* recvBuf, int count);

/// Wrapper for MPI_Allreduce double min with rank.
void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// Wrapper for MPI_Allreduce double max with rank.
void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// Wrapper for MPI_Bcast
void bcastParallel(void* buf, int len, int root);

///  Return non-zero if code was built with MPI active.
int builtWithMpi(void);

// --- Categoria A: device-resident data, NCCL on device pointers ----------

/// NCCL allreduce integer sum directly on device pointers (no memcpy).
void addIntParallelDevice(int* d_sendBuf, int* d_recvBuf, int count);

/// NCCL allreduce real sum directly on device pointers (no memcpy).
void addRealParallelDevice(real_t* d_sendBuf, real_t* d_recvBuf, int count);

#endif

