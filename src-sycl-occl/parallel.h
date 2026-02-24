/// \file
/// Wrappers for distributed communication via OneCCL.

#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "mytype.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Structure for rank-aware min/max reductions.
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

/// Initialize distributed runtime metadata and OneCCL library.
void initParallel(int *argc, char ***argv);

/// Initialize OneCCL communicator and stream.
/// Must be called AFTER the SYCL device has been selected (after SetupGpu).
void initCCL(void);

/// Finalize OneCCL communicator state.
void destroyParallel(void);

/// OneCCL barrier (no-op before initCCL for single-rank startup).
void barrierParallel(void);

/// Point-to-point send/receive implemented with OneCCL send/recv.
int sendReceiveParallel(void* sendBuf, int sendLen, int dest,
                        void* recvBuf, int recvLen, int source);

/// OneCCL allreduce integer sum.
void addIntParallel(int* sendBuf, int* recvBuf, int count);

/// OneCCL allreduce real sum.
void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count);

/// OneCCL allreduce double sum.
void addDoubleParallel(double* sendBuf, double* recvBuf, int count);

/// OneCCL allreduce integer max.
void maxIntParallel(int* sendBuf, int* recvBuf, int count);

/// Rank-aware min reduction (OneCCL allgather + local reduce).
void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// Rank-aware max reduction (OneCCL allgather + local reduce).
void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// OneCCL broadcast.
void bcastParallel(void* buf, int len, int root);

/// Return non-zero if code was built with MPI active.
int builtWithMpi(void);

#ifdef __cplusplus
}
#endif

#endif
