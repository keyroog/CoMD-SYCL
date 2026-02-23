/// \file
/// Wrappers for distributed communication (OneCCL collectives + MPI p2p).

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

/// OneCCL allreduce integer sum.
void addIntParallel(int* sendBuf, int* recvBuf, int count);

/// OneCCL allreduce real sum.
void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count);

/// OneCCL allreduce double sum.
void addDoubleParallel(double* sendBuf, double* recvBuf, int count);

/// OneCCL allreduce integer max.
void maxIntParallel(int* sendBuf, int* recvBuf, int count);

/// MPI_Allreduce double min with rank (MINLOC not in OneCCL).
void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// MPI_Allreduce double max with rank (MAXLOC not in OneCCL).
void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// Wrapper for MPI_Bcast.
void bcastParallel(void* buf, int len, int root);

///  Return non-zero if code was built with MPI active.
int builtWithMpi(void);

#ifdef __cplusplus
}
#endif

#endif
