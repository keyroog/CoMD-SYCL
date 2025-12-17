/// \file
/// Wrappers for MPI functions with oneCCL integration.

#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "mytype.h"

/// Structure for use with MPI_MINLOC and MPI_MAXLOC operations.
typedef struct RankReduceDataSt
{
   double val;
   int rank;
} RankReduceData;

#ifdef __cplusplus
extern "C" {
#endif

/// Return total number of processors.
int getNRanks(void);

/// Return local rank.
int getMyRank(void);

/// Return non-zero if printing occurs from this rank.
int printRank(void);

/// Print a timestamp and message when all tasks arrive.
void timestampBarrier(const char* msg);

/// Wrapper for MPI_Init (also initializes oneCCL if enabled).
void initParallel(int *argc, char ***argv);

/// Wrapper for MPI_Finalize (also destroys oneCCL if enabled).
void destroyParallel(void);

/// Wrapper for MPI_Barrier(MPI_COMM_WORLD).
void barrierParallel(void);

/// Wrapper for MPI_Sendrecv.
int sendReceiveParallel(void* sendBuf, int sendLen, int dest,
                        void* recvBuf, int recvLen, int source);

/// Wrapper for MPI_Allreduce integer sum (uses oneCCL if enabled).
void addIntParallel(int* sendBuf, int* recvBuf, int count);

/// Wrapper for MPI_Allreduce real sum (uses oneCCL if enabled).
void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count);

/// Wrapper for MPI_Allreduce double sum (uses oneCCL if enabled).
void addDoubleParallel(double* sendBuf, double* recvBuf, int count);

/// Wrapper for MPI_Allreduce integer max (uses oneCCL if enabled).
void maxIntParallel(int* sendBuf, int* recvBuf, int count);

/// Wrapper for MPI_Allreduce double min with rank.
void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// Wrapper for MPI_Allreduce double max with rank.
void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count);

/// Wrapper for MPI_Bcast
void bcastParallel(void* buf, int len, int root);

///  Return non-zero if code was built with MPI active.
int builtWithMpi(void);

///  Return non-zero if code was built with oneCCL active.
int builtWithOneCCL(void);

#ifdef __cplusplus
}
#endif

#endif
