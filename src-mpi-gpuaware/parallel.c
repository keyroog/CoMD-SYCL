/// \file
/// Wrappers for distributed communication functions.
///
/// Collective operations are split into two categories:
///
/// Categoria B (host-resident data: timers, atom counts, init):
///   addIntParallel, addRealParallel, addDoubleParallel, maxIntParallel,
///   minRankDoubleParallel, maxRankDoubleParallel, bcastParallel
///   -> plain MPI_Allreduce / MPI_Bcast on host pointers, no device involved.
///
/// Categoria A (device-resident data: energies, update flags):
///   addIntParallelDevice, addRealParallelDevice
///   -> MPI_Allreduce called directly on device pointers (CUDA-aware MPI,
///      GPUDirect). No explicit memcpy needed.
///
/// barrierParallel uses MPI_Barrier.
/// sendReceiveParallel uses MPI_Sendrecv (point-to-point, kept as MPI).

#include "parallel.h"

#ifdef DO_MPI
#include <mpi.h>
#include <cuda_runtime.h>
#endif

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>

static int myRank = 0;
static int nRanks = 1;

#ifdef DO_MPI
#ifdef COMD_SINGLE
#define REAL_MPI_TYPE MPI_FLOAT
#else
#define REAL_MPI_TYPE MPI_DOUBLE
#endif
#endif

int getNRanks()
{
   return nRanks;
}

int getMyRank()
{
   return myRank;
}

/// \details
/// For now this is just a check for rank 0 but in principle it could be
/// more complex.  It is also possible to suppress practically all
/// output by causing this function to return 0 for all ranks.
int printRank()
{
   if (myRank == 0) return 1;
   return 0;
}

void timestampBarrier(const char* msg)
{
   barrierParallel();
   if (! printRank())
      return;
   time_t t= time(NULL);
   char* timeString = ctime(&t);
   timeString[24] = '\0'; // clobber newline
   fprintf(screenOut, "%s: %s\n", timeString, msg);
   fflush(screenOut);
}

void initParallel(int* argc, char*** argv)
{
#ifdef DO_MPI
   MPI_Init(argc, argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#endif
}

void destroyParallel()
{
#ifdef DO_MPI
   MPI_Finalize();
#endif
}

void barrierParallel()
{
#ifdef DO_MPI
   MPI_Barrier(MPI_COMM_WORLD);
#endif
}

/// \param [in]  sendBuf Data to send.
/// \param [in]  sendLen Number of bytes to send.
/// \param [in]  dest    Rank in MPI_COMM_WORLD where data will be sent.
/// \param [out] recvBuf Received data.
/// \param [in]  recvLen Maximum number of bytes to receive.
/// \param [in]  source  Rank in MPI_COMM_WORLD from which to receive.
/// \return Number of bytes received.
int sendReceiveParallel(void* sendBuf, int sendLen, int dest,
                        void* recvBuf, int recvLen, int source)
{
#ifdef DO_MPI
   int bytesReceived;
   MPI_Status status;
   MPI_Sendrecv(sendBuf, sendLen, MPI_BYTE, dest,   0,
                recvBuf, recvLen, MPI_BYTE, source, 0,
                MPI_COMM_WORLD, &status);
   MPI_Get_count(&status, MPI_BYTE, &bytesReceived);

   return bytesReceived;
#else
   assert(source == dest);
   memcpy(recvBuf, sendBuf, sendLen);

   return sendLen;
#endif
}

// ---------------------------------------------------------------------------
// Categoria B — host-resident data: plain MPI, no device involvement
// ---------------------------------------------------------------------------

void addIntParallel(int* sendBuf, int* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, REAL_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void addDoubleParallel(double* sendBuf, double* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void maxIntParallel(int* sendBuf, int* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
   {
      recvBuf[ii].val = sendBuf[ii].val;
      recvBuf[ii].rank = sendBuf[ii].rank;
   }
#endif
}

void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
   {
      recvBuf[ii].val = sendBuf[ii].val;
      recvBuf[ii].rank = sendBuf[ii].rank;
   }
#endif
}

/// \param [in] count Length of buf in bytes.
void bcastParallel(void* buf, int count, int root)
{
#ifdef DO_MPI
   MPI_Bcast(buf, count, MPI_BYTE, root, MPI_COMM_WORLD);
#endif
}

int builtWithMpi(void)
{
#ifdef DO_MPI
   return 1;
#else
   return 0;
#endif
}

// ---------------------------------------------------------------------------
// Categoria A — device-resident data: CUDA-aware MPI on device pointers
// ---------------------------------------------------------------------------

/// Integer allreduce sum directly on device pointers via CUDA-aware MPI
/// (GPUDirect). No explicit D2H/H2D memcpy needed.
void addIntParallelDevice(int* d_sendBuf, int* d_recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(d_sendBuf, d_recvBuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
   cudaMemcpy(d_recvBuf, d_sendBuf, count * sizeof(int), cudaMemcpyDeviceToDevice);
#endif
}

/// Real allreduce sum directly on device pointers via CUDA-aware MPI
/// (GPUDirect). No explicit D2H/H2D memcpy needed.
void addRealParallelDevice(real_t* d_sendBuf, real_t* d_recvBuf, int count)
{
#ifdef DO_MPI
   MPI_Allreduce(d_sendBuf, d_recvBuf, count, REAL_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
#else
   cudaMemcpy(d_recvBuf, d_sendBuf, count * sizeof(real_t), cudaMemcpyDeviceToDevice);
#endif
}
