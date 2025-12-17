/// \file
/// Wrappers for MPI functions with oneCCL integration for collective operations.
/// This version uses oneCCL for Allreduce operations where sensible.

#include "parallel.h"

#ifdef DO_MPI
#include <mpi.h>
#endif

#ifdef USE_ONECCL
#include <oneapi/ccl.hpp>
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

#ifdef USE_ONECCL
// oneCCL communicator and related objects
static ccl::communicator* ccl_comm = nullptr;
static ccl::shared_ptr_class<ccl::kvs> kvs;
static bool ccl_initialized = false;

// Initialize oneCCL
static void initOneCCL() {
    if (ccl_initialized) return;
    
    ccl::init();
    
    // Get KVS address from rank 0 and broadcast to all ranks
    ccl::kvs::address_type main_addr;
    if (myRank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
    }
    
#ifdef DO_MPI
    MPI_Bcast(main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    
    if (myRank != 0) {
        kvs = ccl::create_kvs(main_addr);
    }
    
    ccl_comm = new ccl::communicator(ccl::create_communicator(nRanks, myRank, kvs));
    ccl_initialized = true;
}

static void destroyOneCCL() {
    if (ccl_comm) {
        delete ccl_comm;
        ccl_comm = nullptr;
    }
    ccl_initialized = false;
}
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

#ifdef USE_ONECCL
   initOneCCL();
#endif
}

void destroyParallel()
{
#ifdef USE_ONECCL
   destroyOneCCL();
#endif

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

void addIntParallel(int* sendBuf, int* recvBuf, int count)
{
#ifdef USE_ONECCL
   // Use oneCCL for Allreduce
   ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::sum, *ccl_comm).wait();
#elif defined(DO_MPI)
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count)
{
#ifdef USE_ONECCL
   // Use oneCCL for Allreduce
   ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::sum, *ccl_comm).wait();
#elif defined(DO_MPI)
   MPI_Allreduce(sendBuf, recvBuf, count, REAL_MPI_TYPE, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void addDoubleParallel(double* sendBuf, double* recvBuf, int count)
{
#ifdef USE_ONECCL
   // Use oneCCL for Allreduce
   ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::sum, *ccl_comm).wait();
#elif defined(DO_MPI)
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void maxIntParallel(int* sendBuf, int* recvBuf, int count)
{
#ifdef USE_ONECCL
   // Use oneCCL for Allreduce with max
   ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::max, *ccl_comm).wait();
#elif defined(DO_MPI)
   MPI_Allreduce(sendBuf, recvBuf, count, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#else
   for (int ii=0; ii<count; ++ii)
      recvBuf[ii] = sendBuf[ii];
#endif
}

void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count)
{
   // oneCCL doesn't support MINLOC directly, fall back to MPI
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
   // oneCCL doesn't support MAXLOC directly, fall back to MPI
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

int builtWithOneCCL(void)
{
#ifdef USE_ONECCL
   return 1;
#else
   return 0;
#endif
}
