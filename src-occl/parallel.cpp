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
///   -> ccl::allreduce called directly on device pointers, no memcpy.
///
/// barrierParallel uses OneCCL barrier (device stream, no data).
/// sendReceiveParallel uses MPI_Sendrecv (point-to-point, kept as MPI).

#include "parallel.h"

#ifdef DO_MPI
#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>
#endif

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <memory>
#include <stdexcept>
#include <utility>

static int myRank = 0;
static int nRanks = 1;

#ifdef DO_MPI

// --- OneCCL global state (initialised by initCCL) --------------------------
static sycl::queue*            g_ccl_queue  = nullptr;
static ccl::communicator*      g_ccl_comm   = nullptr;
static ccl::stream*            g_ccl_stream = nullptr;
// We keep the objects alive in static storage so the pointers stay valid.
static ccl::communicator*      s_comm_heap   = nullptr;
static ccl::stream*            s_stream_heap = nullptr;
static std::unique_ptr<sycl::queue> s_ccl_queue;
static ccl::shared_ptr_class<ccl::kvs> s_kvs;

#ifdef COMD_SINGLE
#define REAL_CCL_TYPE ccl::datatype::float32
#define REAL_MPI_TYPE MPI_FLOAT
#else
#define REAL_CCL_TYPE ccl::datatype::float64
#define REAL_MPI_TYPE MPI_DOUBLE
#endif

static sycl::device get_device_for_rank(int rank)
{
   auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
   if (gpus.empty())
      throw std::runtime_error("No SYCL GPU devices available for OneCCL.");

   int idx = rank % static_cast<int>(gpus.size());
   if (idx < 0) idx += static_cast<int>(gpus.size());
   return gpus[idx];
}

#endif // DO_MPI

int getNRanks()
{
   return nRanks;
}

int getMyRank()
{
   return myRank;
}

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
   timeString[24] = '\0';
   fprintf(screenOut, "%s: %s\n", timeString, msg);
   fflush(screenOut);
}

void initParallel(int* argc, char*** argv)
{
#ifdef DO_MPI
   int provided = MPI_THREAD_SINGLE;
   MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

   if (provided < MPI_THREAD_MULTIPLE && myRank == 0) {
      fprintf(stderr,
              "Warning: MPI thread support level is %d, expected %d "
              "(MPI_THREAD_MULTIPLE) for OneCCL.\n",
              provided, MPI_THREAD_MULTIPLE);
   }

   // Initialise OneCCL library (must be called once before any ccl call)
   ccl::init();
#endif
}

/// Call this AFTER MPI init to create OneCCL communicator and stream.
void initCCL()
{
#ifdef DO_MPI
   int rank = myRank;
   int size = nRanks;

   sycl::device dev = get_device_for_rank(rank);
   sycl::context ctx(dev);
   sycl::queue q(ctx, dev, sycl::property::queue::in_order());

   // --- KVS bootstrap via MPI -----------------------------------------------
   ccl::shared_ptr_class<ccl::kvs> kvs;
   ccl::kvs::address_type addr;
   if (rank == 0) {
      kvs  = ccl::create_main_kvs();
      addr = kvs->get_address();
      MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
   } else {
      MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      kvs = ccl::create_kvs(addr);
   }
   s_kvs = kvs;

   // --- Create OneCCL communicator + stream ---------------------------------
   auto ccl_dev = ccl::create_device(q.get_device());
   auto ccl_ctx = ccl::create_context(q.get_context());
   auto comm = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
   auto ccl_stream = ccl::create_stream(q);

   s_ccl_queue = std::make_unique<sycl::queue>(std::move(q));
   g_ccl_queue = s_ccl_queue.get();

   s_comm_heap   = new ccl::communicator(std::move(comm));
   s_stream_heap = new ccl::stream(std::move(ccl_stream));

   g_ccl_comm   = s_comm_heap;
   g_ccl_stream = s_stream_heap;
#endif
}

void destroyParallel()
{
#ifdef DO_MPI
   // Tear down OneCCL state
   delete s_stream_heap; s_stream_heap = nullptr; g_ccl_stream = nullptr;
   delete s_comm_heap;   s_comm_heap   = nullptr; g_ccl_comm   = nullptr;
   s_kvs.reset();
   g_ccl_queue = nullptr;
   s_ccl_queue.reset();

   MPI_Finalize();
#endif
}

void barrierParallel()
{
#ifdef DO_MPI
   if (g_ccl_comm && g_ccl_stream)
      ccl::barrier(*g_ccl_comm, *g_ccl_stream).wait();
   else
      MPI_Barrier(MPI_COMM_WORLD);
#endif
}

/// Point-to-point: kept as MPI_Sendrecv (OneCCL send/recv deadlocks on
/// in-order streams because the NCCL backend requires grouped send+recv).
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

/// MINLOC using MPI_DOUBLE_INT which matches RankReduceData layout exactly.
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

/// MAXLOC using MPI_DOUBLE_INT which matches RankReduceData layout exactly.
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

void bcastParallel(void* buf, int count, int root)
{
#ifdef DO_MPI
   MPI_Bcast(buf, count, MPI_BYTE, root, MPI_COMM_WORLD);
#endif
}

// ---------------------------------------------------------------------------
// Categoria A — device-resident data: OneCCL directly on device pointers
// ---------------------------------------------------------------------------

/// Integer allreduce sum on device pointers. No memcpy — caller owns the
/// device buffers and is responsible for any D2H after this call.
void addIntParallelDevice(int* d_sendBuf, int* d_recvBuf, int count)
{
#ifdef DO_MPI
   ccl::allreduce(d_sendBuf, d_recvBuf, count,
                  ccl::datatype::int32, ccl::reduction::sum,
                  *g_ccl_comm, *g_ccl_stream).wait();
#else
   g_ccl_queue->memcpy(d_recvBuf, d_sendBuf, count * sizeof(int)).wait();
#endif
}

/// Real allreduce sum on device pointers. No memcpy — caller owns the
/// device buffers and is responsible for any D2H after this call.
void addRealParallelDevice(real_t* d_sendBuf, real_t* d_recvBuf, int count)
{
#ifdef DO_MPI
   ccl::allreduce(d_sendBuf, d_recvBuf, count,
                  REAL_CCL_TYPE, ccl::reduction::sum,
                  *g_ccl_comm, *g_ccl_stream).wait();
#else
   g_ccl_queue->memcpy(d_recvBuf, d_sendBuf, count * sizeof(real_t)).wait();
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
