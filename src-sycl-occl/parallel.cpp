/// \file
/// Wrappers for distributed communication functions implemented with OneCCL.

#include "parallel.h"

#ifdef DO_MPI
#include <dpct/dpct.hpp>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>
#endif

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef DO_MPI
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#endif

static int myRank = 0;
static int nRanks = 1;

#ifdef DO_MPI
namespace {

static ccl::communicator* s_comm = nullptr;
static ccl::stream* s_stream = nullptr;
static ccl::communicator* s_comm_owner = nullptr;
static ccl::stream* s_stream_owner = nullptr;
static ccl::shared_ptr_class<ccl::kvs> s_kvs;
static std::string s_kvs_path;

#ifdef COMD_SINGLE
#define REAL_CCL_TYPE ccl::datatype::float32
#else
#define REAL_CCL_TYPE ccl::datatype::float64
#endif

int parse_env_int(const char* value, int fallback)
{
   if (value == nullptr || *value == '\0') return fallback;

   char* end = nullptr;
   long parsed = strtol(value, &end, 10);
   if (end == value || *end != '\0') return fallback;
   if (parsed < 0 || parsed > INT_MAX) return fallback;
   return static_cast<int>(parsed);
}

int read_rank_from_env()
{
   const char* rankVars[] = {
      "CCL_PROCESS_RANK",
      "PMI_RANK",
      "OMPI_COMM_WORLD_RANK",
      "MV2_COMM_WORLD_RANK",
      "RANK",
      "SLURM_PROCID"
   };

   for (const char* name : rankVars)
   {
      int parsed = parse_env_int(getenv(name), -1);
      if (parsed >= 0) return parsed;
   }
   return 0;
}

int read_size_from_env()
{
   const char* sizeVars[] = {
      "CCL_PROCESS_COUNT",
      "PMI_SIZE",
      "OMPI_COMM_WORLD_SIZE",
      "MV2_COMM_WORLD_SIZE",
      "WORLD_SIZE",
      "SLURM_NTASKS"
   };

   for (const char* name : sizeVars)
   {
      int parsed = parse_env_int(getenv(name), -1);
      if (parsed > 0) return parsed;
   }
   return 1;
}

std::string kvs_address_path()
{
   const char* explicitPath = getenv("COMD_CCL_KVS_FILE");
   if (explicitPath != nullptr && *explicitPath != '\0') return std::string(explicitPath);

   const char* jobId = getenv("SLURM_JOB_ID");
   if (jobId != nullptr && *jobId != '\0')
   {
      return std::string("/tmp/comd_ccl_kvs.") + jobId + ".addr";
   }

   return "/tmp/comd_ccl_kvs.addr";
}

bool write_kvs_addr(const std::string& path, const ccl::kvs::address_type& addr)
{
   FILE* fp = fopen(path.c_str(), "wb");
   if (fp == nullptr) return false;

   size_t written = fwrite(addr.data(), 1, addr.size(), fp);
   fclose(fp);
   return written == addr.size();
}

bool read_kvs_addr(const std::string& path, ccl::kvs::address_type& addr)
{
   FILE* fp = fopen(path.c_str(), "rb");
   if (fp == nullptr) return false;

   size_t read = fread(addr.data(), 1, addr.size(), fp);
   fclose(fp);
   return read == addr.size();
}

bool wait_for_kvs_addr(const std::string& path, ccl::kvs::address_type& addr)
{
   constexpr int kPollMs = 100;
   constexpr int kTimeoutMs = 30000;
   const int iters = kTimeoutMs / kPollMs;

   for (int i = 0; i < iters; ++i)
   {
      if (read_kvs_addr(path, addr)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
   }

   return false;
}

sycl::queue& ccl_queue()
{
   return dpct::get_in_order_queue();
}

void require_ccl_initialized(const char* fnName)
{
   if (nRanks <= 1) return;
   if (s_comm != nullptr && s_stream != nullptr) return;

   fprintf(stderr,
           "ERROR: %s called before initCCL() while running with %d ranks.\n",
           fnName, nRanks);
   abort();
}

template <typename T>
void allreduce_device_buffer(T* sendBuf,
                             T* recvBuf,
                             int count,
                             ccl::datatype dtype,
                             ccl::reduction reduction)
{
   if (count <= 0) return;
   if (nRanks <= 1)
   {
      for (int ii = 0; ii < count; ++ii) recvBuf[ii] = sendBuf[ii];
      return;
   }

   require_ccl_initialized("allreduce");
   sycl::queue& q = ccl_queue();

   T* dSend = sycl::malloc_device<T>(count, q);
   T* dRecv = sycl::malloc_device<T>(count, q);
   q.memcpy(dSend, sendBuf, static_cast<size_t>(count) * sizeof(T)).wait();

   auto ev = ccl::allreduce(dSend, dRecv, count, dtype, reduction, *s_comm, *s_stream);
   ev.wait();
   q.wait();

   q.memcpy(recvBuf, dRecv, static_cast<size_t>(count) * sizeof(T)).wait();

   sycl::free(dSend, q);
   sycl::free(dRecv, q);
}

void reduce_rank_data(RankReduceData* sendBuf, RankReduceData* recvBuf, int count, bool doMin)
{
   if (count <= 0) return;
   if (nRanks <= 1)
   {
      for (int ii = 0; ii < count; ++ii)
      {
         recvBuf[ii].val = sendBuf[ii].val;
         recvBuf[ii].rank = sendBuf[ii].rank;
      }
      return;
   }

   require_ccl_initialized(doMin ? "minRankDoubleParallel" : "maxRankDoubleParallel");
   sycl::queue& q = ccl_queue();

   const size_t bytesPerRank = static_cast<size_t>(count) * sizeof(RankReduceData);
   const size_t totalBytes = static_cast<size_t>(nRanks) * bytesPerRank;

   char* dSend = sycl::malloc_device<char>(bytesPerRank, q);
   char* dRecv = sycl::malloc_device<char>(totalBytes, q);

   q.memcpy(dSend, sendBuf, bytesPerRank).wait();

   auto ev = ccl::allgather(dSend, dRecv, bytesPerRank, ccl::datatype::int8,
                            *s_comm, *s_stream);
   ev.wait();
   q.wait();

   std::vector<RankReduceData> gathered(static_cast<size_t>(count) * nRanks);
   q.memcpy(gathered.data(), dRecv, totalBytes).wait();

   for (int item = 0; item < count; ++item)
   {
      RankReduceData best = gathered[item];
      for (int rank = 1; rank < nRanks; ++rank)
      {
         RankReduceData cand = gathered[static_cast<size_t>(rank) * count + item];
         if (doMin)
         {
            if ((cand.val < best.val) || (cand.val == best.val && cand.rank < best.rank))
               best = cand;
         }
         else
         {
            if ((cand.val > best.val) || (cand.val == best.val && cand.rank < best.rank))
               best = cand;
         }
      }
      recvBuf[item] = best;
   }

   sycl::free(dSend, q);
   sycl::free(dRecv, q);
}

} // namespace
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
   if (!printRank()) return;
   time_t t = time(NULL);
   char* timeString = ctime(&t);
   timeString[24] = '\0';
   fprintf(screenOut, "%s: %s\n", timeString, msg);
   fflush(screenOut);
}

void initParallel(int* argc, char*** argv)
{
   (void)argc;
   (void)argv;

#ifdef DO_MPI
   myRank = read_rank_from_env();
   nRanks = read_size_from_env();

   if (nRanks < 1) nRanks = 1;
   if (myRank < 0) myRank = 0;
   if (myRank >= nRanks) nRanks = myRank + 1;

   ccl::init();
#endif
}

void initCCL()
{
#ifdef DO_MPI
   if (nRanks <= 1 || s_comm != nullptr) return;

   sycl::queue& q = ccl_queue();

   ccl::shared_ptr_class<ccl::kvs> kvs;
   ccl::kvs::address_type addr;
   s_kvs_path = kvs_address_path();

   if (myRank == 0)
   {
      kvs = ccl::create_main_kvs();
      addr = kvs->get_address();
      if (!write_kvs_addr(s_kvs_path, addr))
      {
         throw std::runtime_error("Failed to write OneCCL KVS address file: " + s_kvs_path);
      }
   }
   else
   {
      if (!wait_for_kvs_addr(s_kvs_path, addr))
      {
         throw std::runtime_error("Timed out waiting for OneCCL KVS address file: " + s_kvs_path);
      }
      kvs = ccl::create_kvs(addr);
   }

   s_kvs = kvs;

   auto cclDev = ccl::create_device(q.get_device());
   auto cclCtx = ccl::create_context(q.get_context());
   auto comm = ccl::create_communicator(nRanks, myRank, cclDev, cclCtx, kvs);
   auto stream = ccl::create_stream(q);

   s_comm_owner = new ccl::communicator(std::move(comm));
   s_stream_owner = new ccl::stream(std::move(stream));
   s_comm = s_comm_owner;
   s_stream = s_stream_owner;
#endif
}

void destroyParallel()
{
#ifdef DO_MPI
   delete s_stream_owner;
   delete s_comm_owner;
   s_stream_owner = nullptr;
   s_comm_owner = nullptr;
   s_stream = nullptr;
   s_comm = nullptr;
   s_kvs.reset();

   if (myRank == 0 && !s_kvs_path.empty())
   {
      remove(s_kvs_path.c_str());
      s_kvs_path.clear();
   }
#endif
}

void barrierParallel()
{
#ifdef DO_MPI
   if (nRanks <= 1) return;
   if (s_comm == nullptr || s_stream == nullptr) return;

   int in = 1;
   int out = 0;
   allreduce_device_buffer<int>(&in, &out, 1, ccl::datatype::int32, ccl::reduction::sum);
#endif
}

int sendReceiveParallel(void* sendBuf, int sendLen, int dest,
                        void* recvBuf, int recvLen, int source)
{
#ifdef DO_MPI
   if (nRanks <= 1)
   {
      assert(source == dest);
      memcpy(recvBuf, sendBuf, sendLen);
      return sendLen;
   }

   if (dest == myRank && source == myRank)
   {
      int copyLen = (sendLen < recvLen) ? sendLen : recvLen;
      memcpy(recvBuf, sendBuf, copyLen);
      return copyLen;
   }

   require_ccl_initialized("sendReceiveParallel");
   sycl::queue& q = ccl_queue();

   int* dSendLen = sycl::malloc_device<int>(1, q);
   int* dRecvLen = sycl::malloc_device<int>(1, q);
   q.memcpy(dSendLen, &sendLen, sizeof(int)).wait();

   auto recvHeaderEv = ccl::recv(dRecvLen, 1, ccl::datatype::int32, source, *s_comm, *s_stream);
   auto sendHeaderEv = ccl::send(dSendLen, 1, ccl::datatype::int32, dest, *s_comm, *s_stream);
   recvHeaderEv.wait();
   sendHeaderEv.wait();
   q.wait();

   int bytesReceived = 0;
   q.memcpy(&bytesReceived, dRecvLen, sizeof(int)).wait();

   sycl::free(dSendLen, q);
   sycl::free(dRecvLen, q);

   if (bytesReceived < 0 || bytesReceived > recvLen)
   {
      fprintf(stderr,
              "ERROR: sendReceiveParallel received invalid size %d (buffer capacity %d).\n",
              bytesReceived, recvLen);
      abort();
   }

   char* dSend = nullptr;
   char* dRecv = nullptr;

   if (sendLen > 0)
   {
      dSend = sycl::malloc_device<char>(sendLen, q);
      q.memcpy(dSend, sendBuf, static_cast<size_t>(sendLen)).wait();
   }

   if (bytesReceived > 0)
   {
      dRecv = sycl::malloc_device<char>(bytesReceived, q);
   }

   if (bytesReceived > 0 && sendLen > 0)
   {
      auto recvPayloadEv = ccl::recv(dRecv, bytesReceived, ccl::datatype::int8,
                                     source, *s_comm, *s_stream);
      auto sendPayloadEv = ccl::send(dSend, sendLen, ccl::datatype::int8,
                                     dest, *s_comm, *s_stream);
      recvPayloadEv.wait();
      sendPayloadEv.wait();
   }
   else if (bytesReceived > 0)
   {
      auto recvPayloadEv = ccl::recv(dRecv, bytesReceived, ccl::datatype::int8,
                                     source, *s_comm, *s_stream);
      recvPayloadEv.wait();
   }
   else if (sendLen > 0)
   {
      auto sendPayloadEv = ccl::send(dSend, sendLen, ccl::datatype::int8,
                                     dest, *s_comm, *s_stream);
      sendPayloadEv.wait();
   }
   q.wait();

   if (bytesReceived > 0)
      q.memcpy(recvBuf, dRecv, static_cast<size_t>(bytesReceived)).wait();

   if (dSend != nullptr) sycl::free(dSend, q);
   if (dRecv != nullptr) sycl::free(dRecv, q);

   return bytesReceived;
#else
   assert(source == dest);
   memcpy(recvBuf, sendBuf, sendLen);
   return sendLen;
#endif
}

void addIntParallel(int* sendBuf, int* recvBuf, int count)
{
#ifdef DO_MPI
   allreduce_device_buffer<int>(sendBuf, recvBuf, count,
                                ccl::datatype::int32, ccl::reduction::sum);
#else
   for (int ii = 0; ii < count; ++ii) recvBuf[ii] = sendBuf[ii];
#endif
}

void addRealParallel(real_t* sendBuf, real_t* recvBuf, int count)
{
#ifdef DO_MPI
   allreduce_device_buffer<real_t>(sendBuf, recvBuf, count,
                                   REAL_CCL_TYPE, ccl::reduction::sum);
#else
   for (int ii = 0; ii < count; ++ii) recvBuf[ii] = sendBuf[ii];
#endif
}

void addDoubleParallel(double* sendBuf, double* recvBuf, int count)
{
#ifdef DO_MPI
   allreduce_device_buffer<double>(sendBuf, recvBuf, count,
                                   ccl::datatype::float64, ccl::reduction::sum);
#else
   for (int ii = 0; ii < count; ++ii) recvBuf[ii] = sendBuf[ii];
#endif
}

void maxIntParallel(int* sendBuf, int* recvBuf, int count)
{
#ifdef DO_MPI
   allreduce_device_buffer<int>(sendBuf, recvBuf, count,
                                ccl::datatype::int32, ccl::reduction::max);
#else
   for (int ii = 0; ii < count; ++ii) recvBuf[ii] = sendBuf[ii];
#endif
}

void minRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count)
{
#ifdef DO_MPI
   reduce_rank_data(sendBuf, recvBuf, count, true);
#else
   for (int ii = 0; ii < count; ++ii)
   {
      recvBuf[ii].val = sendBuf[ii].val;
      recvBuf[ii].rank = sendBuf[ii].rank;
   }
#endif
}

void maxRankDoubleParallel(RankReduceData* sendBuf, RankReduceData* recvBuf, int count)
{
#ifdef DO_MPI
   reduce_rank_data(sendBuf, recvBuf, count, false);
#else
   for (int ii = 0; ii < count; ++ii)
   {
      recvBuf[ii].val = sendBuf[ii].val;
      recvBuf[ii].rank = sendBuf[ii].rank;
   }
#endif
}

void bcastParallel(void* buf, int count, int root)
{
#ifdef DO_MPI
   if (count <= 0 || nRanks <= 1) return;

   require_ccl_initialized("bcastParallel");
   sycl::queue& q = ccl_queue();

   char* dBuf = sycl::malloc_device<char>(count, q);
   if (myRank == root)
      q.memcpy(dBuf, buf, static_cast<size_t>(count)).wait();

   auto ev = ccl::broadcast(dBuf, count, ccl::datatype::int8, root, *s_comm, *s_stream);
   ev.wait();
   q.wait();

   q.memcpy(buf, dBuf, static_cast<size_t>(count)).wait();
   sycl::free(dBuf, q);
#else
   (void)buf;
   (void)count;
   (void)root;
#endif
}

int builtWithMpi(void)
{
   return 0;
}
