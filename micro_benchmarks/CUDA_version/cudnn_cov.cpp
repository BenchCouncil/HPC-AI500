#include <cstdlib>
#include <cstring>
#include <string>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <chrono>
#include <vector>

// #include "papi.h"
// #include "papi_test.h"

#ifdef PAPI
const int NUM_EVENTS = 3;
#endif

#include "./data_load.h"

#define CUDA_CALL(f)                                               \
  {                                                                \
    cudaError_t err = (f);                                         \
    if (err != cudaSuccess)                                        \
    {                                                              \
      std::cout << " CUDA   Error occurred: " << err << std::endl; \
      std::exit(1);                                                \
    }                                                              \
  }

#define CUDNN_CALL(f)                                                                                  \
  {                                                                                                    \
    cudnnStatus_t err = (f);                                                                           \
    if (err != CUDNN_STATUS_SUCCESS)                                                                   \
    {                                                                                                  \
      std::cout << " CUDNN   Error occurred: " << err << " " << cudnnGetErrorString(err) << std::endl; \
      std::exit(1);                                                                                    \
    }                                                                                                  \
  }

void print(const float *data, int n, int c, int h, int w)
{
  // std::vector<float> buffer;
  float *buffer = new float[n * c * h * w * sizeof(float)];
  CUDA_CALL(cudaMemcpy(buffer, data, n * c * h * w * sizeof(float),
                       cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < c; ++j)
    {
      std::cout << "n=" << i + 1 << ", c=" << j + 1 << ":" << std::endl;
      for (int k = 0; k < h; ++k)
      {
        for (int l = 0; l < w; ++l)
        {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[])
{

  // std::string file_list[] = {"/home/zhangyuchen/out_for_test.h5","",""};
  // initilize mpi
  MPI_Init(&argc, &argv);

  int world_rank, world_size;

  //init cudnn
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // get the current thread rank
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  // get the size of usable threads of communicator
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  CUDA_CALL(cudaSetDevice(world_rank % 8));

  srand(time(NULL) * world_rank);

#ifdef PAPI
  int retval, i;
  int EventSet = PAPI_NULL;
  long long values[NUM_EVENTS];
  /* REPLACE THE EVENT NAME 'PAPI_FP_OPS' WITH A CUDA EVENT 
	   FOR THE CUDA DEVICE YOU ARE RUNNING ON.
	   RUN papi_native_avail to get a list of CUDA events that are 
	   supported on your machine */
  //char *EventName[] = { "PAPI_FP_OPS" };
  string EventName[] = {"cuda:::metric:flop_count_sp:device=", "cuda:::metric:flop_count_sp_add:device=", "cuda:::metric:flop_count_sp_mul:device="};
  for (int i = 0; i < NUM_EVENTS; i++)
  {
    EventName[i] += to_string(world_rank);
    cout << EventName[i] << endl;
  }

  int events[NUM_EVENTS];
  int eventCount = 0;
  int quiet;

  /* Set TESTS_QUIET variable */
  quiet = tests_quiet(argc, argv);

  /* PAPI Initialization */
  retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT)
  {
    if (!quiet)
      printf("PAPI init failed\n");
    test_fail(__FILE__, __LINE__,
              "PAPI_library_init failed", 0);
  }

  if (!quiet)
  {
    printf("PAPI_VERSION     : %4d %6d %7d\n",
           PAPI_VERSION_MAJOR(PAPI_VERSION),
           PAPI_VERSION_MINOR(PAPI_VERSION),
           PAPI_VERSION_REVISION(PAPI_VERSION));
  }

  /* convert PAPI native events to PAPI code */
  for (i = 0; i < NUM_EVENTS; i++)
  {
    retval = PAPI_event_name_to_code((char *)EventName[i].c_str(), &events[i]);
    if (retval != PAPI_OK)
    {
      fprintf(stderr, "PAPI_event_name_to_code failed\n");
      continue;
    }
    eventCount++;
    if (!quiet)
      printf("Name %s --- Code: %#x\n", EventName[i], events[i]);
  }

  /* if we did not find any valid events, just report test failed. */
  if (eventCount == 0)
  {
    if (!quiet)
      printf("Test FAILED: no valid events found.\n");
    test_skip(__FILE__, __LINE__, "No events found", 0);
    return 1;
  }

  retval = PAPI_create_eventset(&EventSet);
  if (retval != PAPI_OK)
  {
    if (!quiet)
      printf("PAPI_create_eventset failed\n");
    test_fail(__FILE__, __LINE__, "Cannot create eventset", retval);
  }

  // If multiple GPUs/contexts were being used,
  // you need to switch to each device before adding its events
  // cudaSetDevice( world_rank );
  // retval = PAPI_add_events( EventSet, events, eventCount );
  // if( retval != PAPI_OK ) {
  // 	fprintf( stderr, "PAPI_add_events failed\n" );
  // }
  for (int i = 0; i < NUM_EVENTS; i++)
  {
    retval = PAPI_add_named_event(EventSet, (char *)EventName[i].c_str());
    if (retval != PAPI_OK)
    {
      fprintf(stderr, "PAPI_add_events failed: %s\n", EventName[i].c_str());
    }
  }

  retval = PAPI_start(EventSet);
  if (retval != PAPI_OK)
  {
    fprintf(stderr, "PAPI_start failed\n");
  }
#endif

  // file dealing with
  // std::string file_name = file_list[world_rank];
  char hostname[256];
  int len;
  MPI_Get_processor_name(hostname, &len);
  double time_end = 0.0;
  double time_start = 0.0;
  // input parameters
  // int in_n = 1;
  int in_c = 10;
  int in_h = 10;
  int in_w = 10;
  auto in_n = std::stoull(argv[1]);
  in_c = std::atoi(argv[2]);
  in_h = std::atoi(argv[3]);
  in_w = std::atoi(argv[4]);
  // std::cout << "Please type in input FOUR paras(NCHW, separate using blank):"
  //           << std::endl;
  // std::cin >> in_n >> in_c >> in_h >> in_w;
  // std::cout << "in_n: " << in_n << std::endl;
  // std::cout << "in_c: " << in_c << std::endl;
  // std::cout << "in_h: " << in_h << std::endl;
  // std::cout << "in_w: " << in_w << std::endl;
  // std::cout << std::endl;

  // filter parameters
  int filt_k = 1;
  int filt_c = 10;
  int filt_h = 2;
  int filt_w = 2;
  filt_k = std::atoi(argv[5]);
  filt_c = std::atoi(argv[6]);
  filt_h = std::atoi(argv[7]);
  filt_w = std::atoi(argv[8]);
  // std::cout << "Please type in filter FOUR paras(out_size, in_size, height, "
  //              "width, separate using blank):"
  //           << std::endl;
  // std::cout
  //     << "the second param (input channels) should equals to the input one"
  //     << std::endl;
  // std::cin >> filt_k >> filt_c >> filt_h >> filt_w;
  // std::cout << "filt_k: " << filt_k << std::endl;
  // std::cout << "filt_c: " << filt_c << std::endl;
  // std::cout << "filt_h: " << filt_h << std::endl;
  // std::cout << "filt_w: " << filt_w << std::endl;
  // std::cout << std::endl;

  // convolution parameters
  int pad_h = 1;
  int pad_w = 1;
  int str_h = 1;
  int str_w = 1;
  int dil_h = 1;
  int dil_w = 1;
  pad_h = std::atoi(argv[9]);
  pad_w = std::atoi(argv[10]);
  str_h = std::atoi(argv[11]);
  str_w = std::atoi(argv[12]);
  dil_h = std::atoi(argv[13]);
  dil_w = std::atoi(argv[14]);
  // std::cout << "Please type in conv SIX paras(pad_h/w,str_h/w,dil_h/w, "
  //              "separate using blank):"
  //           << std::endl;
  // std::cout
  //     << "the second param (input channels) should equals to the input one"
  //     << std::endl;
  // std::cin >> pad_h >> pad_w >> str_h >> str_w >> dil_h >> dil_w;
  // std::cout << "pad_h: " << pad_h << std::endl;
  // std::cout << "pad_w: " << pad_w << std::endl;
  // std::cout << "str_h: " << str_h << std::endl;
  // std::cout << "str_w: " << str_w << std::endl;
  // std::cout << "dil_h: " << dil_h << std::endl;
  // std::cout << "dil_w: " << dil_w << std::endl;
  // std::cout << std::endl;

  // generate random matrix
  // float *h_in_data = new float[in_n * in_c * in_w * in_h];
  // for (int i = 0; i < in_n * in_c * in_w * in_h; ++i) {
  //   *(h_in_data+i) = rand() % 10;
  // }

  res out = data_load(world_rank);
  float *h_in_data = out.out;

  float h_filt_data[filt_w * filt_h * filt_k * filt_c];
  // random_gen(h_filt_data, filt_w * filt_h * filt_k * filt_c);
  for (int j = 0; j < filt_w * filt_h * filt_k * filt_c; ++j)
  {
    h_filt_data[j] = (rand() % 12 - 5) / 10;
  }

  // input descriptor
  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w));
  float *in_data;
  CUDA_CALL(cudaMalloc(&in_data, in_n * in_c * in_h * in_w * sizeof(float)));
  // std::cout << "p 1 " << std::endl;
  // filter descriptor
  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW, filt_k, filt_c,
                                        filt_h, filt_w));

  float *filt_data;
  CUDA_CALL(cudaMalloc(&filt_data,
                       filt_k * filt_c * filt_h * filt_w * sizeof(float)));
  // std::cout << "p 2 " << std::endl;
  // conv descriptor
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CONVOLUTION,
      CUDNN_DATA_FLOAT));
  CUDNN_CALL( cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) );
//  CUDNN_CALL( cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH) );
  // std::cout << "p 3 " << std::endl;
  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;

  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
      conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w));

  // std::cout << "out_n: " << out_n << std::endl;
  // std::cout << "out_c: " << out_c << std::endl;
  // std::cout << "out_h: " << out_h << std::endl;
  // std::cout << "out_w: " << out_w << std::endl;
  // std::cout << std::endl;

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, out_n, out_c, out_h,
                                        out_w));

  float *out_data;
  CUDA_CALL(
      cudaMalloc(&out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  int res_count;
  cudnnConvolutionFwdAlgoPerf_t fwd_perf;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,in_desc,filt_desc,conv_desc,out_desc,1,&res_count,&fwd_perf));
  algo = fwd_perf.algo;
  
  // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
    //   cudnn, in_desc, filt_desc, conv_desc, out_desc,
      // CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  // std::cout << "Convolution algorithm: " << algo << std::endl;
  // std::cout << std::endl;

  // workspace
  uint64_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  // std::cout << "Workspace size: " << ws_size << std::endl;
  // std::cout << std::endl;

  // perform
  float alpha = 1.0f;
  float beta = 0.0f;

  // CUDA_CALL(cudaMalloc(&in_data, sizeof(h_in_data)));
  CUDA_CALL(cudaMemcpy(in_data, h_in_data, in_n * in_c * in_h * in_w * sizeof(float),
                       cudaMemcpyHostToDevice));
  // CUDA_CALL(cudaMalloc(&filt_data, sizeof(h_filt_data)));
  CUDA_CALL(cudaMemcpy(filt_data, h_filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float),
                       cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // start time
  MPI_Barrier(MPI_COMM_WORLD);
  cudaDeviceSynchronize();
  auto start_ = std::chrono::steady_clock::now();
  cudaEventRecord(start);
  time_start = MPI_Wtime();

  // calculation
  for (int i = 0; i < 1000; ++i)
  {

    CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, in_desc, in_data, filt_desc,
                                       filt_data, conv_desc, 
                                       algo, 
                                       ws_data,
                                       ws_size, &beta, out_desc, out_data));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  cudaEventRecord(stop);

  // finish time
  time_end = MPI_Wtime();
  cudaDeviceSynchronize();
  auto end_ = std::chrono::steady_clock::now();
  double duration = time_end - time_start;
  double global;
  // results
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << milliseconds / 1000 << std::endl;

  int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end_ - start_).count() / 1000);
  std::cout << "fwd_time: " << fwd_time << std::endl;
  // results
  // std::cout << "in_data size: " << in_n << "*" << in_c << "*" << in_h << "*"
  //           << in_w << std::endl;
  //print(in_data, in_n, in_c, in_h, in_w);

  // std::cout << "filt_data:" << std::endl;
  // print(filt_data, filt_k, filt_c, filt_h, filt_w);
  // std::cout << "Thread " << world_rank << " has: " << std::endl;
  // std::cout << "out_data:" << std::endl;
  // std::cout << "out_data size: " << out_n << "*" << out_c << "*" << out_h << "*"
  //           << out_w << std::endl;
  // print(out_data, out_n, out_c, out_h, out_w);

  // std::cout << "The time cost of thread " << world_rank << " is: " << time_end - time_start
  //           << " "
  //           << "running on " << hostname << std::endl;
  // Bwd
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  //bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  cudnnConvolutionBwdDataAlgo_t bwd_inputs_algo;
  //bwd_inputs_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  uint64_t bwd_ws_f_size, bwd_ws_d_size;
  float *bwd_ws_f_data,*bwd_ws_d_data;
  float *h_delta = new float [out_n*out_c*out_h*out_w];

  for (int j = 0; j < out_n*out_c*out_h*out_w; ++j)
  {
    *(h_delta + j) = (rand() % 2 - 1);
  }

cudnnConvolutionBwdFilterAlgoPerf_t bwd_f_perf;
cudnnConvolutionBwdDataAlgoPerf_t bwd_d_perf;
int c_f,c_d;
   CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn,in_desc,out_desc,conv_desc,filt_desc,1,&c_f,&bwd_f_perf));
   CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn,filt_desc,out_desc,conv_desc,in_desc,1,&c_d,&bwd_d_perf));
   bwd_filter_algo = bwd_f_perf.algo;
   bwd_inputs_algo = bwd_d_perf.algo;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,in_desc,out_desc,conv_desc,filt_desc,bwd_filter_algo,&bwd_ws_f_size));
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn,filt_desc,out_desc,conv_desc,in_desc,bwd_inputs_algo,&bwd_ws_d_size));
  CUDA_CALL(cudaMalloc(&bwd_ws_f_data, bwd_ws_f_size));
  CUDA_CALL(cudaMalloc(&bwd_ws_d_data, bwd_ws_d_size));
  float *d_W;
  float *d_X;
  float *delta;
  CUDA_CALL(cudaMalloc(&delta, out_n * out_c * out_h * out_w * sizeof(float)));
    CUDA_CALL(cudaMemcpy(delta, h_delta, out_n * out_c * out_h * out_w * sizeof(float),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&d_W, filt_k * filt_c * filt_h * filt_w * sizeof(float)));
  CUDA_CALL(cudaMalloc(&d_X, in_n * in_c * in_h * in_w * sizeof(float)));
  start_ = std::chrono::steady_clock::now();
cudaDeviceSynchronize();

for (int i = 0; i < 1000; ++i)
  {
  CUDNN_CALL(cudnnConvolutionBackwardFilter(cudnn,
                                                         &alpha,
                                                         in_desc,
                                                         in_data,
                                                         out_desc,
                                                         delta,
                                                         conv_desc,
                                                         bwd_filter_algo,
                                                        // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                                                         bwd_ws_f_data,
                                                         bwd_ws_f_size,
                                                         &beta,
                                                         filt_desc,
                                                         d_W));
  }
cudaDeviceSynchronize();

end_ = std::chrono::steady_clock::now();
  fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end_ - start_).count() / 1000);
  std::cout << "bwd_filt_time: " << fwd_time << std::endl;
  start_ = std::chrono::steady_clock::now();
  cudaDeviceSynchronize();

  for (int i = 0; i < 1000; ++i)
  {
  CUDNN_CALL(cudnnConvolutionBackwardData(cudnn,
                                                      &alpha,
                                                      filt_desc,
                                                      filt_data,
                                                      out_desc,
                                                      delta,
                                                      conv_desc,
                                                      bwd_inputs_algo,
                                                      // CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                                      bwd_ws_d_data,
                                                      bwd_ws_d_size,
                                                      &beta,
                                                      in_desc,
                                                      d_X));
  }
cudaDeviceSynchronize();

  end_ = std::chrono::steady_clock::now();
  fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end_ - start_).count() / 1000);
  std::cout << "bwd_data_time: " << fwd_time << std::endl;

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(out_data));
  CUDA_CALL(cudaFree(delta));
  CUDA_CALL(cudaFree(bwd_ws_d_data));
  CUDA_CALL(cudaFree(bwd_ws_f_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));

#ifdef PAPI
  retval = PAPI_stop(EventSet, values);
  if (retval != PAPI_OK)
    fprintf(stderr, "PAPI_stop failed\n");

  retval = PAPI_cleanup_eventset(EventSet);
  if (retval != PAPI_OK)
    fprintf(stderr, "PAPI_cleanup_eventset failed\n");

  retval = PAPI_destroy_eventset(&EventSet);
  if (retval != PAPI_OK)
    fprintf(stderr, "PAPI_destroy_eventset failed\n");

  PAPI_shutdown();

  for (i = 0; i < eventCount; i++)
    if (!quiet)
      printf("%12lld \t\t --> %s \n", values[i], (char *)EventName[i].c_str());
#endif
  MPI_Finalize();
  return 0;
}
