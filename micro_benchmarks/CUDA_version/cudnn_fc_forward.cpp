#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <string>
#include <iostream>
#include <chrono>
#include <mpi.h>
#include <vector>
#include <cstring>
// #include "papi.h"
// #include "papi_test.h"

// #define NUM_EVENTS 3
// #define PAPI 1
#include "./data_load.h"


using namespace std;

#define CUDA_CALL(f)                                                           \
  {                                                                            \
    cudaError_t err = (f);                                                     \
    if (err != cudaSuccess) {                                                  \
      std::cout << "    Error occurred: " << err << std::endl;                 \
      std::exit(1);                                                            \
    }                                                                          \
  }

int main(int argc, char *argv[]) {


  // initilize mpi
  MPI_Init(&argc, &argv);

  // cublasHandle_t cublasHandle;
  cublasStatus_t cublasstat;
  int world_rank, world_size;
  double time_cost = 0.0;
  double time_start = 0.0;

  // get the current thread rank
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  // get the size of usable threads of communicator
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  CUDA_CALL(cudaSetDevice(world_rank%8));
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
  string EventName[] = { "cuda:::metric:flop_count_sp:device=","cuda:::metric:flop_count_sp_add:device=","cuda:::metric:flop_count_sp_mul:device="};
  for(int i = 0; i < NUM_EVENTS; i++){
    EventName[i]+=to_string(world_rank);
    cout << EventName[i] << endl;
  }

	int events[NUM_EVENTS];
	int eventCount = 0;
	int quiet;

	/* Set TESTS_QUIET variable */
	quiet=tests_quiet( argc, argv );
	
	/* PAPI Initialization */
	retval = PAPI_library_init( PAPI_VER_CURRENT );
	if( retval != PAPI_VER_CURRENT ) {
		if (!quiet) printf("PAPI init failed\n");
		test_fail(__FILE__,__LINE__,
			"PAPI_library_init failed", 0 );
	}

	if (!quiet) {
		printf( "PAPI_VERSION     : %4d %6d %7d\n",
			PAPI_VERSION_MAJOR( PAPI_VERSION ),
			PAPI_VERSION_MINOR( PAPI_VERSION ),
			PAPI_VERSION_REVISION( PAPI_VERSION ) );
	}

	/* convert PAPI native events to PAPI code */
	for( i = 0; i < NUM_EVENTS; i++ ){
    retval = PAPI_event_name_to_code( (char *)EventName[i].c_str(), &events[i] );
		if( retval != PAPI_OK ) {
			fprintf( stderr, "PAPI_event_name_to_code failed\n" );
			continue;
		}
		eventCount++;
		if (!quiet) printf( "Name %s --- Code: %#x\n", EventName[i].c_str(), events[i] );
	}

	/* if we did not find any valid events, just report test failed. */
	if (eventCount == 0) {
		if (!quiet) printf( "Test FAILED: no valid events found.\n");
		test_skip(__FILE__,__LINE__,"No events found",0);
		return 1;
	}
	
	retval = PAPI_create_eventset( &EventSet );
	if( retval != PAPI_OK ) {
		if (!quiet) printf( "PAPI_create_eventset failed\n" );
		test_fail(__FILE__,__LINE__,"Cannot create eventset",retval);
	}	

        // If multiple GPUs/contexts were being used, 
        // you need to switch to each device before adding its events
  cudaSetDevice( world_rank );
	retval = PAPI_add_events( EventSet, events, eventCount );
	if( retval != PAPI_OK ) {
		fprintf( stderr, "PAPI_add_events failed\n" );
	}

#endif

  // input descriptor
  int in_n = 1;
  int in_c = 1;
  int in_h = 10;
  int in_w = 10;

  in_n = std::atoi(argv[1]);
  in_c = std::atoi(argv[2]);
  in_h = std::atoi(argv[3]);
  in_w = std::atoi(argv[4]);

  // cudnnTensorDescriptor_t in_desc;
  // CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  // CUDNN_CALL(cudnnSetTensor4dDescriptor(
  //     in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w));
  float *in_data;
  CUDA_CALL(cudaMalloc((void **)&in_data, in_n * in_c * in_h * in_w * sizeof(float)));

  // float *h_in_data = new float[in_n * in_c * in_h * in_w];
  // for (int i = 0; i < in_n * in_c * in_h * in_w; ++i) {
  //   *(h_in_data+i) = rand() % 10;
  // }
  res out = data_load(world_rank);
  float *h_in_data = out.out;

  // CUDA_CALL(cudaMalloc(&in_data, sizeof(h_in_data)));
  CUDA_CALL(cudaMemcpy(in_data, h_in_data, in_n * in_c * in_h * in_w * sizeof(float),
                       cudaMemcpyHostToDevice));
    
  // weight matrix
  // int w_n = 1;
  int w_c = 1;
  int w_h = 10;
  int w_w = 4;
  // cout<<"point 1"<<endl;
  auto w_n = std::stoull(argv[5]);
  w_c = std::atoi(argv[2]);
  w_h = std::atoi(argv[3]);
  w_w = std::atoi(argv[4]);

  // cudnnTensorDescriptor_t w_desc;
  // CUDNN_CALL(cudnnCreateTensorDescriptor(&w_desc));
  // CUDNN_CALL(cudnnSetTensor4dDescriptor(w_desc, CUDNN_TENSOR_NCHW,
  //                                       CUDNN_DATA_FLOAT, w_n, w_c, w_h,
  //                                       w_w));

  float *w_data;
  uint64_t l = w_n * w_c * w_h * w_w;

  float *h_w_data = new float [l];
  for (int i = 0; i < l; ++i) {
    *(h_w_data+i) = rand() % 10;
  }
  // cout<<"point 1"<<endl;
  CUDA_CALL(cudaMalloc((void **)&w_data, w_n * w_c * w_h * w_w * sizeof(float)));
  CUDA_CALL(cudaMemcpy(w_data, h_w_data, w_n * w_c * w_h * w_w * sizeof(float),
                       cudaMemcpyHostToDevice));
  // print test mat
  // std::cout << "matrix A :" << std::endl;
  // for (int i = 0; i < in_n * in_c * in_h * in_w; i++) {
  //   std::cout << h_in_data[i] << " ";
  //   if ((i + 1) % in_w == 0)
  //     std::cout << std::endl;
  // }
  // std::cout << std::endl;
  // std::cout << "matrix B :" << std::endl;
  // for (int i = 0; i < w_n * w_c * w_h * w_w; i++) {
  //   std::cout << h_w_data[i] << " ";
  //   if ((i + 1) % w_w == 0)
  //     std::cout << std::endl;
  // }
  // std::cout << std::endl;
  // cout<<"point 2"<<endl;
  float *d_A, *d_B, *d_C; // device stored matrix
  cudaMalloc((void **)&d_A,
             sizeof(float) * in_n * in_c * in_h * in_w); //allocate memory in device 
  cudaMalloc((void **)&d_B, sizeof(float) * w_n * w_c * w_h * w_w);
  cudaMalloc((void **)&d_C, sizeof(float) * w_n * in_n);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasstat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

 //if(cublasstat != CUBLAS_STATUS_SUCCESS){
   //  cout << cublasstat << endl;
   // } 
  cudaMemcpy(d_A, h_in_data, sizeof(float) * in_n * in_c * in_h * in_w,
             cudaMemcpyHostToDevice); //data copy from host to device
  cudaMemcpy(d_B, h_w_data, sizeof(float) * w_n * w_c * w_h * w_w,
             cudaMemcpyHostToDevice);
// cout<<"point 3"<<endl;
             float a = 1.0f, b = 0.0f;
             cudaEvent_t estart, estop;
  cudaEventCreate(&estart);
  cudaEventCreate(&estop);
  MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();
  auto start_ = std::chrono::steady_clock::now();
  cudaEventRecord(estart);
#ifdef PAPI
	retval = PAPI_start( EventSet );
	if( retval != PAPI_OK ) {
		fprintf( stderr, "PAPI_start failed\n" );
	}
#endif
  time_start = MPI_Wtime();
  for(int i = 0; i < 1000; ++i){
  cublasstat = cublasGemmEx(handle,
              CUBLAS_OP_T, //  A if transpose
              CUBLAS_OP_T, // B if transpose
              in_n,       // A C height
              w_n,       // B C width
              in_c * in_h * in_w, // A width
              &a,    // alpha value
              d_A,   // left mat
              CUDA_R_32F,
              in_c * in_h * in_w, // A width
              d_B, // right mat
              CUDA_R_32F,
              w_n, // B width
              &b,  // beta value
              d_C, // result mat C
              CUDA_R_32F,
              in_n,// C height
              CUDA_R_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP
   
   //             CUBLAS_GEMM_DEFAULT
  );
  }
  // C = alpha*A*B + beta*C


  MPI_Barrier(MPI_COMM_WORLD);
  cudaEventRecord(estop);

  // finish time
  cudaDeviceSynchronize();
  auto end_ = std::chrono::steady_clock::now();

  // results
  cudaEventSynchronize(estop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, estart, estop);
  std::cout << milliseconds / 1000 << std::endl;

  int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end_ - start_).count() / 1000);
  std::cout << fwd_time << std::endl;

  time_cost = MPI_Wtime() - time_start;

  // std::cout << "Thread " << world_rank << " 's result："
            // << std::endl;

  // float *h_C = new float [w_n * in_n];
  // cudaMemcpy(h_C, d_C, sizeof(float) * w_n * in_n, cudaMemcpyDeviceToHost);
  // for (int i = 0; i < w_n * in_n; ++i) {
  //   std::cout << h_C[i] << " ";
  //   if ((i + 1) % w_n == 0)
  //     std::cout << std::endl;
  // }
  std::cout << std::endl;

  std::cout << "The time cost of thread " << world_rank << " is: " << time_cost
  << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  // free(h_C);


  #ifdef PAPI

  retval = PAPI_stop( EventSet, values );
	if( retval != PAPI_OK ){
    fprintf( stderr, "PAPI_stop failed\n" );
  }



	retval = PAPI_cleanup_eventset(EventSet);
	if( retval != PAPI_OK )
		fprintf(stderr, "PAPI_cleanup_eventset failed\n");

	retval = PAPI_destroy_eventset(&EventSet);
	if (retval != PAPI_OK)
		fprintf(stderr, "PAPI_destroy_eventset failed\n");

	PAPI_shutdown();

	for( i = 0; i < eventCount; i++ )
		if (!quiet) printf( "%12lld \t\t --> %s \n", values[i], (char *)EventName[i].c_str() );
#endif
  MPI_Finalize();
  return 0;
}
