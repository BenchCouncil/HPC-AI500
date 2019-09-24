#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <mpi.h>
#include <chrono>
#include <cuda.h>
#include <cudnn.h>
#include <cstring>
// #include "papi.h"
// #include "papi_test.h"

// #define NUM_EVENTS 3
// #define PAPI 1
#include "./data_load.h"


#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

void print(const float *data, int n, int c, int h, int w) {
  float *buffer = new float [n * c * h * w * sizeof(float)];
  CUDA_CALL(cudaMemcpy(buffer, data, n * c * h * w * sizeof(float),
                       cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i+1 << ", c=" << j+1 << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  


  // initilize mpi
  MPI_Init(&argc, &argv);
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));
  int world_rank, world_size;

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
		if (!quiet) printf( "Name %s --- Code: %#x\n", EventName[i], events[i] );
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
        // e.g. cudaSetDevice( 0 );
	retval = PAPI_add_events( EventSet, events, eventCount );
	if( retval != PAPI_OK ) {
		fprintf( stderr, "PAPI_add_events failed\n" );
	}


#endif
  // input
  int in_n = 1;
  int in_c = 1;
  int in_h = 10;
  int in_w = 10;

  in_n = std::atoi(argv[1]);
  in_c = std::atoi(argv[2]);
  in_h = std::atoi(argv[3]);
  in_w = std::atoi(argv[4]);

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

        float *in_data;
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

  // pooling layer
  int w_h = 3;
  int w_w = 3;
  int v_pad = 0;
  int h_pad = 0;
  int v_stride = 1;
  int h_stride = 1;

  w_h = std::atoi(argv[5]);
  w_w = std::atoi(argv[6]);
  v_pad = std::atoi(argv[7]);
  h_pad = std::atoi(argv[8]);
  v_stride = std::atoi(argv[9]);
  h_stride = std::atoi(argv[10]);
  
  cudnnPoolingDescriptor_t pooling_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc));
  cudnnPoolingMode_t algo;
  int pooling_choice = std::atoi(argv[11]);
  if(pooling_choice == 0){
    algo = CUDNN_POOLING_MAX;
  }else if(pooling_choice == 1){
    algo = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
  CUDNN_CALL(cudnnSetPooling2dDescriptor(pooling_desc,algo,CUDNN_NOT_PROPAGATE_NAN,
    w_h,w_w,v_pad,h_pad,v_stride,h_stride));

  

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  cudnnGetPooling2dForwardOutputDim(
    /*const cudnnPoolingDescriptor_t */     pooling_desc,
    /*const cudnnTensorDescriptor_t */      in_desc,
    /*int*/                                &out_n,
    /*int*/                                &out_c,
    /*int*/                                &out_h,
    /*int*/                                &out_w);


  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

        float *out_data;
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // perform
  float alpha = 1.0f;
  float beta = 0.0f;

  // float *h_in_data = new float[in_n * in_c * in_w * in_h];
  // for(int i = 0; i < in_n * in_c * in_w * in_h; ++i){
  //   *(h_in_data+i) = rand()%10;
  // }
  
  res out = data_load(world_rank);
  float *h_in_data = out.out;
  // CUDA_CALL(cudaMalloc(&in_data,sizeof(h_in_data)));
  CUDA_CALL(cudaMemcpy(in_data, h_in_data,in_n * in_c * in_h * in_w * sizeof(float),cudaMemcpyHostToDevice));

  double start = 0.0;
  double end = 0.0;
  // calculation
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
  // time_cost -= MPI_Wtime();
  start = MPI_Wtime();
  for(int i = 0 ; i < 1000 ; ++i){
  CUDNN_CALL(cudnnPoolingForward(
      cudnn, pooling_desc,
      &alpha, in_desc, in_data,
      &beta, out_desc, out_data));
  }


    MPI_Barrier(MPI_COMM_WORLD);
  cudaEventRecord(estop);

  // finish time
  end = MPI_Wtime();
  cudaDeviceSynchronize();
  auto end_ = std::chrono::steady_clock::now();

  // results
  cudaEventSynchronize(estop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, estart, estop);
  std::cout << milliseconds / 1000 << std::endl;

  int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end_ - start_).count() / 1000);
  std::cout << fwd_time << std::endl;
  #ifdef PAPI
  retval = PAPI_stop( EventSet, values );
	if( retval != PAPI_OK )
		fprintf( stderr, "PAPI_stop failed\n" );
  #endif
  // finish time
  // time_cost += MPI_Wtime();
  

  // results
  // std::cout << "in_data size: " 
  // << in_n 
  // << "*"
  // << in_c 
  // << "*" 
  // << in_h 
  // << "*" 
  // << in_w<< std::endl;
  // print(in_data, in_n, in_c, in_h, in_w);
  
  // std::cout << "filt_data:" << std::endl;
  // print(filt_data, filt_k, filt_c, filt_h, filt_w);
  // std::cout << "Thread " << world_rank << " has: " << std::endl;
  // std::cout << "out_data:" << std::endl;
  // std::cout << "out_data size: " 
  // << out_n 
  // << "*"
  // << out_c 
  // << "*" 
  // << out_h 
  // << "*" 
  // << out_w << std::endl;
  //print(out_data, out_n, out_c, out_h, out_w);
  float *h_out_data = new float [out_n * out_c * out_h * out_w * sizeof(float)];
  CUDA_CALL(cudaMemcpy(h_out_data,out_data,out_n * out_c * out_h * out_w * sizeof(float),cudaMemcpyDeviceToHost));
  
  // for(int i=0;i<out_n * out_c * out_h * out_w;++i) {
  //   std::cout<<h_out_data[i]<<" ";
  //   if((i+1)%out_w==0) std::cout<<std::endl;
  // }


  std::cout << "The time cost of thread " << world_rank << " is: " << end - start
  << std::endl;
  // finalizing
  CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroyPoolingDescriptor(pooling_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));

  #ifdef PAPI

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
