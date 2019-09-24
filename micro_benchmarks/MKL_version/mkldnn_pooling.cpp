#include <iostream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <mpi.h>
#include "mkldnn.hpp"
// #include "./data_load.h"

using namespace mkldnn;
using namespace std;

memory::dim product(const memory::dims &dims)
{
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
                           std::multiplies<memory::dim>());
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;

    // get the current thread rank
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // get the size of usable threads of communicator
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Initialize CPU
    engine cpu_engine(engine::kind::cpu, world_rank);

    // Initialize Stream
    stream cpu_stream(cpu_engine);

    // generate test data
    // memory::dims input_size = {1, 3, 55, 55}; // batch, channel, H , W
    // memory::dims kernel_size = {3, 3};// H, W
    // memory::dims pooling_strides = { 2, 2 };
    // memory::dims pooling_padding = { 0, 0 };
    // memory::dims dst_size = {1, 3, 27, 27};

    memory::dims input_size = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4])}; // batch, channel, H , W
    memory::dims kernel_size = {atoi(argv[5]), atoi(argv[6])};                              // H, W
    memory::dims pooling_strides = {atoi(argv[7]), atoi(argv[8])};
    memory::dims pooling_padding = {atoi(argv[9]), atoi(argv[10])};
    memory::dims dst_size = {atoi(argv[1]), atoi(argv[2]), (atoi(argv[3]) - atoi(argv[5])) / atoi(argv[7]) + 1, (atoi(argv[4]) - atoi(argv[6])) / atoi(argv[8]) + 1};

    // vector<float> image(product(input_size));
    vector<float> pooling_weights(product(kernel_size));

    // initialize data
    // for data
    // for (int i = 0; i < image.size(); ++i)
    // {
    //     image[i] = rand() % 5;
    // }
    res out = data_load(world_rank);
    float *h_in_data = out.out;
    int len = out.length;
    vector<float> image(h_in_data, h_in_data+len);

    // create memory descriptor for user data
    auto pooling_src_md = memory::desc(
        {input_size},
        memory::data_type::f32,
        memory::format_tag::nchw);

    auto pooling_des_md = memory::desc(
        {dst_size},
        memory::data_type::f32,
        memory::format_tag::any);

    // create memory
    auto src_mem = memory(pooling_src_md, cpu_engine, image.data());
    int pooling_choice = std::atoi(argv[11]);
    algorithm algo;
    if (pooling_choice == 0)
    {
        algo = algorithm::pooling_max;
    }
    else if (pooling_choice == 1)
    {
        algo = algorithm::pooling_avg;
    }

    // create pooling descriptor and pooling primitive descriptor
    auto pooling_desc = pooling_forward::desc(prop_kind::forward_inference,
                                              algo, pooling_src_md, pooling_des_md,
                                              pooling_strides, kernel_size, pooling_padding, pooling_padding);
    auto pooling_pd = pooling_forward::primitive_desc(pooling_desc, cpu_engine);

    auto pooling_dst_mem = memory(pooling_pd.dst_desc(), cpu_engine);
    auto pooling = pooling_forward(pooling_pd);
    double start = MPI_Wtime();
    pooling.execute(
        cpu_stream,
        {{MKLDNN_ARG_SRC, src_mem},
         {MKLDNN_ARG_DST, pooling_dst_mem}});

    // Wait the stream to complete the execution
    cpu_stream.wait();
    double end = MPI_Wtime();

    float *pooling_res = static_cast<float *>(pooling_dst_mem.get_data_handle());

    // Check the results
    // for ( int i = 0; i < product(dst_size); ++i){
    //     cout << pooling_res[i] << setw(4);
    //     if(i > 0 && i % 27 == 0){
    //         cout << endl;
    //     }
    // }
    cout << "Time cost is " << end - start << endl;
    MPI_Finalize();
    return 0;
}
