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
    // memory::dims kernel_size = {1024, 3, 55, 55};// H, W
    // memory::dims bias_size = {1024};
    // memory::dims dst_size = {1, 1024};

    memory::dims input_size = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};  // batch, channel, H , W
    memory::dims kernel_size = {atoi(argv[5]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4])}; // H, W
    memory::dims bias_size = {atoi(argv[5])};
    memory::dims dst_size = {atoi(argv[1]), atoi(argv[5])};

    // vector<float> image(product(input_size));
    vector<float> fc_weights(product(kernel_size));
    vector<float> fc_bias(product(bias_size));

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
    for (int i = 0; i < fc_weights.size(); ++i)
    {
        fc_weights[i] = rand() % 3 - 1;
    }
    for (int i = 0; i < fc_bias.size(); ++i)
    {
        fc_bias[i] = rand() % 2;
    }

    // create memory descriptor for user data
    auto fc_src_md = memory::desc(
        {input_size},
        memory::data_type::f32,
        memory::format_tag::nchw);

    auto fc_des_md = memory::desc(
        {dst_size},
        memory::data_type::f32,
        memory::format_tag::any);

    auto fc_bias_md = memory::desc(
        {bias_size},
        memory::data_type::f32,
        memory::format_tag::x);

    auto fc_weights_md = memory::desc(
        {kernel_size},
        memory::data_type::f32,
        memory::format_tag::oihw);

    // create memory
    auto src_mem = memory(fc_src_md, cpu_engine, image.data());
    auto fc_weights_mem = memory(fc_weights_md, cpu_engine,
                                 fc_weights.data());
    auto fc_bias_mem = memory(
        fc_bias_md, cpu_engine, fc_bias.data());

    // create fc descriptor and fc primitive descriptor
    auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference,
                                               fc_src_md, fc_weights_md, fc_bias_md, fc_des_md);

    auto fc_pd = inner_product_forward::primitive_desc(fc_desc, cpu_engine);

    auto fc_dst_mem = memory(fc_pd.dst_desc(), cpu_engine);
    auto fc = inner_product_forward(fc_pd);

    double start = MPI_Wtime();
    fc.execute(
        cpu_stream,
        {{MKLDNN_ARG_SRC, src_mem},
         {MKLDNN_ARG_WEIGHTS, fc_weights_mem},
         {MKLDNN_ARG_BIAS, fc_bias_mem},
         {MKLDNN_ARG_DST, fc_dst_mem}});
    double end = MPI_Wtime();
    // Wait the stream to complete the execution
    cpu_stream.wait();

    float *fc_res = static_cast<float *>(fc_dst_mem.get_data_handle());

    // Check the results
    // for ( int i = 0; i < product(dst_size); ++i){
    //     cout << fc_res[i] << setw(4);
    //     if(i > 0 && i % 27 == 0){
    //         cout << endl;
    //     }
    // }
    cout << "Time cost is " << end - start << endl;
    MPI_Finalize();

    return 0;
}
