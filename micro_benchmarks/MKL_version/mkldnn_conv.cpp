#include <iostream>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <math.h>
#include <mpi.h>
#include "mkldnn.hpp"
#include "./data_load.h"

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
    double start = 0.0;
    double end = 0.0;
    // Initialize Stream
    stream cpu_stream(cpu_engine);

    // generate test data
    // memory::dims input_size = {1, 3, 227, 227}; // batch, channel, H , W
    // memory::dims kernel_size = {1, 3, 11, 11};// OC, IC, H, W
    // memory::dims bias_size = {1}; // to each filter
    // memory::dims conv1_strides = { 4, 4 };
    // memory::dims conv1_padding = { 0, 0 };
    // memory::dims dst_size = {1, 1, 55, 55};

    memory::dims input_size = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};  // batch, channel, H , W
    memory::dims kernel_size = {atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8])}; // OC, IC, H, W
    memory::dims bias_size = {atoi(argv[5])};                                                // to each filter
    memory::dims conv1_strides = {atoi(argv[9]), atoi(argv[10])};
    memory::dims conv1_padding = {atoi(argv[11]), atoi(argv[12])};
    memory::dims dst_size = {atoi(argv[1]), atoi(argv[5]), (atoi(argv[3]) - atoi(argv[7]) + 2 * atoi(argv[11])) / atoi(argv[9]) + 1, (atoi(argv[4]) - atoi(argv[8]) + 2 * atoi(argv[12])) / atoi(argv[10]) + 1};
    
    // string file_name = file_list[world_rank];
    // vector<float> image(product(input_size));
    // float *image = new float[product(input_size)];
    vector<float> conv_weights(product(kernel_size));
    vector<float> conv_bias(product(bias_size));
    res out = data_load(world_rank);
    float *h_in_data = out.out;
    int len = out.length;
    vector<float> image(h_in_data, h_in_data+len);

    // initialize data
    // for data
    // for (int i = 0; i < product(input_size); ++i)
    // {
    //     image[i] = rand() % 5;
    // }
    //for weights and bias
    for (int i = 0; i < conv_weights.size(); ++i)
    {
        conv_weights[i] = rand() % 3;
    }
    for (int i = 0; i < conv_bias.size(); ++i)
    {
        conv_bias[i] = rand() % 2;
    }

    // create memory descriptor for user data
    auto conv_src_md = memory::desc(
        {input_size},
        memory::data_type::f32,
        memory::format_tag::nchw);
    auto conv_bias_md = memory::desc(
        {bias_size},
        memory::data_type::f32,
        memory::format_tag::x);
    auto conv_weights_md = memory::desc(
        {kernel_size},
        memory::data_type::f32,
        memory::format_tag::oihw);
    auto conv_des_md = memory::desc(
        {dst_size},
        memory::data_type::f32,
        memory::format_tag::any);

    // create memory
    auto src_mem = memory(conv_src_md, cpu_engine, image.data());
    auto weis_mem = memory(conv_weights_md, cpu_engine, conv_weights.data());
    auto bias_mem = memory(conv_bias_md, cpu_engine, conv_bias.data());
    // auto conv_dst_mem = memory(conv_des_md, cpu_engine);

    // create conv descriptor and conv primitive descriptor
    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
                                               algorithm::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                                               conv_des_md, conv1_strides, conv1_padding, conv1_padding);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, cpu_engine);

    auto conv_dst_mem = memory(conv_pd.dst_desc(), cpu_engine);
    auto conv = convolution_forward(conv_pd);
    start = MPI_Wtime();
    conv.execute(
        cpu_stream,
        {{MKLDNN_ARG_SRC, src_mem},
         {MKLDNN_ARG_WEIGHTS, weis_mem},
         {MKLDNN_ARG_BIAS, bias_mem},
         {MKLDNN_ARG_DST, conv_dst_mem}});

    // Wait the stream to complete the execution
    cpu_stream.wait();
    end = MPI_Wtime();
    // float *conv_res = static_cast<float *>(conv_dst_mem.get_data_handle());
    cout << "Time cost is " << end - start << endl;
    // // Check the results
    // for ( int i = 0; i < product(dst_size); ++i){
    //     cout << conv_res[i] << setw(4);
    //     if(i > 0 && i % 55 == 0){
    //         cout << endl;
    //     }
    // }
    MPI_Finalize();
    return 0;
}
