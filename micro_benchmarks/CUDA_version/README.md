### ENV REQUIREMENT:
```
{
    cuda:9.0
    CUDNN:7.4.1
    OPENMPI:3.1.2
    PAPI:5.7.0
    hdf5:1.10.4
}
```

### ENV INSTALL:
- papi:https://icl.utk.edu/papi/software/index.html
- hdf5(cpp):https://support.hdfgroup.org/HDF5/doc/cpplus_RM/index.html

### FILE DESCRIPTION:
data_load: load data from h5 file.
cudnn_conv*: run convolution operator.
cudnn_fc*: run fully connected operator.
cudnn_pooling*: run pooling operator.
file name with 16 means runing operator with FP16 mode.
in the fc file: you can comment 
```
 cublasstat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
```
in the conv file, it is:
```
  CUDNN_CALL( cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) );
```
to control the tensor core open or not

### COMPILE COMMOND:
```
g++ -std=c++11 -I /usr/local/cuda/include -I /usr/local/.openmpi_install/include -I /usr/local/hdf5/include/ -I papi-5.7.0/src/testlib/ -I papi-5.7.0/src/ -I papi-5.7.0/src/validation_tests/ -L /usr/local/hdf5/lib/  -L /usr/local/cuda/lib64 -L /usr/local/.openmpi_install/lib/ -L /usr/local/lib -L /usr/local/cuda-9.0/extras/CUPTI/lib64/  -lhdf5 -lhdf5_cpp -lmpi -lcuda -lcudart -lcudnn -lcublas -ldl -lcupti cudnn_convolution_papi.cpp data_load.cpp papi-5.7.0/src/testlib/libtestlib.a papi-5.7.0/src/libpapi.a -o cpapi_test -Wl,-rpath /usr/local/hdf5/lib/ -Wl,-rpath /usr/local/cuda-9.0/extras/CUPTI/lib64/
```
- if not using papi, you can delete the papi part in the command.
- "cudnn_pooling_forward.cpp" could change to the file you want to compile, also the output file's name.
- if the file ends with .cu please use nvcc to compile it. Using the commands below:
```
nvcc -std=c++11 -I /usr/local/include/ -I /usr/local/cuda/include -I ../include/  -L ../lib/ -L /usr/local/cuda/lib64 -L /usr/local/lib/ cudnn_fc16.cu data_load.cpp  -lhdf5 -lhdf5_cpp -lmpi -lcuda -lcudart -lcudnn -lcublas  -o fc16_test
```


### RUNNING PARAMETERS:
Before running the program, you need to modify the file list in the data_load.cpp
the data we used is a part from https://extremeweatherdataset.github.io/
```
 mpirun is running with mpi and the -np indicates the number of threads you want. ( should less or equal to the GPU number you have)
 For convolution:
 c_test is the executable file you compiled
 the 14 parameters are the size of input data with NCHW format, the filter size with OIHW format, the paddings, the strides and the dilations
 input size should equal to the size of input data
mpirun -np 1 ./c_test 5 16 768 1152 1 16 2 2 0 0 1 1 1 1

For pooling:
 p_test is the executable file you compiled
 the 11 parameters are the size of input data with NCHW format, the filter size, the padding and the strides, and the mode of pooling(0 for max pooling, 1 for average)
  input size should equal to the size of input data
 mpirun -np 1 ./p_test 5 16 768 1152 5 5 0 0 1 1 0

 For fully connected operate:
 fc_test is the executable file you compiled
 the 5 parameters are the size of input data with NCHW format,  and the output channels 
 so the output size is N * OUT_C
 input size should equal to the size of input data
mpirun -np 1 ./pfc_test 5 16 768 1152 16

```
