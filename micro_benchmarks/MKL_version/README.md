###ENV REQUIREMENT:
```
{
    mkldnn:0.95
    OPENMPI:3.1.2
}
```

### COMPILE COMMOND:
```
g++ -std=c++11 -I /usr/local/.openmpi_install/include -I /usr/local/h
df5/include/ -L /usr/local/hdf5/lib/ -L /usr/local/.openmpi_install/lib/ -L /usr/local/lib -lhdf5 -lhdf5_cpp -lmpi -lmkldnn mkldnn_conv.cpp data_load.cpp -o mkcc_test -Wl,-rpath /usr/local/hdf5/lib/ -Wl,-rpath /usr/local/lib64/
```
"mkldnn_test.cpp" could change to the file you want to compile, also the output file's name

###RUNNING PARAMETERS:
Before running the program, you need to modify the file list in the data_load.h
```
# mpirun is running with mpi and the -np indicates the number of threads you want. ( should less or equal to the GPU number you have)
# For convolution:
# mc_test is the executable file you compiled
# the 12 parameters are the size of input data with NCHW format, the filter size with OIHW format, the strides, the paddings
# input size should equal to the size of input data
mpirun -np 1 ./mc_test 1 3 227 227 1 3 11 11 4 4 0 0


# For pooling:
# mp_test is the executable file you compiled
# the 11 parameters are the size of input data with NCHW format, the filter size, the padding and the strides, and the mode of pooling(0 for max pooling, 1 for average)
# input size should equal to the size of input data
mpirun -np 1 ./mp_test 1 3 55 55 3 3 2 2 0 0 1


# For fully connected operate:
# mfc_test is the executable file you compiled
# the 5 parameters are the size of input data with NCHW format,  and the output channels 
# so the output size is N * OUT_C
# input size should equal to the size of input data
mpirun -np 1 ./mfc_test 10 13 155 155 2048


```