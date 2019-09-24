#ifndef DATA_LOAD_H    
#define DATA_LOAD_H

#include <stdio.h>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include "../include/H5Cpp.h"
#include <cuda_fp16.h>

using namespace H5;
using namespace std;



struct res{
    float *out;
    unsigned long long length;
};

res data_load(int rank);

#endif
