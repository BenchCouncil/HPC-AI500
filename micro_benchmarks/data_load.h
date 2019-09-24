#ifndef DATA_LOAD_H    
#define DATA_LOAD_H

#include <stdio.h>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include "/usr/local/hdf5/include/H5Cpp.h"

using namespace H5;
using namespace std;

string file_list[] = {"/home/zhangyuchen/out_for_test.h5","",""};

struct res{
    float *out;
    int length;
};

res data_load(int rank);

#endif