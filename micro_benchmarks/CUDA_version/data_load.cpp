#include "data_load.h"
static string file_list[] = {"out_for_test.h5","out_for_test.h5","out_for_test.h5","out_for_test.h5","out_for_test.h5","out_for_test.h5","out_for_test.h5","out_for_test.h5"};

res data_load(int rank){
    string file_name = "out_for_test.h5";
    const H5std_string FILE_NAME(file_name);
    H5File file(FILE_NAME, H5F_ACC_RDONLY);
    DataSet images = file.openDataSet("images");
    H5T_class_t type_class = images.getTypeClass();
    DataSpace dataspace = images.getSpace();
    hsize_t dims[4];
    dataspace.getSimpleExtentDims(dims, NULL);
    //cout << dims[0] << dims[1] << dims[2] << dims[3] << endl;
    unsigned long long len = dims[0]*dims[1]*dims[2]*dims[3];
    float *out = new float[len];
    // memset(out, 0, len);
    images.read(out, PredType::NATIVE_FLOAT, dataspace, dataspace);
    return {out,len};
}

// int main(){
//     string file_name = "/home/zyc/weather_data_set_mini/climo_1981_imgs_2.h5";
//     auto [des, length] = data_load(file_name); 
//     cout << *(des + 0) << *(des + 1) << *(des + 2) << endl;
//     cout << endl;
//     cout << length << endl;
// }
