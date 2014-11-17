#define EXPORT extern "C"

EXPORT int cuconv(double* filter,double* full_image, double* feature, int conv_dim, int filter_dim, int image_dim,int sample_num);
EXPORT int cuconvcube(double* filter,double* full_image, double* feature, int conv_dim, int filter_dim, int image_dim,int sample_num);
EXPORT void get_device_info();