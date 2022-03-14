#pragma once

#ifndef __ROTATION__
#define __ROTATION__

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

#include <CL/cl.h>
#include <stdlib.h>


char* get_source_code(const char* file_name, size_t* len);
void build_error(cl_program program, cl_device_id device, cl_int err);
void rotate(const float* input, float* output, const int width, const int height, char* degree);
void image_rotate_on_cpu(const float* input, float* output, const int width, const int height, char* degree);
void show_info(cl_uint num_platforms, cl_platform_id *platforms, cl_uint num_devices, cl_device_id *devices);
#endif