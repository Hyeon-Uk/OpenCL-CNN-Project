#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include "rotation.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<time.h>

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    };
}

void show_info(cl_uint num_platforms, cl_platform_id *platforms) {
	cl_uint num_devices;
	cl_device_id *devices;
	char str[1024];
	cl_device_type device_type;
	size_t max_work_group_size;
	cl_ulong global_mem_size;
	cl_ulong local_mem_size;
	cl_ulong max_mem_alloc_size;
	cl_uint p, d;
	cl_int err;

	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);

	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);
	printf("-----------------SHOW INFO---------------------\n\n");
	printf("Number of platforms: %u\n\n", num_platforms);
	for (p = 0; p < num_platforms; p++)
	{
		printf("platform: %u\n", p);

		err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, str, NULL);
		CHECK_ERROR(err);
		printf("- CL_PLATFORM_NAME\t:%s\n", str);

		err = clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 1024, str, NULL);
		CHECK_ERROR(err);
		printf("- CL_PLATFORM_VENDOR\t:%s\n\n", str);

		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		CHECK_ERROR(err);
		printf("Number of devices:\t%u\n\n", num_devices);

		devices = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
		CHECK_ERROR(err);

		for (d = 0; d < num_devices; d++)
		{
			printf("device: %u\n", d);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_TYPE\t:");
			if (device_type & CL_DEVICE_TYPE_CPU) printf("  CL_DEVICE_TYPE_CPU");
			if (device_type & CL_DEVICE_TYPE_GPU) printf("  CL_DEVICE_TYPE_GPU");
			if (device_type & CL_DEVICE_TYPE_ACCELERATOR) printf("  CL_DEVICE_TYPE_ACCELERATOR");
			if (device_type & CL_DEVICE_TYPE_DEFAULT) printf("  CL_DEVICE_TYPE_DEFAULT");
			if (device_type & CL_DEVICE_TYPE_CUSTOM) printf("  CL_DEVICE_TYPE_CUSTOM");
			printf("\n");

			err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, str, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_NAME\t: %s\n", str);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_MAX_WORK_GROUP_SIZE : %lu\n", max_work_group_size);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_GLOBAL_MEM_SIZE : %lu\n", global_mem_size);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_LOCAL_MEM_SIZE : %lu\n", local_mem_size);

			err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL);
			CHECK_ERROR(err);
			printf("- CL_DEVICE_MAX_MEM_ALLOC_SIZE : %lu\n\n", local_mem_size);

		}

		free(devices);
	}

	free(platforms);
	printf("--------------------------------------------------\n\n");
}

void image_rotate_on_cpu(const float* input, float* output, const int W, const int H, char* _theta) {
	int dest_x, dest_y;
	float x0 = W / 2.0f;
	float y0 = H / 2.0f;
	const float theta = atof(_theta) * M_PI / 180;
	const float sin_theta = sinf(theta);
	const float cos_theta = cosf(theta);
	for (dest_y = 0; dest_y < H; dest_y++) {
		for (dest_x = 0; dest_x < W; dest_x++) {
			float xOff = dest_x - x0;
			float yOff = dest_y - y0;
			int src_x = (int)(xOff * cos_theta + yOff * sin_theta + x0);
			int src_y = (int)(yOff * cos_theta - xOff * sin_theta + y0);
			if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H))
				output[dest_y * W + dest_x] = input[src_y * W + src_x];
			else
				output[dest_y * W + dest_x] = 0.0f;
		}
	}
}

void rotate(const float* input, float* output, const int width, const int height, char *degree) {
    const float theta = atof(degree) * M_PI / 180;
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);
	clock_t start, end;
	float res;

    cl_int err;

	//GetPlatforms
	cl_uint num_platforms;
	cl_platform_id *platforms;
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);

	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
	err=clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);

	show_info(num_platforms, platforms);

	int s_platform, s_device;
	printf("\n\nSelect Platform id : ");
	scanf("%d", &s_platform);
	printf("\n\nSelect Device id : ");
	scanf("%d", &s_device);

	//GetDevices
	cl_device_id *devices,device;
	cl_uint num_devices;
	err = clGetDeviceIDs(platforms[s_platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	CHECK_ERROR(err);

	devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
	err = clGetDeviceIDs(platforms[s_platform], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	CHECK_ERROR(err);

	//CreateContext
	cl_context context;
	device = devices[s_device];
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);


	//Create Command Queue
	cl_command_queue *queues,queue;
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);

    // Create Program Object
    size_t kernel_source_size;
    char* kernel_source = get_source_code("kernel.cl", &kernel_source_size);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    // Build Program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    build_error(program, device, err);
    CHECK_ERROR(err);

    /*
     * 여기서부터 병렬 처리를 위한 호스트 코드를 작성하세요.
     */

	//Build Kernel Object
	cl_kernel kernel_rotation_img;
	kernel_rotation_img = clCreateKernel(program, "imgRotation", &err);
	CHECK_ERROR(err);


	//Create Vector ANSWER
	float *ANSWER;
	ANSWER = (float *)malloc(sizeof(float)*width*height);

	//Run on CPU
	start = clock();

	image_rotate_on_cpu(input, ANSWER, width, height, degree);

	end = clock();
	res = (float)(end - start) / CLOCKS_PER_SEC;
	printf("On CPU = %.3f\n", res);

	//Build Buffer Object
	cl_mem bufInput, bufOutput;
	bufInput = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*width*height, NULL, &err);
	CHECK_ERROR(err);
	bufOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*width*height, NULL, &err);
	CHECK_ERROR(err);

	//Write Buffer
	err = clEnqueueWriteBuffer(queue, bufInput, CL_TRUE, 0, sizeof(float)*width*height, input, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufOutput, CL_TRUE, 0, sizeof(float)*width*height, output, 0, NULL, NULL);
	CHECK_ERROR(err);

	//Set Kernel Arg
	err = clSetKernelArg(kernel_rotation_img, 0, sizeof(bufInput), &bufInput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_rotation_img, 1, sizeof(bufOutput), &bufOutput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_rotation_img, 2, sizeof(int), &width);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_rotation_img, 3, sizeof(int), &height);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_rotation_img, 4, sizeof(int), &sin_theta);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_rotation_img, 5, sizeof(int), &cos_theta);
	CHECK_ERROR(err);


	//Excute Kernel
	size_t global_size[2] = { width,height };
	size_t local_size[2] = { 8,8 };

	start = clock();
	clEnqueueNDRangeKernel(queue, kernel_rotation_img, 2, NULL, global_size, local_size, 0, NULL, NULL);
	end = clock();
	res = (float)(end - start) / CLOCKS_PER_SEC;
	printf("On GPU = %.3f\n", res);

	//Read Buffer
	err = clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, sizeof(float)*width*height, output, 0, NULL, NULL);
	CHECK_ERROR(err);
 }