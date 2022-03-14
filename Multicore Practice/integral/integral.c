#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include<time.h>

#define VECTOR_SIZE 16777216
#define LOCAL_SIZE 256

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char *get_source_code(const char *file_name, size_t *len) {
	char *source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	FILE *file = fopen(file_name, "r");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char *)malloc(length + 1);
	fread(source_code, length, 1, file);

	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') cnt++;
	}
	source_code[length - cnt] = '\0';
	fclose(file);
	*len = length - cnt;
	return source_code;
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

double f(double x) {
	return (double)(3.0 * x*x + 2.0 * x + 1.0);
}



int main() {
	srand(time(NULL));
	int SIZE=538870912;
	cl_int err, i, j;
	clock_t start, end;
	cl_event read_event;

	//GetPlatforms
	cl_uint num_platforms;
	cl_platform_id *platforms;

	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);

	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);

	show_info(num_platforms, platforms);

	int s_platform, s_device;
	printf("\n\nSelect Platform id : ");
	scanf("%d", &s_platform);
	printf("\n\nSelect Device id : ");
	scanf("%d", &s_device);


	printf("expression = 3*x*x + 2*x + 1\n");
	printf("range 0~1\n");
	printf("N = 538,870,912\n");
	//GetDevices
	cl_device_id *devices, device;
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
	cl_command_queue *queues, queue;
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	CHECK_ERROR(err);


	//Get SourceCode
	char *kernel_source;
	size_t kernel_source_size;
	kernel_source = get_source_code("int_kernel.cl", &kernel_source_size);

	//Create Program
	cl_program program;
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	//Build Program
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	CHECK_ERROR(err);

	//Create Kernel Object
	cl_kernel kernel_integral;
	kernel_integral = clCreateKernel(program, "integral", &err);
	CHECK_ERROR(err);

	//Create Vector A,B
	double *A;
	A = (double *)malloc(sizeof(double)*(SIZE/LOCAL_SIZE));
	double answer = 0, xpos;
	double dx = (1.0 / (double)SIZE);
	start = clock();
	for (xpos = 0; xpos < 1; xpos += dx) {
		answer += dx*f(xpos);
	}
	end = clock();
	printf("On CPU = %f sec\n", (double)((end - start) / CLK_TCK));


	//Create Buffer Object
	cl_mem bufA;
	bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)*(SIZE/LOCAL_SIZE), NULL, &err);
	CHECK_ERROR(err);


	//Write Buffer
	err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(double)*(SIZE / LOCAL_SIZE), A, 0, NULL,NULL);
	CHECK_ERROR(err);

	//Set Kernel Arg
	err = clSetKernelArg(kernel_integral, 0, sizeof(cl_mem), (void *)&bufA);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_integral, 1, LOCAL_SIZE * sizeof(double), NULL);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_integral, 2, sizeof(cl_int), (void *)&SIZE);

	//Excute Kernel
	size_t global_size = SIZE;
	size_t local_size = LOCAL_SIZE;

	clEnqueueNDRangeKernel(queue, kernel_integral, 1, NULL, &global_size, &local_size, 0, NULL, &read_event);

	//Read Buffer
	err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0, (SIZE) / (LOCAL_SIZE) * sizeof(double), (void *)A, 0, NULL, NULL);
	CHECK_ERROR(err);

	//Result Accuracy
	double compare = 0.0;
	for (int i = 0; i < SIZE / LOCAL_SIZE; i++) compare += A[i];

	clFinish(queue);
	cl_ulong k_start, k_end;
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &k_start, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &k_end, NULL);
	
	printf("On GPU = %lf s\n", (double)(k_end - k_start)/1000000000);
	printf("On CPU Result = %lf\nOn GPU Result = %lf\n", answer, compare);

	return 0;
}
