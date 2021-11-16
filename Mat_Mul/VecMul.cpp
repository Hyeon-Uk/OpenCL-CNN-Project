#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include<time.h>
#include<iostream>
#include<chrono>

using namespace std;
using namespace std::chrono;

#define VECTOR_SIZE 1000

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
		if (buf[0] == '\n') cnt ++ ;
	}
	source_code[length - cnt] = '\0';
	fclose(file);
	*len = length - cnt;
	return source_code;
}

int main() {
	srand((unsigned)time(NULL));
	cl_int err;
	cl_uint i,j,k,num_platforms,ROW_A,COL_A,COL_B;
	cl_platform_id *platforms;
	system_clock::time_point start, end;
	nanoseconds nano;

	//GetPlatforms
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);

	platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
	err=clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err);


	//GetDevices
	cl_device_id *devices,device;
	cl_uint num_devices;
	err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	CHECK_ERROR(err);

	devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
	err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	CHECK_ERROR(err);

	//CreateContext
	cl_context context;
	device = devices[1];
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);


	//Create Command Queue
	cl_command_queue *queues,queue;
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	CHECK_ERROR(err);


	//Get SourceCode
	char *kernel_source;
	size_t kernel_source_size;
	kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	
	//Create Program
	cl_program program;
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);

	//Build Program
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	CHECK_ERROR(err);

	//Create Kernel Object
	cl_kernel kernel_mat_mul;
	kernel_mat_mul = clCreateKernel(program, "matmul", &err);
	CHECK_ERROR(err);

	//Create Vector A,B,C
	float *A, *B, *C, *ANSWER;
	ROW_A = VECTOR_SIZE;
	COL_A = VECTOR_SIZE;
	COL_B = VECTOR_SIZE;
	A = (float *)malloc(sizeof(float)*ROW_A*COL_A);
	B = (float *)malloc(sizeof(float)*COL_A*COL_B);
	C = (float *)malloc(sizeof(float)*ROW_A*COL_B);
	ANSWER = (float *)malloc(sizeof(float)*ROW_A*COL_B);

	//Random Value Setting
	for (i = 0; i < ROW_A; i++) {
		for (j = 0; j < COL_B; j++) {
			*(A+(i*COL_A + j)) = (float)(rand() % 1000 / 100.f);
		}
	}
	for (i = 0; i < COL_A; i++) {
		for (j = 0; j < COL_B; j++) {
			*(B+(i*COL_B + j)) = (float)(rand() % 1000 / 100.f);
		}
	}

	start = system_clock::now();
	for (i = 0; i < ROW_A; i++) {
		for (j = 0; j < COL_B; j++) {
			*(ANSWER + (i*COL_B + j)) = 0.0f;
			for (k = 0; k < COL_A; k++) {
				*(ANSWER + (i*COL_B + j)) += *(A + (i*COL_A + k))* *(B + (k*COL_B + j));
			}
		}
	}
	end = system_clock::now();
	nano = end - start;
	cout << "On CPU = " << nano.count() << "ns\n";


	//Create Buffer Object
	cl_mem bufA, bufB, bufC;
	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(float)*ROW_A*COL_A,NULL,&err);
	CHECK_ERROR(err);
	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*COL_A*COL_B, NULL, &err);
	CHECK_ERROR(err);
	bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*ROW_A*COL_B, NULL, &err);
	CHECK_ERROR(err);


	//Write Buffer
	err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(float)*ROW_A*COL_A, A, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeof(float)*COL_A*COL_B, B, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, sizeof(float)*ROW_A*COL_B, C, 0, NULL, NULL);
	CHECK_ERROR(err);

	//Set Kernel Arg
	err = clSetKernelArg(kernel_mat_mul, 0, sizeof(bufA), &bufA);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 1, sizeof(bufB), &bufB);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 2, sizeof(bufC), &bufC);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 3, sizeof(int), &ROW_A);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 4, sizeof(int), &COL_A);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_mat_mul, 5, sizeof(int), &COL_B);
	CHECK_ERROR(err);


	//Excute Kernel
	size_t global_size[2] = { COL_B,ROW_A };
	size_t local_size[2] = {100,100};

	start = system_clock::now();
	clEnqueueNDRangeKernel(queue, kernel_mat_mul, 2, NULL, global_size, local_size, 0, NULL, NULL);
	end = system_clock::now();
	nano = end - start;
	cout << "On GPU = " << nano.count() << "ns\n";

	//Read Buffer
	err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float)*ROW_A*COL_B, C, 0, NULL, NULL);
	CHECK_ERROR(err);

	//Result Accuracy
	int correct = 0;
	for (i = 0; i < ROW_A; i++) {
		for (j = 0; j < COL_B; j++) {
			if (ANSWER[i*COL_B + j] - C[i*COL_B + j] < 0.01f || C[i*COL_B + j] - ANSWER[i*COL_B + j] < 0.01f) correct++;
		}
	}
	printf("Accuracy = %0.3lf %%\n", 100*(double)correct / (ROW_A*COL_B));

	return 0;
}