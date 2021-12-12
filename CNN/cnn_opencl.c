#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>


#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL err %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }
#define TS 16

extern const char* CLASS_NAME[];

const int INPUT_DIM[] = {
    3, 64,
    64,

    64,128,
    128,

    128, 256, 256,
    256,

    256, 512, 512,
    512,

    512, 512, 512,
    512,

    512,
    512,
    512
};

const int OUTPUT_DIM[] = {
    64, 64,
    64,

    128, 128,
    128,

    256, 256, 256,
    256,

    512, 512, 512,
    512,

    512, 512, 512,
    512,

    512,
    512,
    10
};

const int NBYN[] = {
    32, 32,
    16,

    16, 16,
    8,

    8, 8, 8,
    4,

    4, 4, 4,
    2,

    2, 2, 2,
    1,

    1,
    1,
    1
};

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue,readQueue,writeQueue;
cl_program program;
char* kernel_source;
size_t kernel_source_size;
cl_kernel kernel_conv, kernel_conv_ex, kernel_pool, kernel_fclayer;
cl_int err;
int i_offset, f_offset;
cl_mem bufImg, bufFilter, bufConvInput, bufConvOutput, bufConvExtend, bufPoolInput, bufPoolOutput,
bufFCInput, bufFCOutput;

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

void build_err(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler err:\n%s\n", log);
        free(log);
        exit(0);
    };
}


void cnn_init(void) {
    // Get Platform ID
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    //Get Device ID
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    //Create Context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    //Create Command Queue
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);
    readQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);
    writeQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);

    //Get Kernel Source
    kernel_source = get_source_code("kernel.cl", &kernel_source_size);

    //Build Program
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source,&kernel_source_size, &err);
    CHECK_ERROR(err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    build_err(program, device, err);
    CHECK_ERROR(err);


    //Create Kernel
    kernel_conv = clCreateKernel(program, "conv", &err);
    CHECK_ERROR(err);
    kernel_conv_ex = clCreateKernel(program, "conv_ex", &err);
    CHECK_ERROR(err);
    kernel_pool = clCreateKernel(program, "pool", &err);
    CHECK_ERROR(err);
    kernel_fclayer = clCreateKernel(program, "fclayer", &err);
    CHECK_ERROR(err);
}

void convolution(cl_mem* inputs, cl_mem* outputs, cl_mem* networks, int input_dim, int output_dim, int nbyn) {
    //Set Group Size
    size_t global_size[2] = { nbyn * nbyn * output_dim,1 };
    size_t local_size[2] = { 256,1 };

    //Set Kernel Arg
    err = clSetKernelArg(kernel_conv_ex, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv_ex, 1, sizeof(cl_mem), &bufConvExtend);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv_ex, 2, sizeof(cl_int), &output_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv_ex, 3, sizeof(cl_int), &nbyn);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv_ex, 4, sizeof(cl_int), &i_offset);
    CHECK_ERROR(err);

    //Enqueue
    err = clEnqueueNDRangeKernel(queue, kernel_conv_ex, 1, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_ERROR(err);

    global_size[0] = nbyn * nbyn;
    global_size[1] = output_dim;
    local_size[0] = TS;
    local_size[1] = TS;

    if (global_size[0] < TS) global_size[0] = TS;

    err = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), &bufConvExtend);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 2, sizeof(cl_mem), networks);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 3, sizeof(int), &input_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 4, sizeof(int), &output_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 5, sizeof(int), &nbyn);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_conv, 6, sizeof(int), &f_offset);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, kernel_conv, 2, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_ERROR(err);

}

void max_pooling(cl_mem* inputs, cl_mem* outputs, int input_dim, int nbyn) {
    //Set Kernel Arg
    err = clSetKernelArg(kernel_pool, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_pool, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_pool, 2, sizeof(cl_int), &input_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_pool, 3, sizeof(cl_int), &nbyn);
    CHECK_ERROR(err);
    
    //Set Group Size
    int output_nbyn = nbyn / 2;
    size_t global_size[2] = { input_dim * output_nbyn , output_nbyn };
    size_t local_size[2] = { TS,TS };

    err = clEnqueueNDRangeKernel(queue, kernel_pool, 2, NULL, global_size, NULL, 0, NULL, NULL);
}

void fc_layer(cl_mem* inputs, cl_mem* outputs, cl_mem* networks, int input_dim, int output_dim) {
    //Set Kernel Arg
    err = clSetKernelArg(kernel_fclayer, 0, sizeof(cl_mem), inputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fclayer, 1, sizeof(cl_mem), outputs);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fclayer, 2, sizeof(cl_mem), networks);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fclayer, 3, sizeof(cl_int), &input_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fclayer, 4, sizeof(cl_int), &output_dim);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fclayer, 5, sizeof(cl_int), &f_offset);
    CHECK_ERROR(err);

    //Set Group Size
    size_t global_size = output_dim < input_dim ? 256 : 512;
    size_t local_size = 256;
    //Enqueue
    err = clEnqueueNDRangeKernel(queue, kernel_fclayer, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);
}

static void softmax(float* input, int N) {
    int i;
    float max = input[0];
    for (i = 1; i < N; i++) {
        if (max < input[i]) max = input[i];
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(input[i] - max);
    }
    for (i = 0; i < N; i++) {
        input[i] = exp(input[i] - max) / (sum + 1e-7);
    }
}

static int find_max(float* input, int classNum) {
    int i;
    int maxIndex = 0;
    float max = 0;
    for (i = 0; i < classNum; i++) {
        if (max < input[i]) {
            max = input[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image) {
    //Create and Write Buffer
    bufImg = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * num_of_image * 32 * 32 * 3, NULL, &err);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, bufImg, CL_FALSE, 0, sizeof(float) * num_of_image * 32 * 32 * 3,images, 0, NULL, NULL);
    CHECK_ERROR(err);

    bufFilter= clCreateBuffer(context, CL_MEM_READ_ONLY,  60980520, NULL, &err);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, bufFilter, CL_FALSE, 0,  60980520, network, 0, NULL, NULL);
    CHECK_ERROR(err);

    bufConvInput= clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64*32*32, NULL, &err);
    CHECK_ERROR(err);
    bufConvOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64 * 32 * 32, NULL, &err);
    CHECK_ERROR(err);
    bufConvExtend = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64 * 32 * 32*9, NULL, &err);
    CHECK_ERROR(err);
    bufPoolInput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64 * 32 * 32, NULL, &err);
    CHECK_ERROR(err);
    bufPoolOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 64 * 32 * 32, NULL, &err);
    CHECK_ERROR(err);
    bufFCInput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);
    bufFCOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512, NULL, &err);
    CHECK_ERROR(err);

    //Allocate Layer arrays
    float* layer=(float*)malloc(sizeof(float) * 10);
	

    for (int i = 0; i < num_of_image; ++i) {
        i_offset = i * 3 * 32 * 32; f_offset = 0;
        convolution(&bufImg, &bufConvOutput, &bufFilter, INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
        i_offset = 0; f_offset += (3 * 3 * INPUT_DIM[0] * OUTPUT_DIM[0]) + OUTPUT_DIM[0];
        convolution(&bufConvOutput, &bufConvInput, &bufFilter, INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
        f_offset += (3 * 3 * INPUT_DIM[2] * OUTPUT_DIM[2]) + OUTPUT_DIM[2];
        max_pooling(&bufConvInput, &bufPoolOutput, INPUT_DIM[2], NBYN[2] * 2);

        convolution(&bufPoolOutput, &bufConvOutput, &bufFilter, INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
        f_offset += (3 * 3 * INPUT_DIM[3] * OUTPUT_DIM[3]) + OUTPUT_DIM[3];
        convolution(&bufConvOutput, &bufConvInput, &bufFilter, INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
        f_offset += (3 * 3 * INPUT_DIM[5] * OUTPUT_DIM[5]) + OUTPUT_DIM[5];
        max_pooling(&bufConvInput, &bufPoolOutput, INPUT_DIM[5], NBYN[5] * 2);

        convolution(&bufPoolOutput, &bufConvOutput, &bufFilter, INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
        f_offset += (3 * 3 * INPUT_DIM[6] * OUTPUT_DIM[6]) + OUTPUT_DIM[6];
        convolution(&bufConvOutput, &bufConvInput, &bufFilter, INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
        f_offset += (3 * 3 * INPUT_DIM[7] * OUTPUT_DIM[7]) + OUTPUT_DIM[7];
        convolution(&bufConvInput, &bufConvOutput, &bufFilter, INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
        f_offset += (3 * 3 * INPUT_DIM[9] * OUTPUT_DIM[9]) + OUTPUT_DIM[9];
        max_pooling(&bufConvOutput, &bufPoolOutput, INPUT_DIM[9], NBYN[9] * 2);

        convolution(&bufPoolOutput, &bufConvOutput, &bufFilter, INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
        f_offset += (3 * 3 * INPUT_DIM[10] * OUTPUT_DIM[10]) + OUTPUT_DIM[10];
        convolution(&bufConvOutput, &bufConvInput, &bufFilter, INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
        f_offset += (3 * 3 * INPUT_DIM[11] * OUTPUT_DIM[11]) + OUTPUT_DIM[11];
        convolution(&bufConvInput, &bufConvOutput, &bufFilter, INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
        f_offset += (3 * 3 * INPUT_DIM[13] * OUTPUT_DIM[13]) + OUTPUT_DIM[13];
        max_pooling(&bufConvOutput, &bufPoolOutput, INPUT_DIM[13], NBYN[13] * 2);

        convolution(&bufPoolOutput, &bufConvOutput, &bufFilter, INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
        f_offset += (3 * 3 * INPUT_DIM[14] * OUTPUT_DIM[14]) + OUTPUT_DIM[14];
        convolution(&bufConvOutput, &bufConvInput, &bufFilter, INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
        f_offset += (3 * 3 * INPUT_DIM[15] * OUTPUT_DIM[15]) + OUTPUT_DIM[15];
        convolution(&bufConvInput, &bufConvOutput, &bufFilter, INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
        f_offset += (3 * 3 * INPUT_DIM[17] * OUTPUT_DIM[17]) + OUTPUT_DIM[17];
        max_pooling(&bufConvOutput, &bufPoolOutput, INPUT_DIM[17], NBYN[17] * 2);

        fc_layer(&bufPoolOutput, &bufFCOutput, &bufFilter, INPUT_DIM[18], OUTPUT_DIM[18]);
        f_offset += (INPUT_DIM[18] * OUTPUT_DIM[18]) + OUTPUT_DIM[18];
        fc_layer(&bufFCOutput, &bufFCInput, &bufFilter, INPUT_DIM[19], OUTPUT_DIM[19]);
        f_offset += (INPUT_DIM[19] * OUTPUT_DIM[19]) + OUTPUT_DIM[19];
        fc_layer(&bufFCInput, &bufFCOutput, &bufFilter, INPUT_DIM[20], OUTPUT_DIM[20]);

        err = clEnqueueReadBuffer(queue, bufFCOutput, CL_TRUE, 0, sizeof(float) * 10, layer, 0, NULL, NULL);
        CHECK_ERROR(err);


        softmax(layer, 10);
        
        labels[i] = find_max(layer, 10);
        confidences[i] = layer[labels[i]];
    }
}
