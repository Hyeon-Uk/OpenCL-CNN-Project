#include <CL/cl.h>
#include "cnn.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }


extern const int INPUT_DIM[];
extern const int OUTPUT_DIM[];
extern const int NBYN[];

cl_int err;
cl_event* write_events,*operation_events,read_events;
cl_uint num_platforms;
cl_platform_id* platforms;
cl_device_id* devices, device;
cl_uint num_devices;
cl_context context;
cl_command_queue write_queue, queue, read_queue;
cl_program program;
char* kernel_source;
size_t kernel_source_size;
cl_kernel kernel_conv, kernel_pool, kernel_fclayer;
cl_mem bufImg, bufNetwork, bufConvInput, bufConvMiddle, bufConvOutput, bufPoolInput,bufResult;
cl_uint image_offset, filter_offset;
int image_index;

char* get_source_code(const char* file_name, size_t* len) {
    char* source_code;
    char buf[2] = "\0";
    int cnt = 0;
    size_t length;
    FILE* file = fopen(file_name, "r");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    length = (size_t)ftell(file);
    rewind(file);

    source_code = (char*)malloc(length + 1);
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

void cnn_init() {
    //Get Platform number
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_ERROR(err);

    //GetPlatforms
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    CHECK_ERROR(err);

    //Get Device number;
    cl_uint num_devices;
    err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    CHECK_ERROR(err);

    //Get Devices
    devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
    err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    CHECK_ERROR(err);

    //Set device
    device = devices[1];

    //Create Context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    //Create Command Queue
    write_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);

    read_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);

    //Get SourceCode
    kernel_source = get_source_code("kernel.cl", &kernel_source_size);

    //Create Program
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    //Build Program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    CHECK_ERROR(err);

    //Create Kernel Object
    kernel_conv = clCreateKernel(program, "conv", &err);
    CHECK_ERROR(err);

    kernel_pool = clCreateKernel(program, "pool", &err);
    CHECK_ERROR(err);

    kernel_fclayer = clCreateKernel(program, "fclayer", &err);
    CHECK_ERROR(err);
}

//void convolution_cnn(cl_mem* input, cl_mem* output, cl_mem* networks, int input_dim, int output_dim, int nbyn,int start) {
//    //Set Convolution Kernel Arg
//    err = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), input);
//    CHECK_ERROR(err);
//
//    err = clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), output);
//    CHECK_ERROR(err);
//
//    err = clSetKernelArg(kernel_conv, 2, sizeof(cl_mem), networks);
//    CHECK_ERROR(err);
//
//    err = clSetKernelArg(kernel_conv, 3, sizeof(int), &input_dim);
//    CHECK_ERROR(err);
//
//    err = clSetKernelArg(kernel_conv, 4, sizeof(int), &output_dim);
//    CHECK_ERROR(err);
//
//    err = clSetKernelArg(kernel_conv, 5, sizeof(int), &nbyn);
//    CHECK_ERROR(err);
//
//    err = clSetKernelArg(kernel_conv, 6, sizeof(int), &filter_offset);
//    CHECK_ERROR(err);
//
//    err = clSetKernelArg(kernel_conv, 7, sizeof(int), &image_offset);
//    CHECK_ERROR(err);
//
//    size_t global_size[2] = { nbyn * output_dim,nbyn };
//    //Enqueue
//    if (start == 0) {
//        err = clEnqueueNDRangeKernel(queue, kernel_conv, 2, NULL, global_size, NULL, 0, NULL, NULL);
//        CHECK_ERROR(err);
//    }
//    else {
//        err = clEnqueueNDRangeKernel(queue, kernel_conv, 2, NULL, global_size, NULL, 1, &write_events[image_index], NULL);
//        CHECK_ERROR(err);
//    }
//}

void convolution_cnn(cl_mem* input, cl_mem* output, cl_mem* networks, int input_dim, int output_dim, int nbyn, int start) {
    //Set Convolution Kernel Arg
    err = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), input);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), output);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_conv, 2, sizeof(cl_mem), networks);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_conv, 3, sizeof(int), &input_dim);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_conv, 4, sizeof(int), &output_dim);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_conv, 5, sizeof(int), &nbyn);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_conv, 6, sizeof(int), &filter_offset);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_conv, 7, sizeof(int), &image_offset);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_conv, 8, input_dim*sizeof(float), NULL);
    CHECK_ERROR(err);

    size_t global_size[2] = { nbyn * nbyn * input_dim, output_dim };
    size_t local_size[2] = { input_dim,1};
    //Enqueue
    if (start == 0) {
        err = clEnqueueNDRangeKernel(queue, kernel_conv, 2, NULL, global_size, local_size, 0, NULL, NULL);
        CHECK_ERROR(err);
    }
    else {
        err = clEnqueueNDRangeKernel(queue, kernel_conv, 2, NULL, global_size, local_size, 1, &write_events[image_index], NULL);
        CHECK_ERROR(err);
    }
}

void max_pooling_cnn(cl_mem* input, cl_mem* output, int DIM, int nbyn) {
    //Set Pooling Kernel Arg
    err = clSetKernelArg(kernel_pool, 0, sizeof(cl_mem), input);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_pool, 1, sizeof(cl_mem), output);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_pool, 2, sizeof(int), &DIM);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_pool, 3, sizeof(int), &nbyn);
    CHECK_ERROR(err);

    int output_nbyn = nbyn / 2;

    size_t global_size[2] = { DIM * output_nbyn , output_nbyn };

    err = clEnqueueNDRangeKernel(queue, kernel_pool, 2, NULL, global_size, NULL, 0, NULL, NULL);
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
    }
    CHECK_ERROR(err);
}

void fc_layer_cnn(cl_mem* input, cl_mem* output, cl_mem* networks, int inDIM, int outDIM,int end){
    //Set Kernel Arg
    err = clSetKernelArg(kernel_fclayer, 0, sizeof(cl_mem), input);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_fclayer, 1, sizeof(cl_mem), output);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_fclayer, 2, sizeof(cl_mem), networks);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_fclayer, 3, sizeof(int), &inDIM);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_fclayer, 4, sizeof(int), &outDIM);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_fclayer, 5, sizeof(int), &filter_offset);
    CHECK_ERROR(err);
    size_t global_size = outDIM;

    //enqueue
    if (end == 0) {
        err = clEnqueueNDRangeKernel(queue, kernel_fclayer, 1, 0, &global_size, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);
    }
    else {
        err = clEnqueueNDRangeKernel(queue, kernel_fclayer, 1, 0, &global_size, NULL, 0, NULL, &operation_events[image_index]);
        CHECK_ERROR(err);
    }
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


void cnn(float* images, float* network, int* labels, float* confidences, int num_images) {
    //TODO
    //Global Buffer Create and Write
    bufImg = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * num_images * 32 * 32 * 3, NULL, &err);
    CHECK_ERROR(err);

    //err = clEnqueueWriteBuffer(queue, bufImg, CL_TRUE, 0, sizeof(float) * num_images * 32 * 32 * 3, images, 0, NULL, NULL);
    //CHECK_ERROR(err);

    bufNetwork = clCreateBuffer(context, CL_MEM_READ_ONLY, 60980520, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, bufNetwork, CL_FALSE, 0, 60980520, network, 0, NULL, &err);
    CHECK_ERROR(err);

    bufConvInput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 512, NULL, &err);
    CHECK_ERROR(err);

    bufConvMiddle = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 512, NULL, &err);
    CHECK_ERROR(err);

    bufConvOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 512, NULL, &err);
    CHECK_ERROR(err);

    bufPoolInput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 512, NULL, &err);
    CHECK_ERROR(err);

    bufResult= clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * 10, NULL, &err);
    CHECK_ERROR(err);
    // allocate memory for layer
    float** layer;
    layer = (float**)malloc(sizeof(float*) * num_images);

    for (int i = 0; i < num_images; ++i) {
        layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[20] * NBYN[20] * NBYN[20]);
        if (layer[i] == NULL) {
            perror("malloc error");
        }
    }

    write_events = (cl_event*)malloc(sizeof(cl_event) * num_images);
    operation_events = (cl_event*)malloc(sizeof(cl_event) * num_images);
    clock_t start, end;
    for (image_index = 0; image_index < num_images; image_index++) {
        image_offset = image_index * 32 * 32 * 3; filter_offset = 0;
        start=clock();
        err = clEnqueueWriteBuffer(write_queue, bufImg, CL_TRUE, sizeof(float) * image_index * 32 * 32 * 3, sizeof(float) * 32 * 32 * 3, images + image_offset, 0, NULL,&write_events[image_index]);
        CHECK_ERROR(err);
        //clFinish(write_queue);
        convolution_cnn(&bufImg, &bufConvOutput, &bufNetwork, INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0],1);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[0] * OUTPUT_DIM[0] + OUTPUT_DIM[0]);
        convolution_cnn(&bufConvOutput, &bufPoolInput, &bufNetwork, INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[1] * OUTPUT_DIM[1] + OUTPUT_DIM[1]);
        max_pooling_cnn(&bufPoolInput, &bufConvInput, INPUT_DIM[2], NBYN[2] * 2);
        
        convolution_cnn(&bufConvInput, &bufConvOutput, &bufNetwork, INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[3] * OUTPUT_DIM[3] + OUTPUT_DIM[3]);
        convolution_cnn(&bufConvOutput, &bufPoolInput, &bufNetwork, INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[4] * OUTPUT_DIM[4] + OUTPUT_DIM[4]);
        max_pooling_cnn(&bufPoolInput, &bufConvInput, INPUT_DIM[5], NBYN[5] * 2);

        convolution_cnn(&bufConvInput, &bufConvMiddle, &bufNetwork, INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[6] * OUTPUT_DIM[6] + OUTPUT_DIM[6]);
        convolution_cnn(&bufConvMiddle, &bufConvOutput, &bufNetwork, INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[7] * OUTPUT_DIM[7] + OUTPUT_DIM[7]);
        convolution_cnn(&bufConvOutput, &bufPoolInput, &bufNetwork, INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[8] * OUTPUT_DIM[8] + OUTPUT_DIM[8]);
        max_pooling_cnn(&bufPoolInput, &bufConvInput, INPUT_DIM[9], NBYN[9] * 2);
        
        convolution_cnn(&bufConvInput, &bufConvMiddle, &bufNetwork, INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[10] * OUTPUT_DIM[10] + OUTPUT_DIM[10]);
        convolution_cnn(&bufConvMiddle, &bufConvOutput, &bufNetwork, INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[11] * OUTPUT_DIM[11] + OUTPUT_DIM[11]);
        convolution_cnn(&bufConvOutput, &bufPoolInput, &bufNetwork, INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[12] * OUTPUT_DIM[12] + OUTPUT_DIM[12]);
        max_pooling_cnn(&bufPoolInput, &bufConvInput, INPUT_DIM[13], NBYN[13] * 2);
        
        convolution_cnn(&bufConvInput, &bufConvMiddle, &bufNetwork, INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[14] * OUTPUT_DIM[14] + OUTPUT_DIM[14]);
        convolution_cnn(&bufConvMiddle, &bufConvOutput, &bufNetwork, INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[15] * OUTPUT_DIM[15] + OUTPUT_DIM[15]);
        convolution_cnn(&bufConvOutput, &bufPoolInput, &bufNetwork, INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16],0);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[16] * OUTPUT_DIM[16] + OUTPUT_DIM[16]);
        max_pooling_cnn(&bufPoolInput, &bufConvInput, INPUT_DIM[17], NBYN[17] * 2);
        
        fc_layer_cnn(&bufConvInput, &bufConvOutput, &bufNetwork, INPUT_DIM[18], OUTPUT_DIM[18],0);
        filter_offset += INPUT_DIM[18] * OUTPUT_DIM[18] + OUTPUT_DIM[18];
        fc_layer_cnn(&bufConvOutput, &bufConvInput, &bufNetwork, INPUT_DIM[19], OUTPUT_DIM[19],0);
        filter_offset += INPUT_DIM[19] * OUTPUT_DIM[19] + OUTPUT_DIM[19];
        fc_layer_cnn(&bufConvInput, &bufResult, &bufNetwork, INPUT_DIM[20], OUTPUT_DIM[20],1);
        
        err = clEnqueueReadBuffer(read_queue, bufResult, CL_TRUE, 0, sizeof(float) * 10, layer[image_index], 1 ,&operation_events[image_index], NULL);
        CHECK_ERROR(err);
        softmax(layer[image_index], 10);

        labels[image_index] = find_max(layer[image_index], 10);
        confidences[image_index] = layer[image_index][labels[image_index]];
    }
}
