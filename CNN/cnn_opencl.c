#include <CL/cl.h>
#include "cnn.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

extern const int INPUT_DIM[];
extern const int OUTPUT_DIM[];
extern const int NBYN[];

cl_int err;
cl_uint num_platforms;
cl_platform_id* platforms;
cl_device_id* devices, device;
cl_uint num_devices;
cl_context context;
cl_command_queue queue;
cl_program program;
char* kernel_source;
size_t kernel_source_size;
cl_kernel kernel_conv,kernel_poll;
cl_mem bufImg, bufNetwork, bufConvInput, bufConvOutput;
cl_uint image_offset, filter_offset;

char* get_source_code(const char* file_name, size_t * len) {
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
    queue = clCreateCommandQueue(context, device, 0, &err);
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
}

void convolution_cnn(cl_mem* input, cl_mem* output, cl_mem* networks, int input_dim, int output_dim, int nbyn) {
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

    size_t global_size[2] = {nbyn*output_dim,nbyn};
    //size_t local_size[2] = {4,4};
    //Enqueue
    err = clEnqueueNDRangeKernel(queue, kernel_conv, 2, NULL, global_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

static void convolution(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn) {
    memset(outputs, 0, nbyn * nbyn * outDim * sizeof(float));
    int x = 0, y = 0;
    int offset = nbyn * nbyn;
    float sum = 0, temp;
    float* input, * output;
    for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
        input = inputs;
        for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
            output = outputs;
            for (int row = 0; row < nbyn; ++row) {
                for (int col = 0; col < nbyn; ++col) {
                    sum = 0;
                    for (int fRow = 0; fRow < 3; ++fRow) {
                        for (int fCol = 0; fCol < 3; ++fCol) {
                            x = col + fCol - 1;
                            y = row + fRow - 1;

                            if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                                sum += input[nbyn * y + x] * filter[3 * fRow + fCol];
                            }

                        }
                    }
                    *(output++) += sum;
                }
            }
            filter += 9;
            input += offset;
        }
        for (int i = 0; i < offset; ++i) {
            (*outputs) = (*outputs)+(*biases);
            if (*outputs < 0) (*outputs) = 0;	//ReLU
            outputs++;
        }
        ++biases;
    }
}

static void max_pooling(float* input, float* output, int DIM, int nbyn) {
	float max, temp;
	int n, row, col, x, y;
	for (n = 0; n < DIM; ++n) {
		for (row = 0; row < nbyn; row += 2) {
			for (col = 0; col < nbyn; col += 2) {
				//max = -FLT_MAX;
				max = 0;
				for (y = 0; y < 2; ++y) {
					for (x = 0; x < 2; ++x) {
						temp = input[nbyn * (row + y) + col + x];
						if (max < temp) max = temp;
					}
				}
				*(output++) = max;
			}
		}
		input += nbyn * nbyn;
	}
}


void fc_layer(float* input, float* output, float* weights, float* biases, int inDim, int outDim) {
	float sum;
	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		sum = 0;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			sum += input[inNeuron] * (*weights++);
		}
		sum += biases[outNeuron];
		if (sum > 0) output[outNeuron] = sum;	//ReLU
		else output[outNeuron] = 0;
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

    err = clEnqueueWriteBuffer(queue, bufImg, CL_FALSE, 0, sizeof(float) * num_images * 32 * 32 * 3, images, 0, NULL, NULL);
    CHECK_ERROR(err);

    bufNetwork = clCreateBuffer(context, CL_MEM_READ_ONLY, 60980520, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, bufNetwork, CL_FALSE, 0, 60980520, network, 0, NULL, &err);
    CHECK_ERROR(err);

    bufConvInput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 512, NULL, &err);
    CHECK_ERROR(err);

    bufConvOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 512, NULL, &err);
    CHECK_ERROR(err);
    float* w[21];
    float* b[21];
    int offset = 0;
    // link weights and biases to network
    for (int i = 0; i < 17; ++i) {
        if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
        w[i] = network + offset;
        offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
        b[i] = network + offset;
        offset += OUTPUT_DIM[i];
    }
    for (int i = 18; i < 21; ++i) {
        w[i] = network + offset;
        offset += INPUT_DIM[i] * OUTPUT_DIM[i];
        b[i] = network + offset;
        offset += OUTPUT_DIM[i];
    }


    // allocate memory for layer
    float* layer[21];
    for (int i = 0; i < 21; ++i) {
        layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
        if (layer[i] == NULL) {
            perror("malloc error");
        }
    }

    for (int i = 0; i < num_images; i++) {
        image_offset = i * 32 * 32 * 3 ; filter_offset = 0;
        convolution_cnn(&bufImg, &bufConvOutput, &bufNetwork, INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
        image_offset = 0; filter_offset += (9 * INPUT_DIM[0] * OUTPUT_DIM[0]+OUTPUT_DIM[0]);
        convolution_cnn(&bufConvOutput, &bufConvInput, &bufNetwork, INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
        err = clEnqueueReadBuffer(queue, bufConvInput, CL_TRUE, 0, sizeof(float) * OUTPUT_DIM[1] * NBYN[1] * NBYN[1], layer[1], 0, NULL, NULL);
        CHECK_ERROR(err);
        clFinish(queue);
        max_pooling(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2);

        convolution(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
        convolution(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
        max_pooling(layer[4], layer[5], INPUT_DIM[5], NBYN[5] * 2);

        convolution(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
        convolution(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
        convolution(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
        max_pooling(layer[8], layer[9], INPUT_DIM[9], NBYN[9] * 2);

        convolution(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
        convolution(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
        convolution(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
        max_pooling(layer[12], layer[13], INPUT_DIM[13], NBYN[13] * 2);

        convolution(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
        convolution(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
        convolution(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
        max_pooling(layer[16], layer[17], INPUT_DIM[17], NBYN[17] * 2);

        fc_layer(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
        fc_layer(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
        fc_layer(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);

        softmax(layer[20], 10);

        labels[i] = find_max(layer[20], 10);
        confidences[i] = layer[20][labels[i]];
    }
}