__kernel void conv(__global float *inputs,__global float *outputs,__global float *filters
					,int input_dim,int output_dim,int nbyn,int filter_offset){
    //Tile Size
    int TS = 16;

    //filter matrix
    int ROW_A = output_dim; 
    int COL_A = input_dim * 3 * 3;

    //input matrix
    int ROW_B = input_dim * 3 * 3;
    int COL_B = nbyn * nbyn;
    

    __global float* input = inputs;
    __global float* output = outputs;
    __global float* filter = filters + filter_offset;
    __global float* b = filters + filter_offset + (input_dim*output_dim*9);

    __local float filterSub[16][16];
    __local float inputSub[16][16];

    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int g_row = get_group_id(0) * TS + row;//global_index
    const int g_col = get_group_id(1) * TS + col;//output_dim

    __private float sum = 0.0f;

    //Matrix Multiple With Tilling
    #pragma unroll
    for (int i = 0; i < COL_A; i += TS) {
        const int temp_row = i + row;
        const int temp_col = i + col;

        filterSub[col][row] = (g_col < output_dim && temp_row < COL_A) ? filter[g_col * COL_A + temp_row] : 0;
        inputSub[col][row] = (temp_col < ROW_B&& g_row < COL_B) ? input[temp_col * COL_B + g_row] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);   

        #pragma unroll
        for (int k = 0; k < TS; k++) {
            sum += filterSub[col][k] * inputSub[k][row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_col < ROW_A && g_row < COL_B) {
        output[g_col * COL_B + g_row] = (sum + b[g_col] > 0 ? sum + b[g_col] : 0);
    }
}

__kernel void conv_ex(__global float *inputs,__global float *outputs,
						int output_dim,int nbyn,int image_offset){
	 int g_j = get_global_id(0);

    __global float* input = inputs + image_offset;
    __global float* output = outputs;


    int rows = g_j / nbyn; 
    int col = g_j % nbyn; 
    int dim = rows / nbyn; 
    int row = rows - dim * nbyn; 

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            int x = col + j - 1;
            int y = row + i - 1;
            if (x >= 0 && x < nbyn && y >= 0 && y < nbyn)
                output[((dim * 3 * 3) + (3 * i + j)) * (nbyn * nbyn) + (row * nbyn + col)] = input[((dim * nbyn) + y) * nbyn + x];
            else
                output[((dim * 3 * 3) + (3 * i + j)) * (nbyn * nbyn) + (row * nbyn + col)] = 0.0f;
        }
    }
}


__kernel void pool(__global float *inputs, __global float *outputs, int input_dim,int nbyn){
	int g_i=get_global_id(0);
	int g_j=get_global_id(1);
	int output_nbyn=nbyn/2;
	int group_num=(g_i/output_nbyn);

	int frow=g_j%output_nbyn;
	int fcol=g_i%output_nbyn;

	__global float *input=inputs+(nbyn*nbyn)*group_num;
	__global float *output=outputs+(output_nbyn*output_nbyn)*group_num;
	float max=0.0f;

	#pragma unroll
	for(int y=0;y<2;y++){
		#pragma unroll
		for(int x=0;x<2;x++){
			float temp=input[nbyn*(2*frow+y)+(2*fcol+x)];
			if(max<temp) max=temp;
		}
	}
	output[output_nbyn*frow+fcol]=max;
}


__kernel void fclayer(__global float *inputs,__global float *outputs,__global float *filters,
					int input_dim,int output_dim,int f_offset){
    int TS=16;
    int l_i=get_local_id(0);

    int output_group=get_group_id(0)*TS+l_i;
	if(output_group>=output_dim) return;

	__global float *w=filters+f_offset+(input_dim*output_group);
	__global float *b=filters+f_offset+(input_dim*output_dim)+output_group;

	float sum=0.0f;

	#pragma unroll
	for(int i=0;i<input_dim;i++){
		sum += inputs[i]* (*(w+i));
	}
	sum+= (*b);
	outputs[output_group]= ( sum<=0 ? 0.0f : sum);
}
