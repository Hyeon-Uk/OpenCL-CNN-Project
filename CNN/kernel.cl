__kernel void conv(__global float *inputs,__global float *outputs,__global float *filters
					,int input_dim,int output_dim,int nbyn,int filter_offset){
	const int TS = 16;

    const int ROW_A = output_dim; 
    const int COL_B = nbyn * nbyn;

    const int ROW_B = input_dim * 3 * 3;
    const int COL_A = input_dim * 3 * 3;
    

    __global float* input = inputs;
    __global float* filter = filters + filter_offset;
    __global float* b = filters + filter_offset + (ROW_A * COL_A);
    __global float* output = outputs;

    __local float Asub[16][16];
    __local float Bsub[16][16];

    const int j = get_local_id(0);
    const int i = get_local_id(1);
    const int gj = get_group_id(0) * TS + j;
    const int gi = get_group_id(1) * TS + i;

    float sum = 0.0f;

    #pragma unroll
    for (int t = 0; t < COL_A; t += TS) {

        const int tj = t + j;
        const int ti = t + i;

       
        Asub[i][j] = (gi < ROW_A&& tj < COL_A) ? filter[gi * COL_A + tj] : 0;
        Bsub[i][j] = (ti < ROW_B&& gj < COL_B) ? input[ti * COL_B + gj] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);   

        #pragma unroll
        for (int k = 0; k < TS; k++) {
            sum += Asub[i][k] * Bsub[k][j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gi < ROW_A && gj < COL_B) {
        output[gi * COL_B + gj] = (sum + b[gi] > 0 ? sum + b[gi] : 0);// Relu
    }
}

__kernel void conv_ex(__global float *inputs,__global float *outputs,
						int output_dim,int nbyn,int image_offset){
	 int g_j = get_global_id(0);

    __global float* input = inputs + image_offset;
    __global float* output = outputs;


    int rows = g_j / nbyn; // x*z 인덱스를 나타냄
    int col = g_j % nbyn; // y 인덱스를 나타냄
    int channel = rows / nbyn; // z 인덱스  
    int row = rows - channel * nbyn; // x 인덱스 

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            int x = col + j - 1;
            int y = row + i - 1;
            if (x >= 0 && x < nbyn && y >= 0 && y < nbyn)
                output[(((channel * 3 * 3) + (3 * i + j)) * (nbyn * nbyn)) + (row * nbyn + col)] = input[((channel * nbyn) + y) * nbyn + x];
            else
                output[(((channel * 3 * 3) + (3 * i + j)) * (nbyn * nbyn)) + (row * nbyn + col)] = 0.0f;
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
	int output_group=get_global_id(0);
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
