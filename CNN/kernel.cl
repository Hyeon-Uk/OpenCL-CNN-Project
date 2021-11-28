__kernel void conv(__global float *inputs,__global float *outputs,__global float *filters
					,int input_dim,int output_dim,int nbyn,int filter_offset,int image_offset){
	int g_i=get_global_id(0);
	int g_j=get_global_id(1);
	__global float *input = inputs+image_offset;
	int group_num=(g_i/nbyn);

	__global float *output = outputs+(nbyn*nbyn*group_num);
	__global float *w=filters+filter_offset+(3*3*input_dim*group_num);
	__global float *b=filters+filter_offset+(3*3*input_dim*output_dim)+group_num;

	int x=0;
	int y=0;

	int frow=g_j%nbyn;
	int fcol=g_i%nbyn;


	float sum=0.0f;
	int offset=nbyn*nbyn;
	for(int i=0;i<input_dim;i++){
		for(int row=0;row<3;row++){
			for(int col=0;col<3;col++){
				x=col+fcol-1;
				y=row+frow-1;
				if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
					sum += (input[nbyn*y+x]*w[3*row+col]);
				}
			}
		}
		input+=offset;
		w+=9;
	}
	sum+=(*b);
	if(sum<0) sum=0.0f;
	output[nbyn*frow+fcol]=sum;
}

__kernel void pool(__global float *inputs, __global float *outputs, int INPUT_DIM,int nbyn){
	int g_i=get_global_id(0);
	int g_j=get_global_id(1);
	int output_nbyn=nbyn/2;
	int group_num=(g_i/output_nbyn);

	int frow=g_j%output_nbyn;
	int fcol=g_i%output_nbyn;

	__global float *input=inputs+(nbyn*nbyn)*group_num;
	__global float *output=outputs+(output_nbyn*output_nbyn)*group_num;
	float max=0.0f;
	for(int y=0;y<2;y++){
		for(int x=0;x<2;x++){
			float temp=input[nbyn*(2*frow+y)+(2*fcol+x)];
			if(max<temp) max=temp;
		}
	}
	output[output_nbyn*frow+fcol]=max;
}

__kernel void fclayer(__global float *inputs,__global float *outputs,__global float *filters,
					int inDIM,int outDIM,int filter_offset){
	int output_id=get_global_id(0);

	__global float *input=inputs;
	__global float *w=filters+filter_offset+(inDIM)*output_id;
	__global float *b=filters+filter_offset+(inDIM*outDIM)+output_id;

	float sum=0.0f;

	for(int i=0;i<inDIM;i++){
		sum+= input[i]* *(w+i);
	}
	sum+=(*b);
	if(sum<=0) sum=0.0f;
	outputs[output_id]=sum;
}
