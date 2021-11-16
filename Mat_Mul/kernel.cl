__kernel void matmul(__global float *A,__global float *B,__global float *C,int ROW_A,int COL_A,int COL_B){
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k;
	float temp=0.0f;

	for(k=0;k<COL_A;k++){
		temp+=A[i*COL_A+k]*B[k*COL_B+j];
	}
	C[i*COL_B+j]=temp;
}
