__kernel void matmul1(__global float *A,__global float *B,__global float *C,int ROW_A,int COL_A,int COL_B){
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k;
	float temp=0.0f;

	for(k=0;k<COL_A;k++){
		C[i*COL_B+j]+=A[i*COL_A+k]*B[k*COL_B+j];
	}
}

__kernel void matmul2(__global float *A,__global float *B,__global float *C,int ROW_A,int COL_A,int COL_B){
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k;
	float temp=0.0f;


	for(k=0;k<COL_A;k++){
		temp+=A[i*COL_A+k]*B[k*COL_B+j];
	}
	C[i*COL_B+j]=temp;
}

__kernel void matmul3(__global float *A,__global float *B,__global float *C,int ROW_A,int COL_A,int COL_B,int TS,__local float *Asub,__local float *Bsub){
	int i=get_local_id(0);
	int j=get_local_id(1);
	
	int gi=get_global_id(0)*TS+i;
	int gj=get_global_id(1)*TS+j;
	float temp=0.0f;
	for(int t=0;t<COL_A;t+=TS){
		const int ti=t+j;
		const int tj=t+i;

		Asub[TS*j+i]=A[COL_A*gj+ti];
		Bsub[TS*j+i]=B[COL_B*tj+gi];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for(int k=0;k<TS;k++)
			temp+=Asub[TS*j+k]*Bsub[TS*k+i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
