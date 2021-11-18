
__kernel void imgRotation(__global float *INPUT,__global float *OUTPUT,int W,int H,
								const float sin_theta,const float cos_theta){
	int i=get_global_id(0);
	int j=get_global_id(1);

	float x0 = W/2.0f;
	float y0 = W/2.0f;
	float xOff = i-x0;
	float yOff = j-y0;

	int src_x = (int)(xOff * cos_theta + yOff * sin_theta + x0);
	int src_y = (int)(yOff * cos_theta - xOff * sin_theta + y0);
	if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H))
		OUTPUT[j * W + i] = INPUT[src_y * W + src_x];
	else
		OUTPUT[j * W + i] = 0.0f;

}
