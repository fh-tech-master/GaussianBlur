__kernel void test(
	__global uchar* r,
	__global uchar* g,
	__global uchar* b,
	__global uchar* rOut,
	__global uchar* gOut,
	__global uchar* bOut,
	__global const int* kernelSize,
	__global const double* blurKernel,
	__local uchar* tempR,
	__local uchar* tempG,
	__local uchar* tempB
	)
{
  // for accessing the correct pixel
  size_t px = get_global_id(0);
  size_t py = get_global_id(1);
  size_t width = get_global_size(0);
  size_t globalIndex = py * width + px;

  // for storing it at the right place
  size_t localIndex = get_local_id(0) + get_local_id(1);

  // for knowing the length of the local arrays
  size_t size = get_local_size(0) * get_local_size(1);

  // for knowing if it is the horizontal or vertical blur
  bool isHorizontalBlur = get_local_size(1) == 1;

  if (isHorizontalBlur) {
	  tempR[localIndex] = r[globalIndex];
	  tempG[localIndex] = g[globalIndex];
	  tempB[localIndex] = b[globalIndex];
  }
  else {
	  tempR[localIndex] = rOut[globalIndex];
	  tempG[localIndex] = gOut[globalIndex];
	  tempB[localIndex] = bOut[globalIndex];
  }

  // waiting for the local arrays to be fully initialzed accross the workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  int kSize = *kernelSize;

  double rBlur = 0.0;
  double gBlur = 0.0;
  double bBlur = 0.0;

  for (int i = 0; i < kSize; i++) {
    int x = localIndex - (kSize / 2) + i;
    if (x < 0) x = 0;
    if (x >= size) x = size - 1;

    rBlur += (double)tempR[x] * blurKernel[i];
    gBlur += (double)tempG[x] * blurKernel[i];
    bBlur += (double)tempB[x] * blurKernel[i];
  }

  if (isHorizontalBlur) {
	  rOut[globalIndex] = (unsigned char)round(rBlur);
	  gOut[globalIndex] = (unsigned char)round(gBlur);
	  bOut[globalIndex] = (unsigned char)round(bBlur);
  }
  else {
	  r[globalIndex] = (unsigned char)round(rBlur);
	  g[globalIndex] = (unsigned char)round(gBlur);
	  b[globalIndex] = (unsigned char)round(bBlur);
  }
}