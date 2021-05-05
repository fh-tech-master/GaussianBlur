__kernel void test(
	__global uchar* r,
	__global uchar* g,
	__global uchar* b,
	__global uchar* rOut,
	__global uchar* gOut,
	__global uchar* bOut,
	__global int* kernelSize,
	__global double* blurKernel
	)
{
  size_t px = get_global_id(0);
  size_t py = get_global_id(1);

  size_t width = get_global_size(0);
  size_t height = get_global_size(1);

  int kSize = *kernelSize;

  double rBlur = 0.0;
  double gBlur = 0.0;
  double bBlur = 0.0;

  int x = px - (kSize / 2);

  for (int i = 0; i < kSize; i++) {
    int y = py - (kSize / 2);
    for (int j = 0; j < kSize; j++) {
      int newX = x;
      int newY = y;
      if (newX < 0) newX = 0;
      if (newX >= width) newX = width - 1;
      if (newY < 0) newY = 0;
      if (newY >= height) newY = height - 1;

      int index = newY * width + newX;
      int blurIndex = i * kSize + j;

      rBlur += (double)r[index] * blurKernel[blurIndex];
      gBlur += (double)g[index] * blurKernel[blurIndex];
      bBlur += (double)b[index] * blurKernel[blurIndex];

      y++;
    }
    x++;
  }

  int index = py * width + px;

  rOut[index] = (unsigned char)round(rBlur);
  gOut[index] = (unsigned char)round(gBlur);
  bOut[index] = (unsigned char)round(bBlur);
}