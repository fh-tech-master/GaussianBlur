__kernel void test(
	__global uchar* r,
	__global uchar* g,
	__global uchar* b,
	__global uchar* rOut,
	__global uchar* gOut,
	__global uchar* bOut)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);

    size_t width = get_global_size(0);
    size_t height = get_global_size(1);


	if(x % 100 == 0 && y % 100 == 0) {
		printf("x = %d\n", x);
		printf("y = %d\n", y);
	}

	rOut[0] = 255;
	gOut[0] = 255;
	bOut[0] = 255;
}