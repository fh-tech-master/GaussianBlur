__kernel void test(
	__global uchar* r,
	__global uchar* g,
	__global uchar* b,
	__global uchar* rOut,
	__global uchar* gOut,
	__global uchar* bOut)
{
	size_t id = get_global_id(0);

	printf("r[%d] = %d, g[%d] = %d, b[%d] = %d\n", id, r[id], id, g[id], id, b[id]);

	rOut[id] = 1;
	gOut[id] = 2;
	bOut[id] = 3;
}