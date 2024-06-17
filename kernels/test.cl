

__kernel void test_capabilities(__global unsigned char *output)
{
	for (size_t i = 0; i < 10; i++) {
		output[i] = i * 10;
	}
}
