//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef FLOAT_T
#define FLOAT_T float
#endif

__kernel void add(__constant FLOAT_T *rhs, unsigned int w_rhs,
		  __constant FLOAT_T *lhs, unsigned int w_lhs,
		  __global FLOAT_T *output)
{
	// Calculate the sum of two Vectors.
	for (size_t i = 0; i < min(w_rhs, w_lhs); i++) {
		output[i] = lhs[i] + rhs[i];
	}
}
