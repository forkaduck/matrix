//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef FLOAT_T
#define FLOAT_T float
#endif

#ifndef SIZE_T
#define SIZE_T unsigned long
#endif

__kernel void add(__constant FLOAT_T *rhs, SIZE_T w_rhs,
		  __constant FLOAT_T *lhs, SIZE_T w_lhs,
		  __global FLOAT_T *output)
{
	for (size_t i = 0; i < min(w_rhs, w_lhs); i++) {
		output[i] = lhs[i] + rhs[i];
	}
}

__kernel void sub(__constant FLOAT_T *rhs, SIZE_T w_rhs,
		  __constant FLOAT_T *lhs, SIZE_T w_lhs,
		  __global FLOAT_T *output)
{
	for (size_t i = 0; i < min(w_rhs, w_lhs); i++) {
		output[i] = lhs[i] - rhs[i];
	}
}
