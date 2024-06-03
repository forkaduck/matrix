
// Define some fallback macros, so that clang can do type checks.
#ifndef FLOAT_T
#warning "Missing float type!"
#define FLOAT_T float
#endif

#define SIZE_T unsigned long

#ifndef OPERATOR
#warning "Missing operator!"
#define OPERATOR +
#endif

#ifndef KERNEL_NAME
#warning "Missing kernel name!"
#define KERNEL_NAME add
#endif

__kernel void KERNEL_NAME(__constant FLOAT_T *rhs, SIZE_T w_rhs,
			  __constant FLOAT_T *lhs, SIZE_T w_lhs,
			  __global FLOAT_T *output)
{
	for (size_t i = 0; i < min(w_rhs, w_lhs); i++) {
		output[i] = lhs[i] OPERATOR rhs[i];
	}
}
