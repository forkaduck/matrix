
// Define some fallback macros, so that clang can do type checks.
#ifndef TYPE_T
#warning "Missing float type!"
#define TYPE_T float
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

#define CAT_I(a, b) a##b
#define CAT(a, b) CAT_I(a, b)

__kernel void KERNEL_NAME(__constant TYPE_T *rhs, SIZE_T w_rhs,
			  __constant TYPE_T *lhs, SIZE_T w_lhs,
			  __global TYPE_T *output)
{
	for (SIZE_T i = 0; i < min(w_rhs, w_lhs); i++) {
		output[i] = lhs[i] OPERATOR rhs[i];
	}
}

__kernel void CAT(KERNEL_NAME, _down)(__global TYPE_T *rhs, SIZE_T w_rhs,
				      __global TYPE_T *output)
{
	for (SIZE_T i = 0; i < w_rhs; i++) {
		*output = *output OPERATOR rhs[i];
	}
}
