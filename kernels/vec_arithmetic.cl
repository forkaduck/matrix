
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

#include "helpers.h"

__kernel void KERNEL_NAME(__constant TYPE_T *rhs, SIZE_T w_rhs,
			  __constant TYPE_T *lhs, SIZE_T w_lhs,
			  __global TYPE_T *output)
{
	for (SIZE_T i = get_local_id(0); i < min(w_rhs, w_lhs);
	     i += get_local_size(0)) {
		output[i] = lhs[i] OPERATOR rhs[i];
	}
}

__kernel void CAT(KERNEL_NAME, _down)(__global TYPE_T *rhs, SIZE_T w_rhs)
{
	for (SIZE_T k = 0; k < (SIZE_T)native_sqrt(w_rhs) + 1; k++) {
		SIZE_T correction = 2 << k;

		for (SIZE_T ix = correction * get_local_id(0); ix < w_rhs - 1;
		     ix += correction * get_local_size(0)) {
			SIZE_T io = 0;

			if (correction == 2) {
				io = ix + 1;
			} else {
				io = ix + (1 << k);
			}

			if (io < w_rhs) {
				rhs[ix] = rhs[ix] OPERATOR rhs[io];
			}
		}
		print_array(rhs, w_rhs);
	}
}
