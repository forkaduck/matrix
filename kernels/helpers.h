
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

#ifndef HELPERS
#define HELPERS

inline void print_array(__global TYPE_T *array, SIZE_T len)
{
#ifdef DEBUG
	if (get_local_id(0) == 0) {
		printf("[");
		for (SIZE_T i = 0; i < len; i++) {
			printf("%.1f, ", array[i]);
		}

		printf("]\n");
	}
#endif
}
#endif
