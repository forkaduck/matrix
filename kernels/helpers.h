
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
