
unsigned int min(unsigned int w_rhs, unsigned int w_lhs)
{
	if (w_lhs > w_rhs) {
		return w_rhs;
	} else {
		return w_lhs;
	}
}

__kernel void add(__constant float *rhs, unsigned int w_rhs,
		  __constant float *lhs, unsigned int w_lhs,
		  __global float *output)
{
	// Calculate the sum of two Vectors.
	for (size_t i = 0; i < min(w_rhs, w_lhs); i++) {
		output[i] = lhs[i] + rhs[i];
	}
}
