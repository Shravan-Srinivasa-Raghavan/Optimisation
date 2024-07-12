#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



kernel void add_kernel(global float* A, global float* B, global float* C) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	const uint n = get_global_id(0);
	C[n] = A[n]+B[n];
}

kernel void mul_kernel(global float* A, global float* B, global float* C) {
	const uint n = get_global_id(0);
	C[n] = A[n]*B[n];
	return;
}

kernel void matMul (__global float* A, __global float *B, __global float *C, int aCol, int cRow, int cCol) {
	const uint n = get_global_id(0);
	uint i = n/aCol, j = n%aCol;
	for (uint k = 0 ; k < aCol ; k++){
		C[n] += A[i*aCol + k]*B[k*cRow + j];
	}
	return;
}


);} // ############################################################### end of OpenCL C code #####################################################################
