#ifndef THREAD
#define THREAD
#include "matrix.h"

#define NUM_THREADS 8

void threadSum(matrix* result, const matrix* input, uint64_t start, uint64_t end, uint axis, uint64_t cols);

void threadOnes(matrix* result, uint64_t start, uint64_t end);

void threadFabs(matrix* result, const matrix* input, uint64_t start, uint64_t end);

void threadExp(matrix* result, const matrix* input, uint64_t start, uint64_t end);

void threadTanh(matrix* result, const matrix* input, uint64_t start, uint64_t end);

void threadLogb(matrix* result, const matrix* input, float base, uint64_t start, uint64_t end);

void threadLog(matrix* result, const matrix* input, uint64_t start, uint64_t end);

void threadSqrt(matrix* result, const matrix* input, uint64_t start, uint64_t end);

#endif
