#include "threads.hpp"

void threadSum(matrix* result, const matrix* input, uint64_t start, uint64_t end, uint axis, uint64_t cols){
    for (uint64_t i = start ; i < end ; i++){
        (*result)(axis == 0 ? 0 : i/cols, axis == 0 ? i%cols : 0) += (*input)(i);
    }
}


void threadOnes(matrix* result, uint64_t start, uint64_t end){
    for (uint64_t i = start ; i < end ; i++){
        (*result)(i) = 1;
    }
}

void threadFabs(matrix* result, const matrix* input, uint64_t start, uint64_t end){
    for (uint64_t i = start ; i < end ; i++){
        (*result)(i) = std::fabs((*input)(i));
    }
}

void threadExp(matrix* result, const matrix* input, uint64_t start, uint64_t end){
    for (uint64_t i = start ; i < end ; i++){
        (*result)(i) = std::exp((*input)(i));
    }
}

void threadTanh(matrix* result, const matrix* input, uint64_t start, uint64_t end){
    for (uint64_t i = start ; i < end ; i++){
        (*result)(i) = std::tanh((*input)(i));
    }
}

void threadLogb(matrix* result, const matrix* input, float base, uint64_t start, uint64_t end){
    for (uint64_t i = start ; i < end ; i++){
        (*result)(i) = std::log((*input)(i))/std::log(base);
    }
}

void threadLog(matrix* result, const matrix* input, uint64_t start, uint64_t end){
    for (uint64_t i = start ; i < end ; i++){
        (*result)(i) = std::log((*input)(i));
    }
}

void threadSqrt(matrix* result, const matrix* input, uint64_t start, uint64_t end){
    for (uint64_t i = start ; i < end ; i++){
        (*result)(i) = std::sqrt((*input)(i));
    }
}