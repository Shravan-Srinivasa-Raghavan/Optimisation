#include "matrix.h"
#include "threads.hpp"

matrix::matrix(uint64_t rowNum, uint64_t colNum){
    data = new float[rowNum*colNum];
    for (uint64_t i = 0 ; i < rowNum*colNum ; i++) data[i] = 0;
    rows = rowNum;
    cols = colNum;
}

matrix::matrix(uint64_t size){
    data = new float[size];
    for (uint64_t i = 0 ; i < size ; i++) data[i] = 0;
    rows = size;
    cols = 1;
}

matrix::matrix(__size dim){
    data = new float[dim.first*dim.second];
    for (uint64_t i = 0 ; i < dim.first*dim.second ; i++) data[i] = 0;
    rows = dim.first;
    cols = dim.second;
}

matrix::matrix(vf v){
    uint64_t size = v.size();
    data = new float[size];
    for (uint64_t i = 0 ; i < size ; i++) data[i] = v[i];
    rows = size;
    cols = 1;    
}

// Constructor assumes that v is a 2d vector and that every vector v[i] has the same size
matrix::matrix(vvf M){
    __size dim = {M.size(),M[0].size()};
    data = new float[dim.first*dim.second];
    rows = dim.first;
    cols = dim.second;
    for (uint64_t i = 0 ; i < rows ; i++){
        for (uint64_t j = 0 ; j < cols ; j++){
            data[i*cols + j] = M[i][j];
        }
    }
}

matrix::matrix(float* arr, uint64_t size){
    data = new float[size];
    for (uint64_t i = 0 ; i < size ; i++) data[i] = arr[i];
    rows = size;
    cols = 1;      
}


matrix::matrix(float** Matrix ,__size dim){
    data = new float[dim.first*dim.second];
    rows = dim.first;
    cols = dim.second;
    for (uint64_t i = 0 ; i < rows ; i++){
        for (uint64_t j = 0 ; j < cols ; j++){
            data[i*cols + j] = Matrix[i][j];
        }
    }    
}

matrix::matrix(){

}

matrix::matrix(const matrix& other) {
    data = other.data;
    rows = other.rows;
    cols = other.cols;
}

matrix& matrix::operator=(const matrix& other) {
    rows = other.rows;
    cols = other.cols;
    data = new float[rows*cols];
    for (uint64_t i = 0 ; i < rows*cols ; i++){
        this->data[i] = other.data[i];
    }

    return *this;
}

matrix operator+(const matrix& first, const matrix& second){
    if (first.rows!=second.rows && first.cols!=second.cols){
        throw std::invalid_argument("Cannot add ( "+ std::to_string(first.rows) +" , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )" );
    }
    else if (first.rows == second.rows && first.cols == second.cols) {
        __size dim = first.shape();
        uint64_t N = dim.first*dim.second;
        if (N > 1e3){
            matrix sum(dim);
            __m256 r1,r2,r3;
            for(uint64_t i = 0 ; i < N - N%8 ; i += 8){
                r1 = _mm256_loadu_ps(&first(i));
                r2 = _mm256_loadu_ps(&second(i));
                r3 = _mm256_add_ps(r1,r2);
                _mm256_storeu_ps(&sum(i),r3);
            }
            for (uint64_t i = N - N%8 ; i < N ; i++) {
                sum(i) = first(i) + second(i);
            }
            return sum;
        }
        else {
            matrix sum(dim);
            for (uint64_t i = 0 ; i < N ; i++){
                sum(i) = first(i) + second(i); 
            }
            return sum;
        }

    }
    else if (first.rows == second.rows) {
        if (first.cols == 1) { 
            matrix sum(second.rows, second.cols);
            for (uint64_t i = 0 ; i < second.rows ; i++) {
                for (uint64_t j = 0 ; j < second.cols ; j++) {
                    sum(i,j) = second(i,j) + first(i,0);
                }
            }
            return sum;
        }
        else if (second.cols == 1) {
            matrix sum(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows; i++) {
                for (uint64_t j = 0; j < first.cols; j++) {
                    sum(i,j) = first(i,j) + second(i,0);
                }
            }
            return sum;
        }
        else {
            throw std::invalid_argument("Cannot add ( "+ std::to_string(first.rows) +" , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )" );
        }
    }
    else if (first.cols == second.cols) {
        if (first.rows == 1) { 
            matrix sum(second.rows, second.cols);
            for (uint64_t i = 0 ; i < second.rows ; i++) {
                for (uint64_t j = 0 ; j < second.cols ; j++) {
                    sum(i,j) = second(i,j) + first(0,j);
                }
            }
            return sum;
        }
        else if (second.rows == 1) {
            matrix sum(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    sum(i,j) = first(i,j) + second(0,j);
                }
            }
            return sum;
        }
        else {
            throw std::invalid_argument("Cannot add ( "+ std::to_string(first.rows) +" , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )" );
        }
    }
    return zeros(1,1);
}

matrix operator-(const matrix& first, const matrix& second){
    if (first.rows!=second.rows && first.cols!=second.cols){
        throw std::invalid_argument("Cannot subtract ( "+ std::to_string(second.rows) +" , " + std::to_string(second.cols) + " ) from ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " )" );
        }
    else if (first.rows == second.rows && first.cols == second.cols) {
        __size dim = first.shape();
        uint64_t N = dim.first*dim.second;
        if (N > 1e3){
            matrix diff(dim);
            __m256 r1,r2,r3;
            for(uint64_t i = 0 ; i < N - N%8 ; i += 8){
                r1 = _mm256_loadu_ps(&first(i));
                r2 = _mm256_loadu_ps(&second(i));
                r3 = _mm256_sub_ps(r1,r2);
                _mm256_storeu_ps(&diff(i),r3);
            }
            for (uint64_t i = N - N%8 ; i < N ; i++) {
                diff(i) = first(i) - second(i);
            }
            return diff;
        }
        else {
            matrix diff(dim);
            for (uint64_t i = 0 ; i < N ; i++){
                diff(i) = first(i) - second(i);
            }
            return diff;
        }
    }
    else if (first.rows == second.rows) {
        if (first.cols == 1) { 
            matrix diff(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    diff(i,j) = first(i,0) - second(i,j);
                }
            }
            return diff;
        }
        else if (second.cols == 1) {
            matrix diff(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    diff(i,j) = first(i,j) - second(i,0);
                }
            }
            return diff;
        }
        else {
            throw std::invalid_argument("Cannot subtract ( "+ std::to_string(second.rows) +" , " + std::to_string(second.cols) + " ) from ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " )" );
        }
    }
    else if (first.cols == second.cols) {
        if (first.rows == 1) { 
            matrix diff(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    diff(i,j) = first(0,j) - second(i,j);
                }
            }
            return diff;
        }
        else if (second.rows == 1) {
            matrix diff(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    diff(i,j) = first(i,j) - second(0,j);
                }
            }
            return diff;
        }
        else {
            throw std::invalid_argument("Cannot subtract ( "+ std::to_string(second.rows) +" , " + std::to_string(second.cols) + " ) from ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " )" );
        }
    }
    return zeros(1,1);
}

matrix operator*(const matrix& first, const matrix& second){
    if (first.rows!=second.rows && first.cols!=second.cols){
        throw std::invalid_argument("Cannot multiply(elementwise) ( "+ std::to_string(first.rows) +" , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )" );
    }
    else if (first.rows == second.rows && first.cols == second.cols) {
        __size dim = first.shape();
        uint64_t N = dim.first*dim.second;
        if (N > 1e3){
            matrix prod(dim);
            __m256 r1,r2,r3;
            for(uint64_t i = 0 ; i < N - N%8 ; i += 8){
                r1 = _mm256_loadu_ps(&first(i));
                r2 = _mm256_loadu_ps(&second(i));
                r3 = _mm256_mul_ps(r1,r2);
                _mm256_storeu_ps(&prod(i),r3);
            }
            for (uint64_t i = N - N%8 ; i < N ; i++) {
                prod(i) = first(i)*second(i);
            }
            return prod;
        }
        else {
            matrix prod(dim);
            for (uint64_t i = 0 ; i < N ; i++) {
                prod(i) = first(i)*second(i);
            }
            return prod;
        }
    }
    else if (first.rows == second.rows) {
        if (first.cols == 1) { 
            matrix prod(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    prod(i,j) = first(i,0)*second(i,j);
                }
            }
            return prod;
        }
        else if (second.cols == 1) {
            matrix prod(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    prod(i,j) = first(i,j)*second(i,0);
                }
            }
            return prod;
        }
        else {
            throw std::invalid_argument("Cannot multiply ( "+ std::to_string(first.rows) +" , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )" );
        }
    }
    else if (first.cols == second.cols) {
        if (first.rows == 1) { 
            matrix prod(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    prod(i,j) = first(0,j)*second(i,j);
                }
            }
            return prod;
        }
        else if (second.rows == 1) {
            matrix prod(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    prod(i,j) = first(i,j)*second(0,j);
                }
            }
            return prod;
        }
        else {
            throw std::invalid_argument("Cannot multiply ( "+ std::to_string(first.rows) +" , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )" );
        }
    }
    return zeros(1,1);
}

matrix operator/(const matrix& first, const matrix& second){
    if (first.rows!=second.rows && first.cols!=second.cols){
        throw std::invalid_argument("Cannot divide ( "+ std::to_string(second.rows) +" , " + std::to_string(second.cols) + " ) from ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " )" );
    }
    else if (first.rows == second.rows && first.cols == second.cols) {
        __size dim = first.shape();
        uint64_t N = dim.first*dim.second;
        if (N > 1e3){
            matrix quotient(dim);
            __m256 r1,r2,r3;
            for(uint64_t i = 0 ; i < N - N%8 ; i += 8){
                r1 = _mm256_loadu_ps(&first(i));
                r2 = _mm256_loadu_ps(&second(i));
                r3 = _mm256_div_ps(r1,r2);
                _mm256_storeu_ps(&quotient(i),r3);
            }
            for (uint64_t i = N - N%8 ; i < N ; i++) {
                quotient(i) = first(i)/second(i);
            }
            return quotient;
        }
        else {
            matrix quotient(dim);
            for (uint64_t i = 0 ; i < N ; i++) {
                quotient(i) = first(i)/second(i);
            }
            return quotient;
        }
    }
    else if (first.rows == second.rows) {
        if (first.cols == 1) { 
            matrix quotient(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    quotient(i,j) = first(i,0)/second(i,j);
                }
            }
            return quotient;
        }
        else if (second.cols == 1) {
            matrix quotient(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    quotient(i,j) = first(i,j)/second(i,0);
                }
            }
            return quotient;
        }
        else {
            throw std::invalid_argument("Cannot divide ( "+ std::to_string(second.rows) +" , " + std::to_string(second.cols) + " ) from ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " )" );
        }
    }
    else if (first.cols == second.cols) {
        if (first.rows == 1) { 
            matrix quotient(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    quotient(i,j) = first(0,j)/second(i,j);
                }
            }
            return quotient;
        }
        else if (second.rows == 1) {
            matrix quotient(first.rows, first.cols);
            for (uint64_t i = 0 ; i < first.rows ; i++) {
                for (uint64_t j = 0 ; j < first.cols ; j++) {
                    quotient(i,j) = first(i,j)/second(0,j);
                }
            }
            return quotient;
        }
        else {
            throw std::invalid_argument("Cannot divide ( "+ std::to_string(second.rows) +" , " + std::to_string(second.cols) + " ) from ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " )" );
        }
    }
    return zeros(1,1);
}

matrix operator+(const float t, const matrix& first){
    __size dim = first.shape();
    uint64_t N = dim.first*dim.second;
    if (N > 1e4){
        matrix sum(dim);
        float v[8] = {t,t,t,t,t,t,t,t};
        __m256 r1,r2,r3;
        r1 = _mm256_loadu_ps(v);
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8) {
            r2 = _mm256_loadu_ps(&first(i));
            r3 = _mm256_add_ps(r1,r2);
            _mm256_storeu_ps(&sum(i),r3);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++){
            sum(i) = t + first(i);
        }
        return sum;
    }
    else {
        matrix sum(dim);
        for (uint64_t i = 0 ; i < N ; i++){
            sum(i) = t + first(i);
        }
        return sum;
    }
}

matrix operator-(const float t, const matrix& first){
    __size dim = first.shape();
    uint64_t N = dim.first*dim.second;
    if (N > 1e4){
        matrix diff(dim);
        float v[8] = {t,t,t,t,t,t,t,t};
        __m256 r1,r2,r3;
        r1 = _mm256_loadu_ps(v);
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8) {
            r2 = _mm256_loadu_ps(&first(i));
            r3 = _mm256_sub_ps(r1,r2);
            _mm256_storeu_ps(&diff(i),r3);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++){
            diff(i) = t - first(i);
        }
        return diff;
    }
    else {
        matrix diff(dim);
        for (uint64_t i = 0 ; i < N ; i++){
            diff(i) = t - first(i);
        }
        return diff;
    }
   
}

matrix operator*(const float t, const matrix& first){
    __size dim = first.shape();
    uint64_t N = dim.first*dim.second;
    if (N > 1e4){
        matrix prod(dim);
        float v[8] = {t,t,t,t,t,t,t,t};
        __m256 r1,r2,r3;
        r1 = _mm256_loadu_ps(v);
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8) {
            r2 = _mm256_loadu_ps(&first(i));
            r3 = _mm256_mul_ps(r1,r2);
            _mm256_storeu_ps(&prod(i),r3);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++){
            prod(i) = t*first(i);
        }
        return prod;
    }
    else {
        matrix prod(dim);
        for (uint64_t i = 0 ; i < N ; i++){
            prod(i) = t*first(i);
        }
        return prod;
    }
   
}

matrix operator/(const float t, const matrix& first){
    __size dim = first.shape();
    uint64_t N = dim.first*dim.second;
    if (N > 1e4){
        matrix quotient(dim);
        float v[8] = {t,t,t,t,t,t,t,t};
        __m256 r1,r2,r3;
        r1 = _mm256_loadu_ps(v);
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8) {
            r2 = _mm256_loadu_ps(&first(i));
            r3 = _mm256_div_ps(r1,r2);
            _mm256_storeu_ps(&quotient(i),r3);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++){
            quotient(i) = t/first(i);
        }
        return quotient;        
    }
    else {
        matrix quotient(dim);
        for (uint64_t i = 0 ; i < N ; i++){
            quotient(i) = t/first(i);
        }
        return quotient;
    }   
}

matrix operator+(const matrix& first, const float t) {
    __size dim = first.shape();
    uint64_t N = dim.first*dim.second;
    if (N > 1e4){
        matrix sum(dim);
        float v[8] = {t,t,t,t,t,t,t,t};
        __m256 r1,r2,r3;
        r2 = _mm256_loadu_ps(v);
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8) {
            r1 = _mm256_loadu_ps(&first(i));
            r3 = _mm256_add_ps(r1,r2);
            _mm256_storeu_ps(&sum(i),r3);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++){
            sum(i) = first(i) + t;
        }
        return sum;        
    }
    else {
        matrix sum(dim);
        for (uint64_t i = 0 ; i < N ; i++){
            sum(i) = first(i) + t;
        }
        return sum;
    }
}

matrix operator-(const matrix& first, const float t) {
    __size dim = first.shape();
    uint64_t N = dim.first*dim.second;
    if (N > 1e4){
        matrix diff(dim);
        float v[8] = {t,t,t,t,t,t,t,t};
        __m256 r1,r2,r3;
        r2 = _mm256_loadu_ps(v);
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8) {
            r1 = _mm256_loadu_ps(&first(i));
            r3 = _mm256_sub_ps(r1,r2);
            _mm256_storeu_ps(&diff(i),r3);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++){
            diff(i) = first(i) - t;
        }
        return diff;        
    }
    else {
        matrix diff(dim);
        for (uint64_t i = 0 ; i < N ; i++){
            diff(i) = first(i) - t;
        }
        return diff;
    }
}

matrix operator*(const matrix& first, const float t) {
    __size dim = first.shape();
    uint64_t N = dim.first*dim.second;
    if (N > 1e4){
        matrix prod(dim);
        float v[8] = {t,t,t,t,t,t,t,t};
        __m256 r1,r2,r3;
        r2 = _mm256_loadu_ps(v);
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8) {
            r1 = _mm256_loadu_ps(&first(i));
            r3 = _mm256_mul_ps(r1,r2);
            _mm256_storeu_ps(&prod(i),r3);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++){
            prod(i) = first(i)*t;
        }
        return prod;
    }
    else {
        matrix prod(dim);
        for (uint64_t i = 0 ; i < N ; i++){
            prod(i) = first(i)*t;
        }
        return prod;
    }
}

matrix operator/(const matrix& first, const float t) {
    __size dim = first.shape();
    uint64_t N = dim.first*dim.second;
    if (N > 1e4){
        matrix quotient(dim);
        float v[8] = {t,t,t,t,t,t,t,t};
        __m256 r1,r2,r3;
        r2 = _mm256_loadu_ps(v);
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8) {
            r1 = _mm256_loadu_ps(&first(i));
            r3 = _mm256_div_ps(r1,r2);
            _mm256_storeu_ps(&quotient(i),r3);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++){
            quotient(i) = first(i)/t;
        }
        return quotient;
    }
    else {
        matrix quotient(dim);
        for (uint64_t i = 0 ; i < N ; i++){
            quotient(i) = first(i)/t;
        }
        return quotient;
    }
}

matrix& operator+=(matrix& self,const matrix& other){
    __size d1 = self.shape(), d2 = other.shape();
    if (d1 != d2){
        throw std::invalid_argument("Cannot add ( "+ std::to_string(other.rows) +" , " + std::to_string(other.cols) + 
        " ) to ( " + std::to_string(self.rows) + " , " + std::to_string(self.cols) + " )" );
    }
    for (uint64_t i = 0 ; i < self.rows ; i++){
        for (uint64_t j  = 0 ; j < self.cols ; j++){
            self(i,j) += other(i,j);
        }
    }
    return self;
}

matrix& operator-=(matrix& self,const matrix& other){
    __size d1 = self.shape(), d2 = other.shape();
    if (d1 != d2){
        throw std::invalid_argument("Cannot subtract ( "+ std::to_string(other.rows) +" , " + std::to_string(other.cols) + 
        " ) from ( " + std::to_string(self.rows) + " , " + std::to_string(self.cols) + " )" );
    }
    for (uint64_t i = 0 ; i < self.rows ; i++){
        for (uint64_t j  = 0 ; j < self.cols ; j++){
            self(i,j) -= other(i,j);
        }
    }
    return self;
}

matrix& operator*=(matrix& self,const matrix& other){
    __size d1 = self.shape(), d2 = other.shape();
    if (d1 != d2){
        throw std::invalid_argument("Cannot multiply(elementwise) ( "+ std::to_string(other.rows) +" , " + std::to_string(other.cols) + 
        " ) with ( " + std::to_string(self.rows) + " , " + std::to_string(self.cols) + " )" );
    }
    for (uint64_t i = 0 ; i < self.rows ; i++){
        for (uint64_t j  = 0 ; j < self.cols ; j++){
            self(i,j) *= other(i,j);
        }
    }
    return self;
}

matrix& operator/=(matrix& self,const matrix& other){
    __size d1 = self.shape(), d2 = other.shape();
    if (d1 != d2){
        throw std::invalid_argument("Cannot divide(elementwise) ( "+ std::to_string(other.rows) +" , " + std::to_string(other.cols) + 
        " ) with ( " + std::to_string(self.rows) + " , " + std::to_string(self.cols) + " )" );
    }
    for (uint64_t i = 0 ; i < self.rows ; i++){
        for (uint64_t j  = 0 ; j < self.cols ; j++){
            self(i,j) /= other(i,j);
        }
    }
    return self;
}

matrix operator-(const matrix& M){
    matrix negation(M.rows,M.cols);
    for (uint64_t i = 0 ; i < M.rows ; i++){
        for (uint64_t j = 0 ; j < M.cols ; j++){
            negation(i,j) = -M(i,j);
        }
    }
    return negation;
}

matrix matmul(const matrix& first, const matrix& second){
    __size dim1 = first.shape();
    __size dim2 = second.shape();
    if( dim1.second != dim2.first){
        throw std::invalid_argument("Cannot matmul ( "+ std::to_string(dim1.first) +" , " + std::to_string(dim1.second) + " ) with ( " + std::to_string(dim2.first) + " , " + std::to_string(dim2.second) + " )" );
    }
    else { 
        uint64_t N = std::max(dim1.first*dim1.second,dim2.first*dim2.second);
        if (N > 1e5){
            uint64_t n = dim1.first, m = dim1.second, p = dim2.second;
            matrix Result(n,p);
            Device device (select_device_with_most_flops());
            Memory<float> A(device,n*m); 
            Memory<float> B(device,m*p); 
            Memory<float> C(device,n*p); 
            
            Kernel matMul(device,n*p,"matMul",A,B,C,n,m,p);
            
            for (uint64_t i = 0 ; i < n ; i++) {
                for (uint64_t j = 0 ; j < m ; j++) {
                    A[i*m + j] = first(i,j);
                }
            }

            for (uint64_t i = 0 ; i < m ; i++) {
                for (uint64_t j = 0 ; j < p ; j++) {
                    B[i*p + j] = second(i,j);
                }
            }
            for (uint64_t i = 0 ; i < n ; i++) {
                for (uint64_t j = 0 ; j < p ; j++) {
                    C[i*p + j] = 0;
                }
            }
            
            A.write_to_device(); 
            B.write_to_device();
            matMul.run(); 
            C.read_from_device();

            for (uint64_t i = 0 ; i < m ; i++) {
                for (uint64_t j = 0 ; j < p ; j++) {
                    Result(i,j) = C[i*p + j] ;
                }
            }
            return Result;
        }
        else {
            matrix net(dim1.first,dim2.second);
            for(uint64_t k = 0 ; k < dim1.second ; k++){
                for(uint64_t i = 0 ; i < dim1.first ; i++){
                    for(uint64_t j = 0 ;j < dim2.second ; j++){
                        net(i,j) +=first(i,k)*second(k,j);
                    }
                }
            }
            return net;
        }
    }
}

float dot(const matrix& first, const matrix& second){
    __size dim1 = first.shape();
    __size dim2 = second.shape();
    if (dim1.first != 1 || dim1.second != dim2.first || dim2.second != 1){
        throw std::invalid_argument("Cannot dot vector with dimensions ( "+ std::to_string(dim1.first) +" , " + std::to_string(dim1.second) + " ) with  a vector with dimensions ( " + std::to_string(dim2.first) + " , " + std::to_string(dim2.second) + " )" );
    }
    uint64_t N = dim1.second;
    float d = 0;
    if (N < 100000) {
        for (uint64_t i = 0 ; i < N ; i++){
            d += first(0,i)*second(i,0);
        }
        return d;
    }
    else {
        __m256 v1,v2,v3;
        __m128 v4,zero = _mm_setzero_ps();
        float d1,d2;
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8){
            v1 = _mm256_loadu_ps(&first(0,i));
            v2 = _mm256_loadu_ps(&second(i,0));
            v3 = _mm256_mul_ps(v1,v2);
            v4 = _mm256_extractf128_ps(v3,0);
            v4 = _mm_hadd_ps(v4,zero);
            v4 = _mm_hadd_ps(v4,zero);
            d1 = _mm_cvtss_f32(v4);
            v4 = _mm256_extractf128_ps(v3,1);
            v4 = _mm_hadd_ps(v4,zero);
            v4 = _mm_hadd_ps(v4,zero);
            d2 = _mm_cvtss_f32(v4);
            d += (d1 + d2);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++ ){
            d += first(0,i)*second(i,0);
        }
        return d;
    }
}

float norm(const matrix& v){
    __size dim = v.shape();
    if (dim.first != 1 && dim.second != 1){
        throw std::invalid_argument("Cannot compute norm of vector with dimensions ( "+ std::to_string(dim.first)+" , "+ std::to_string(dim.second) + " )");
    } 
    uint64_t N = std::max(dim.first,dim.second);
    float n = 0;
    if (N < 1e5) {
        for (uint64_t i = 0 ; i < N ; i++){
            n += v(i)*v(i);
        }
        return std::sqrt(n);
    }
    else {
        __m256 v1,v2,v3;
        __m128 v4,zero = _mm_setzero_ps();
        float d1,d2;
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8){
            v1 = _mm256_loadu_ps(&v(i));
            v2 = _mm256_loadu_ps(&v(i));
            v3 = _mm256_mul_ps(v1,v2);
            v4 = _mm256_extractf128_ps(v3,0);
            v4 = _mm_hadd_ps(v4,zero);
            v4 = _mm_hadd_ps(v4,zero);
            d1 = _mm_cvtss_f32(v4);
            v4 = _mm256_extractf128_ps(v3,1);
            v4 = _mm_hadd_ps(v4,zero);
            v4 = _mm_hadd_ps(v4,zero);
            d2 = _mm_cvtss_f32(v4);
            n += (d1 + d2);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++ ){
            n += v(i)*v(i);
        }
        return std::sqrt(n);
    }
}

matrix matrix::inverse(){
    matrix a = *this;
    __size dim = a.shape();
    if (dim.first != dim.second) 
        throw std::invalid_argument("Cannot invert ( "+ std::to_string(dim.first) +" , " + std::to_string(dim.second) + " )");
    uint64_t n = a.rows;
    matrix augmented(n, 2 * n);
    // Initialize the augmented matrix with the identity matrix on the right
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            augmented(i, j) = a(i, j);
            augmented(i, j + n) = (i == j) ? 1.0 : 0.0;
        }
    }
    // Perform Gauss-Jordan elimination
    for (uint64_t i = 0; i < n; ++i) {
        // Find the pivot
        float pivot = augmented(i, i);
        if (pivot == 0.0) {
            throw runtime_error("Matrix is singular and cannot be inverted.");
        }

        // Normalize the pivot row
        for (uint64_t j = 0; j < 2 * n; ++j) {
            augmented(i, j) /= pivot;
        }

        // Eliminate the current column in other rows
        for (uint64_t k = 0; k < n; ++k) {
            if (k != i) {
                float factor = augmented(k, i);
                for (uint64_t j = 0; j < 2 * n; ++j) {
                    augmented(k, j) -= factor * augmented(i, j);
                }
            }
        }
    }
    // Extract the inverse matrix from the augmented matrix
    matrix result(n, n);
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            result(i, j) = augmented(i, j + n);
        }
    }
    return result;
}

matrix matrix::transpose(){
    __size dim = this->shape();
    matrix T(dim.second,dim.first);
    for (uint64_t i = 0 ; i < dim.first ; i++){
        for (uint64_t j = 0 ; j < dim.second ; j++){
            T(j,i) = (*this)(i,j);
        }
    }
    return T;
}

float matrix::determinant(){
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to calculate determinant");
    }
    uint64_t n = rows;
    matrix a(*this); // Make a copy of the matrix

    float det = 1;
    for (uint64_t i = 0; i < n; ++i) {
        // Find the pivot
        uint64_t pivot = i;
        for (uint64_t j = i + 1; j < n; ++j) {
            if (abs(a(j,i)) > abs(a(pivot,i))) {
                pivot = j;
            }
        }

        // Swap rows if needed
        if (pivot != i) {
            for (uint64_t k = 0; k < n; ++k) {
                std::swap(a(i,k), a(pivot,k));
            }
            det *= -1; // Swap changes the sign of the determinant
        }

        // Check for zero pivot
        if (a(i,i) == 0) {
            return 0; // Determinant is zero
        }

        // Eliminate the column
        for (uint64_t j = i + 1; j < n; ++j) {
            float factor = a(j,i) / a(i,i);
            for (uint64_t k = i; k < n; ++k) {
                a(j,k) -= factor * a(i,k);
            }
        }

        // Multiply the diagonal elements
        det *= a(i,i);
    }

    return det;
}

float matrix::sum(){
    float Sum = 0;
    uint64_t N = this->rows*this->cols;
    if (N > 1e4){
        __m256 r1;
        __m128 r2,r3,r4,zero = _mm_setzero_ps();
        for (uint64_t i = 0 ; i < N - N%8 ; i += 8){
            r1 = _mm256_loadu_ps(&(*this)(i));
            r2 = _mm256_extractf128_ps(r1,1);
            r3 = _mm256_extractf128_ps(r1,0);
            r4 = _mm_hadd_ps(r2,r3);
            r4 = _mm_hadd_ps(r4,zero);
            r4 = _mm_hadd_ps(r4,zero);
            Sum += _mm_cvtss_f32(r4);
        }
        for (uint64_t i = N - N%8 ; i < N ; i++){
            Sum += (*this)(i);
        }
    }
    else {
        for (uint64_t i = 0 ; i  < N ; i++){
            Sum += (*this)(i);
        }
    }
    return Sum;
}

matrix matrix::sum(uint axis){
    if (axis < 0 || axis > 1) throw std::invalid_argument("Axis must be 0 or 1");
    __size dim = this->shape();
    uint64_t n = dim.first, m = dim.second;
    uint64_t N = n*m;
    matrix Sum(axis == 0 ? 1 : n, axis == 0 ? m : 1);
    if (N > 1e6){
        uint64_t thread_size = N/8;
        thread* T = new thread[NUM_THREADS];
        for (uint i = 0 ; i <  8 ; i++){
            T[i] = thread(threadSum,&Sum,this,i*thread_size,(i + 1)*thread_size,axis,m);
        }
        for (uint i = 0 ; i < 8 ; i++) T[i].join();
        for (uint64_t i = N - N%8 ; i < N ; i++){
            Sum(axis == 0 ? 0 : i/m, axis == 0 ? i%m : 0) += (*this)(i);
        }
        delete [] T;

    }
    else {
        for (uint64_t i = 0 ; i < N ; i++){
            Sum(axis == 0 ? 0 : i/m, axis == 0 ? i%m : 0) += (*this)(i);
        }
    }
    return Sum;
}

matrix zeros(uint64_t rows, uint64_t cols){
    return matrix(rows,cols);
}

matrix zeros(uint64_t size){
    return matrix(size);
}

matrix zeros(__size dim){
    return zeros(dim.first,dim.second);
}

matrix eye(uint64_t size){
    matrix diag(size,size);
    for(uint64_t i = 0 ; i < size ; i++){
        diag(i,i) = 1;
    }
    return diag;
}

matrix eye(uint64_t rows, uint64_t cols){
    matrix diag(rows,cols);
    for(uint64_t i = 0 ; i < min(rows,cols) ; i++){
        diag(i,i)=1;
    }
    return diag;
}

matrix identity(uint64_t size){
    return eye(size);
}

matrix max(matrix &arr,int axis) {
    if (axis < 0 || axis > 1) throw std::invalid_argument("Axis must be 0 or 1");
    
    float twoRows,twoCols;
    twoRows = arr.shape().first;
    twoCols = arr.shape().second;

    matrix result(axis == 0 ? 1 : twoRows, axis == 0 ? twoCols : 1);

    if (axis == 0) {  //largest each col 
        for (uint64_t col = 0; col < twoCols; ++col) {
            float max_value = arr(0, col);
            for (uint64_t row = 1; row < twoRows; ++row) {
                if (arr(row, col) > max_value) {
                    max_value = arr(row, col);
                }
            }
            result(0, col) = max_value;
        }
    } 
    else {  //largest for each row
        for (uint64_t row = 0; row < twoRows; ++row) {
            float max_value = arr(row, 0);
            for (uint64_t col = 1; col < twoCols; ++col) {
                if (arr(row, col) > max_value) {
                    max_value = arr(row, col);
                }
            }
            result(row, 0) = max_value;
        }
    }
    return result;
}

matrix max (matrix &arr) {
    float arrRows, arrCols;
    arrRows = arr.shape().first;
    arrCols = arr.shape().second;
    if (arrRows != 1 && arrCols != 1) {
        matrix result(1, arrCols);
        for (uint64_t col = 0; col < arrCols; ++col) {
            float max_value = arr(0, col);
            uint64_t max_index = 0;
            for (uint64_t row = 1; row < arrRows; ++row) {
                if (arr(row, col) > max_value) {
                    max_value = arr(row, col);
                    max_index = row;
                }
            }
            result(0, col) = max_value;
        }
        return result;
    }
    else if (arrRows == 1) {
        matrix t(1,1);
        t(0,0) = arr(0,0);
        for (uint64_t col = 1; col < arrCols; ++col) {
            t(0,0) = max(t(0,0),arr(0,col));
        }
        return t;
    }
    else {
        matrix t(1,1);
        t(0,0) = arr(0,0);
        for (uint64_t row = 1; row < arrRows; ++row) {
            t(0,0) = max(t(0,0),arr(row,0));
        }
        return t;
    }
}

matrix argmax(matrix &arr,int axis) {
    if (axis < 0 || axis > 1) throw std::invalid_argument("Axis must be 0 or 1");
    
    float twoRows,twoCols;
    twoRows = arr.shape().first;
    twoCols = arr.shape().second;

    matrix result(axis == 0 ? 1 : twoRows, axis == 0 ? twoCols : 1);

    if (axis == 0) {  // Argmax along columns (resulting in row vector)
        for (uint64_t col = 0; col < twoCols; ++col) {
            float max_value = arr(0, col);
            uint64_t max_index = 0;
            for (uint64_t row = 1; row < twoRows; ++row) {
                if (arr(row, col) > max_value) {
                    max_value = arr(row, col);
                    max_index = row;
                }
            }
            result(0, col) = max_index;
        }
    } else {  // Argmax along rows (resulting in column vector)
        for (uint64_t row = 0; row < twoRows; ++row) {
            float max_value = arr(row, 0);
            uint64_t max_index = 0;
            for (uint64_t col = 1; col < twoCols; ++col) {
                if (arr(row, col) > max_value) {
                    max_value = arr(row, col);
                    max_index = col;
                }
            }
            result(row, 0) = max_index;
        }
    }

    return result;
}

matrix argmax (matrix &arr) {
    float arrRows, arrCols;
    arrRows = arr.shape().first;
    arrCols = arr.shape().second;
    if (arrRows != 1 && arrCols != 1) {
        matrix result(1, arrCols);
        for (uint64_t col = 0; col < arrCols; ++col) {
            float max_value = arr(0, col);
            uint64_t max_index = 0;
            for (uint64_t row = 1; row < arrRows; ++row) {
                if (arr(row, col) > max_value) {
                    max_value = arr(row, col);
                    max_index = row;
                }
            }
            result(0, col) = max_index;
        }
        return result;
    }
    else if (arrRows == 1) {
        matrix t(1,1);
        t(0,0) = 0;
        for (uint64_t col = 1; col < arrCols; ++col) {
            if (arr(0,t(0,0)) < arr(col)) {
                t(0,0) = col;
            }
        }
        return t;
    }
    else {
        matrix t(1,1);
        t(0,0) = 0;
        for (uint64_t row = 1; row < arrRows; ++row) {
            if (arr(t(0,0),0) < arr(row)) {
                t(0,0) = row;
            }
        }
        return t;
    }
}

matrix min(matrix &arr,int axis) {
    if (axis < 0 || axis > 1) throw std::invalid_argument("Axis must be 0 or 1");
    
    float twoRows,twoCols;
    twoRows = arr.shape().first;
    twoCols = arr.shape().second;

    matrix result(axis == 0 ? 1 : twoRows, axis == 0 ? twoCols : 1);

    if (axis == 0) {  //largest each col 
        for (uint64_t col = 0; col < twoCols; ++col) {
            float max_value = arr(0, col);
            for (uint64_t row = 1; row < twoRows; ++row) {
                if (arr(row, col) < max_value) {
                    max_value = arr(row, col);
                }
            }
            result(0, col) = max_value;
        }
    } 
    else {  //largest for each row
        for (uint64_t row = 0; row < twoRows; ++row) {
            float max_value = arr(row, 0);
            for (uint64_t col = 1; col < twoCols; ++col) {
                if (arr(row, col) < max_value) {
                    max_value = arr(row, col);
                }
            }
            result(row, 0) = max_value;
        }
    }
    return result;
}

matrix min (matrix &arr) {
    float arrRows, arrCols;
    arrRows = arr.shape().first;
    arrCols = arr.shape().second;
    if (arrRows != 1 && arrCols != 1) {
        matrix result(1, arrCols);
        for (uint64_t col = 0; col < arrCols; ++col) {
            float max_value = arr(0, col);
            uint64_t max_index = 0;
            for (uint64_t row = 1; row < arrRows; ++row) {
                if (arr(row, col) < max_value) {
                    max_value = arr(row, col);
                    max_index = row;
                }
            }
            result(0, col) = max_value;
        }
        return result;
    }
    else if (arrRows == 1) {
        matrix t(1,1);
        t(0,0) = arr(0,0);
        for (uint64_t col = 1; col < arrCols; ++col) {
            t(0,0) = min(t(0,0),arr(0,col));
        }
        return t;
    }
    else {
        matrix t(1,1);
        t(0,0) = arr(0,0);
        for (uint64_t row = 1; row < arrRows; ++row) {
            t(0,0) = min(t(0,0),arr(row,0));
        }
        return t;
    }
}

matrix argmin(matrix &arr,int axis) {
    if (axis < 0 || axis > 1) throw std::invalid_argument("Axis must be 0 or 1");
    
    float twoRows,twoCols;
    twoRows = arr.shape().first;
    twoCols = arr.shape().second;

    matrix result(axis == 0 ? 1 : twoRows, axis == 0 ? twoCols : 1);

    if (axis == 0) {  // Argmax along columns (resulting in row vector)
        for (uint64_t col = 0; col < twoCols; ++col) {
            float max_value = arr(0, col);
            uint64_t max_index = 0;
            for (uint64_t row = 1; row < twoRows; ++row) {
                if (arr(row, col) < max_value) {
                    max_value = arr(row, col);
                    max_index = row;
                }
            }
            result(0, col) = max_index;
        }
    } 
    else {  // Argmax along rows (resulting in column vector)
        for (uint64_t row = 0; row < twoRows; ++row) {
            float max_value = arr(row, 0);
            uint64_t max_index = 0;
            for (uint64_t col = 1; col < twoCols; ++col) {
                if (arr(row, col) < max_value) {
                    max_value = arr(row, col);
                    max_index = col;
                }
            }
            result(row, 0) = max_index;
        }
    }
    return result;
}

matrix argmin (matrix &arr) {
    float arrRows, arrCols;
    arrRows = arr.shape().first;
    arrCols = arr.shape().second;
    if (arrRows != 1 && arrCols != 1) {
        matrix result(1, arrCols);
        for (uint64_t col = 0; col < arrCols; ++col) {
            float max_value = arr(0, col);
            uint64_t max_index = 0;
            for (uint64_t row = 1; row < arrRows; ++row) {
                if (arr(row, col) < max_value) {
                    max_value = arr(row, col);
                    max_index = row;
                }
            }
            result(0, col) = max_index;
        }
        return result;
    }
    else if (arrRows == 1) {
        matrix t(1,1);
        t(0,0) = 0;
        for (uint64_t col = 1; col < arrCols; ++col) {
            if (arr(0,t(0,0)) > arr(col)) {
                t(0,0) = col;
            }
        }
        return t;
    }
    else {
        matrix t(1,1);
        t(0,0) = 0;
        for (uint64_t row = 1; row < arrRows; ++row) {
            if (arr(t(0,0),0) > arr(row)) {
                t(0,0) = row;
            }
        }
        return t;
    }
}

matrix ones(uint64_t rows, uint64_t cols) {
    uint64_t N = rows*cols;
    matrix res(rows,cols);
    if (N > 1e6){
        uint64_t thread_size = N/8;
        thread* T = new thread[NUM_THREADS];
        for (uint i = 0 ; i < 8 ; i++){
            T[i] = thread(threadOnes,&res,i*thread_size,(i + 1)*thread_size);
        }
        for (uint i = 0 ; i < 8 ; i++) T[i].join();
        for (uint64_t i = N - N%8 ; i < N ; i++) res(i) = 1;
    }
    else {
        for (uint64_t i = 0; i < N ; i++) {
            res(i) = 1;
        }
    }
    return res;
}

matrix ones(__size dim){
    return ones(dim.first,dim.second);
}

matrix fabs(matrix &a) {
    __size dim = a.shape();
    matrix res(dim);
    uint64_t N = dim.first*dim.second;
    if (N > 1e6){
        uint64_t thread_size = N/8;
        thread* T = new thread[NUM_THREADS];
        for (uint i = 0 ; i < 8 ; i++){
            T[i] = thread(threadFabs,&res,&a,i*thread_size,(i + 1)*thread_size);
        }
        for (uint i = 0 ; i < 8 ; i++) T[i].join();
        for (uint64_t i = N - N%8 ; i < N ; i++) res(i) = std::fabs(a(i));
    }
    else {
        for (uint64_t i = 0 ;i < N ; i++){
            res(i) = std::fabs(a(i));
        }
    }
    return res;
}

matrix exp(matrix &a) {
    __size dim = a.shape();
    matrix res(dim);
    uint64_t N = dim.first*dim.second;
    if (N > 1e6){
        uint64_t thread_size = N/8;
        thread* T = new thread[NUM_THREADS];
        for (uint i = 0 ; i < 8 ; i++){
            T[i] = thread(threadExp,&res,&a,i*thread_size,(i + 1)*thread_size);
        }
        for (uint i = 0 ; i < 8 ; i++) T[i].join();
        for (uint64_t i = N - N%8 ; i < N ; i++) res(i) = std::exp(a(i));
    }
    else {
        for (uint64_t i = 0 ;i < N ; i++){
            res(i) = std::exp(a(i));
        }
    }
    return res;
}

matrix tanh(matrix &a) {
    __size dim = a.shape();
    matrix res(dim);
    uint64_t N = dim.first*dim.second;
    if (N > 1e6){
        uint64_t thread_size = N/8;
        thread* T = new thread[NUM_THREADS];
        for (uint i = 0 ; i < 8 ; i++){
            T[i] = thread(threadTanh,&res,&a,i*thread_size,(i + 1)*thread_size);
        }
        for (uint i = 0 ; i < 8 ; i++) T[i].join();
        for (uint64_t i = N - N%8 ; i < N ; i++) res(i) = std::tanh(a(i));
    }
    else {
        for (uint64_t i = 0 ;i < N ; i++){
            res(i) = std::tanh(a(i));
        }
    }
    return res;
}

matrix log(matrix &a, float logbase) {
    __size dim = a.shape();
    matrix res(dim);
    uint64_t N = dim.first*dim.second;
    if (N > 1e6){
        uint64_t thread_size = N/8;
        thread* T = new thread[NUM_THREADS];
        for (uint i = 0 ; i < 8 ; i++){
            T[i] = thread(threadLogb,&res,&a,logbase,i*thread_size,(i + 1)*thread_size);
        }
        for (uint i = 0 ; i < 8 ; i++) T[i].join();
        for (uint64_t i = N - N%8 ; i < N ; i++) res(i) = std::log(a(i))/std::log(logbase);
    }
    else {
        for (uint64_t i = 0 ;i < N ; i++){
            res(i) = std::log(a(i))/std::log(logbase);
        }
    }
    return res;
}

matrix log(matrix &a) {
    __size dim = a.shape();
    matrix res(dim);
    uint64_t N = dim.first*dim.second;
    if (N > 1e6){
        uint64_t thread_size = N/8;
        thread* T = new thread[NUM_THREADS];
        for (uint i = 0 ; i < 8 ; i++){
            T[i] = thread(threadLog,&res,&a,i*thread_size,(i + 1)*thread_size);
        }
        for (uint i = 0 ; i < 8 ; i++) T[i].join();
        for (uint64_t i = N - N%8 ; i < N ; i++) res(i) = std::log(a(i));
    }
    else {
        for (uint64_t i = 0 ;i < N ; i++){
            res(i) = std::log(a(i));
        }
    }
    return res;
}

matrix sqrt(matrix &a) {
    __size dim = a.shape();
    matrix res(dim);
    uint64_t N = dim.first*dim.second;
    if (N > 1e6){
        uint64_t thread_size = N/8;
        thread* T = new thread[NUM_THREADS];
        for (uint i = 0 ; i < 8 ; i++){
            T[i] = thread(threadSqrt,&res,&a,i*thread_size,(i + 1)*thread_size);
        }
        for (uint i = 0 ; i < 8 ; i++) T[i].join();
        for (uint64_t i = N - N%8 ; i < N ; i++) res(i) = std::sqrt(a(i));
    }
    else {
        for (uint64_t i = 0 ;i < N ; i++){
            res(i) = std::sqrt(a(i));
        }
    }
    return res;
}