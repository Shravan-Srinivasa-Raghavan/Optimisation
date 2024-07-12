#ifndef MATRIX
#define MATRIX
#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>
#include <immintrin.h>
#include <ammintrin.h>
#include <thread>
#include "opencl.hpp"

#define PB push_back

using namespace std;

typedef pair<unsigned long,unsigned long> __size;
typedef vector<float> vf;
typedef vector<vf> vvf;

class matrix{
    private:
        // The Matrix
        uint64_t rows; // Number of rows of the matrix

        uint64_t cols; // Number of columns of the matrix 

        float* data; // The data that the matrix has i.e. matrix itself

    public:
        // Constructor
        matrix(uint64_t rowNum, uint64_t colNum); // Constructor, creates a matrix of 0s of dimensions (rowNum, colNum)

        matrix(uint64_t size); // 1D matrix constructor, creates a 1D matrix (vector) of dimensions (size,1)

        matrix(__size dim); // A 2D matrix constructor, creates a 2D matrix of dimensions (dim.first,dim.second) 

        matrix(vf v); // A vector of floats
        
        matrix(vvf M); // A 2d vector of floats

        matrix(float* arr, uint64_t size); // An array of floats
        
        matrix(float** Matrix, __size dim); // A 2d array of floats

        matrix(); // Default constructor

        // Destructor
        ~matrix(){}; // Default destructor
        
        // Utility Functions

        // Usage : To print a matrix M, use M.printMatrix()  
        void printMatrix(){ 
            for (int i = 0 ; i < rows ; i++){
                for (int j = 0; j < cols ; j++){
                    cout << data[i * cols + j]<<" ";
                }
                cout << "\n";
            }
        }

        // Usage : To access a reference of data[i*cols + j], use M(i,j)
        float& operator()(uint64_t row_i, uint64_t col_i) {
            if(row_i >= rows ||col_i >= cols) throw out_of_range("Index out of bounds");
            else return data[row_i * cols + col_i];
        }

        // Usage : To access a const reference of data[i*cols + j], use M(i,j)
        const float& operator()(uint64_t row_i, uint64_t col_i) const{
            if(row_i >= rows ||col_i >= cols) throw out_of_range("Index out of bounds");
            else return data[row_i * cols + col_i];
        }

        // Usage : To access a reference of data[i], use M(i) where M is the matrix object
        float& operator()(uint64_t i) {
            return (*this).data[i];
        }
        
        // Usage : To access a const reference of data[i], use M(i) where M is the matrix object
        const float& operator()(uint64_t i)const{
            return (*this).data[i];
        }

        // Usage : To access a rows of the matrix, use M[i] where M is the matrix object
        matrix operator[](uint64_t i){ 
            matrix v(1,cols);
            for (uint64_t j = 0 ; j < rows ; j++){
                v(0,j) = (*this)(i,j);
            }
            return v;
        }

        // This function returns the dimensions of the matrix
        __size shape() const{
            return make_pair(rows,cols);
        }

        // Operators

        // Copy constructor, creates a matrix with same rows, cols and data as `other`
        matrix(const matrix& other);                                    
        
        // Equality operator, same functionality as copy constructor                                                                                                            
        matrix& operator=(const matrix& other);                               

        // Elementwise addition operator, A[i][j] = first[i][j] + second[i][j]
        friend matrix operator+(const matrix& first, const matrix& second);    
        
        // Elementwise subtraction operator, A[i][j] = first[i][j] - second[i][j]                                                                                        
        friend matrix operator-(const matrix& first, const matrix& second); 

        // Elementwise multiplication operator, A[i][j] = first[i][j]*second[i][j]
        friend matrix operator*(const matrix& first, const matrix& second); 

        // Elementwise division operator, A[i][j] = first[i][j]/second[i][j]                                                                                                    
        friend matrix operator/(const matrix& first, const matrix& second);     

        // float addition operator : A[i][j] = first[i][j]+t
        friend matrix operator+(const matrix& first, const float t);                                

        // float subtraction operator : A[i][j] = first[i][j]-t
        friend matrix operator-(const matrix& first, const float t);

        // float multiplication operator : A[i][j] = first[i][j]*t
        friend matrix operator*(const matrix& first, const float t);                          
        
        // float division operator : A[i][j] = first[i][j]/t
        friend matrix operator/(const matrix& first, const float t);

        // float addition operator : A[i][j] = first[i][j] + t
        friend matrix operator+(const float t, const matrix& first); 
        
        // float subtraction operator : A[i][j] = t - first[i][j]
        friend matrix operator-(const float t, const matrix& first); 
       
        // float multiplication operator : A[i][j] = first[i][j]*t
        friend matrix operator*(const float t, const matrix& first); 
        
        // float division operator : A[i][j] = t/first[i][j]
        friend matrix operator/(const float t, const matrix& first);
       
        // Adds other elementwise to self and stores it in self
        friend matrix& operator+=(matrix& self, const matrix& other); 

        // Subtracts other elementwise from self and stores it in self
        friend matrix& operator-=(matrix& self, const matrix& other); 

        // Multiplies other elementwise with self and stores it in self
        friend matrix& operator*=(matrix& self, const matrix& other); 
        
        // Divides other elementwise with self and stores it in self
        friend matrix& operator/=(matrix& self, const matrix& other); 

        // Negates the entire matrix
        friend matrix operator-(const matrix& M); 

        // Matrix Operations

        // Transpose function : A.transpose()(i,j) = A(j,i)
        matrix transpose(); 

        // Inverse function : matmul(A.inverse(),A) = I
        matrix inverse(); 

        // Determinant function : A.determinant() = det(A)
        float determinant(); 

        // Sum function : float S = A.sum() = \sum_{i = 0}^{i = rows - 1} \sum_{j = 0}^{j = cols - 1} A(i,j)
        float sum(); 
        
        // Sum function : matrix S = A.sum(axis) = sums along rows if axis = 0 and sums along columns if axis = 1
        matrix sum(uint axis);
};

matrix matmul(const matrix& first, const matrix& second); // matmul function, A = first x second

float dot(const matrix& first, const matrix& second); // dot product of first and second, d = first o second

float norm(const matrix& v); // norm of a vector(l2 norm only) defined as \root(\Sigma v(i,0)*v(i,0)). Defined only for 1d matrices

// Basic Functions

matrix zeros(uint64_t rows, uint64_t cols); // zeros function : A[i][j] = 0 for all i in [0,rows) and for all j in [0,cols)

matrix zeros(__size dim); // zeros function : A[i][j] = 0 for all i in [0,dim.first) and for all j in [0,dim.second)

matrix zeros(uint64_t size); // 1D zeros function : A[i] = 0 for all i in [0,size)

matrix eye(uint64_t size); // 1D eye function : A = I(size) where I(x) is identity matrix with dimensions x https://en.wikipedia.org/wiki/Identity_matrix

matrix eye(uint64_t rows, uint64_t cols); // eye function : A = I(min(rows,cols)), zeros everywhere else

matrix ones(uint64_t rows, uint64_t cols); // ones function : A = ones((rows,cols))

matrix ones(__size dim); // ones function : A = ones(dim.first,dim.second)

matrix identity(uint64_t size); // identity matrix function : A = I(size)

// Minmax Functions
/*

An explanation of what max(arr,axis) does :
Suppose arr has shape (n,m) then max(arr,0) returns a matrix of shape (1,m) where max(arr,0)(0,j) = maximum over all i {arr[i][j]}
max(arr,1) returns a matrix of shape (n,1) where max(arr,1)(i,0) = maximum over all j {arr[i][j]}
You should be able to infer min, argmax and argmin similarly

*/

// Specific max function : A = max(arr,axis)
matrix max(matrix &arr,int axis); 

// General max function : A = max(arr,0)
matrix max(matrix &arr); 
// Specific argmax function : A = argmax(arr,axis)                                
matrix argmax(matrix &arr,int axis); 

// General argmax function : A = argmax(arr,0)
matrix argmax(matrix &arr); 

// Specific min function : A = min(arr,axis)
matrix min(matrix &arr,int axis);

// General min function : A = min(arr,0)
matrix min(matrix &arr); 
// Specific argmin function : A = argmin(arr,axis)
matrix argmin(matrix &arr,int axis); 

// General argmin function : A = argmin(arr,0)
matrix argmin(matrix &arr); 

// Elementwise Operations

// Elementwise tanh function : matr[i][j] = tanh(matr[i][j])
matrix tanh(matrix &a); 

// Elementwise fabs function : matr[i][j] = fabs(matr[i][j])
matrix fabs(matrix &a); 

// Elementwise exponentiator : matr[i][j] = exp(matr[i][j])
matrix exp(matrix &a); 

// Elementwise log operator : matr[i][j] = log_{logbase}(matr[i][j])
matrix log(matrix &a,float logbase);

// Elementwise log with logbase e
matrix log(matrix &a); 

// Elementwise sqrt function : matr[i][j] = sqrt(matr[i][j]) 
matrix sqrt(matrix &a); 

#endif