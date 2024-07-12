#include "matrix.h"
#include <chrono>

int main(int argc, char* argv[]){
    uint64_t N = atoi(argv[1]);
    matrix A(N,N);
    matrix B(N,N);
    matrix C(N,N);
    for (uint64_t i = 0 ; i < N*N ; i++){
        A(i) = 5.0;
        B(i) = 2.0;
    }
    auto start = chrono::high_resolution_clock::now();
    matrix D = ones(A.shape());
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::duration<double>>(end - start);
    D.printMatrix();
    cout << "Time elapsed " << elapsed.count() << "\n";
}