#include "kernel.h"

dim3 threadsPerBlock(16, 8);


// Kernel function for matrix multiplication
__global__ void Mul_cuda(float* A, float* B, float* C, int numARows, int numAColumns, int numBColumns) {
    // Calculate row index of the C element and A
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate column index of C element and B
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        // Each thread computes one element of the block sub-matrix
        float value = 0;
        for (int k = 0; k < numAColumns; ++k) {
            value += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        // Write the block sub-matrix to device memory;
        // each thread writes one element
        C[row * numBColumns + col] = value;
    }
}


__global__ void Tpose_cuda(const float* A, float* A_T, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        A_T[col * rows + row] = A[row * cols + col];
    }
}


void Mul(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output){
    //printf("%d %d %d %d %d %d\n", input1.row_count, input1.col_count, input2.row_count, input2.col_count, output.row_count, output.col_count);
    dim3 numBlocks((input2.col_count + threadsPerBlock.x - 1) / threadsPerBlock.x, (input1.row_count + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Mul_cuda<<<numBlocks, threadsPerBlock>>>(input1.data_ptr, input2.data_ptr, output.data_ptr, input1.row_count, input1.col_count, input2.col_count);
    //cudaDeviceSynchronize(); 
    }
void Tpose(array2d_t<float>& input1, array2d_t<float>& output){
    dim3 numBlocks((input1.col_count + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (input1.row_count + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Tpose_cuda<<<numBlocks, threadsPerBlock>>>(input1.data_ptr, output.data_ptr, input1.row_count, input1.col_count);
    //cudaDeviceSynchronize(); 
    }
