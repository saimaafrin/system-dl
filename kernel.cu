#include "kernel.h"

dim3 threadsPerBlock(16, 8);


__global__ void vectorAdd(float *A, float *B, float *C, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // Total number of threads

    for (int i = index; i < W; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// Kernel function for matrix multiplication
__global__ void Mul(float* A, float* B, float* C, int numARows, int numAColumns, int numBColumns) {
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


__global__ void Tpose(const float* A, float* A_T, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        A_T[col * rows + row] = A[row * cols + col];
    }
}


void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm){;}
void gspmmve(graph_t& graph, array2d_t<float>& input1, array1d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmme(graph_t& graph, array1d_t<float>& edge_input, array1d_t<float>& output, op_t op, bool reverse){;}
void gspmme2d(graph_t& graph, array2d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmmve2d(graph_t& graph, array3d_t<float>& input1, array2d_t<float>& edge_input, array3d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve(graph_t& graph, array1d_t<float>& input_left, array1d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve2d(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv2d(graph_t& graph, array3d_t<float>& input_left, array3d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void test_2out(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test3(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test4(array3d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, int t){;}
void vectorAdd(array1d_t<float>& input1, array1d_t<float>& input2, array1d_t<float>& output, int W){
    vectorAdd<<<3, 32>>>(input1.data_ptr, input2.data_ptr, output.data_ptr, W);
    cudaDeviceSynchronize(); 
    }
void Mul(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output, int numARows, int numAColumns, int numBColumns){
    dim3 numBlocks((numBColumns + threadsPerBlock.x - 1) / threadsPerBlock.x, (numARows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Mul<<<numBlocks, threadsPerBlock>>>(input1.data_ptr, input2.data_ptr, output.data_ptr, numARows, numAColumns, numBColumns);
    cudaDeviceSynchronize(); 
    }
void Tpose(array2d_t<float>& input1, array2d_t<float>& output, int rows, int cols){
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    Tpose<<<numBlocks, threadsPerBlock>>>(input1.data_ptr, output.data_ptr, rows, cols);
    cudaDeviceSynchronize(); 
    }