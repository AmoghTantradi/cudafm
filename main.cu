#include "./util/fmatrix.h"
#include "data.h"
#include "fm.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cusparse.h>

// =================
// Helper Functions
// =================

// Command Line Option Processing

// ==============
// Main Function
// ==============


void printCudaSparse(cusparseSpMatDescr_t sparse_descr) {
  double* values_dev;
  int32_t* row_indices_dev;
  int32_t* col_indices_dev;
  int64_t rows;
  int64_t cols;
  int64_t nnz;
  cusparseIndexType_t rowidx_type;
  cusparseIndexType_t colidx_type;
  cusparseIndexBase_t idx_base;
  cudaDataType cuda_data_type;

  cusparseCsrGet(
    sparse_descr,
    &rows,
    &cols,
    &nnz,
    (void**)&row_indices_dev,
    (void**)&col_indices_dev,
    (void**)&values_dev,
    &rowidx_type,
    &colidx_type,
    &idx_base,
    &cuda_data_type
  );
  double * values_host = new double[nnz];
  int32_t* row_indices_host = new int32_t[nnz];
  int32_t* col_indices_host = new int32_t[nnz];
  cudaMemcpy(values_host, values_dev, nnz*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(row_indices_host, row_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(col_indices_host, col_indices_dev, nnz*sizeof(int32_t), cudaMemcpyDeviceToHost);
  std::cout << "nnz: " << nnz << " " << std::endl;
  
  for (int64_t i = 0 ; i < nnz; i++) {
        std::cout << values_host[i] << " " ;
        std::cout << ": " << col_indices_host[i] << " ";
  }
  std::cout << std::endl;
  delete [] values_host;
}


/*
void printCudaDense(cusparseDnMatDescr_t descrC) {
    double* valuesdv;
    int64_t rows;
    int64_t cols;
    int64_t ld;
    cudaDataType cuda_data_type;
    cusparseOrder_t order;
    cusparseDnMatGet(descrC, &rows, &cols, &ld, (void**)&valuesdv, &cuda_data_type, &order);
    double* h_C = new double[9];
    cudaMemcpy(h_C, valuesdv, 9 * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Result: " << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;
    delete[] h_C;
}
*/

int main(int argc, char** argv) {
    auto start_time = std::chrono::steady_clock::now();
    
        // Host problem definition
    int   A_num_rows      = 4;
    int   A_num_cols      = 4;
    int   A_nnz           = 9;
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = 3;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
    int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f };
    float hB[]            = { 1.0f,  2.0f,  3.0f,  4.0f,
                              5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f };
    float hC[]            = { 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f };
    float hC_result[]     = { 19.0f,  8.0f,  51.0f,  52.0f,
                              43.0f, 24.0f, 123.0f, 120.0f,
                              67.0f, 40.0f, 195.0f, 188.0f };
    float alpha           = 1.0f;
    float beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    ( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) );
    ( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    );
    ( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  );
    ( cudaMalloc((void**) &dB,         B_size * sizeof(float)) );
    ( cudaMalloc((void**) &dC,         C_size * sizeof(float)) );

    ( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) );
    ( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) );
    ( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) );
    ( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) );
    ( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) );
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    ( cusparseCreate(&handle) );
    // Create sparse matrix A in CSR format
    ( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    // Create dense matrix B
    ( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) );
    // Create dense matrix C
    ( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) );
    // allocate an external buffer if needed
    ( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
    ( cudaMalloc(&dBuffer, bufferSize) );

    // execute SpMM
    ( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );

  
    //--------------------------------------------------------------------------
    // device result check
    ( cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) );
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            if (hC[i + j * ldc] != hC_result[i + j * ldc]) {
                correct = 0; // direct floating point comparison is not reliable
                break;
            }
        }
    }
    if (correct)
        printf("spmm_csr_example test PASSED\n");
    else
        printf("spmm_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------





    //now modify dB

    float* temp = (float* ) malloc(sizeof(float) * B_size);

    cudaMemcpy(temp, dB, sizeof(float) * B_size, cudaMemcpyDeviceToHost);

    temp[0] = 10.0f; //changing from 1.0f to 10.0f
    temp[1] = -20.0f; //changing from 2.0f to -20.0f

    cudaMemcpy(dB, temp, sizeof(float) * B_size, cudaMemcpyHostToDevice); // changing and copying the values back 

    //now perform the spmm again

    int correct2 = 1;

    ( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
    ( cudaMalloc(&dBuffer, bufferSize) );

    // execute SpMM
    ( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );



    ( cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) );

    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            if (hC[i + j * ldc] != hC_result[i + j * ldc]) {
                correct2 = 0; // direct floating point comparison is not reliable
                break;
            }
        }
    }
    if (correct2)
        printf("spmm_csr_example test 2 PASSED\n");
    else
        printf("spmm_csr_example test 2 FAILED: wrong result\n"); //what we want to happen








      // destroy matrix/vector descriptors
    ( cusparseDestroySpMat(matA) );
    ( cusparseDestroyDnMat(matB) );
    ( cusparseDestroyDnMat(matC) );
    ( cusparseDestroy(handle) );


    // device memory deallocation
    ( cudaFree(dBuffer) );
    ( cudaFree(dA_csrOffsets) );
    ( cudaFree(dA_columns) );
    ( cudaFree(dA_values) );
    ( cudaFree(dB) );
    ( cudaFree(dC) );

    auto end_time = std::chrono::steady_clock::now();

	std::chrono::duration<double> diff = end_time - start_time;
	double seconds = diff.count();

		// Finalize
	std::cout << "Simulation Time = " << seconds << " seconds \n";
}
