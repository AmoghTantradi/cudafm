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

    /*
    const std::string param_train_file	= "../data/ml-tag.test.libfm"; // "libfm_test.txt";
    // const std::string param_train_file	= "../scripts/libfm_test_data_large.txt"; //
    "libfm_test.txt"; Data train; train.load(param_train_file); for (int i = 0; i < 10; i++) {
    	std :: cout << train.target[i] << " ";
    	for (int j = 0; j < train.data[i]->size; j++) {
    		std::cout << train.data[i]->data[j].id << ":" << train.data[i]->data[j].value << "
    ";
    	}
    	std::cout << std :: endl;
    }
    fm_model fm(train.num_feature, 8);
    fm.params.learn_rate = 0.05;
    fm.params.task = 1;
    fm.params.min_target = train.min_target;
    fm.params.max_target = train.max_target;
    auto start_time = std::chrono::steady_clock::now();
    // fm.learn(&train, &train, 2);
    */
    // [[1 0 2]
    // [0 3 0]
    // [4 0 5]]
    /*
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    double values[5] = {1, 2, 3, 4, 5};
    int colIdx[5] = {0, 2, 1, 0, 2};
    int rowPtr[4] = {0, 2, 3, 5};


    double* d_A_values;
    int* devrows;
    int* devcols;
    cudaMalloc((void**)&d_A_values, 5 * sizeof(double));
    cudaMalloc((void**) &devrows, 4 * sizeof(int));
    cudaMalloc((void**)&devcols, 5 * sizeof(int));


    cudaMemcpy(d_A_values, values, 5 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devrows, rowPtr, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devcols, colIdx, 5 * sizeof(int), cudaMemcpyHostToDevice);


    cusparseSpMatDescr_t descrA;
    cusparseCreateCsr(&descrA, 3, 3, 5, devrows, devcols, d_A_values, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);


    // cudaMemcpy(a_C, d_A_values, 5 * sizeof(double), cudaMemcpyDeviceToHost);wrong
    std:: cout << "Current matrix A" << std::endl;
    //printCudaSparse(descrA);

    double *Ahost = (double*) malloc(sizeof(double) * 5);
    cudaMemcpy(Ahost, d_A_values, sizeof(double) * 5, cudaMemcpyDeviceToHost);

    for (int i=0; i < 5; i++) {
        std::cout << Ahost[i] << " ";
    }
    std::cout << std::endl;

    // turns out that any modifications to descrA also modifies device ptr d_A_values, so we might not need printCudaSparse



    double values2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    cusparseDnMatDescr_t descrB;

    double* d_B_values;
    cudaMalloc((void**)&d_B_values, 9 * sizeof(double));
    cudaMemcpy(d_B_values, values2, 9 * sizeof(double), cudaMemcpyHostToDevice);

    cusparseCreateDnMat(&descrB, 3, 3, 3, d_B_values, CUDA_R_64F, CUSPARSE_ORDER_ROW); //ld is number of rows of matrix

    double* b_C = (double*)malloc(9 * sizeof(double));

    //cudaMemcpy(b_C, d_B_values, 9 * sizeof(double), cudaMemcpyDeviceToHost); wrong 

    double* d_values;
    int* d_colIdx;
    int* d_rowPtr;

    cudaMalloc((void**)&d_values, 5 * sizeof(double));
    cudaMalloc((void**)&d_colIdx, 5 * sizeof(int));
    cudaMalloc((void**)&d_rowPtr, 4 * sizeof(int));

    cudaMemcpy(d_values, values, 5 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, rowPtr, 4 * sizeof(int), cudaMemcpyHostToDevice);


    std::cout << "Current matrix B" << std::endl;

    double* Bhost = (double*) malloc(sizeof(double) * 9);
    cudaMemcpy(Bhost, d_B_values, sizeof(double) * 9, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9; i++){
        std::cout << Bhost[i] << " " ;
    }
    std::cout << std::endl;

    //printCudaDense(descrB);

    /// double* d_C;
    // cudaMalloc((void**)&d_C, 9 * sizeof(double));

    cusparseDnMatDescr_t descrC;
    double valuesC[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double* d_C_values;
    cudaMalloc((void**)&d_C_values, 9 * sizeof(double));
    cudaMemcpy(d_C_values, valuesC, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cusparseCreateDnMat(&descrC, 3, 3, 3, d_C_values, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    
    std::cout << "Current matrix C" << std::endl;

    double* dHost = (double *) malloc(sizeof(double) * 9);
    cudaMemcpy(dHost, d_C_values, sizeof(double) * 9, cudaMemcpyDeviceToHost);


       
    for(int i = 0; i < 9; i++) {
        std::cout << dHost[i] << " " ;
    }
    std:: cout << std::endl;

    size_t buffer_size = 0;

    int nnzC = 0;
    int* nnzTotalDevHostPtr = &nnzC;  // what's this for ? 

    const float alpha = 1.0;
    const float beta = 0;
    
    //cusparseSpMMAlg_t alg = ;

    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, (void*)&alpha, descrA, descrB,
                            (void*)&beta, descrC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);

    void* buffer = NULL;
    std::cout <<"buffer_size " << buffer_size << std::endl;
    cudaMalloc(&buffer, buffer_size);

    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 (void*)&alpha, descrA, descrB, (void*)&beta, descrC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, buffer);


    std:: cout << "Result of matmul" <<std::endl;
    //printCudaDense(descrC);    
   
    cudaMemcpy(dHost, d_C_values, sizeof(double) * 9, cudaMemcpyDeviceToHost);


    std::cout << "DC values " << std:: endl; // this will store all the modified values as well
    for(int i = 0; i < 9; i++) {
        std::cout << dHost[i] << " " ;
    }






    cudaFree(d_C_values);

    */

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


     // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int));

    cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int));
    cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float));

    cudaMalloc((void**) &dB,         B_size * sizeof(float));

    cudaMalloc((void**) &dC,         C_size * sizeof(float));

    cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice);


    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    // Create dense matrix B
    cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    // Create dense matrix C
   cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize)
                                 ;
    cudaMalloc(&dBuffer, bufferSize);



    // execute SpMM
    for (int i = 0; i<10; i++) {
    cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    
    }

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);  
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);


    cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost);
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


    auto start_time = std::chrono::steady_clock::now();
    // construct a fm object
    const std::string param_train_file	= "../data/ml-tag.test.libfm"; // "libfm_test.txt";
    // const std::string param_train_file	= "../scripts/libfm_test_data_large.txt"; //
    "libfm_test.txt";
    Data train;
    train.load(param_train_file);
    for (int i = 0; i < 10; i++) {
    	std :: cout << train.target[i] << " ";
    	for (int j = 0; j < train.data[i]->size; j++) {
    		std::cout << train.data[i]->data[j].id << ":" << train.data[i]->data[j].value << "\n";
    	}
    }
    fm_model fm(train.num_feature, 8);
    fm.params.learn_rate = 0.05;
    fm.params.task = 1;
    fm.params.min_target = train.min_target;
    fm.params.max_target = train.max_target;
    std::vector<std::pair<cusparseSpMatDescr_t, cusparseSpMatDescr_t>> batches;
    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(train.data.begin(), train.data.end(), rng);
    fm.batchSamples(&train, batches);

    std::cout << batches.size() << " batches for " << train.data.size() << " samples \n";
    int idx = 0;


    //check dimensions of fm.V
    /*
    { // checking dimensions

        double* valuesdv;
        int64_t r;
        int64_t c;
        int64_t ld;
        cudaDataType cuda_data_type;
        cusparseOrder_t order;
        cusparseDnMatGet(fm.V, &r, &c, &ld, (void**)&valuesdv, &cuda_data_type, &order);

        std::cout<<"Rows of fm.V " << r << " Cols of fm.V " << c <<std::endl;



    }
    */

    for (int i = 0; i < 10000; i++) {
        /*
        std::cout << "here" << std::endl;
        printCudaSparse(batches[i].first);

        printCudaSparse(batches[i].second);
        */
        // make result matrix
        cusparseDnMatDescr_t result;
        // have to allocate device memory too

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

        //std::cout << "here1: " << nnz << std::endl;
        cusparseCsrGet(
            batches[i].first,
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

        double* host = (double*) malloc(sizeof(double) * rows * 8);
        double* devptr;
        //std::cout << rows << " " << cols << "\n";
        
        cudaMalloc((void**)&devptr, sizeof(double) * rows * 8); 

        //cudaMemcpy(devptr, host, sizeof(double) * rows * 8, cudaMemcpyHostToDevice);
        //std::cout << "here2: " << rows << std::endl;
        //std::cout << "here3: " << cols << std::endl;
        cusparseCreateDnMat(&result, rows, 8, cols, &devptr, CUDA_R_64F, CUSPARSE_ORDER_ROW); 
        //std::cout << "created dn\n";
        fm.matMul(batches[i].first, fm.V, result);

 
    //   break;
    }


    //device memory deallocation
    cudaFree(dBuffer);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dB);
    cudaFree(dC);
    

    auto end_time = std::chrono::steady_clock::now();

	std::chrono::duration<double> diff = end_time - start_time;
	double seconds = diff.count();

		// Finalize
	std::cout << "Simulation Time = " << seconds << " seconds \n";
}
