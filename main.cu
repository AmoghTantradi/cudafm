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

#include <cuda_runtime.h>
#include <cusparse.h>

// =================
// Helper Functions
// =================

// Command Line Option Processing

// ==============
// Main Function
// ==============

/*
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
  std:: cout << "Host values " << std::endl;
  for (int64_t i = 0 ; i < nnz; i++) {
        std::cout << values_host[i] << std::endl;
  }
  std::cout << std::endl;
  delete [] values_host;
}

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
    // const std::string param_train_file	= "../data/ml-tag.test.libfm"; // "libfm_test.txt";
    // // const std::string param_train_file	= "../scripts/libfm_test_data_large.txt"; //
    // "libfm_test.txt"; Data train; train.load(param_train_file); for (int i = 0; i < 10; i++) {
    // 	std :: cout << train.target[i] << " ";
    // 	for (int j = 0; j < train.data[i]->size; j++) {
    // 		std::cout << train.data[i]->data[j].id << ":" << train.data[i]->data[j].value << "
    // ";
    // 	}
    // 	std::cout << std :: endl;
    // }
    // fm_model fm(train.num_feature, 8);
    // fm.params.learn_rate = 0.05;
    // fm.params.task = 1;
    // fm.params.min_target = train.min_target;
    // fm.params.max_target = train.max_target;
    auto start_time = std::chrono::steady_clock::now();
    // fm.learn(&train, &train, 2);

    // [[1 0 2]
    // [0 3 0]
    // [4 0 5]]

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

    // cudaFree(d_values);
    // cudaFree(d_colIdx);
    // cudaFree(d_rowPtr);
    // // cudaFree(d_C);
    // cusparseDestroySpMat(descrA);
    // cusparseDestroyDnMat(descrB);
    // cusparseDestroyDnMat(descrC);
    // cusparseDestroy(handle);

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    // Finalize
    std::cout << "Simulation Time = " << seconds << " seconds \n";
    /*
srand ( time(NULL) );
    try {

            const std::string param_task		= "c";
            const std::string param_train_file	= "libfm_test.txt";
            const std::string param_test_file	= "libfm_test.txt";

            double param_init_stdev	= 0.1;
            int param_num_iter	= 100;
            double param_learn_rate	= 0.01;
            const std::string param_method		= "sgd";

            const std::string param_do_sampling	= "do_sampling";
            const std::string param_do_multilevel	= "do_multilevel";
            const std::string param_num_eval_cases  = "num_eval_cases";
            // (1) Load the data
            std::cout << "Loading train...\t" << std::endl;
            Data train(
                    0,
                    ! (!param_method.compare("mcmc")), // no original data for mcmc
                    ! (!param_method.compare("sgd") || !param_method.compare("sgda")) // no
transpose data for sgd, sgda
            );
            train.load(param_train_file);

            std::cout << "Loading test... \t" << std::endl;
            Data test(
                    0,
                    ! (!param_method.compare("mcmc")), // no original data for mcmc
                    ! (!param_method.compare("sgd") || !param_method.compare("sgda")) // no
transpose data for sgd, sgda
            );
            test.load(param_test_file);

            Data* validation = NULL;

            // (2) Setup the factorization machine
            fm_model fm;
            {
        uint num_all_attribute = std::max(train.num_feature, test.num_feature);
                    fm.num_attribute = num_all_attribute;
                    fm.init_stdev = param_init_stdev;
                    // set the number of dimensions in the factorization
                    {
                            std::vector<int> dim(3);
            dim[0] = 1;
            dim[1] = 1;
            dim[2] = 8;
                            assert(dim.size() == 3);
                            fm.k0 = dim[0] != 0;
                            fm.k1 = dim[1] != 0;
                            fm.num_factor = dim[2];
                    }
                    fm.init();

            }

            // (3) Setup the learning method:
            fm_learn* fml;
            if (! param_method.compare("sgd")) {
                    fml = new fm_learn();
                    fml->num_iter = param_num_iter;
            } else {
                    throw "unknown method";
            }
            fml->fm = &fm;
            fml->max_target = train.max_target;
            fml->min_target = train.min_target;
            if (! param_task.compare("r") ) {
                    fml->task = 0;
            } else if (! param_task.compare("c") ) {
                    fml->task = 1;
                    for (uint i = 0; i < train.target.dim; i++) { if (train.target(i) <= 0.0) {
train.target(i) = -1.0; } else {train.target(i) = 1.0; } } for (uint i = 0; i < test.target.dim;
i++) { if (test.target(i) <= 0.0) { test.target(i) = -1.0; } else {test.target(i) = 1.0; } } } else
{ throw "unknown task";
            }

    fml->init();
    // set the regularization; for standard SGD, groups are not supported
    {
        std::vector<double> reg(3);
        reg[2] == 0.01;
        assert((reg.size() == 0) || (reg.size() == 1) || (reg.size() == 3));
        if (reg.size() == 0) {
            fm.reg0 = 0.0;
            fm.regw = 0.0;
            fm.regv = 0.0;
        } else if (reg.size() == 1) {
            fm.reg0 = reg[0];
            fm.regw = reg[0];
            fm.regv = reg[0];
        } else {
            fm.reg0 = reg[0];
            fm.regw = reg[1];
            fm.regv = reg[2];
        }
    }
            {
                    std::vector<double> lr(1, param_learn_rate);
                    assert((lr.size() == 1) || (lr.size() == 3));
                    if (lr.size() == 1) {
                            fml->learn_rate = lr[0];
                            fml->learn_rates.init(lr[0]);
                    } else {
                            fml->learn_rate = 0;
                            fml->learn_rates(0) = lr[0];
                            fml->learn_rates(1) = lr[1];
                            fml->learn_rates(2) = lr[2];
                    }
            }

            auto start_time = std::chrono::steady_clock::now();

            // () learn
            fml->learn(&train, &test);

            // () Prediction at the end  (not for mcmc and als)
            std::cout << "Final\t" << "Train=" << fml->evaluate(&train) << "\tTest=" <<
fml->evaluate(&test) << std::endl;

            auto end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end_time - start_time;
            double seconds = diff.count();

            // Finalize
            std::cout << "Simulation Time = " << seconds << " seconds \n";


    } catch (std::string &e) {
            std::cerr << std::endl << "ERROR: " << e << std::endl;
    } catch (char const* &e) {
            std::cerr << std::endl << "ERROR: " << e << std::endl;
    }
    */
}
