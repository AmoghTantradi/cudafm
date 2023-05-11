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

// I/O routines
void save(std::ofstream& fsave, particle_t* parts, int num_parts, double size) {
    static bool first = true;

    if (first) {
        fsave << num_parts << " " << size << std::endl;
        first = false;
    }

    for (int i = 0; i < num_parts; ++i) {
        fsave << parts[i].x << " " << parts[i].y << std::endl;
    }

    fsave << std::endl;
}


/*
// Particle Initialization
void init_particles(particle_t* parts, int num_parts, double size, int part_seed) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());

    int sx = (int)ceil(sqrt((double)num_parts));
    int sy = (num_parts + sx - 1) / sx;

    std::vector<int> shuffle(num_parts);
    for (int i = 0; i < shuffle.size(); ++i) {
        shuffle[i] = i;
    }

    for (int i = 0; i < num_parts; ++i) {
        // Make sure particles are not spatially sorted
        std::uniform_int_distribution<int> rand_int(0, num_parts - i - 1);
        int j = rand_int(gen);
        int k = shuffle[j];
        shuffle[j] = shuffle[num_parts - i - 1];

        // Distribute particles evenly to ensure proper spacing
        parts[i].x = size * (1. + (k % sx)) / (1 + sx);
        parts[i].y = size * (1. + (k / sx)) / (1 + sy);

        // Assign random velocities within a bound
        std::uniform_real_distribution<float> rand_real(-1.0, 1.0);
        parts[i].vx = rand_real(gen);
        parts[i].vy = rand_real(gen);
    }
}
*/


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

int main(int argc, char** argv) {

    // construct a fm object
    const std::string param_train_file	= "../data/ml-tag.train.libfm"; // "libfm_test.txt";
    // const std::string param_train_file	= "../scripts/libfm_test_data_large.txt"; //
    //"libfm_test.txt";
    Data train;
    train.load(param_train_file);
    /*
    for (int i = 0; i < 10; i++) {
    	std :: cout << train.target[i] << " ";
    	for (int j = 0; j < train.data[i]->size; j++) {
    		std::cout << train.data[i]->data[j].id << ":" << train.data[i]->data[j].value << "\n";
    	}
    }
    */

    std::cout << "Num feature " << train.num_feature << std::endl;
    fm_model fm(train.num_feature, 8);
    fm.params.learn_rate = 0.05;
    fm.params.task = 1;
    fm.params.min_target = train.min_target;
    fm.params.max_target = train.max_target;
    std::vector<trainBatch> batches;
    //std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
    //std::shuffle(train.data.begin(), train.data.end(), rng);
    fm.batchSamples(&train, batches);
    std::cout << batches.size() << " batches for " << train.data.size() << " samples \n";
    auto start_time = std::chrono::steady_clock::now();
    fm.learn(batches, 1); 

    /*
    for (int i = 0; i < 1; i++) {
        
        std::cout << "here" << std::endl;
        printCudaSparse(batches[i].x);

        //printCudaSparse(batches[i].second);
        
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
            batches[i].x,
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
        cusparseCreateDnMat(&result, rows, 8, cols, devptr, CUDA_R_64F, CUSPARSE_ORDER_ROW); 
        //std::cout << "created dn\n";
        fm.matMul(batches[i].x, fm.V, result);


 
    //   break;
    }
*/




    //device memory deallocation
    //cudaFree(dBuffer);
    // cudaFree(dA_csrOffsets);
    // cudaFree(dA_columns);
    // cudaFree(dA_values);
    // cudaFree(dB);
    // cudaFree(dC);
    

    auto end_time = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end_time - start_time;
	double seconds = diff.count();

		// Finalize
	std::cout << "Total Simulation Time for SGD = " << seconds << " seconds \n";
    std::cout << "Simulation Time for predict" << fm.predictTime << std::endl;
    std::cout << "Time for SGD without predict" << seconds - fm.predictTime << std::endl;
}
