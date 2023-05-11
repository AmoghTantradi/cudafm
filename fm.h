#ifndef FM_MODEL_
#define FM_MODEL_

#include "./util/fmatrix.h"

#include "fm_data.h"
#include "data.h"
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>


struct fm_params {
	double learn_rate;
	int task = 0;
	uint num_attribute; //num_features (f or n) 
	int num_factor; // num_factors (k) 
	double min_target;
	double max_target;
};

struct cudaArgs {
	double* w0;
	double* w;
	double* v;
	double* v2;
	fm_params params;
	double* ret;
};

struct trainBatch {
	cusparseSpMatDescr_t x;
	cusparseSpMatDescr_t x2;
	double* target;
	int* xCols;
	double* xVals;
	int* rowidx;
	int size;
	int nnz;
};

void matMul(cusparseSpMatDescr_t &A, cusparseDnMatDescr_t& B, cusparseDnMatDescr_t& result);
//assume all pointers exist in CUDA
class fm_model {
	public:
		double* w0;
		double* w;
		double* v;
		double* v2;
		fm_params params;
		cudaArgs* cuda_args;
		double* ret;



	public:
		// the following values should be set:
		
		double init_stdev;
		double init_mean;
		int maxBatch = 1;

		double* xiv;
		double* x2iv2;
		double* xiw;

		void* bufferMatmul;
		void* bufferSpMv;
		int maxBufferSize = 0;
		double predictTime = 0;

		cusparseDnMatDescr_t xv;
		cusparseDnMatDescr_t x2v2;



		
		std::vector<int> pointsPerBatch;  // stores the number of rows per batch. This avoids having to make a call to cusparseSpMatGet everytime. 
		cusparseDnMatDescr_t V	;
		cusparseDnMatDescr_t V_2;


		cusparseDnVecDescr_t w_vec;

		cusparseDnVecDescr_t xw;

		cusparseHandle_t handle = NULL; // should be only created once.


		fm_model(int n, int k); // n features, k is the rank 
		void init();
		double predict(sparse_row_v<FM_FLOAT>* x);
		double predict(sparse_row_v<FM_FLOAT>* x, double* sum, double* sum_sqr);
		void predict(trainBatch batch, int batchSize,  double* preds);
		void predict(Data* data, double* out);
		double evaluate(Data* data);
		void learn(std::vector<trainBatch> &training_batches, const int num_iter);
		double evaluate_classification(Data* data);
		double evaluate_regression(Data* data);
		void SGD(sparse_row_v<FM_FLOAT>* x, const double multiplier, double *sum);
		void batchSamples(Data* train, std::vector<trainBatch> &batches);
		void matMul(cusparseSpMatDescr_t &A, cusparseDnMatDescr_t& B, cusparseDnMatDescr_t& result);
		void setSize(int batchSize);
};

 

#endif