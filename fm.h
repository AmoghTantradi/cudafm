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
void matMul(cusparseSpMatDescr_t &A, cusparseDnMatDescr_t& B, cusparseDnMatDescr_t& result);
//assume all pointers exist in CUDA
class fm_model {
	public:
		double* w0;
		double* w;
		double* v;
		double* v2;
		double* m_sum;
		double* m_sum_sqr;
		fm_params params;
		cudaArgs* cuda_args;
		double* ret;



	public:
		// the following values should be set:
		
		double init_stdev;
		double init_mean;
		
		std::vector<int> pointsPerBatch;  // stores the number of rows per batch. This avoids having to make a call to cusparseSpMatGet everytime. 
		cusparseDnMatDescr_t V	;
		cusparseDnMatDescr_t V_2;


		cusparseHandle_t handle = NULL; // should be only created once.


		fm_model(int n, int k); // n features, k is the rank 
		void init();
		double predict(sparse_row_v<FM_FLOAT>* x);
		double predict(sparse_row_v<FM_FLOAT>* x, double* sum, double* sum_sqr);
		void predict(std::pair<cusparseSpMatDescr_t, cusparseSpMatDescr_t> batch, int batchSize,  double* preds);
		void predict(Data* data, double* out);
		double evaluate(Data* data);
		void learn(Data* train, Data* test, int num_iter);
		double evaluate_classification(Data* data);
		double evaluate_regression(Data* data);
		void SGD(sparse_row_v<FM_FLOAT>* x, const double multiplier, double *sum);
		void batchSamples(Data* train, std::vector<std::pair<cusparseSpMatDescr_t, cusparseSpMatDescr_t>> &batches);
		void matMul(cusparseSpMatDescr_t &A, cusparseDnMatDescr_t& B, cusparseDnMatDescr_t& result);
};

 

#endif