#ifndef FM_MODEL_
#define FM_MODEL_

#include "./util/fmatrix.h"

#include "fm_data.h"
#include "data.h"
#include <vector>


struct fm_params {
	double learn_rate;
	int task = 0;
	uint num_attribute;
	int num_factor;
	double min_target;
	double max_target;
}; 
//assume all pointers exist in CUDA
class fm_model {
	public:
		double* w0;
		double* w;
		double* v;
		double* m_sum;
		double* m_sum_sqr;
		fm_params params;
		fm_params* params_ptr;

	public:
		// the following values should be set:
		
		double init_stdev;
		double init_mean;
		
		fm_model(int n, int k); // n features, k is the rank 
		void init();
		double predict(sparse_row_v<FM_FLOAT>* x);
		double predict(sparse_row_v<FM_FLOAT>* x, double* sum, double* sum_sqr);
		void predict(Data* data, double* out);
		double evaluate(Data* data);
		void learn(Data* train, Data* test, int num_iter);
		double evaluate_classification(Data* data);
		double evaluate_regression(Data* data);
		void SGD(sparse_row_v<FM_FLOAT>* x, const double multiplier, double *sum);
		//double predict(sparse_entry<FM_FLOAT>* x, int xsize, double* sum, double* sum_sqr);
		//void SGD(sparse_entry<FM_FLOAT>* x, int xsize);
		//void learn(std::vector<std::pair<sparse_entry<FM_FLOAT>*, int>> trainX, std::vector<double> trainY, std::vector<std::pair<sparse_entry<FM_FLOAT>*,int>> testX, std::vector<double> testY, int num_iter);
	
};

 

#endif