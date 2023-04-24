#ifndef FM_MODEL_
#define FM_MODEL_

#include "./util/matrix.h"
#include "./util/fmatrix.h"

#include "./fm_data.h"
#include <vector>

//assume all pointers exist in CUDA
class fm_model {
	public:
		double* w0;
		double* w;
		double* v;
		double* m_sum;
		double* m_sum_sqr;
		double learn_rate;
		int task = 0;

	public:
		// the following values should be set:
		uint* num_attribute;
		
		int* num_factor;
		
		double init_stdev;
		double init_mean;
		
		fm_model(int n, int k); // n features, k is the rank 
		void init();
		double predict(sparse_entry<FM_FLOAT>* x, int xsize);
		double predict(sparse_entry<FM_FLOAT>* x, int xsize, double* sum, double* sum_sqr);
		void SGD(sparse_entry<FM_FLOAT>* x, int xsize);
		void learn(std::vector<std::pair<sparse_entry<FM_FLOAT>*, int>> trainX, std::vector<double> trainY, std::vector<std::pair<sparse_entry<FM_FLOAT>*,int>> testX, std::vector<double> testY, int num_iter);
	
};

 

#endif