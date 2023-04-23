#ifndef FM_MODEL_H_
#define FM_MODEL_H_

#include "./util/matrix.h"
#include "./util/fmatrix.h"

#include "fm_data.h"


class fm_model {
	private:
		DVector<double> m_sum, m_sum_sqr;
	public:
		double w0;
		DVectorDouble w;
		DMatrixDouble v;

	public:
		// the following values should be set:
		uint num_attribute;
		
		bool k0, k1;
		int num_factor;
		
		double reg0;
		double regw, regv;
		
		double init_stdev;
		double init_mean;
		
		fm_model();
		void debug();
		void init();
		double predict(sparse_row<FM_FLOAT>& x);
		double predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr);
	
};



void fm_SGD(fm_model* fm, const double& learn_rate, sparse_row<FM_FLOAT> &x, const double multiplier, DVector<double> &sum);

#endif