#include "fm.h"
#include "fm_sgd.h"

void fm_SGD(fm_model* fm, const double& learn_rate, sparse_row<FM_FLOAT> &x, const double multiplier, DVector<double> &sum) {
	if (fm->k0) {
		double& w0 = fm->w0;
		w0 -= learn_rate * (multiplier + fm->reg0 * w0);
	}
	if (fm->k1) {
		for (uint i = 0; i < x.size; i++) {
			double& w = fm->w(x.data[i].id);
			w -= learn_rate * (multiplier * x.data[i].value + fm->regw * w);
		}
	}
	for (int f = 0; f < fm->num_factor; f++) {
		for (uint i = 0; i < x.size; i++) {
			double& v = fm->v(f, x.data[i].id);
			double grad = sum(f) * x.data[i].value - v * x.data[i].value * x.data[i].value; 
			v -= learn_rate * (multiplier * grad + fm->regv * v);
		}
	}
}