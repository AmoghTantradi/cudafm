#ifndef FM_SGD_H_
#define FM_SGD_H_

void fm_SGD(fm_model* fm, const double& learn_rate, sparse_row<FM_FLOAT> &x, const double multiplier, DVector<double> &sum);

#endif