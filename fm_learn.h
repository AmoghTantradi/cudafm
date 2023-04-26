#ifndef FM_LEARN_H_
#define FM_LEARN_H_

#include <cmath>
#include "Data.h"
#include "fm.h"
#include <iostream>

class fm_learn {
 public:
  fm_learn();
  virtual void init();
  virtual double evaluate(Data* data);
  virtual void learn(Data* train, Data* test);
  void SGD(sparse_row<DATA_FLOAT> *x, const double multiplier, DVector<double> *sum);

  virtual void predict(Data* data, DVector<double>* out);
  fm_model* fm;
  double min_target;
  double max_target;

  int num_iter;
  double learn_rate;
  DVector<double> learn_rates;

  int task; // 0=regression, 1=classification
  const static int TASK_REGRESSION = 0;
  const static int TASK_CLASSIFICATION = 1;
  
 protected:
  // these functions can be overwritten (e.g. for MCMC)
  virtual double evaluate_classification(Data* data);
  virtual double evaluate_regression(Data* data);
  virtual double predict_case(Data* data);

  DVector<double> sum, sum_sqr;
  DMatrix<double> pred_q_term;
};

#endif /*FM_LEARN_H_*/