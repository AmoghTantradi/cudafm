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
  virtual double evaluate(Data& data);
  virtual void learn(Data& train, Data& test);
  virtual void predict(Data& data, DVector<double>& out) = 0;
  fm_model* fm;
  double min_target;
  double max_target;

  int task; // 0=regression, 1=classification
  const static int TASK_REGRESSION = 0;
  const static int TASK_CLASSIFICATION = 1;

  Data* validation;
  
 protected:
  // these functions can be overwritten (e.g. for MCMC)
  virtual double evaluate_classification(Data& data);
  virtual double evaluate_regression(Data& data);
  virtual double predict_case(Data& data);

  DVector<double> sum, sum_sqr;
  DMatrix<double> pred_q_term;
};

// Implementation
double fm_learn::predict_case(Data& data) {
  return fm->predict(data.data->getRow());
}

fm_learn::fm_learn() {
  task = 0;
}

void fm_learn::init() {
  sum.setSize(fm->num_factor);
  sum_sqr.setSize(fm->num_factor);
  pred_q_term.setSize(fm->num_factor, 2);
}

double fm_learn::evaluate(Data& data) {
  assert(data.data != NULL);
  if (task == TASK_REGRESSION) {
    return evaluate_regression(data);
  } else if (task == TASK_CLASSIFICATION) {
    return evaluate_classification(data);
  } else {
    throw "unknown task";
  }
}

void fm_learn::learn(Data& train, Data& test) {
}


double fm_learn::evaluate_classification(Data& data) {
  int num_correct = 0;
  for (data.data->begin(); !data.data->end(); data.data->next()) {
    double p = predict_case(data);
    if (((p >= 0) && (data.target(data.data->getRowIndex()) >= 0)) || ((p < 0) && (data.target(data.data->getRowIndex()) < 0))) {
      num_correct++;
    }
  }

  return (double) num_correct / (double) data.data->getNumRows();
}

double fm_learn::evaluate_regression(Data& data) {
  double rmse_sum_sqr = 0;
  double mae_sum_abs = 0;
  for (data.data->begin(); !data.data->end(); data.data->next()) {
    double p = predict_case(data);
    p = std::min(max_target, p);
    p = std::max(min_target, p);
    double err = p - data.target(data.data->getRowIndex());
    rmse_sum_sqr += err*err;
    mae_sum_abs += std::abs((double)err);
  }

  return std::sqrt(rmse_sum_sqr/data.data->getNumRows());
}

#endif /*FM_LEARN_H_*/