#ifndef FM_LEARN_SGD_H_
#define FM_LEARN_SGD_H_

#include "fm_learn.h"
#include "fm_sgd.h"
#include "util/matrix.h"


// Implementation
void fm_learn_sgd::init() {
  fm_learn::init();
  learn_rates.setSize(3);
}

void fm_learn_sgd::learn(Data& train, Data& test) {
  fm_learn::learn(train, test);
  std::cout << "learnrate=" << learn_rate << std::endl;
  std::cout << "learnrates=" << learn_rates.get(0) << "," << learn_rates.get(1) << "," << learn_rates.get(2) << std::endl;
  std::cout << "#iterations=" << num_iter << std::endl;

  std::cout.flush();
}

void fm_learn_sgd::SGD(sparse_row<DATA_FLOAT> &x, const double multiplier, DVector<double> &sum) {
  fm_SGD(fm, learn_rate, x, multiplier, sum);
}

void fm_learn_sgd::predict(Data& data, DVector<double>& out) {
  assert(data.data->getNumRows() == out.dim);
  for (data.data->begin(); !data.data->end(); data.data->next()) {
    double p = predict_case(data);
    if (task == TASK_REGRESSION ) {
      p = std::min(max_target, p);
      p = std::max(min_target, p);
    } else if (task == TASK_CLASSIFICATION) {
      p = 1.0/(1.0 + exp(-p));
    } else {
      throw "task not supported";
    }
    out(data.data->getRowIndex()) = p;
  }
}

#endif /*FM_LEARN_SGD_H_*/
