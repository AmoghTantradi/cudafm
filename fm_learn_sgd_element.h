#ifndef FM_LEARN_SGD_ELEMENT_H_
#define FM_LEARN_SGD_ELEMENT_H_

#include <iomanip>
#include "fm_learn_sgd.h"

bool LOG = false;

class fm_learn_sgd_element: public fm_learn_sgd {
 public:
  virtual void init();
  virtual void learn(Data& train, Data& test);
};

// Implementation
void fm_learn_sgd_element::init() {
  fm_learn_sgd::init();
}

void fm_learn_sgd_element::learn(Data& train, Data& test) {
  fm_learn_sgd::learn(train, test);

  std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl;
  // SGD
  for (int i = 0; i < num_iter; i++) {

    for (train.data->begin(); !train.data->end(); train.data->next()) {
      double p = fm->predict(train.data->getRow(), sum, sum_sqr);
      double mult = 0;
      if (task == 0) {
        p = std::min(max_target, p);
        p = std::max(min_target, p);
        mult = -(train.target(train.data->getRowIndex())-p);
      } else if (task == 1) {
        mult = -train.target(train.data->getRowIndex())*(1.0-1.0/(1.0+exp(-train.target(train.data->getRowIndex())*p)));
      }
      SGD(train.data->getRow(), mult, sum);
    }
    double rmse_train = evaluate(train);
    double rmse_test = evaluate(test);
    std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
    if (log != NULL && LOG) {
      std::cout << "rmse_train " << rmse_train << std::endl;
      std::cout << '\n' << std::endl;
    }
  }
}

#endif /*FM_LEARN_SGD_ELEMENT_H_*/