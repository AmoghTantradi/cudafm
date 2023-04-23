#ifndef FM_LEARN_SGD_H_
#define FM_LEARN_SGD_H_

class fm_learn_sgd: public fm_learn {
 public:
  virtual void init();
  virtual void learn(Data& train, Data& test);
  void SGD(sparse_row<DATA_FLOAT> &x, const double multiplier, DVector<double> &sum);

  virtual void predict(Data& data, DVector<double>& out);

  int num_iter;
  double learn_rate;
  DVector<double> learn_rates;
};



#endif /*FM_LEARN_SGD_H_*/