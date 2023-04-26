#ifndef DATA_H_
#define DATA_H_

#include <limits>
#include "./util/matrix.h"
#include "./util/fmatrix.h"
#include "fm_data.h"

typedef FM_FLOAT DATA_FLOAT;


class Data {
 public:
  Data(uint64 cache_size, bool has_x, bool has_xt);
  void load(std::string filename);

  LargeSparseMatrix<DATA_FLOAT>* data_t;
  LargeSparseMatrix<DATA_FLOAT>* data;
  DVector<DATA_FLOAT> target;

  int num_feature;
  uint num_cases;

  DATA_FLOAT min_target;
  DATA_FLOAT max_target;


 protected:
  void create_data_t();

  uint64 cache_size;
  bool has_xt;
  bool has_x;
};


#endif