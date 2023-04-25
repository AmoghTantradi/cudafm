#ifndef DATA_H_
#define DATA_H_

#include <limits>
#include "./util/fmatrix.h"
#include "fm_data.h"
#include <assert.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

typedef FM_FLOAT DATA_FLOAT;

class Data {
 public:
  void load(std::string filename);

  std::vector<sparse_row_v<DATA_FLOAT>*> data;
  std::vector<DATA_FLOAT> target;

  int num_feature;

  DATA_FLOAT min_target;
  DATA_FLOAT max_target;
};


#endif