#include "data.h"

void Data::load(std::string filename) {
  int num_rows = 0;
  uint64 num_values = 0;
  num_feature = 0;
  min_target = +std::numeric_limits<DATA_FLOAT>::max();
  max_target = -std::numeric_limits<DATA_FLOAT>::max();

  // (1) determine the number of rows and the maximum feature_id
  std::ifstream fData(filename.c_str());
  if (! fData.is_open()) {
    throw "unable to open " + filename;
  }
  DATA_FLOAT _value;
  int nchar, _feature;
  std::vector<int> row_sizes;
  while (!fData.eof()) {
    std::string line;
    std::getline(fData, line);
    const char *pline = line.c_str();
    while ((*pline == ' ')  || (*pline == 9)) { pline++; } // skip leading spaces
    if ((*pline == 0)  || (*pline == '#')) { continue; }  // skip empty rows
    if (sscanf(pline, "%f%n", &_value, &nchar) >=1) {
      pline += nchar;
      min_target = std::min(_value, min_target);
      max_target = std::max(_value, max_target);
      num_rows++;
      int row_size = 0;
      while (sscanf(pline, "%d:%f%n", &_feature, &_value, &nchar) >= 2) {
        pline += nchar;
        num_feature = std::max(_feature+1, num_feature);
        num_values++;
        row_size++;
      }
      row_sizes.push_back(row_size);
      while ((*pline != 0) && ((*pline == ' ')  || (*pline == 9))) { pline++; } // skip trailing spaces
    }
  }
  fData.close();
  std::cout << "num_rows=" << num_rows << "\tnum_values=" << num_values << "\tnum_features=" << num_feature << "\tmin_target=" << min_target << "\tmax_target=" << max_target << std::endl;
  data.resize(num_rows);
  target.resize(num_rows);
  {
    std::ifstream fData(filename.c_str());
    if (! fData.is_open()) {
      throw "unable to open " + filename;
    }
    int row_id = 0;
    DATA_FLOAT _value;
    int nchar, _feature;
    while (!fData.eof()) {
      std::string line;
      std::getline(fData, line);
      const char *pline = line.c_str();
      while ((*pline == ' ')  || (*pline == 9)) { pline++; } // skip leading spaces
      if ((*pline == 0)  || (*pline == '#')) { continue; }  // skip empty rows
      data[row_id] = (sparse_row_v<DATA_FLOAT>*)malloc(sizeof(sparse_row_v<DATA_FLOAT>) + row_sizes[row_id]*sizeof(sparse_entry<DATA_FLOAT>));
      data[row_id]->size = row_sizes[row_id];
      if (sscanf(pline, "%f%n", &_value, &nchar) >=1) {
        pline += nchar;
        assert(row_id < num_rows);
        target[row_id] = _value;
        int entry_id = 0;
        while (sscanf(pline, "%d:%f%n", &_feature, &_value, &nchar) >= 2) {
          pline += nchar;
          data[row_id]->data[entry_id].id = _feature;
          data[row_id]->data[entry_id].value = _value;
          entry_id++;
        }
        row_id++;

        while ((*pline != 0) && ((*pline == ' ')  || (*pline == 9))) { pline++; } // skip trailing spaces
      }
    }
    fData.close();
    
  }
}