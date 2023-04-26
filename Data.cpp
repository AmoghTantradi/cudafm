#include <limits>
#include "./util/matrix.h"
#include "./util/fmatrix.h"
#include "fm_data.h"
#include "Data.h"

typedef FM_FLOAT DATA_FLOAT;


// BEGIN: util.h
inline bool fileexists(std::string filename) {
  std::ifstream in_file (filename.c_str());
  return in_file.is_open();
}
// END: util.h

Data::Data(uint64 cache_size, bool has_x, bool has_xt) {
  this->data_t = NULL;
  this->data = NULL;
  this->cache_size = cache_size;
  this->has_x = has_x;
  this->has_xt = has_xt;
}

void Data::load(std::string filename) {

  std::cout << "has x = " << has_x << std::endl;
  std::cout << "has xt = " << has_xt << std::endl;
  assert(has_x || has_xt);

  int load_from = 0;
  if ((! has_x || fileexists(filename + ".data")) && (! has_xt || fileexists(filename + ".datat")) && fileexists(filename + ".target")) {
    load_from = 1;
  } else if ((! has_x || fileexists(filename + ".x")) && (! has_xt || fileexists(filename + ".xt")) && fileexists(filename + ".y")) {
    load_from = 2;
  }


  if (load_from > 0) {
    uint num_values = 0;
    uint64 this_cs = cache_size;
    if (has_xt && has_x) { this_cs /= 2; }

    if (load_from == 1) {
      this->target.loadFromBinaryFile(filename + ".target");
    } else {
      this->target.loadFromBinaryFile(filename + ".y");
    }
    if (has_x) {
      std::cout << "data... ";
      if (load_from == 1) {
        this->data = new LargeSparseMatrixHD<DATA_FLOAT>(filename + ".data", this_cs);
      } else {
        this->data = new LargeSparseMatrixHD<DATA_FLOAT>(filename + ".x", this_cs);
      }
      assert(this->target.dim == this->data->getNumRows());
      this->num_feature = this->data->getNumCols();
      num_values = this->data->getNumValues();
    } else {
      data = NULL;
    }
    if (has_xt) {
      std::cout << "data transpose... ";
      if (load_from == 1) {
        this->data_t = new LargeSparseMatrixHD<DATA_FLOAT>(filename + ".datat", this_cs);
      } else {
        this->data_t = new LargeSparseMatrixHD<DATA_FLOAT>(filename + ".xt", this_cs);
      }
      this->num_feature = this->data_t->getNumRows();
      num_values = this->data_t->getNumValues();
    } else {
      data_t = NULL;
    }

    if (has_xt && has_x) {
      assert(this->data->getNumCols() == this->data_t->getNumRows());
      assert(this->data->getNumRows() == this->data_t->getNumCols());
      assert(this->data->getNumValues() == this->data_t->getNumValues());
    }
    min_target = +std::numeric_limits<DATA_FLOAT>::max();
    max_target = -std::numeric_limits<DATA_FLOAT>::max();
    for (uint i = 0; i < this->target.dim; i++) {
      min_target = std::min(this->target(i), min_target);
      max_target = std::max(this->target(i), max_target);
    }
    num_cases = target.dim;

    std::cout << "num_cases=" << this->num_cases << "\tnum_values=" << num_values << "\tnum_features=" << this->num_feature << "\tmin_target=" << min_target << "\tmax_target=" << max_target << std::endl;
    return;
  }

  this->data = new LargeSparseMatrixMemory<DATA_FLOAT>();

  DVector< sparse_row<DATA_FLOAT> >& data = ((LargeSparseMatrixMemory<DATA_FLOAT>*)this->data)->data;

  int num_rows = 0;
  uint64 num_values = 0;
  num_feature = 0;
  bool has_feature = false;
  min_target = +std::numeric_limits<DATA_FLOAT>::max();
  max_target = -std::numeric_limits<DATA_FLOAT>::max();

  // (1) determine the number of rows and the maximum feature_id
  {
    std::ifstream fData(filename.c_str());
    if (! fData.is_open()) {
      throw "unable to open " + filename;
    }
    DATA_FLOAT _value;
    int nchar, _feature;
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
        while (sscanf(pline, "%d:%f%n", &_feature, &_value, &nchar) >= 2) {
          pline += nchar;
          num_feature = std::max(_feature, num_feature);
          has_feature = true;
          num_values++;
        }
        while ((*pline != 0) && ((*pline == ' ')  || (*pline == 9))) { pline++; } // skip trailing spaces
      }
    }
    fData.close();
  }

  if (has_feature) {
    num_feature++; // number of feature is bigger (by one) than the largest value
  }
  std::cout << "num_rows=" << num_rows << "\tnum_values=" << num_values << "\tnum_features=" << num_feature << "\tmin_target=" << min_target << "\tmax_target=" << max_target << std::endl;
  data.setSize(num_rows);
  target.setSize(num_rows);

  ((LargeSparseMatrixMemory<DATA_FLOAT>*)this->data)->num_cols = num_feature;
  ((LargeSparseMatrixMemory<DATA_FLOAT>*)this->data)->num_values = num_values;

  sparse_entry<DATA_FLOAT>* cache = new sparse_entry<DATA_FLOAT>[num_values];

  // (2) read the data
  {
    std::ifstream fData(filename.c_str());
    if (! fData.is_open()) {
      throw "unable to open " + filename;
    }
    int row_id = 0;
    uint64 cache_id = 0;
    DATA_FLOAT _value;
    int nchar, _feature;
    while (!fData.eof()) {
      std::string line;
      std::getline(fData, line);
      const char *pline = line.c_str();
      while ((*pline == ' ')  || (*pline == 9)) { pline++; } // skip leading spaces
      if ((*pline == 0)  || (*pline == '#')) { continue; }  // skip empty rows
      if (sscanf(pline, "%f%n", &_value, &nchar) >=1) {
        pline += nchar;
        assert(row_id < num_rows);
        target.value[row_id] = _value;
        data.value[row_id].data = &(cache[cache_id]);
        data.value[row_id].size = 0;

        while (sscanf(pline, "%d:%f%n", &_feature, &_value, &nchar) >= 2) {
          pline += nchar;
          assert(cache_id < num_values);
          cache[cache_id].id = _feature;
          cache[cache_id].value = _value;
          cache_id++;
          data.value[row_id].size++;
        }
        row_id++;

        while ((*pline != 0) && ((*pline == ' ')  || (*pline == 9))) { pline++; } // skip trailing spaces
      }
    }
    fData.close();

    assert(num_rows == row_id);
    assert(num_values == cache_id);
  }

  num_cases = target.dim;
}
