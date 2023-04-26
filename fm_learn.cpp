#include "fm_learn.h"
#include <cmath>
#include "Data.h"
#include "fm.h"
#include <iostream>
#include <iomanip>
#include <omp.h>

bool LOG = false;

//207.989 for omp
//221.908

double fm_learn::predict_case(Data* data) {
  return fm->predict(data->data->getRow());
}

//done
fm_learn::fm_learn() {
  task = 0;
}

//done
void fm_learn::init() {
  sum.setSize(fm->num_factor);
  sum_sqr.setSize(fm->num_factor);
  pred_q_term.setSize(fm->num_factor, 2);
  learn_rates.setSize(3);
}

//done
double fm_learn::evaluate(Data* data) {
  //assert(data.data != NULL);
  if (task == TASK_REGRESSION) {
    return evaluate_regression(data);
  } else if (task == TASK_CLASSIFICATION) {
    return evaluate_classification(data);
  } else {
    throw "unknown task";
  }
}

//done
void fm_learn::learn(Data* train, Data* test) {
    std::cout << "learnrate=" << learn_rate << std::endl;
    std::cout << "learnrates=" << learn_rates.get(0) << "," << learn_rates.get(1) << "," << learn_rates.get(2) << std::endl;
    std::cout << "#iterations=" << num_iter << std::endl;

    std::cout.flush();
      std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl;
    // SGD
    //#pragma omp parallel for 
    for (int i = 0; i < num_iter; i++) {

        for (train->data->begin(); !train->data->end(); train->data->next()) {
        double p = fm->predict(train->data->getRow(), sum, sum_sqr);
        double mult = 0;
        if (task == 0) {
            p = std::min(max_target, p);
            p = std::max(min_target, p);
            mult = -(train->target(train->data->getRowIndex())-p);
        } else if (task == 1) {
            mult = -train->target(train->data->getRowIndex())*(1.0-1.0/(1.0+exp(-train->target(train->data->getRowIndex())*p)));
        }
        SGD(&train->data->getRow(), mult, &sum);
        }
        double rmse_train = evaluate(train);
        double rmse_test = evaluate(test);
        std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
        if (LOG) {
        std::cout << "rmse_train " << rmse_train << std::endl;
        std::cout << '\n' << std::endl;
        }
    }
}

//done
void fm_learn::SGD(sparse_row<DATA_FLOAT> *x, const double multiplier, DVector<double> *sum) {
  if (fm->k0) {
		double& w0 = fm->w0;
		w0 -= learn_rate * (multiplier + fm->reg0 * w0);
	}
	if (fm->k1) {
    #pragma omp for 
		for (uint i = 0; i < x->size; i++) {
			double& w = fm->w(x->data[i].id);
			w -= learn_rate * (multiplier * x->data[i].value + fm->regw * w);
		}
	}
  #pragma omp parallel for collapse(2)
	for (int f = 0; f < fm->num_factor; f++) {
		for (uint i = 0; i < x->size; i++) {
			double& v = fm->v(f, x->data[i].id);
			double grad = (*sum)(f) * x->data[i].value - v * x->data[i].value * x->data[i].value; 
			v -= learn_rate * (multiplier * grad + fm->regv * v);
		}
	}
}

//predict
void fm_learn::predict(Data* data, DVector<double>* out) {
  assert(data->data->getNumRows() == out->dim);
  for (data->data->begin(); !data->data->end(); data->data->next()) {
    double p = predict_case(data);
    if (task == TASK_REGRESSION ) {
      p = std::min(max_target, p);
      p = std::max(min_target, p);
    } else if (task == TASK_CLASSIFICATION) {
      p = 1.0/(1.0 + exp(-p));
    } else {
      throw "task not supported";
    }
    (*out)(data->data->getRowIndex()) = p;
  }
}

double fm_learn::evaluate_classification(Data* data) {
  int num_correct = 0;
  for (data->data->begin(); !data->data->end(); data->data->next()) {
    double p = predict_case(data);
    if (((p >= 0) && (data->target(data->data->getRowIndex()) >= 0)) || ((p < 0) && (data->target(data->data->getRowIndex()) < 0))) {
      num_correct++;
    }
  }

  return (double) num_correct / (double) data->data->getNumRows();
}

double fm_learn::evaluate_regression(Data* data) {
  double rmse_sum_sqr = 0;
  double mae_sum_abs = 0;
  for (data->data->begin(); !data->data->end(); data->data->next()) {
    double p = predict_case(data);
    p = std::min(max_target, p);
    p = std::max(min_target, p);
    double err = p - data->target(data->data->getRowIndex());
    rmse_sum_sqr += err*err;
    mae_sum_abs += std::abs((double)err);
  }

  return std::sqrt(rmse_sum_sqr/data->data->getNumRows());
}
