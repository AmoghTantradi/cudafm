#include "fm.h"
#include <cuda.h>
#include <vector>
#include <iomanip>
#include "util/random.h"

#define NUM_THREADS 256


fm_model::fm_model(int n, int k) {
	/*
	cudaMalloc(&num_attribute, sizeof(int));
	cudaMalloc(&num_factor, sizeof(int));

	cudaMalloc(&w, sizeof(double) * n);

	cudaMalloc(&v, sizeof(double) * n * k);

	cudaMalloc(&w0, sizeof(double));
	
	*num_attribute = n; //do in cuda memory
	*num_factor = k; //do in cuda memory
	init_mean = 0;
	init_stdev = 0.01;

	double* vTemp = (double*)malloc(n*k * sizeof(double));//initialize random, cuda malloc and copy into v
	for (int i = 0; i < n*k; i++) {
		vTemp[i] = ran_gaussian(init_mean, init_stdev);
	}

	cudaMemcpy(v, vTemp, sizeof(double) * n * k, cudaMemcpyHostToDevice);

	
	free(vTemp);
	double* wTemp = (double*)malloc(n * sizeof(double));//initialize random, cuda malloc and copy into v
	for (int i = 0; i < n; i++) {
		wTemp[i] = ran_gaussian(init_mean, init_stdev);
	}
	
	cudaMemcpy(w, wTemp, sizeof(double) * n, cudaMemcpyHostToDevice);
	
	free(wTemp);
	double w0Temp = 0; //copy to cuda
	cudaMemcpy(w0, &w0Temp, sizeof(double), cudaMemcpyHostToDevice);
	//cuda malloc m_sum and m_sum_sqr
	learn_rate = 0.01;
	*/
	init_mean = 0;
	init_stdev = 0.01;

	double* vTemp = (double*)malloc(n*k * sizeof(double));//initialize random, cuda malloc and copy into v
	for (int i = 0; i < n*k; i++) {
		vTemp[i] = ran_gaussian(init_mean, init_stdev);
	}
	v = vTemp;
	double* wTemp = (double*)malloc(n * sizeof(double));//initialize random, cuda malloc and copy into v
	for (int i = 0; i < n; i++) {
		wTemp[i] = ran_gaussian(init_mean, init_stdev);
	}
	w = wTemp;
	w0 = (double*)malloc(sizeof(double));
	*w0 = 0;
	m_sum = (double*) malloc(n * sizeof(double));
	m_sum_sqr = (double*) malloc(n * sizeof(double));
	params.num_attribute = n;
	params.num_factor = k;
}

/*
__global__ void cudaPredict(sparse_entry<FM_FLOAT>* x, int xsize, int n, double* w0, double* w, double* v, double * pred) {

}
*/

__global__ void cudaPredict(sparse_entry<FM_FLOAT>* x, int xsize, uint* n, int* k, double* w0, double* w, double* v, double* sum, double* sum_sqr, double* pred) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid == 0) {
		*pred += *w0;
	}
	if (tid >= xsize)
        return;
	for (uint i = 0; i <xsize; i++) {
		*pred += w[x[i].id] * x[i].value;
	}
	for (int f = 0; f < (*k); f++) {
		sum[f] = 0;
		sum_sqr[f] = 0;
		for (uint i = 0; i < xsize; i++) {
			double d = v[f*(*n)+x[i].id] * x[i].value;
			sum[f] += d;
			sum_sqr[f] += d*d;
		}
		*pred += 0.5 * (sum[f]*sum[f] - sum_sqr[f]);
	}
}

/*
double fm_model::predict(sparse_entry<FM_FLOAT>* x, int xsize) {
	return predict(x, xsize, m_sum, m_sum_sqr);
}

double fm_model::predict(sparse_entry<FM_FLOAT>* x, int xsize, double* sum, double* sum_sqr) {
	
	double* pred; //cudamalloc this
	double hostPred; 
	cudaMalloc(&pred, sizeof(double));
	
	int blks = (xsize + NUM_THREADS-1)/NUM_THREADS;
	cudaPredict<<<blks, NUM_THREADS>>>(x, xsize, num_attribute, num_factor, w0, w, v, sum, sum_sqr, pred);
	//bring pred to host
	cudaMemcpy(&hostPred, pred, sizeof(long), cudaMemcpyDeviceToHost);
	cudaFree(pred);
	
	return hostPred;
}

__global__ void cudaSGD(sparse_entry<FM_FLOAT>* x, int xsize, uint* n, int* k, double* w0, double* w, double* v, const double multiplier, double lr,  double* sum) {
    //__syncthreads();
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid == 0) {
		*w0 -= lr * (multiplier);
    }
	if (tid >= xsize)
        return;
	w[x[tid].id] -= lr * (multiplier * x[tid].value);
    for (int f = 0; f < *k; f++) {
        double grad = sum[f] * x[tid].value - v[f*(*n)+x[tid].id] * x[tid].value * x[tid].value; 
        //v -= lr * (multiplier * grad + fm->regv * v); 
        v[f*(*n)+x[tid].id] -= lr * (multiplier * grad);
    }

}
*/

// X data must be stored as vector of  where the sparse entry resides in cuda memory, Y data is stored as vector of doubles
//std::vector<std::pair<sparse_entry<FM_FLOAT>*, int>> trainX, std::vector<double> trainY, std::vector<std::pair<sparse_entry<FM_FLOAT>*,int>> testX, std::vector<double> testY
/*
void fm_model::learn(std::vector<std::pair<sparse_entry<FM_FLOAT>*, int>> trainX, std::vector<double> trainY, std::vector<std::pair<sparse_entry<FM_FLOAT>*,int>> testX, std::vector<double> testY, int num_iter) {
    std::cout << "learnrate=" << learn_rate << std::endl;
    std::cout << "#iterations=" << num_iter << std::endl;

    std::cout.flush();
    std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl;
    // SGD

    for (int i = 0; i < num_iter; i++) {
		//double rmse
        for (int sample = 0; sample < trainX.size(); sample++) {
          double p = predict(trainX[sample].first, trainX[sample].second);
          double mult = 0;
          if (task == 0) {
              //p = std::min(max_target, p);
              //p = std::max(min_target, p);
              mult = -(trainY[sample]-p);
          } else if (task == 1) {
              mult = -trainY[sample]*(1.0-1.0/(1.0+exp(-trainY[sample]*p)));
          }
		  int blks = (NUM_THREADS+trainX[sample].second-1)/NUM_THREADS;
		  cudaSGD<<<blks, NUM_THREADS>>>(trainX[sample].first, trainX[sample].second, num_attribute, num_factor, w0, w, v, mult, learn_rate, sum);
        }
		/*
        double rmse_train = evaluate(train);
        double rmse_test = evaluate(test);
        std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
        if (LOG) {
        std::cout << "rmse_train " << rmse_train << std::endl;
        std::cout << '\n' << std::endl;
        }
    }
}
*/

//done
double fm_model::evaluate(Data* data) {
  assert(data.data != NULL);
  if (params.task == 0) {
    return evaluate_regression(data);
  } else if (params.task == 1) {
    return evaluate_classification(data);
  } else {
    throw "unknown task";
  }
}

//done
void fm_model::learn(Data* train, Data* test, int num_iter) {
    std::cout << "learnrate=" << params.learn_rate << std::endl;
	std::cout << "#iterations=" << num_iter << std::endl;
    std::cout.flush();
    std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl;
	// SGD
    for (int i = 0; i < num_iter; i++) {

        for (int j = 0; j < train->data.size(); j++) {
        	double p = predict(train->data[j], m_sum, m_sum_sqr);
        	double mult = 0;
			if (params.task == 0) {
				p = std::min(params.max_target, p);
				p = std::max(params.min_target, p);
				mult = -(train->target[j]-p);
			} else if (params.task == 1) {
				mult = -train->target[j]*(1.0-1.0/(1.0+exp(-train->target[j]*p)));
			}
        	SGD(train->data[j], mult, m_sum);

        }
        double rmse_train = evaluate(train);
        double rmse_test = evaluate(test);
        std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
    }
}

//done
void fm_model::SGD(sparse_row_v<FM_FLOAT>* x, const double multiplier, double *sum) {
	*w0 -= params.learn_rate * (multiplier);
	for (uint i = 0; i < x->size; i++) {
		w[i] -= params.learn_rate * (multiplier * x->data[i].value);
	}
	for (int f = 0; f < params.num_factor; f++) {
		for (uint i = 0; i < x->size; i++) {
			double& v1 = v[f*params.num_attribute + x->data[i].id];
			double grad = sum[f] * x->data[i].value - v1 * x->data[i].value * x->data[i].value; 
			v1 -= params.learn_rate * (multiplier * grad);
		}
	}
}

double fm_model::predict(sparse_row_v<FM_FLOAT>* x) {
	return predict(x, m_sum, m_sum_sqr);
}

double fm_model::predict(sparse_row_v<FM_FLOAT>* x, double* sum, double* sum_sqr) {
	double result = 0;
	result += *w0;
	for (uint i = 0; i < x->size; i++) {
		assert(x.data[i].id < num_attribute);
		result += w[x->data[i].id] * x->data[i].value;
	}
	for (int f = 0; f < params.num_factor; f++) {
		sum[f] = 0;
		sum_sqr[f] = 0;
		for (uint i = 0; i < x->size; i++) {
			double d = v[f*params.num_attribute + x->data[i].id] * x->data[i].value;
			sum[f] += d;
			sum_sqr[f] += d*d;
		}
		result += 0.5 * (sum[f]*sum[f] - sum_sqr[f]);
	}
	return result;
}

//predict
void fm_model::predict(Data* data, double* out) {
  for (int i = 0; i < data->data.size(); i++) {
    double p = predict(data->data[i]);
    if (params.task == 0 ) {
      p = std::min(params.max_target, p);
      p = std::max(params.min_target, p);
    } else if (params.task == 1) {
      p = 1.0/(1.0 + exp(-p));
    } else {
      throw "task not supported";
    }
    out[i] = p;
  }
}

double fm_model::evaluate_classification(Data* data) {
  int num_correct = 0;
  for (int i = 0; i < data->data.size(); i++) {
    double p = predict(data->data[i]);
    if (((p >= 0) && (data->target[i] >= 0)) || ((p < 0) && (data->target[i] < 0))) {
      num_correct++;
    }
  }

  return (double) num_correct / (double) data->data.size();
}

double fm_model::evaluate_regression(Data* data) {
  double rmse_sum_sqr = 0;
  double mae_sum_abs = 0;
  for (int i = 0; i < data->data.size(); i++) {
    double p = predict(data->data[i]);
    p = std::min(params.max_target, p);
    p = std::max(params.min_target, p);
    double err = p - data->target[i];
    rmse_sum_sqr += err*err;
    mae_sum_abs += std::abs((double)err);
  }

  return std::sqrt(rmse_sum_sqr/data->data.size());
}