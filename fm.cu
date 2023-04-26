#include "fm.h"
#include <cuda.h>
#include <vector>
#include <iomanip>
#include "util/random.h"

#define NUM_THREADS 256


fm_model::fm_model(int n, int k) {

	init_mean = 0;
	init_stdev = 0.01;

	double* vTemp = (double*)malloc(n*k * sizeof(double));//initialize random, cuda malloc and copy into v
	for (int i = 0; i < n*k; i++) {
		vTemp[i] = ran_gaussian(init_mean, init_stdev);
	}
	cudaMalloc((void**)&v, sizeof(double) * n * k);
	cudaMemcpy(v, vTemp, sizeof(double) * n * k, cudaMemcpyHostToDevice);
	free(vTemp);

	double* wTemp = (double*)malloc(n * sizeof(double));//initialize random, cuda malloc and copy into v
	for (int i = 0; i < n; i++) {
		wTemp[i] = ran_gaussian(init_mean, init_stdev);
	}
	cudaMalloc((void**)&w, sizeof(double) * n);
	cudaMemcpy(w, wTemp, sizeof(double) * n, cudaMemcpyHostToDevice);
	free(wTemp);
	
	double w0Temp = 0;
	cudaMalloc((void**)&w0, sizeof(double));
	cudaMemcpy(w0, &w0Temp, sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&m_sum, n * sizeof(double));
	cudaMalloc((void**)&m_sum_sqr, n * sizeof(double));
	//no cuda
	params.num_attribute = n;
	params.num_factor = k;
}


// X data must be stored as vector of  where the sparse entry resides in cuda memory, Y data is stored as vector of doubles


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
	for (int i = 0; i < train->data.size(); i++) {
		sparse_row_v<DATA_FLOAT>* sample;
		int memsize = sizeof(sparse_row_v<DATA_FLOAT>) + train->data[i]->size*sizeof(sparse_entry<DATA_FLOAT>);
		cudaMalloc((void**)&sample, memsize);
		cudaMemcpy(sample, train->data[i], memsize, cudaMemcpyHostToDevice);
		//free(train->data[i]);
		train->data[i] = sample;
	}

	cudaMalloc((void**)&cuda_args, sizeof(cudaArgs));
	cudaArgs args;
	args.w0 = w0;
	args.w = w;
	args.v = v;
	args.params = params;
	cudaMalloc((void**)&ret, sizeof(double));
	args.ret = ret;
	cudaMemcpy(cuda_args, &args, sizeof(cudaArgs), cudaMemcpyHostToDevice);
	

    for (int i = 0; i < num_iter; i++) {
		// new version


		// old version
        for (int j = 0; j < train->data.size(); j++) {
        	double p = predict(train->data[j], m_sum, m_sum_sqr);
			//std:: cout << p << "\n";
			// double p = 0;
        	double mult = 0;
			if (params.task == 0) {
				p = std::min(params.max_target, p);
				p = std::max(params.min_target, p);
				mult = -(train->target[j]-p);
			} else if (params.task == 1) {
				mult = -train->target[j]*(1.0-1.0/(1.0+exp(-train->target[j]*p)));
			}
        	SGD(train->data[j], mult, m_sum);

			// predict(train -> data[j], m_sum, m_sum_sqr, args);
        }
        double rmse_train = evaluate(train);
		std::cout << rmse_train << "\n";

		//std::cout << i << "\n";
        //double rmse_test = evaluate(test);
        //std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
    }

}


__global__ void cudaSGD(sparse_row_v<DATA_FLOAT>* x, const double multiplier, double *sum, cudaArgs* args) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid == 0) {
		*args->w0 -= args->params.learn_rate * (multiplier);
	}
	if (tid >= x->size * args -> params.num_factor)
        return;

	/*
	for (uint i = 0; i < x->size; i++) {
		args->w[i] -= args->params.learn_rate * (multiplier * x->data[i].value);
	}
	*/
	int nf = args -> params.num_factor;

	int f = int(tid / nf); // corresponds to f
	int idx = tid - f * nf; // corresponds to i


	if(f == 0) {
		args->w[idx] -= args->params.learn_rate * (multiplier * x->data[idx].value);
	}

	double& v1 = args->v[f * args->params.num_attribute + x->data[idx].id];
	double grad = sum[f] * x->data[idx].value - v1 * x->data[idx].value * x->data[idx].value;
	v1 -= args->params.learn_rate * (multiplier * grad);


	/**
	for (int f = 0; f < args->params.num_factor; f++) {
		for (uint i = 0; i < x->size; i++) {
			double& v1 = args->v[f*args->params.num_attribute + x->data[i].id];
			double grad = sum[f] * x->data[i].value - v1 * x->data[i].value * x->data[i].value; 
			v1 -= args->params.learn_rate * (multiplier * grad);
		}
	}
	*/
}
//done
void fm_model::SGD(sparse_row_v<DATA_FLOAT>* x, const double multiplier, double *sum) {
	/*
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
	*/
	cudaSGD<<<1, NUM_THREADS>>>(x, multiplier, sum, cuda_args);
}


/*
__device__ void cudaDotProd(double *v, sparse_entry<DATA_FLOAT> x, ) {
	atomicAdd(, v[x.id] * x.value);
}
*/

__global__ void cudaPredict(sparse_row_v<DATA_FLOAT>* x, double* sum, double* sum_sqr, cudaArgs* args) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int nf = args -> params.num_factor;
	if (tid >= x->size * nf) {
		return;
	}

	//tid = face_value * num_factor + i where i < x.size 
	int f = int(tid / nf); // corresponds to f
	int idx = tid - f * nf; // corresponds to i

	if (f == 0) {
		atomicAdd(args->ret, args -> w[x->data[idx].id] * x->data[idx].value);
	} // adds the linear terms 
	
	double val = args -> v[f *args->params.num_attribute + x->data[idx].id] * x->data[idx].value;

	atomicAdd(&sum[f], val); // stores in sum[Faceval]
	atomicAdd(&sum_sqr[f], val * val);
}

__global__ void aggregate(double * ret, double * sum, double * sum_sqr, cudaArgs * args) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= args -> params.num_factor) {
		return;
	}
	//ret should be initialized to 0 first 
	atomicAdd(ret, 0.5 * (sum[tid] * sum[tid] - sum_sqr[tid]));
}

double fm_model::predict(sparse_row_v<DATA_FLOAT>* x) {
	return predict(x, m_sum, m_sum_sqr);
}

double fm_model::predict(sparse_row_v<DATA_FLOAT>* x, double* sum, double* sum_sqr) {
	double pred;
	cudaMemcpy(ret, w0, sizeof(double), cudaMemcpyDeviceToDevice);

	cudaMemset(sum, 0, sizeof(int) * params.num_factor);
	cudaMemset(sum_sqr, 0, sizeof(int) * params.num_factor);

	cudaPredict<<<1, NUM_THREADS>>>(x, sum, sum_sqr, cuda_args);
	aggregate<<<1, NUM_THREADS>>>(ret, sum, sum_sqr, cuda_args);

	cudaMemcpy(&pred, ret, sizeof(double), cudaMemcpyDeviceToHost);

	return pred;
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