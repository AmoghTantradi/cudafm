#include "fm.h"
#include <cuda.h>
#include <vector>

#define NUM_THREADS 256

#include "fm.h"

fm_model::fm_model(int n, int k) {
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

// X data must be stored as vector of  where the sparse entry resides in cuda memory, Y data is stored as vector of doubles
//std::vector<std::pair<sparse_entry<FM_FLOAT>*, int>> trainX, std::vector<double> trainY, std::vector<std::pair<sparse_entry<FM_FLOAT>*,int>> testX, std::vector<double> testY

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
		*/
    }
}
