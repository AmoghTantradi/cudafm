#include "fm.h"
#include <cuda.h>
#include <vector>
#include <iomanip>
#include "util/random.h"
#include <unordered_set>

#define NUM_THREADS 256

fm_model::fm_model(int n, int k) {
	init_mean = 0;
	init_stdev = 0.01;

	double* vTemp = (double*)malloc(n*k * sizeof(double));
	double* vTempsq = (double*) malloc(n * k * sizeof(double));
	for (int i = 0; i < n*k; i++) {
		vTemp[i] = ran_gaussian(init_mean, init_stdev);
		vTempsq[i] = vTemp[i] * vTemp[i];
	}
	cudaMalloc((void**)&v, sizeof(double) * n * k);
	cudaMalloc((void**)&v2, sizeof(double) * n * k);


	cudaMemcpy(v, vTemp, sizeof(double) * n * k, cudaMemcpyHostToDevice);
	cudaMemcpy(v2, vTempsq, sizeof(double) * n * k, cudaMemcpyHostToDevice);

	free(vTemp);
	free(vTempsq);

	double* wTemp = (double*)malloc(n * sizeof(double));
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
	params.num_attribute = n;
	params.num_factor = k;


	cusparseCreate(&handle);

	// initializes cusparseSpMatDescr_t V, cusparseSpMatDescr_t V_2 
	cusparseCreateDnMat(&V, params.num_attribute, params.num_factor, params.num_factor, v, CUDA_R_64F, CUSPARSE_ORDER_ROW);
	cusparseCreateDnMat(&V_2, params.num_attribute, params.num_factor, params.num_factor, v2, CUDA_R_64F, CUSPARSE_ORDER_ROW);
}



__global__ void cudaPredict(sparse_row_v<DATA_FLOAT>* x, double* sum, double* sum_sqr, cudaArgs* args) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int nf = args -> params.num_factor;
	if (tid >= x->size * nf) {
		return;
	}

	int f = int(tid / nf);
	int idx = tid - f * nf;

	if (f == 0) {
		atomicAdd(args->ret, args -> w[x->data[idx].id] * x->data[idx].value);
	}
	
	double val = args -> v[f *args->params.num_attribute + x->data[idx].id] * x->data[idx].value;

	atomicAdd(&sum[f], val);
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

double fm_model::predict(sparse_row_v<DATA_FLOAT>* x, double* sum, double* sum_sqr) {

	// want to batch x 
	//also want to make sure our x doesn't have any overlap in our features 


    double pred;
    cudaMemcpy(ret, w0, sizeof(double), cudaMemcpyDeviceToDevice);

    cudaMemset(sum, 0, sizeof(double) * params.num_factor);
    cudaMemset(sum_sqr, 0, sizeof(double) * params.num_factor);

    int num_blocks = (3 * params.num_factor + NUM_THREADS - 1) / NUM_THREADS;
    size_t shared_mem_size = 2 * params.num_factor * sizeof(double);

    cudaPredict<<<num_blocks, NUM_THREADS, shared_mem_size>>>(x, sum, sum_sqr, cuda_args);
    aggregate<<<1, NUM_THREADS>>>(ret, sum, sum_sqr, cuda_args);

    cudaMemcpy(&pred, ret, sizeof(double), cudaMemcpyDeviceToHost);

    return pred;
}

// double fm_model::predict(sparse_row_v<DATA_FLOAT>* x, double* sum, double* sum_sqr) {
// 	double pred;
// 	cudaMemcpy(ret, w0, sizeof(double), cudaMemcpyDeviceToDevice);

// 	cudaMemset(sum, 0, sizeof(int) * params.num_factor);
// 	cudaMemset(sum_sqr, 0, sizeof(int) * params.num_factor);

// 	cudaPredict<<<1, NUM_THREADS>>>(x, sum, sum_sqr, cuda_args);  // multiply X and V, X2 and V2  (get quadratic term). multiply X with w to get linear term. mu
// 	aggregate<<<1, NUM_THREADS>>>(ret, sum, sum_sqr, cuda_args);

// 	cudaMemcpy(&pred, ret, sizeof(double), cudaMemcpyDeviceToHost);

// 	return pred;
// }

void fm_model::batchSamples(Data* train, std::vector<std::pair<cusparseSpMatDescr_t, cusparseSpMatDescr_t>> &batches) {
	int* csr_offsets = new int[train->data.size()+1];
	csr_offsets[0] = 0;
	int cnt = 0;
	for (int i = 0; i < train->data.size(); i++) {
		cnt += train->data[i]->size;
		csr_offsets[i+1] = cnt;
	}
	int* csr_columns = new int[cnt];
	double* csr_values = (double *) malloc(sizeof(double)*cnt);
	double* csr_values2 = (double *) malloc(sizeof(double)*cnt);
	int idx = 0;
	for (int i = 0; i < train->data.size(); i++) {
		for(int j = 0; j < train->data[i]->size; j++) {
			csr_columns[idx] = train->data[i]->data[j].id;
			csr_values[idx] = train->data[i]->data[j].value;
			csr_values2[idx] = train->data[i]->data[j].value*train->data[i]->data[j].value;
			idx++;
		}
	}
	//cuda put this into big sparse matrices
	int start = 0;
	idx = 0;
	int rows = 0;
	std::unordered_set<int> feature_ids;
	for (int i = 0; i < train->data.size(); i++) {
		bool repeat = false;
		csr_offsets[i] -= start;
		for (int j = 0; j < train->data[i]->size; j++) {
			if (feature_ids.count(train->data[i]->data[j].id)) {
				repeat = true;
				break;
			}
		}
		if (repeat) {
			int nnz = idx - start;
			int   *dX_csrOffsets, *dX_columns;
    		double *dX_values, *dX2_values;
			cudaMalloc((void**) &dX_csrOffsets,
                           (rows + 1) * sizeof(int));

			cudaMalloc((void**) &dX_columns, nnz * sizeof(int));
			cudaMalloc((void**) &dX_values,  nnz * sizeof(double));
			cudaMalloc((void**) &dX2_values,  nnz * sizeof(double));

			cudaMemcpy(dX_csrOffsets, csr_offsets+i-rows,
								(rows + 1) * sizeof(int),
								cudaMemcpyHostToDevice);
			cudaMemcpy(dX_columns, csr_columns + start, nnz * sizeof(int),
								cudaMemcpyHostToDevice);
			cudaMemcpy(dX_values, csr_values + start, nnz * sizeof(double),
								cudaMemcpyHostToDevice);
			cudaMemcpy(dX2_values, csr_values2 + start, nnz * sizeof(double),
								cudaMemcpyHostToDevice);

			cusparseSpMatDescr_t matX;
			// Create sparse matrix A in CSR format
			cusparseCreateCsr(&matX, rows, (int)params.num_attribute, nnz,
											dX_csrOffsets, dX_columns, dX_values,
											CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
			
			cusparseSpMatDescr_t matX2;
			// Create sparse matrix A in CSR format
			cusparseCreateCsr(&matX2, rows, (int)params.num_attribute, nnz,
											dX_csrOffsets, dX_columns, dX2_values,
											CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
			rows = 0;
			start = idx;
			batches.push_back(std::make_pair(matX, matX2));
			feature_ids.clear();
		}
		++rows;
		for (int j = 0; j < train->data[i]->size; ++j) {
			feature_ids.insert(train->data[i]->data[j].id);
		}
		idx += train->data[i]->size;
	}

	{
		csr_offsets[train->data.size()] -= start;
		int nnz = idx - start;
		int   *dX_csrOffsets, *dX_columns;
		double *dX_values, *dX2_values;
		cudaMalloc((void**) &dX_csrOffsets,
						(rows + 1) * sizeof(int));

		cudaMalloc((void**) &dX_columns, nnz * sizeof(int));
		cudaMalloc((void**) &dX_values,  nnz * sizeof(double));
		cudaMalloc((void**) &dX2_values,  nnz * sizeof(double));

		cudaMemcpy(dX_csrOffsets, csr_offsets+train->data.size()-rows,
							(rows + 1) * sizeof(int),
							cudaMemcpyHostToDevice);
		cudaMemcpy(dX_columns, csr_columns + start, nnz * sizeof(int),
							cudaMemcpyHostToDevice);
		cudaMemcpy(dX_values, csr_values + start, nnz * sizeof(double),
							cudaMemcpyHostToDevice);
		cudaMemcpy(dX2_values, csr_values2 + start, nnz * sizeof(double),
							cudaMemcpyHostToDevice);

		cusparseSpMatDescr_t matX;
		// Create sparse matrix A in CSR format
		cusparseCreateCsr(&matX, rows, (int)params.num_attribute, nnz,
										dX_csrOffsets, dX_columns, dX_values,
										CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
										CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
		
		cusparseSpMatDescr_t matX2;
		// Create sparse matrix A in CSR format
		cusparseCreateCsr(&matX2, rows, (int)params.num_attribute, nnz,
										dX_csrOffsets, dX_columns, dX2_values,
										CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
										CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
		rows = 0;
		start = idx;
		batches.push_back(std::make_pair(matX, matX2));
	}

	
}


//take in batch (can be x or x2) and multiply by v and store in result 
void fm_model::matMul(cusparseSpMatDescr_t &A, cusparseDnMatDescr_t& B, cusparseDnMatDescr_t& result) {
	void* dBuffer = NULL;
    size_t bufferSize = 0;


	double alpha           = 1.0;
    double beta            = 0.0;



	cusparseSpMM_bufferSize(
							handle,
							CUSPARSE_OPERATION_NON_TRANSPOSE,
							CUSPARSE_OPERATION_NON_TRANSPOSE,
							&alpha, A, B, &beta, result, CUDA_R_64F,
							CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

	cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, A, B, &beta, result, CUDA_R_64F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);


}

void fm_model::learn(Data* train, Data* test, int num_iter) {

	//call batch samples here 

	std::vector<std::pair<cusparseSpMatDescr_t, cusparseSpMatDescr_t>> training_batches;
	batchSamples(train, training_batches);

	cudaMalloc((void**)&cuda_args, sizeof(cudaArgs));
	cudaArgs args;
	args.w0 = w0;
	args.w = w;
	args.v = v;

	args.v2 = v2;


	args.params = params;
	cudaMalloc((void**)&ret, sizeof(double));
	args.ret = ret;
	cudaMemcpy(cuda_args, &args, sizeof(cudaArgs), cudaMemcpyHostToDevice);



    for (int i = 0; i < num_iter; i++) {
        for (int j = 0; j < train->data.size(); j++) {
        	double p = predict(train->data[j], m_sum, m_sum_sqr);  // prediction is here 
        	double mult = 0;
			if (params.task == 0) {
				p = std::min(params.max_target, p);
				p = std::max(params.min_target, p);
				mult = -(train->target[j]-p);
			} else if (params.task == 1) {
				mult = -train->target[j]*(1.0-1.0/(1.0+exp(-train->target[j]*p)));
			}
        	SGD(train->data[j], mult, m_sum); // serialize this (kernels will probably make it slower)
        }
        double rmse_train = evaluate(train);
    }
}


__global__ void cudaSGD(sparse_row_v<DATA_FLOAT>* x, const double multiplier, double *sum, cudaArgs* args) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid == 0) {
		*args->w0 -= args->params.learn_rate * (multiplier);
	}
	if (tid >= x->size * args -> params.num_factor) return;
	int nf = args -> params.num_factor;

	int f = int(tid / nf);
	int idx = tid - f * nf;


	if(f == 0) {
		args->w[idx] -= args->params.learn_rate * (multiplier * x->data[idx].value);
	}

	double& v1 = args->v[f * args->params.num_attribute + x->data[idx].id];
	double grad = sum[f] * x->data[idx].value - v1 * x->data[idx].value * x->data[idx].value;
	v1 -= args->params.learn_rate * (multiplier * grad);
}

void fm_model::SGD(sparse_row_v<DATA_FLOAT>* x, const double multiplier, double *sum) {

	//serialize 
	cudaSGD<<<1, NUM_THREADS>>>(x, multiplier, sum, cuda_args);
}



double fm_model::predict(sparse_row_v<DATA_FLOAT>* x) {
	return predict(x, m_sum, m_sum_sqr);
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

double fm_model::evaluate(Data* data) {
  //assert(data->data != NULL);
  if (params.task == 0) {
    return evaluate_regression(data);
  } else if (params.task == 1) {
    return evaluate_classification(data);
  } else {
    throw "unknown task";
  }
}

