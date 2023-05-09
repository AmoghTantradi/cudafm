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
	params.num_attribute = n;
	params.num_factor = k;



	cudaMalloc((void **) &bufferSpMv, sizeof(double));


	cusparseCreate(&handle);


	//initialize cusparseDnvec w
	cusparseCreateDnVec(&w_vec, n, w, CUDA_R_64F);

	// initializes cusparseSpMatDescr_t V, cusparseSpMatDescr_t V_2 
	cusparseCreateDnMat(&V, params.num_attribute, params.num_factor, params.num_factor, v, CUDA_R_64F, CUSPARSE_ORDER_ROW);
	cusparseCreateDnMat(&V_2, params.num_attribute, params.num_factor, params.num_factor, v2, CUDA_R_64F, CUSPARSE_ORDER_ROW);
}

void fm_model::batchSamples(Data* train, std::vector<trainBatch> &batches) {
	int* csr_offsets = new int[train->data.size()+1];
	double* target = new double[train->data.size()];
	csr_offsets[0] = 0;
	int cnt = 0;
	for (int i = 0; i < train->data.size(); i++) {
		cnt += train->data[i]->size;
		csr_offsets[i+1] = cnt;
		target[i] = train->target[i];
	}
	int* csr_columns = new int[cnt];
	int* row_idx = new int[cnt];
	double* csr_values = (double *) malloc(sizeof(double)*cnt);
	double* csr_values2 = (double *) malloc(sizeof(double)*cnt);
	int idx = 0;
	for (int i = 0; i < train->data.size(); i++) {
		for(int j = 0; j < train->data[i]->size; j++) {
			csr_columns[idx] = train->data[i]->data[j].id;
			csr_values[idx] = train->data[i]->data[j].value;
			csr_values2[idx] = train->data[i]->data[j].value*train->data[i]->data[j].value;
			++idx;
		}
	}
	double* cudaTarget;
	cudaMalloc((void**) &cudaTarget, sizeof(double)*train->data.size());
	cudaMemcpy(cudaTarget, target, sizeof(double)*train->data.size(), cudaMemcpyHostToDevice);
	//cuda put this into big sparse matrices
	int start = 0;
	idx = 0;
	int rows = 0;
	std::unordered_set<int> feature_ids;
	for (int i = 0; i < train->data.size(); i++) {
		bool repeat = false;
		csr_offsets[i] -= start;
		for (int j = 0; j < train->data[i]->size; j++) {
			row_idx[idx] = rows;
			idx++;
			if (feature_ids.count(train->data[i]->data[j].id)) {
				repeat = (rows > 2000);
			}
		}
		if (repeat) {
			int nnz = idx - train->data[i]->size - start;
			int   *dX_csrOffsets, *dX_columns, *dX_row_idx;
    		double *dX_values, *dX2_values;
			cudaMalloc((void**) &dX_csrOffsets,
                           (rows + 1) * sizeof(int));

			cudaMalloc((void**) &dX_columns, nnz * sizeof(int));
			cudaMalloc((void**) &dX_row_idx, nnz * sizeof(int));
			cudaMalloc((void**) &dX_values,  nnz * sizeof(double));
			cudaMalloc((void**) &dX2_values,  nnz * sizeof(double));

			cudaMemcpy(dX_csrOffsets, csr_offsets+i-rows,
								(rows + 1) * sizeof(int),
								cudaMemcpyHostToDevice);
			cudaMemcpy(dX_columns, csr_columns + start, nnz * sizeof(int),
								cudaMemcpyHostToDevice);
			cudaMemcpy(dX_row_idx, row_idx + start, nnz * sizeof(int),
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
			pointsPerBatch.push_back(rows);
			maxBatch = std::max(maxBatch, rows);
			trainBatch tbatch;
			tbatch.x = matX;
			tbatch.x2 = matX2;
			tbatch.target = cudaTarget + i - rows;
			tbatch.size = rows;
			tbatch.xCols = dX_columns;
			tbatch.xVals = dX_values;
			tbatch.rowidx = dX_row_idx;
			tbatch.nnz = nnz;
			batches.push_back(tbatch);
			rows = 0;
			start = idx;
			feature_ids.clear();
		}
		++rows;
		for (int j = 0; j < train->data[i]->size; ++j) {
			feature_ids.insert(train->data[i]->data[j].id);
		}
	}

	{
		csr_offsets[train->data.size()] -= start;
		int nnz = idx - start;
		int   *dX_csrOffsets, *dX_columns, *dX_row_idx;
		double *dX_values, *dX2_values;
		cudaMalloc((void**) &dX_csrOffsets,
						(rows + 1) * sizeof(int));

		cudaMalloc((void**) &dX_columns, nnz * sizeof(int));
		cudaMalloc((void**) &dX_row_idx, nnz * sizeof(int));
		cudaMalloc((void**) &dX_values,  nnz * sizeof(double));
		cudaMalloc((void**) &dX2_values,  nnz * sizeof(double));

		cudaMemcpy(dX_csrOffsets, csr_offsets+train->data.size()-rows,
							(rows + 1) * sizeof(int),
							cudaMemcpyHostToDevice);
		cudaMemcpy(dX_columns, csr_columns + start, nnz * sizeof(int),
							cudaMemcpyHostToDevice);
		cudaMemcpy(dX_row_idx, row_idx + start, nnz * sizeof(int),
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
		pointsPerBatch.push_back(rows);
		maxBatch = std::max(maxBatch, rows);
		trainBatch tbatch;
		tbatch.x = matX;
		tbatch.x2 = matX2;
		tbatch.target = cudaTarget +train->data.size()-rows;
		tbatch.xCols = dX_columns;
		tbatch.xVals = dX_values;
		tbatch.rowidx = dX_row_idx;
		tbatch.nnz = nnz;
		tbatch.size = rows;
		batches.push_back(tbatch);
	}
	cudaMalloc((void**) &xiv, sizeof(double) * maxBatch * params.num_factor);
	cudaMalloc((void**) &x2iv2, sizeof(double) * maxBatch * params.num_factor);

	cudaMalloc((void ** )&xiw, sizeof(double) * maxBatch);

	cudaMalloc((void**) &bufferMatmul, (8*((maxBatch + 7)/8)));
	
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
	// if (bufferSize > maxBufferSize) {
	// 	maxBufferSize = bufferSize;
	// 	std::cout << maxBatch << " " << bufferSize << std::endl;
	// }
    cudaMalloc(&dBuffer, bufferSize);

	

	cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, A, B, &beta, result, CUDA_R_64F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);


}


void fm_model::setSize(int batchSize) {
	//cuda
	//cudamalloc
	
	cusparseCreateDnVec(&xw,
                    batchSize,
                    xiw,
                    CUDA_R_64F); // creates dense vector for linear term

	
	cusparseCreateDnMat(&xv, batchSize, params.num_factor, params.num_factor, xiv, CUDA_R_64F, CUSPARSE_ORDER_ROW);
	cusparseCreateDnMat(&x2v2, batchSize, params.num_factor, params.num_factor, x2iv2, CUDA_R_64F, CUSPARSE_ORDER_ROW);
}

__global__ void doNothing(double *XV, double* X2V2, double* XW, double* W0) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx == 0) {
		
	}
	printf("hi\n");
	__syncthreads();
}
__global__ void sumColumns(double *XV, double* X2V2, double* XW, double* W0, double* preds, int batchSize, int num_factors) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("%d", batchSize);
    if (idx < batchSize) {
		//printf("test1\n");
        double sum = 0.0;
		preds[idx] = 0.0;
        for (int j = 0; j < num_factors; j++) {
            sum += XV[idx*num_factors + j];
			preds[idx] -= 0.5*X2V2[idx*num_factors+j];
        }
        preds[idx] += *W0 + XW[idx] + 0.5*sum*sum;
    } else {
		//printf("test2\n");
	}
}

void fm_model::predict(trainBatch batch, int batchSize,  double* preds) {

	// step 3, step 4 done here 	double* x2iv2; //x^2v^2
	setSize(batchSize);

	//create intermediate result for sotring the product of Xi and V 
	//matmul. Only does SPMM!
	matMul(batch.x, V, xv); // xv, xiv will now contain XV 
	matMul(batch.x2, V_2, x2v2); //x2v2, x2iv2 will now contain x^2 v^2
	//inline matvec 
	
	//size_t bufferSize = 0;
	//void* buffer;
	double alpha           = 1.0;
    double beta            = 0.0;

/*
	cusparseSpMV_bufferSize(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        batch.x,  // non-const descriptor supported
                        w_vec,  // non-const descriptor supported
                        &beta,
                        xw,
                        CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT,
                        &bufferSize);

	cudaMalloc(&buffer, bufferSize);
	if (bufferSize > 8) std::cout<< bufferSize << "\n";
*/
	//std::cout<<"bufferSize " << batchSize << " " << bufferSize << std::endl;

	//cudaMalloc(&buffer, 8);
	cusparseSpMV(handle,
             CUSPARSE_OPERATION_NON_TRANSPOSE,
              &alpha,
             batch.x,  // non-const descriptor supported
             w_vec,  // non-const descriptor supported
             &beta,
             xw,
             CUDA_R_64F,
             CUSPARSE_SPMV_ALG_DEFAULT,
             bufferSpMv);

	// compute preds for NUM_THREADS threads
	int blks = (batchSize + NUM_THREADS - 1) / NUM_THREADS;
	//std::cout << "Calling sum " << blks << std::endl;
	/*
	cudaDeviceSynchronize();
	cudaError_t err = cudaPeekAtLastError();
	//doNothing<<<1, 1>>>(xiv, x2iv2, xiw, w0);
	if (err != cudaSuccess) {
	 	std::cerr << "Error0: " << cudaGetErrorString(err) << std::endl;
	}
	*/
	// sumColumns<<<blks, NUM_THREADS>>>(xv, x2v2, xw, w0, preds, batchSize, params.num_factor);
	sumColumns<<<blks, NUM_THREADS>>>(xiv, x2iv2, xiw, w0, preds, batchSize, params.num_factor);
	//std::cout << "completed sum " << blks << std::endl;
	// cudaError_t
	/*
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "Error1: " << cudaGetErrorString(err) << std::endl;
	}
	*/
	// err = cudaPeekAtLastError();
	// if (err != cudaSuccess) {
	// 	std::cerr << "Error3: " << cudaGetErrorString(err) << std::endl;
	// }
	/*
	for (int i = 0; i < batchSize; ++i) {
        sumColumns(xv, x2v2, xw, w0, preds, batchSize, params.num_factor);
		double pred = preds[i];
        double target = batch.target[i];
        double mult = -target * (1 - 1 / (1 + fexp(-pred * target))); // TODO: replace with fast sigmoid

        // Store the prediction in the corresponding spot in preds array
        preds[batch.startIdx + i] = pred;
    }*/

	
	
	
	//will have to free our cudaMallocs too!
	//will have to also destroy our intermediate cusparseDnMats too!
}


__global__ void cudaSGD(double* pred, double* target, double* xv, cudaArgs* args, trainBatch batch) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	//calculate multiplier via -target*(1-1/(1+e^(-pred*target)))
	if (tid >= batch.nnz) {
		return;
	}
	int sample = batch.rowidx[tid];
	double multiplier = -target[sample] * (1 - 1 / (1 + __expf(-pred[sample] * target[sample])));
	int col = batch.xCols[tid];

	
	*args->w0 -= args->params.learn_rate * (multiplier);
	args->w[col] -= args->params.learn_rate * multiplier * batch.xVals[tid];
	for (int i = 0; i < args->params.num_factor; i++) {
		double& v1 = args->v[args->params.num_factor * col + i];
		double grad = batch.xVals[tid] * xv[sample*args->params.num_factor+i] - v1 * batch.xVals[tid] * batch.xVals[tid];
		//v1 -= args->params.learn_rate * (multiplier * grad);
	}

	/*
	for (uint i = 0; i < x->size; i++) {
		args->w[i] -= args->params.learn_rate * (multiplier * x->data[i].value);
	}
	int nf = args -> params.num_factor;

	int f = int(tid / nf); // corresponds to f
	int idx = tid - f * nf; // corresponds to i


	if(f == 0) {
		args->w[idx] -= args->params.learn_rate * (multiplier * x->data[idx].value);
	}

	double& v1 = args->v[f * args->params.num_attribute + x->data[idx].id];
	double grad = sum[f] * x->data[idx].value - v1 * x->data[idx].value * x->data[idx].value;
	v1 -= args->params.learn_rate * (multiplier * grad);
	*/
}

void fm_model::learn(std::vector<trainBatch> &training_batches, const int num_iter) {

	//call batch samples here 

	std::cout <<"Learn called" << std::endl;

	
	std::cout << "batched training samples" << std::endl;

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
	
	double* preds;
	cudaMalloc((void**)&preds, sizeof(double)*maxBatch);
	

    //for (int i = 0; i < num_iter; i++) {

		//use batches here instead 
		/*
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
        }*/
		// loop over predictions, create our own mult array / vector based off of those predictions
		//for (int j = 0; j < training_batches.size(); j++) {
	bool broken = false;
	for (int num_iters = 0; num_iters < 100; num_iters++) {
		int correct = 0;
	for (int b1 = 0; b1 < training_batches.size(); b1++){
		//std::cout << "Made prediction vector" << std::endl;		
		//std::cout << "Batch size for batch " << b << " is: " << pointsPerBatch[b] << std ::endl;
		predict(training_batches[b1], training_batches[b1].size, preds);
		int blks = (training_batches[b1].nnz+NUM_THREADS-1)/NUM_THREADS;
		cudaSGD<<<blks, NUM_THREADS>>>(preds, training_batches[b1].target, xiv, cuda_args, training_batches[b1]);	
		cudaError_t err = cudaPeekAtLastError();
		if (err != cudaSuccess) {
			std::cerr << "Error69: " << cudaGetErrorString(err) << std::endl;
			broken = true;
			std::cout << num_iters << "\n";
		}
		
		if (broken) {
			break;
		}
		
		{
			//std::cout << training_batches[b1].size << "\n";
		double* p = (double*)malloc(sizeof(double)*training_batches[b1].size);
		//std::cout <<"trying to copy" << std::endl;
		cudaMemcpy(p, preds, training_batches[b1].size*sizeof(double), cudaMemcpyDeviceToHost);
		double* p1 = (double*)malloc(sizeof(double)*training_batches[b1].size);
		//std::cout <<"trying to copy" << std::endl;
		cudaMemcpy(p1, training_batches[b1].target, training_batches[b1].size*sizeof(double), cudaMemcpyDeviceToHost);
		
		cudaError_t err = cudaPeekAtLastError();
		if (err != cudaSuccess) {
			std::cerr << "Error420: " << cudaGetErrorString(err) << std::endl;
			broken = true;
			std::cout << num_iters << "\n";
		}
		int numzero = 0;
		for (int i = 0; i < training_batches[b1].size; i++) {
				if(p[i]*p1[i] > 0) {
				++correct;
				}
				if (p[i] == 0) {
				++numzero;
				}
				if(b1 == 0) {
					if(i < 10) std::cout<<p[i] << " ";
				}
		}
		free(p);
		free(p1);
		}

			//}
			//double rmse_train = evaluate(train);
			/*
		for (int i = 0; i < pointsPerBatch[b]; i++) {
			std::cout << "Prediction for data point " << i << ": " << p[i] << std::endl;
		}
		*/
	}
	std::cout << correct << "\n";
	if (broken) {
		break;
	}
	}


    //}
}

/*
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






//predict
void fm_model::predict(Data* data, double* out) { // should take in a batch 
  for (int i = 0; i < data->data.size(); i++) {
    double p = predict(data->data[i]); // supposed to call predict(x, m_sum, m_sum_sqr) 
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




double fm_model::predict(sparse_row_v<FM_FLOAT>& x) {
	return predict(x, m_sum, m_sum_sqr);		
}

double fm_model::predict(sparse_row_v<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr) {
	double result = 0;
	if (k0) {	
		result += w0;
	}
	if (k1) {
		for (uint i = 0; i < x.size; i++) {
			assert(x.data[i].id < num_attribute);
			result += w(x.data[i].id) * x.data[i].value;
		}
	}
	for (int f = 0; f < num_factor; f++) {
		sum(f) = 0;
		sum_sqr(f) = 0;
		for (uint i = 0; i < x.size; i++) {
			double d = v(f,x.data[i].id) * x.data[i].value;
			sum(f) += d;
			sum_sqr(f) += d*d;
		}
		result += 0.5 * (sum(f)*sum(f) - sum_sqr(f));
	}
	return result;
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
*/
