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
    if (idx < batchSize) {
        double sum = 0.0;
		preds[idx] = 0.0;
        for (int j = 0; j < num_factors; j++) {
            sum += XV[idx*num_factors + j];
			preds[idx] -= 0.5*X2V2[idx*num_factors+j];
        }
        preds[idx] += *W0 + XW[idx] + 0.5*sum*sum;
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
	double alpha           = 1.0;
    double beta            = 0.0;

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


	int blks = (batchSize + NUM_THREADS - 1) / NUM_THREADS;
	sumColumns<<<blks, NUM_THREADS>>>(xiv, x2iv2, xiw, w0, preds, batchSize, params.num_factor);

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
	atomicAdd(&args->w[col], -(args->params.learn_rate * multiplier * batch.xVals[tid])/3.0);
	for (int i = 0; i < args->params.num_factor; i++) {
		double v1 = args->v[args->params.num_factor * col + i];
		double grad = batch.xVals[tid] * xv[sample*args->params.num_factor+i] - v1 * batch.xVals[tid] * batch.xVals[tid];
		if (fabs(grad) < 0.5) {
			args->v[args->params.num_factor * col + i] -= args->params.learn_rate * (multiplier * grad + 0.01 * v1);
			v1 = args->v[args->params.num_factor * col + i];
			args->v2[args->params.num_factor * col + i] = v1*v1;
		}
	}
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
	
	bool broken = false;
	for (int num_iters = 0; num_iters < 100; num_iters++) {
		int correct = 0;
	for (int b1 = 0; b1 < training_batches.size(); b1++){
		predict(training_batches[b1], training_batches[b1].size, preds);
		int blks = (training_batches[b1].nnz+NUM_THREADS-1)/NUM_THREADS;
		cudaSGD<<<blks, NUM_THREADS>>>(preds, training_batches[b1].target, xiv, cuda_args, training_batches[b1]);	
		cudaDeviceSynchronize();
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
		double* p = (double*)malloc(sizeof(double)*training_batches[b1].size);
		cudaMemcpy(p, preds, training_batches[b1].size*sizeof(double), cudaMemcpyDeviceToHost);
		double* p1 = (double*)malloc(sizeof(double)*training_batches[b1].size);
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
	}
	std::cout << correct << "\n";
	if (broken) {
		break;
	}
	}

}