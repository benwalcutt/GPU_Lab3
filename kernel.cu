/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

	//const int width = m * m;
	
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	float C_value = 0;
	
	if ((row < m) && (col < n)) {
		for (int i = 0; i < k; i++) {

			C_value += A[row*k + i] * B[i*n + col];
		}
		
		C[row*m + col] = C_value;
		C_value  = 0;
	}
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = 16; // Use 16x16 thread blocks

    //INSERT CODE HERE

	//const int width = m * n;
	const int num_blocks_m = ceil(m * 1.0 / BLOCK_SIZE * 1.0);
	const int num_blocks_n = ceil(k * 1.0 / BLOCK_SIZE * 1.0);
	dim3 dimGrid(num_blocks_m, num_blocks_n);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE




}


