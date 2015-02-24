/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

	__shared__ float As[TILE_SIZE][TILE_SIZE];
	__shared__ float Bs[TILE_SIZE][TILE_SIZE];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int Row = by * TILE_SIZE + ty;
	int Col = bx * TILE_SIZE + tx;
	
	
	
	float Cvalue = 0.0;
	
	
	for (int i = 0; i < (k / TILE_SIZE + 1); i++) {
		if ((Row < m) && (tx + i*TILE_SIZE < k)) {
			As[ty][tx] = A[Row*k + i*TILE_SIZE + tx];
		}
		else {
			As[ty][tx] = 0.0;
		}
		
		if ((Col < k) && (ty + i*TILE_SIZE < k)) {
			Bs[ty][tx] = B[(i*TILE_SIZE + ty) * n + Col];
		}
		else {
		Bs[ty][tx] = 0.0;
		}
		
		__syncthreads();
		
		for (int j = 0; j < TILE_SIZE; ++j) {
			Cvalue += As[ty][j] * Bs[j][tx];
		}
		
		__syncthreads();
		
		
		
	}
	if ((Row < m) && (Col < n)) {
		C[Row*n + Col] = Cvalue;
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

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE

	const int num_blocks_m = ceil(m * 1.0 / BLOCK_SIZE * 1.0);
	const int num_blocks_n = ceil(n * 1.0 / BLOCK_SIZE * 1.0);
	dim3 dimGrid(num_blocks_m, num_blocks_n);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);



    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE




}


