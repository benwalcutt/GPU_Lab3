/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "kernel.cu"
#include "support.h"

using namespace std;
int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;
    time_t t;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    } else if (argc == 2) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    } else if (argc == 4) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./sgemm                # All matrices are 1000 x 1000"
           "\n    Usage: ./sgemm <m>            # All matrices are m x m"
           "\n    Usage: ./sgemm <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
           "\n");
        exit(0);
    }

    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    C_sz = matArow*matBcol;

    /* Intializes random number generator */
    srand((unsigned) time(&t));    


    A_h = (float*) malloc( sizeof(float)*A_sz + 1);
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz + 1);
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*C_sz );
	
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
        matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

	if (cudaSuccess != cudaMalloc((void**)&A_d, sizeof(float)*A_sz)) {
		printf("Error allocating memory.\n");
		return 0;
	}	// request space for A on device
	if (cudaSuccess != cudaMalloc((void**)&B_d, sizeof(float)*B_sz)) {
		printf("Error allocating memory.\n");
		return 0;
	}   // request space for B on device
	if (cudaSuccess != cudaMalloc((void**)&C_d, sizeof(float)*C_sz)) {
		printf("Error allocating memory.\n");
		return 0;
	}	// request space for C on device

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

	if (cudaSuccess != cudaMemcpy(A_d, A_h, sizeof(float) * A_sz, cudaMemcpyHostToDevice)) {
		printf("Error copying memory to device.\n");
		return 0;
	}// copy A to device
	if (cudaSuccess != cudaMemcpy(B_d, B_h, sizeof(float) * B_sz, cudaMemcpyHostToDevice)) {
		printf("Error copying memory to device.\n");
		return 0;
	}		// copy B to device

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicSgemm('N', 'N', matArow, matBcol, matBrow, 1.0f, \
		A_d, matArow, B_d, matBcol, 0.0f, C_d, matBrow);

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

	if (cudaSuccess != cudaMemcpy(C_h, C_d, sizeof(float) * C_sz, cudaMemcpyDeviceToHost)) {
		printf("Error copying data from device to host.\n");
		return 0;
	}

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, matArow, matAcol, matBcol);

	
	
    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE

	if (cudaSuccess != cudaFree(A_d)) {
		printf("Error freeing device memory.\n");
		return 0;
	}				// free device memory for A 
	if (cudaSuccess != cudaFree(B_d)) {
		printf("Error freeing device memory.\n");
		return 0;
	}			// free device memory for B 
	if (cudaSuccess != cudaFree(C_d)) {
		printf("Error freeing device memory.\n");
		return 0;
	}			// free device memory for C 


    return 0;

}

