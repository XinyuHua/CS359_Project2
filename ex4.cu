#include <stdio.h>
#include <assert.h>

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

__global__ void SetMatrixA( float *d_a )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_a[idx] = blockIdx.x + 0.1 * threadIdx.x;
}

__global__ void SetMatrixB( float *d_a )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_a[idx] = 3 * blockIdx.x + 0.5 * threadIdx.x;
}

// Warning: While use this function, dim should not exceed 32
__global__ void MultiplicationSingle( float *d_a, float *d_b, float *d_c, int dim)
{
	int idx = threadIdx.x;
	int row = threadIdx.x / dim;
	int col = threadIdx.x % dim;
	float rst = 0;
	for(int i = 0; i < dim; ++i)
	{
		rst = d_a[row * dim + i]*d_b[i * dim + col] + rst;
	}
	d_c[row * dim + col] = rst;
}

__global__ void MultiplicationMulti( float *d_a, float *d_b, float *d_c)
{
	int dim = blockDim.x;
	int row = blockIdx.x;
	int col = threadIdx.x;
	float rst = 0;
	for(int i = 0; i < dim; ++i)
	{
	    rst = d_a[row * dim + i]*d_b[i * dim + col] + rst;
	}
	d_c[row * dim + col] = rst;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

typedef struct{
	int width;
	int height;
	float *vals;
}Matrix;

void displayMatrix(Matrix m)
{
	printf("----------------------------------\n");
	for(int i = 0; i < m.height; ++i)
	{	
		printf("Row %d\n",i);
		for(int j = 0; j < m.width; ++j)
		{
			printf("%.2f\n", m.vals[m.width * i + j]);
		}
		printf("\n\n");
	}
}


int main( int argc, char** argv) 
{
	float *h_a, *h_b, *h_c;
	float *d_a, *d_b, *d_c;
	
	int dim = 64;
	
	int numBlocks = dim;
	int numThreadsPerBlock = dim;
	
	size_t memSize = numBlocks * numThreadsPerBlock * sizeof(float);
	
	h_a = (float *) malloc(memSize);
	h_b = (float *) malloc(memSize);
	h_c = (float *) malloc(memSize);

    cudaMalloc( &d_a, memSize );
	cudaMalloc( &d_b, memSize );
	cudaMalloc( &d_c, memSize );

//  Initialize Matrix a and Matrix b
	dim3 dimGrid( numBlocks );
    dim3 dimBlock( numThreadsPerBlock  );
    
	SetMatrixA<<< numBlocks , numThreadsPerBlock >>>( d_a );
	cudaThreadSynchronize();

	SetMatrixB<<< numBlocks , numThreadsPerBlock >>>( d_b );
	cudaThreadSynchronize();

	checkCUDAError("kernel execution");
	cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost  );
	cudaMemcpy(h_b, d_b, memSize, cudaMemcpyDeviceToHost  );
	checkCUDAError("cudaMemcpy");

//  Perform the Multiplication
	MultiplicationMulti<<< dim, dim >>>(d_a, d_b, d_c);
	cudaThreadSynchronize();

	checkCUDAError("kernel execution");
	cudaMemcpy(h_c, d_c, memSize, cudaMemcpyDeviceToHost  );
	checkCUDAError("cudaMemcpy");

	Matrix ma = {dim,dim,h_a};
	Matrix mb = {dim,dim,h_b};
	Matrix mc = {dim,dim,h_c};
	//displayMatrix(ma);
	//displayMatrix(mb);
	displayMatrix(mc);
	
	free(h_a);
	free(h_b);
	free(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
    cudaDeviceReset();
	return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}
