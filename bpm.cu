#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cufft.h>
#include <cuda.h>

#define N 1024


__host__ void readInWavFile(float* output){
	
	//Open wave file in read mode
	FILE * infile = fopen("classic.wav","rb");        
   // For counting number of frames in wave file.
    int count = 0;                        
    /// short int used for 16 bit as input data format is 16 bit PCM audio
    short int buff16;
   
    printf("%s\n","start" );

    if (infile)
    {
       
        fseek(infile,44,SEEK_SET);

        while (count < N){
            fread(&buff16,sizeof(buff16),1,infile);        // Reading data in chunks of BUFSIZE
            output[count] = buff16;
            count++;                    
    	}
   printf("the first value is %f", output[0]);
	}
}

__global__ void cuda_derivative(float* sample, float* differential) {

	//determine the index of array to be handled by the thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //sample rate of digital quality audio
    int constant = 44100;

    //if the index # is 0 or 1
    if (index == 0 || index == N - 1) {
        differential[index] = sample[index];
    }

    else{
    	//calculate the difference between the previous and next element
        differential[index] = constant * (sample[index+1]-sample[index-1])/2;

    }

}


__global__ void float_to_real(float* sample, cufftReal* differential) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	differential[index] = sample[index];


}

__host__ void oneDMatrixMultiplication(float * A, float * B, float * result){
	for(int i = 0; i < N; i++){
		result[i] = A[i] * B[i];
	}
}
__host__ void matrixCompare(float * cudaD, float * hostD){
	
	int result = 1;
	int i; 
	
	for (i = 0; i < N; i++){

		if (cudaD[i] != hostD[i]){
			result = 0;
		}
	}

	fprintf(stdout, "the result of the comparison is: %d ", result);
}


__host__ void host_derivative(float* sample, float* differential){

	int constant = 44100;

	for (int i = 0; i < N; i++){

		if (i == 0 || i == N -1){

			differential[i] = sample [i];

		}

		else {

			differential[i] = constant * (sample[i+1]-sample[i-1])/2;
		}

	}

}

__host__ void host_combFilter(int bpm, float* result){
	int Ti = 60 * 44100/bpm;
	for(int i = 0; i < N; i++)
		result[i] = 0;
	for(int j = 0; j< N; j++){
		if(j % Ti == 0)
			result[j] = 1;
	}	
}
__host__ void host_calculateSignalEnergy(float* energy, float* inputData){
	*energy = 0;
	for(int i = 0; i < N; i++){
		inputData[i] = inputData[i] * inputData[i];
		*energy = *energy + inputData[i];
	}
}

__host__ void host_convolve(float* Signal, size_t length, float* Kernel, float* result){
	size_t n;
	for (n = 0; n < length + length - 1; n++){
		size_t kmin, kmax, k;
		result[n] = 0;
		kmin = (n >= length - 1) ? n - (length - 1) : 0;
		kmax = (n < length - 1) ? n : length - 1;
		for (k = kmin; k <= kmax; k++){
			result[n] += Signal[k] * Kernel[n - k];
		}

	}
}

int main (int argc, char ** argv)
{

	//call in fuction to read in the file 

	float * A, *derivative, *cudaResult, *combFilter, *combresult;
	float sum = 0;
	int i;
	size_t length = N;
	A = (float*) malloc(N*sizeof(float));
	derivative = (float*) malloc(N*sizeof(float));
	cudaResult = (float*) malloc(N*sizeof(float));
	combFilter = (float*) malloc(N*sizeof(float));
	combresult = (float*) malloc(2*N*sizeof(float));
	readInWavFile(A);
	//Fill in random values to test
	//move memory to 
	float *d_A, *d_derivative;
	cudaMalloc((void**) &d_A, N*sizeof(float));
	cudaMalloc((void**) &d_derivative, N*sizeof(float));

	cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);


	//Call kernel setup
	dim3 threadsPerBlock(N, 1, 1);
    dim3 blocksPerGrid(1, 1, 1);

   	//differentiate the sample
	cuda_derivative<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_derivative);
	cudaDeviceSynchronize();

	/*transfer results back to host*/
	cudaMemcpy(cudaResult, d_derivative, N*sizeof(float), cudaMemcpyDeviceToHost);


	host_derivative(A, derivative);
	host_calculateSignalEnergy(&sum,derivative);
	host_combFilter(30,combFilter);
	host_convolve(derivative,length,combFilter,combresult);
	host_calculateSignalEnergy(&sum,combresult);
	printf("Signal Energy is %f \n", sum);
	matrixCompare(cudaResult, derivative);


	//move data through cuda from float to complex

	cufftReal *data; 
	cudaMalloc((void**)&data, sizeof(cufftReal)*(N)); 

	cufftComplex *fftout; 
	cudaMalloc((void**)&fftout, sizeof(cufftComplex)*(N/2+1)); 



   	//differentiate the sample
	float_to_real<<<blocksPerGrid,threadsPerBlock>>>(d_derivative, data);
	cudaDeviceSynchronize();

	//take the fft of the sample

	cufftHandle plan; 

	if (cudaGetLastError() != cudaSuccess){ 
		fprintf(stderr, "Cuda error: Failed to allocate\n"); 
		return 1;	

	} 

	if (cufftPlan1d(&plan, N, CUFFT_R2C, 1) != CUFFT_SUCCESS){ 
		fprintf(stderr, "CUFFT error: Plan creation failed"); 
		return 1;	
	}	

	 //Use the CUFFT plan to transform the signal in place.  
	if (cufftExecR2C(plan, data, fftout) != CUFFT_SUCCESS){ 
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed"); 
		return 1;	
	} 


	if (cudaDeviceSynchronize() != cudaSuccess){ 
		fprintf(stderr, "Cuda error: Failed to synchronize\n"); 
		return 1;	
	} 

	fprintf(stderr, "Finished\n"); 


	cufftDestroy(plan); 
	cudaFree(data); 

	//generate combs

	//multiply with sample

	//compare results 

	//pick the winner

}

