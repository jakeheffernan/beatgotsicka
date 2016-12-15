#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cufft.h>
#include <cuda.h>

#define N 8192


__global__ void cuda_derivative(float* sample, cufftReal* differential) {

	//determine the index of array to be handled by the thread
    int index;
    index = blockIdx.x * blockDim.x + threadIdx.x;

   /// printf("index is: %i \n", index);

    //sample rate of digital quality audio
    int constant;
    constant = 44100;

    //if the index # is 0 or 1
    if (index == 0 || index == (N - 1)) {
        differential[index] = sample[index];
    }

    else{
    	//calculate the difference between the previous and next element
        differential[index] = constant * (sample[index+1]-sample[index-1])/2;
        //printf("value is: %d \n", sample[index]);

    }

    printf("der index: %i , der value: %i \n", index, differential[index]);
	//printf("der value: %i \n", differential[index]);

}


__global__ void float_to_real(float* sample, cufftReal* differential) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	differential[index] = sample[index];


}

__host__ void readInWavFile(float* output){
	
	//Open wave file in read mode
	FILE * infile = fopen("classic.wav","rb");        
   // For counting number of frames in wave file.
    int count = 0;                        
    /// short int used for 16 bit as input data format is 16 bit PCM audio
    int buff16;
   
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


//function to take the differential derivative of a signal
void host_derivative(float* sample, cufftReal* differential){

	int constant = 44100;
	int i;

	for (i = 0; i < N; i++){

		if (i == 0 || i == N -1){

			differential[i] = sample [i];

		}

		else {

			differential[i] = constant * (sample[i+1]-sample[i-1])/2;
		
		}

	}

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

__host__ void generateCombFilters(int bpm, int sampleSize, int numbOfCombs, cufftReal* combFilters){

	int ti;
	int offset = 0;

	for (int j = 0; j < numbOfCombs; j++){


		ti = 60 * 44100/(bpm + j);
		offset = sampleSize * j;

		for(int i = 0; i < sampleSize; i++){
	
			if(i % ti == 0)
				combFilters[i + offset] = 1;
			else
				combFilters[i + offset] = 0;	
		}
	}
		
}

__host__ void combFilterFFT(int numberOfCombs, cufftReal* combFilters, cufftComplex* fftOfCombFilters) {
    // Assign Variables
    cufftHandle plan;
    cufftReal* deviceDataIn;
  	cufftComplex* deviceDataOut;
 
    // Malloc Variables
    cudaMalloc((void**)&deviceDataIn, sizeof(cufftReal) * numberOfCombs * N);
     // Malloc Variables
    cudaMalloc((void**)&deviceDataOut, sizeof(cufftComplex) * numberOfCombs * N);
    //Generate all Combs
    int n[1] = {N};
    cudaMemcpy(deviceDataIn, combFilters, numberOfCombs * N * sizeof(cufftReal), cudaMemcpyHostToDevice);

    //malloc room for output fft
 
    if (cufftPlanMany(&plan, 1, n, NULL, 1, N, NULL, 1, N, CUFFT_R2C, numberOfCombs) != CUFFT_SUCCESS) {
        printf("CUFFT Error - plan creation failed\n");
        exit(-1);
    }
    if (cufftExecR2C(plan, deviceDataIn, deviceDataOut) != CUFFT_SUCCESS) {
        printf("CUFFT Error - execution of FFT failed\n");
        exit(-1);
    }

	cudaDeviceSynchronize();

    // Cleanup
    if (cufftDestroy(plan) != CUFFT_SUCCESS) {
      printf("CUFFT Error - plan destruction failed\n");
      exit(-1);
    }

    /*transfer results back to host*/
	cudaMemcpy(fftOfCombFilters, deviceDataOut, N*numberOfCombs*sizeof(cufftComplex), cudaMemcpyDeviceToHost);


    cudaFree(deviceDataIn);
    cudaFree(deviceDataOut);
    free(combFilters);

    return;
}

__host__ int combFilterAMultiplication(cufftComplex* fftDerivativeArray, int numberOfCombs, cufftComplex* fftCombs){

	double* magnitudes = (double*)malloc(sizeof(double) * numberOfCombs * (N/2+1));
	double* sumarray = (double*)malloc(sizeof(double) * numberOfCombs);
	double a = 0;
	double b = 0;
	float sum = 0;


	//multiply step
	for(int j = 0; j < numberOfCombs; j++){
		for(int i = 0; i < (N/2+1); i++){
	    
	        a = fftDerivativeArray[i].x* fftCombs[i + (j*N)].x - fftDerivativeArray[i].y * fftCombs[i + (j*N)].y ;	       
	        b = fftDerivativeArray[i].x * fftCombs[i + (j*N)].y + fftDerivativeArray[i].y * fftCombs[i + (j*N)].x ;

	        //find magnitude
	   		magnitudes[i + (j*N)] =  a*a + b*b;
	    }
	}

	for(int num = 0; num < numberOfCombs; num++){
	    for(int j = num * N; j < (num + 1) * N; j++){
	        //find signal energy
	        sum = sum + magnitudes[j];
	    }
	    //put signal energy of comb #num into sumArray[num]
	    sumarray[num] = sum;
	}
	int k = 0;
	int max = sumarray[k];
	//find max
	for (int ct = 0; ct < numberOfCombs; ++ct){
	    if (sumarray[ct] > max){
	        max = sumarray[ct];
	        k = ct; //k is index of max signal energy
	    }
	}
	free(magnitudes);
	free(sumarray);

	//printf(" the thing is %d", a);

	return k;
}


int main (int argc, char ** argv)
{

	//call in fuction to read in the file 

	float * A;
	A = (float*)malloc(N*sizeof(float));

	readInWavFile(A);


	//move memory to device
	float *d_A;
	cudaMalloc((void**) &d_A, N*sizeof(float));
	

	cufftReal *d_derivative; 
	cudaMalloc((void**) &d_derivative, N*sizeof(cufftReal)); 

	cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);

	//Call kernel setup
	int numberofThreads = 1024;
	int numberofBlocks = N/1024;
   	//differentiate the sample
	cuda_derivative<<<numberofBlocks,numberofThreads>>>(d_A, d_derivative);

	cudaDeviceSynchronize();

	cufftReal* testStuff = (cufftReal*)malloc(sizeof(cufftReal) * N);
	cudaMemcpy(testStuff, d_derivative, N*sizeof(cufftReal), cudaMemcpyDeviceToHost);

	//host_derivative(A, testStuff);

	printf("\nThe derivative is: %d \n", testStuff[1375]);


	//take the fft of the sample

	cufftComplex *fftout; 
	cudaMalloc((void**)&fftout, sizeof(cufftComplex)*(N/2+1)); 

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
	if (cufftExecR2C(plan, d_derivative, fftout) != CUFFT_SUCCESS){ 
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed"); 
		return 1;	
	} 


	if (cudaDeviceSynchronize() != cudaSuccess){ 
		fprintf(stderr, "Cuda error: Failed to synchronize\n"); 
		return 1;	
	} 

	/*transfer results back to host*/
	cufftComplex *fftoutput = (cufftComplex*)malloc( sizeof(cufftComplex)*(N/2+1)); 
	cudaMemcpy(fftoutput,fftout, sizeof(cufftComplex)*(N/2+1), cudaMemcpyDeviceToHost);

	//printf("\n the fft of signal is %d", fftoutput[1000].x);

	cufftDestroy(plan); 
	cudaFree(fftout); 

	//generate combs
	int minBPM = 50;
	int maxBPM = 150;
	int numberOfCombs = maxBPM - minBPM;

	cufftReal* combFilters = (cufftReal*)malloc(sizeof(cufftReal) * numberOfCombs * N);
	generateCombFilters(minBPM, N, numberOfCombs, combFilters);



	//FFT of the combBabies 

	printf("\nhere");

	cufftComplex* fftOfCombFilters = (cufftComplex*)malloc(sizeof(cufftComplex) * numberOfCombs * N);
	combFilterFFT(numberOfCombs, combFilters, fftOfCombFilters);

	//printf("\n the fft of comb filter is %d", fftOfCombFilters[1000].x);

	//multiply with sample

	int beat = 0;
	beat = combFilterAMultiplication(fftoutput, numberOfCombs, fftOfCombFilters);

	//compare results 
	printf("\n the resulting beat is: %i", minBPM + beat);

	//pick the winner


}

