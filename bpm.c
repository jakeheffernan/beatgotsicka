
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include "kiss_fft.h"


#define N 1024


void readInWavFile(float* output){
	
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

//function to multiply two arrays together
void oneDMatrixMultiplication(float * A, float * B, float * result){
	int i;
	for(i = 0; i < N; i++){
		result[i] = A[i] * B[i];
	}
}

//function to take the differential derivative of a signal
void host_derivative(float* sample, float* differential){

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
//function to generate a comb filter 
void host_combFilter(int bpm, float* result){
	//Convert input tempo to beats per minute
	int Ti = 60 * 44100/bpm;
	int i;

	//set elements at each beat to 1 else set to 0
	for(i = 0; i < N; i++){

		if(i % Ti == 0)
			result[i] = 1;
		else
			result[i] = 0;
	}
	
}
//function that calculates signal energy by taking each element and squaring it then adding it to a total
void host_calculateSignalEnergy(float* energy, float* inputData){
	*energy = 0;
	int i;
	for(i = 0; i < N; i++){
		inputData[i] = inputData[i] * inputData[i];
		*energy = *energy + inputData[i];
	}
}

//function to convolve to signals together
void host_convolve(float* Signal, size_t length, float* Kernel, float* result){
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


	host_derivative(A, derivative);
	host_calculateSignalEnergy(&sum,derivative);
	host_combFilter(30,combFilter);
	host_convolve(derivative,length,combFilter,combresult);
	host_calculateSignalEnergy(&sum,combresult);
	printf("Signal Energy is %f \n", sum);



}

