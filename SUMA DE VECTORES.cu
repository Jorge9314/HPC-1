//SUMA DE VECTORES EN CUDA Y C++ CON TIEMPO
#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<cuda.h>

using namespace std;

#define SIZE 1024 //tamaño de los vectores


__host__ void inicializaVec(int* X){
	srand(time(NULL));
	for(int i=0;i<SIZE;i++){
                X[i]=rand()%10;
        }
} 

__host__ void imprimeVec(int* X){
        for(int i=0;i<SIZE;i++){
                cout<<X[i]<<" ";
        }
}

__host__ void SumaVec(int* X,int* Y,int* Z){
	for(int i=0;i<SIZE;i++){
                Z[i]=X[i]+Y[i];
        }

}

__global__ void SumaVecCU(int *A,int *B, int*C){
	int tid=threadIdx.x;
	//int tid=blockIdx.x;
	if(tid<SIZE)
		C[tid]=A[tid]+B[tid];

}


int main(void){
	clock_t startCPU,endCPU,startGPU,endGPU;  
	int *A, *B, *C, *d_A, *d_B, *d_C,*h_C; //vectores a los cuales se le van a realizar las operaciones

	//-------------------------------CPU--------------------------------------------------------------------
	//iniciamos la cuenta del reloj
	startCPU = clock();	

	//Reservamos memoria para el host
	A=(int*)malloc(SIZE*sizeof(int)); 
	B=(int*)malloc(SIZE*sizeof(int));
	C=(int*)malloc(SIZE*sizeof(int));

	inicializaVec(B);
	inicializaVec(A);
	
	SumaVec(A,B,C);

	//terminamos la cuenta del reloj
	endCPU = clock();
	
	//imprimeVec(C);
	double time_CPU=((double)(endCPU-startCPU))/CLOCKS_PER_SEC;
	cout<<endl<<"El tiempo transcurrido en la CPU fue: "<<time_CPU<<endl;
	//-------------------------------GPU--------------------------------------------------------------------
	h_C=(int*)malloc(SIZE*sizeof(int));
	
	//iniciamos la cuenta del reloj
	startGPU = clock();

	//Reservamos memoria para el device
	cudaMalloc((void**)&d_A,SIZE*sizeof(int));
	cudaMalloc((void**)&d_B,SIZE*sizeof(int));
	cudaMalloc((void**)&d_C,SIZE*sizeof(int));

	cudaMemcpy(d_A,A,SIZE*sizeof(int),cudaMemcpyHostToDevice);//destino d_A y origen A
	cudaMemcpy(d_B,B,SIZE*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 dimblock(SIZE,1,1);//vamos a utilicar un bloque con size threads
	dim3 dimGrid(1,1,1);
	
	SumaVecCU<<<dimGrid,dimblock>>>(d_A,d_B,d_C);
	cudaDeviceSynchronize();//espera que termine la funcion anterior 
	cudaMemcpy(h_C,d_C,SIZE*sizeof(int),cudaMemcpyDeviceToHost);//copia la operacion relizada en el device al host en el vector C

	//terminamos la cuenta del reloj	
	endGPU = clock();
	
	imprimeVec(h_C);
	double time_GPU=((double)(endGPU-startGPU))/CLOCKS_PER_SEC;
	cout<<endl<<"El tiempo transcurrido en la GPU fue: "<<time_GPU<<endl;
	//------------------------------------------------------------------------------------------------------	
	free(A);free(B);free(C);free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
	cout<<endl<<"El tiempo de aceleramiento fue: "<<time_CPU/time_GPU<<endl;
	return 0;
}
