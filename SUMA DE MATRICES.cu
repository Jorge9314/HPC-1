//SUMA DE MATRICES EN C++ Y CUDA CON TIEMPO
#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<cuda.h>
using namespace std; 

__global__ void SumaCU(int* A,int* B,int* C,int m,int n){//matriz[m][n]
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if((row<m)&&(col<n)){
		C[row*n+col]=A[row*n+col]+B[row*n+col];
	}	
}

__host__ void suma(int* A,int* B,int* C,int filas,int columnas){//Para sumar dos matrices deben tener las mismas dimensiones
	for(int i = 0; i < filas ; i++){
        	for(int j = 0; j < columnas ; j++){
                	C[(i*columnas)+j]=A[(i*columnas)+j]+B[(i*columnas)+j];
        	}
   	}
}

__host__ void imprime(int* A,int filas, int columnas){//imprime como si fuera una matriz
	for(int i = 0; i < filas; i++){
        	for(int j = 0; j < columnas; j++){
            		cout<<A[(i*columnas)+j];
        	}
        cout<<endl;
    }
}	

__host__ void inicializa(int *A,int filas, int columnas){//inicializa arreglos
	for(int i=0;i<filas*columnas;i++){
		A[i]=1;
	}
}

int main(void){

	clock_t startCPU,endCPU,startGPU,endGPU;  
	int *A,*B,*C,*h_C,*d_A,*d_B,*d_C;
	int filas=2048,columnas=2048,SIZE=filas*columnas*sizeof(int);

	//-------------------------------CPU--------------------------------------------------------------------
	startCPU = clock();	

	A=(int*)malloc(SIZE); 
	B=(int*)malloc(SIZE);
	C=(int*)malloc(SIZE);

	inicializa(A,filas,columnas);
	inicializa(B,filas,columnas);	

	suma(A,B,C,filas,columnas);
	
	endCPU = clock();

	//imprime(C,filas,columnas);
	double time_CPU=((double)(endCPU-startCPU))/CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la CPU fue: "<<time_CPU<<endl;
	//-------------------------------GPU--------------------------------------------------------------------	
	h_C=(int*)malloc(SIZE);
	
	startGPU = clock();

	cudaMalloc((void**)&d_A,SIZE);
	cudaMalloc((void**)&d_B,SIZE);
	cudaMalloc((void**)&d_C,SIZE);

	cudaMemcpy(d_A,A,SIZE,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,SIZE,cudaMemcpyHostToDevice);

	dim3 dimblock(32,32,1);//ya que es una matriz 10x10
	dim3 dimGrid(ceil(filas/32),ceil(columnas/32),1);
	
	SumaCU<<<dimGrid,dimblock>>>(d_A,d_B,d_C,filas,columnas);
	cudaDeviceSynchronize();//espera que termine la funcion anterior 
	cudaMemcpy(h_C,d_C,SIZE,cudaMemcpyDeviceToHost);//copia la operacion relizada en el device al host en el vector C
	
	endGPU = clock();
	
	imprime(h_C,filas,columnas);
	double time_GPU=((double)(endGPU-startGPU))/CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la GPU fue: "<<time_GPU<<endl;
	//------------------------------------------------------------------------------------------------------	
	free(A);free(B);free(C);free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cout<<"El tiempo de aceleramiento fue: "<<time_CPU/time_GPU<<endl;
	
	return 0;
}

