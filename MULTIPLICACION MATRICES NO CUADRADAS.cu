//MULTIPLICACIÓN DE MATRICES(APLANADAS)CUADRADAS EN C++ y CUDA con tiempo
#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<cuda.h>
using namespace std; 


__global__ void MultiplicaMatricesCU(int* A,int filA,int colA,int* B,int filB,int colB,int* C){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if((row<ancho)&&(col<ancho)){
		int suma=0;
		for(int k=0;k<filB;k++){//Se mueve entre las filas de B 
			suma=suma+A[(row*colA)+k]*B[(k*colB)+col];
		}
		C[(row*ancho)+col]=suma;
	}	
}

__host__ void multiplicaMatrices(int* X,int filX,int colX,int* Y,int filY,int colY,int* Z){
	for(int i=0;i<filX;i++){
		for(int j=0;j<colY;j++){
			int suma=0;
			for(int k=0;k<filY;k++){
				suma=suma+X[(i*colX)+k]*Y[(k*colY)+j];

			}
			Z[(i*colY)+j]=suma;
		}	
	}
}

__host__ void imprime(int* A,int filas, int columnas){//imprime como si fuera una matriz
	for(int i = 0; i < filas; i++){
        	for(int j = 0; j < columnas; j++){
            		cout<<A[(i*columnas)+j]<<" ";
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
	int *A,*B,*C; //A[filA][colA],B[filB][colB],C[filA][colB]
	int *d_A,*d_B,*d_C,*h_C;
	int filA=1024,colA=1024,filB=1024,colB=1024;
	//int filA=5,colA=10,filB=10,colB=1;
	//-------------------------------CPU--------------------------------------------------------------------
	startCPU = clock();	

	A=(int*)malloc(filA*colA*sizeof(int)); 
	B=(int*)malloc(filB*colB*sizeof(int));
	C=(int*)malloc(filA*colB*sizeof(int));

	inicializa(A,filA,colA);
	inicializa(B,filB,colB);
	
	if(colA==filB){//para que sean multiplicables
		//multiplicaMatrices(A,filA,colA,B,filB,colB,C);
		//imprime(C,filA,colB);
	}else{
		cout<<"Error, no se pueden multiplicar"<<endl;
		return 0;
	}
	
	endCPU = clock();

	double time_CPU=((double)(endCPU-startCPU))/CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la CPU fue: "<<time_CPU<<endl;
	//-------------------------------GPU--------------------------------------------------------------------
	h_C=(int*)malloc(filA*colB*sizeof(int));

	startGPU = clock();

	cudaMalloc((void**)&d_A,filA*colA*sizeof(int));
	cudaMalloc((void**)&d_B,filB*colB*sizeof(int));
	cudaMalloc((void**)&d_C,filA*colB*sizeof(int));	
	
	cudaMemcpy(d_A,A,filA*colA*sizeof(int),cudaMemcpyHostToDevice);//destino d_A y origen A
	cudaMemcpy(d_B,B,filB*colB*sizeof(int),cudaMemcpyHostToDevice);	

	//Depende directamente de la dimensión de las matrices
	dim3 dimblock(32,32,1);
	dim3 dimGrid(ceil(filA/32),ceil(colB/32),1);
	
	MultiplicaMatricesCU<<<dimGrid,dimblock>>>(d_A,filA,colA,d_B,filB,colB,d_C);

	cudaDeviceSynchronize();

	cudaMemcpy(h_C,d_C,filA*colB*sizeof(int),cudaMemcpyDeviceToHost);
	
	endGPU = clock();

	imprime(h_C,filA,colB);
	double time_GPU=((double)(endGPU-startGPU))/CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la GPU fue: "<<time_GPU<<endl;
	//-----------------------------------------------------------------------------------
	cout<<"El tiempo de aceleramiento fue: "<<time_CPU/time_GPU<<endl;
	free(A);free(B);free(C);free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
