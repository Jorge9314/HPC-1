//MULTIPLICACIÓN DE MATRICES(APLANADAS)NO CUADRADAS EN C++ y CUDA con tiempo
#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<cuda.h>
using namespace std; 
#define TILE_WIDTH 32

/*	
* TILE_WIDTH_X debe ser el resultado de columnas de A
* TILE_WIDTH_Y debe ser el resultado de las filas de B
* ya que C[colA][filB] 
* TILE_WIDTH_X es el común entre A y B
*/

__global__ void MultiplicaMatricesCU(int* A,int filA,int colA,int* B,int filB,int colB,int* C){//filC=filA,colC=colB
	
	__shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

	// Identifico la fila y la columna de el elemento a trabajar
	int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;//la dimensión del bloque(en shared memory) va a ser el TILE_WIDTH
	int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

	int suma = 0;
	
	for (int m=0; m < colA/TILE_WIDTH; m++){//¿ hasta el TILE_WIDTH mayor ?

		//sacamos los pedazos con los que vamos a trabajar
		A_s[threadIdx.y][threadIdx.x] = A[Row*colA + ( (m*TILE_WIDTH) + threadIdx.x )];//(Row*colA + k), donde k-> 0..filB (filB = colA)
		// (m*TILE_WIDTH_X) + threadIdx.x aquí nos movemos entre las columnas 
		B_s[threadIdx.y][threadIdx.x] = B[( (m*TILE_WIDTH + threadIdx.y)*colB ) + Col];//(k*colB)+Col, donde k-> 0..filB 
		//(m*TILE_WIDTH_Y + threadIdx.y) se mueve entre las filas 
				
		__syncthreads();//espera a todos los hilos
		
		for (int k=0; k < TILE_WIDTH; ++k) {//es como si se moviera entre 0 y (filB || colA) que son las mismas 
			suma += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
		}
		__syncthreads();
	}
	C[Row*colB + Col] = suma;//(row*colB)+col
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
        cudaError_t error = cudaSuccess;
	int *A,*B,*C; //A[filA][colA],B[filB][colB],C[filA][colB]
	int *d_A,*d_B,*d_C,*h_C;
	int filA=700,colA=1024,filB=1024,colB=1;
	//int filA=5,colA=10,filB=10,colB=1;
	//-------------------------------CPU--------------------------------------------------------------------
	A=(int*)malloc(filA*colA*sizeof(int)); 
	B=(int*)malloc(filB*colB*sizeof(int));
	C=(int*)malloc(filA*colB*sizeof(int));

	inicializa(A,filA,colA);
	inicializa(B,filB,colB);
	
	if(colA==filB){//para que sean multiplicables
		startCPU = clock();		
		multiplicaMatrices(A,filA,colA,B,filB,colB,C);
		endCPU = clock();
		//imprime(C,filA,colB);
	}else{
		cout<<"Error, no se pueden multiplicar"<<endl;
		return 0;
	}

	double time_CPU=((double)(endCPU-startCPU))/CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la CPU fue: "<<time_CPU<<endl;
	//-------------------------------GPU--------------------------------------------------------------------
	h_C=(int*)malloc(filA*colB*sizeof(int));

	startGPU = clock();

	error=cudaMalloc((void**)&d_A,filA*colA*sizeof(int));
        if(error != cudaSuccess){
            cout<<"Error reservando memoria para d_A"<<endl;
            return -1;
        }
    
	cudaMalloc((void**)&d_B,filB*colB*sizeof(int));
        if(error != cudaSuccess){
            cout<<"Error reservando memoria para d_B"<<endl;
            return -1;
        }
        
	cudaMalloc((void**)&d_C,filA*colB*sizeof(int));	
        if(error != cudaSuccess){
            cout<<"Error reservando memoria para d_C"<<endl;
            return -1;
        }
	
	cudaMemcpy(d_A,A,filA*colA*sizeof(int),cudaMemcpyHostToDevice);//destino d_A y origen A
	cudaMemcpy(d_B,B,filB*colB*sizeof(int),cudaMemcpyHostToDevice);	

	//Depende directamente de la dimensión de las matrices
	dim3 dimblock(32,32,1);
	dim3 dimGrid(ceil((double)(colA/32)),ceil((double)(filA/32)),1);
	
	MultiplicaMatricesCU<<<dimGrid,dimblock>>>(d_A,filA,colA,d_B,filB,colB,d_C);

	cudaDeviceSynchronize();

	cudaMemcpy(h_C,d_C,filA*colB*sizeof(int),cudaMemcpyDeviceToHost);
	
	endGPU = clock();

	//imprime(h_C,filA,colB);
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
