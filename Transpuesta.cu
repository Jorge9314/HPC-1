//Transpuesta de una matriz
#include<iostream>
#include<stdio.h>
#include<malloc.h>

using namespace std;


__host__
void T(int *A, int filas, int columnas, int* B){
  for(int j = 0; j < columnas; j++){
    for(int i = 0; i < filas; i++){
      B[j*filas+i] = A[i*columnas+j];
    }
  }
}

__global__
void T(int *A, int filas, int columnas, int* B){

    int i = blockIdx.y*blockDim.y+threadIdx.y;//filas
    int j = blockIdx.x*blockDim.x+threadIdx.x;//columnas

    if(i < filas && j < columnas)
      B[j*filas+i] = A[i*columnas+j];
}

__host__
void imprime(int* A,int filas, int columnas){//imprime como si fuera una matriz
	for(int i = 0; i < filas; i++){
        	for(int j = 0; j < columnas; j++){
            		cout<<A[(i*columnas)+j]<<" ";
        	}
        cout<<endl;
    }
}

__host__
void inicializa(int *A,int filas, int columnas){//inicializa arreglos
	for(int i=0;i<filas*columnas;i++){
		A[i]=i;
	}
}

__host__
bool compara(int *A, int *B, int filas, int columnas){
	for(int i = 0; i < filas; i++){
		for(int j = 0; j < columnas; j++){
			if(A[i*columnas+j] != B[i*columnas+j]) return false;
		}
	}
	return true;
}

int main(void){

    cudaError_t error = cudaSuccess;//Para controlar errores
    int *matriz, *Tmatriz, *h_matriz, *d_matriz;
    int filas = 3, columnas = 2;

    int size = filas*columnas*sizeof(int);

    //----------------------CPU-------------------------

    //Separamos memoria para el host
    matriz = (int*)malloc(size);
    Tmatriz = (int*)malloc(size);

    //Inicializamos la matriz
    inicializa(matriz, filas, columnas);

    clock_t t = clock();//Iniciamos la cuenta de reloj

    //Hacemos la transpuesta
    T(matriz, filas, columnas, Tmatriz);

    t = clock() - t;//Terminamos la cuenta de reloj

    //Mostramos el resultado
    /*
    cout << "Original: " << endl;
    imprime(matriz, filas, columnas);
    cout << "transpuesta: " << endl;
    imprime(Tmatriz, columnas, filas);
    */

    double time_CPU = ((double)t) / CLOCKS_PER_SEC;
  	cout<<"El tiempo transcurrido en la CPU fue: "<<time_CPU<<endl;

    //----------------------GPU--------------------------------------

    //Separamos memoria para el device

    

    //Liberamos memoria
    free(matriz); free(Tmatriz);


    return 0;

}
