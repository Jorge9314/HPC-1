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
void TCU(int *A, int filas, int columnas, int* B){

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
  int *matriz, *Tmatriz, *h_matriz, *d_matriz, *d_Tmatriz;
  int filas = 1024, columnas = 2048;

  int size = filas*columnas*sizeof(int);

  //----------------------CPU-------------------------
  clock_t t = clock();//Iniciamos la cuenta de reloj
  //Separamos memoria para el host
  matriz = (int*)malloc(size);
  Tmatriz = (int*)malloc(size);

  //Inicializamos la matriz
  inicializa(matriz, filas, columnas);

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
  h_matriz = (int*)malloc(size);//Este va a ser el resultado después de copiar los datos
  //del device al host

  t = clock();//Iniciamos la cuenta de reloj

  //Separamos memoria para el device
  error = cudaMalloc((void**)&d_matriz,size);
  if(error != cudaSuccess){
    cout<<"Error reservando memoria para d_matriz"<<endl;
    //return -1;
  }

	cudaMalloc((void**)&d_Tmatriz,size);
  if(error != cudaSuccess){
      cout<<"Error reservando memoria para d_Tmatriz"<<endl;
      //return -1;
  }

  //Copiamos datos del host al device
  error = cudaMemcpy(d_matriz,matriz,size,cudaMemcpyHostToDevice);//destino d_matriz y origen matriz
  if(error != cudaSuccess){
      printf("Error copiando los datos de matriz a d_matriz \n");
      //exit(-1);
  }

  //Lanzamos el kernel

  dim3 dimblock(32,32,1);
  //dim3 dimGrid(1,1,1);
  dim3 dimGrid(ceil((double)(columnas/32)),ceil((double)(filas/32)),1);

	TCU<<<dimGrid,dimblock>>>(d_matriz, filas, columnas, d_Tmatriz);

	cudaDeviceSynchronize();

  //Copiamos el resultado
	error = cudaMemcpy(h_matriz,d_Tmatriz,size,cudaMemcpyDeviceToHost);
  if(error != cudaSuccess){
      printf("Error copiando los datos de d_Tmatriz a h_matriz \n");
      //exit(-1);
  }

  t = clock() - t;//Terminamos la cuenta de reloj

  double time_GPU = ((double)t) / CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la GPU fue: "<<time_GPU<<endl;
  //------------------------------------------------------------
  cout<<"El tiempo de aceleramiento fue: "<<time_CPU/time_GPU<<endl;

	if(compara(h_matriz, Tmatriz, filas, columnas)) cout << "Buen cálculo" << endl;
	else cout << "Mal cálculo" << endl;

  //Liberamos memoria
  free(matriz); free(Tmatriz);
  free(h_matriz);
  cudaFree(d_matriz); cudaFree(d_Tmatriz);

  return 0;

}
