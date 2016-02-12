//Determinante de una matriz
#include<iostream>
#include<time.h>

using namespace std;

__global__
void Det1_CU(int *M, int filas, int columnas, int &suma){

	//int i = blockIdx.y*blockDim.y+threadIdx.y;//filas
	int j = blockIdx.x*blockDim.x+threadIdx.x;//columnas

	if(j < columnas){
		int k = j, aux = columnas, l = 0, mult = 1;
		while(aux--){
			if(k == columnas) k = 0;
			mult *= M[(l*columnas)+k];
			__syncthreads();//espera a que todos los hilos hagan la misma multiplicacion
			k++; l++;
		}
		suma += mult;
	}
}

__global__
void Det2_CU(int *M, int filas, int columnas, int &suma){

	//int i = blockIdx.y*blockDim.y+threadIdx.y;//filas
	int j = blockIdx.x*blockDim.x+threadIdx.x;//columnas

	if(j < columnas){

		int k = j, aux = columnas, l = filas - 1, mult = 1;
		while(aux--){
			if(k == columnas) k = 0;
			mult *= M[(l*columnas)+k];
			__syncthreads();//espera a que todos los hilos hagan la misma multiplicacion
			k++; l--;
		}
		suma += mult;
	}
}

__host__
int Det1(int *M, int filas, int columnas){
	int suma = 0;
	for(int j = 0; j < columnas; j++){
		int k = j, aux = columnas, l = 0, mult = 1;
		while(aux--){
			if(k == columnas) k = 0;
			mult *= M[(l*columnas)+k];
			k++; l++;
		}
		suma += mult;
	}
	return suma;
}

__host__
int Det2(int *M, int filas, int columnas){
	int suma = 0;
	for(int j = 0; j < columnas; j++){
		int k = j, aux = columnas, l = filas - 1, mult = 1;
		while(aux--){
			if(k == columnas) k = 0;
			mult *= M[(l*columnas)+k];
			k++; l--;
		}
		suma += mult;
	}
	return suma;
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
	/* initialize random seed: */
	srand (time(NULL));
	int valor;
	for(int i=0;i<filas*columnas;i++){
		/* generate secret number between 1 and 10: */
		valor = rand() % 10 + 1;
		A[i] = valor;
	}
}

__host__
void llena(int *A, int filas, int columnas){
	int valor;
	for(int i = 0; i < filas; i++){
			for(int j = 0; j < columnas; j++){
				cin >> valor;
				A[(i*columnas)+j] = valor;
			}
	}
}

int main(void){

	cudaError_t error = cudaSuccess;//Para controlar errores
	int *d_matriz, *matriz;
	int filas = 3, columnas = 3, h_det, d_det;

	int size = filas*columnas*sizeof(int);

	//----------------------CPU-------------------------

	//Separamos memoria para el host
	matriz = (int*)malloc(size);

	//Inicializamos los valores de la matriz
	//llena(matriz, filas, columnas);
	inicializa(matriz, filas, columnas);

	clock_t t = clock();//Iniciamos la cuenta de reloj

	//imprime(matriz, filas, columnas);
	//Calculamos el determinante
	h_det = Det1(matriz, filas, columnas) - Det2(matriz, filas, columnas);
	//cout << endl << "Determinante = " << h_det << endl;

	t = clock() - t;//Terminamos la cuenta de reloj

	double time_CPU = ((double)t) / CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la CPU fue: "<<time_CPU<<endl;

	//---------------------GPU------------------------------
	t = clock();//Iniciamos la cuenta de reloj

  //Separamos memoria para el device
  error = cudaMalloc((void**)&d_matriz,size);
  if(error != cudaSuccess){
    cout << "Error reservando memoria para d_matriz" << endl;
    //return -1;
  }

	//Copiamos datos del host al device
  error = cudaMemcpy(d_matriz,matriz,size,cudaMemcpyHostToDevice);//destino d_matriz y origen matriz
  if(error != cudaSuccess){
      cout << "Error copiando los datos de matriz a d_matriz" << endl;
      //exit(-1);
  }

	//Lanzamos el kernel

  dim3 dimblock(3,1,1);//solo necesitamos los hilos de las columnas
  dim3 dimGrid(1,1,1);
  //dim3 dimGrid(ceil((double)(columnas/32)),ceil((double)(filas/32)),1);

	int ans1 = 0, ans2 = 0;
	//Det1_CU<<<dimGrid,dimblock>>>(d_matriz, filas, columnas, ans1);
	//Det2_CU<<<dimGrid,dimblock>>>(d_matriz, filas, columnas, ans2);

	cudaDeviceSynchronize();

	d_det = ans1 - ans2;

  t = clock() - t;//Terminamos la cuenta de reloj

  double time_GPU = ((double)t) / CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la GPU fue: "<<time_GPU<<endl;
  //------------------------------------------------------------
  cout<<"El tiempo de aceleramiento fue: "<<time_CPU/time_GPU<<endl;

	if(h_det == d_det) cout << "Buen cálculo" << endl;
	else cout << "Mal cálculo" << endl;

	//Liberamos memoria
	free(matriz);
	cudaFree(d_matriz);

	return 0;
}
