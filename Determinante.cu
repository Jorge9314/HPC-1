//Determinante de una matriz
#include<iostream>
#include<time.h>

using namespace std;

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
		int k = j, aux = columnas, l = filas -1, mult = 1; 
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
  	int *matriz; 
  	int filas = 3, columnas = 3, det;

  	int size = filas*columnas*sizeof(int);

  	//----------------------CPU-------------------------
  	clock_t t = clock();//Iniciamos la cuenta de reloj
  	
  	//Separamos memoria para el host
  	matriz = (int*)malloc(size);
	
	//Inicializamos los valores de la matriz
	//llena(matriz, filas, columnas);
	inicializa(matriz, filas, columnas);
	
	imprime(matriz, filas, columnas);	
	//Calculamos el determinante
	det = Det1(matriz, filas, columnas) - Det2(matriz, filas, columnas);
	cout << endl << "Determinante = " << det << endl;
	
	return 0;
}
