#include<iostream>
#include<stdio.h>
#include<malloc.h>
#include<cuda.h>
using namespace std;

#define TILE_WIDTH 32

__global__
void MatrixMulKernel(int* Md,int fil_Md,int col_Md,int* Nd,int fil_Nd,int col_Nd,int* Pd)
{
	//Memoria compartida para un subconjunto de Mds y Nds
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	//Para saber en qué bloque y qué hilo estamos
	int bx = blockIdx.x; 	int by = blockIdx.y;
  int tx = threadIdx.x;	int ty = threadIdx.y;
	int gx = gridDim.x;	int gy = gridDim.y;

	//Calcula el indice de la fila del elemento Pd y A
	int row = by * TILE_WIDTH + ty;
	//Calcula el indice de la columna del elemento Pd y B
	int col = bx * TILE_WIDTH + tx;

	//para llevar la Pvalue de las multiplicaciones
	int Pvalue = 0;

	int n = 0, m = 0;
	while(m < gx && n < gy)
		{
			if(( ( m * TILE_WIDTH ) + tx ) < col_Md && row < fil_	Md) //Si no se pasa
				// Trae un elemento, cada uno de Mds en la memoria compartida.
				Mds[ty][tx] = Md[ (row * col_Md) + ( ( m * TILE_WIDTH ) + tx )];
			else Mds[ty][tx] = 0;

			if(( n * TILE_WIDTH + ty) < fil_Nd && col < col_Nd)
				Nds[ty][tx] = Nd[( ( n * TILE_WIDTH + ty) * col_Nd ) + col ];//(k*colB)+Col, donde k-> 0..filB
			else Nds[ty][tx] = 0;

			m++; n++;

			//espera a todos los hilos del bloque y espera que el tile este en memoria compartida
			__syncthreads();

    // cada hilo calcula un elemento de la submatriz del bloque
			for (int k=0; k < TILE_WIDTH ; ++k) {
				Pvalue += Mds[ty][k] * Nds[k][tx]; //Acumula el subconjunto del producto punto
			}
			__syncthreads();
		}
	if(row < fil_Md && col < col_Nd)
		//Pd[fil_Md][col_Nd]
		Pd[ (row * col_Nd) + col] = Pvalue;
}

__host__
void multiplicaMatrices(int* A, int fil_A, int col_A, int* B, int fil_B, int col_B, int* C)
{
	for(int i=0;i<fil_A;i++)
	{
		for(int j=0;j<col_B;j++)
		{
			int Pvalue=0;
			for(int k=0;k<fil_B;k++)
			{
				Pvalue=Pvalue+A[(i*col_A)+k]*Y[(k*col_B)+j];
			}
			//Escribe al final el resultado en la memoria globall
			C[(i*col_B)+j]=Pvalue;
		}
	}
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
		A[i]=1;
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

	clock_t inicio_gpu, fin_gpu, inicio_cpu, fin_cpu;
  	cudaError_t error = cudaSuccess;
	int *A,*B,*C; //A[fil_Md][col_Md],B[fil_Nd][col_Nd],C[fil_Md][col_Nd]
	int *d_A,*d_B,*d_C,*h_C;
	//int filA=2048,colA=2048,filB=2048,colB=2048;
	int fil_A=1,col_A=1024,fil_B=1024,col_B=1;
	//-------------------------------CPU--------------------------------------------------------------------
	A=(int*)malloc(fil_A*col_A*sizeof(int));
	B=(int*)malloc(fil_B*col_B*sizeof(int));
	C=(int*)malloc(fil_A*col_B*sizeof(int));

	inicializa(A,fil_A,col_A);
	inicializa(B,fil_B,col_B);

	if(col_A==fil_B){//para que sean multiplicables
		inicio_cpu = clock();
		multiplicaMatrices(A,fil_A,col_A,B,fil_B,col_B,C);
		fin_cpu = clock();
		//imprime(C,filA,colB);
	}else{
		cout<<"Error, no se pueden multiplicar"<<endl;
		return 0;
	}

	double time_CPU=((double)(fin_cpu-inicio_cpu))/CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la CPU fue: "<<time_CPU<<endl;
	//-------------------------------GPU--------------------------------------------------------------------
	h_C=(int*)malloc(fil_A*col_B*sizeof(int));

	inicio_gpu = clock();

	error=cudaMalloc((void**)&d_A,fil_A*col_A*sizeof(int));
        if(error != cudaSuccess){
            cout<<"Error reservando memoria para d_A"<<endl;
            //return -1;
        }

	cudaMalloc((void**)&d_B,fil_B*col_B*sizeof(int));
        if(error != cudaSuccess){
            cout<<"Error reservando memoria para d_B"<<endl;
            //return -1;
        }

	cudaMalloc((void**)&d_C,fil_A*col_B*sizeof(int));
        if(error != cudaSuccess){
            cout<<"Error reservando memoria para d_C"<<endl;
            //return -1;
        }

	cudaMemcpy(d_A,A,fil_A*col_A*sizeof(int),cudaMemcpyHostToDevice);//destino d_A y origen A
	cudaMemcpy(d_B,B,fil_B*col_B*sizeof(int),cudaMemcpyHostToDevice);

	//Depende directamente de la dimensión de las matrices
	dim3 dimblock(32,32,1);
	dim3 dimGrid(32,32,1);
  	//dim3 dimGrid(ceil((double)(colB/32)),ceil((double)(filA/32)),1);

	MatrixMulKernel<<<dimGrid,dimblock>>>(d_A,fil_A,col_A,d_B,fil_B,col_B,d_C);

	cudaDeviceSynchronize();

	cudaMemcpy(h_C,d_C,fil_A*col_B*sizeof(int),cudaMemcpyDeviceToHost);

	fin_gpu = clock();

	//imprime(h_C,filA,colB);
	double time_GPU=((double)(fin_gpu-inicio_gpu))/CLOCKS_PER_SEC;
	cout<<"El tiempo transcurrido en la GPU fue: "<<time_GPU<<endl;
	//-----------------------------------------------------------------------------------
	cout<<"El tiempo de aceleramiento fue: "<<time_CPU/time_GPU<<endl;

	if(compara(h_C, C, fil_A, col_B)) cout << "Buen cálculo" << endl;
	else cout << "Mal cálculo" << endl;

	free(A);free(B);free(C);free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
