#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define CHUNKSIZE 10

#define N = 123

void filling_vector(int *vector, int len){
   for (int i = 0; i < len; i++) {
    vector[i] = ((int)rand() / (int)(RAND_MAX)) ;
  }
}

void print(int *V, int len){
  for (int i = 0; i < len; i++) {
    printf("%.2f ", V[i]);
  }
  printf("\n");
}

void add(int *v_a, int *v_b, int *result, int len){
  int i, tid, chunk = CHUNKSIZE, nthreads;
  	#pragma omp parallel shared(v_a, v_b, result, chunk, nthreads) private(tid, i)
  	{
	  	tid = omp_get_thread_num();
	  	if(tid == 0){
	  		nthreads = omp_get_num_threads();
	  		printf("Numero de hilos: %d\n", nthreads);
	  	}
	  	printf("Hilo %d iniciando..\n", tid);
	  	#pragma omp for schedule(dynamic, chunk)
		  	for (i = 0; i < len; i++) {
		    	result[i] = v_a[i] + v_b[i];
		  	}
  	}
}


int main(){
  int len = N;
  clock_t start, end;

  start = clock();
  //Memoria para los vectores
  int *v_a = (int*)malloc(len * sizeof(int));
  int *v_b = (int*)malloc(len * sizeof(int));
  int *result = (int*)malloc(len * sizeof(int));

  //Generando vectores aleatorios
  filling_vector(v_a, len);
  filling_vector(v_b, len);

  //Realizando la suma
  add(v_a, v_b, result, len);


  end = clock();

  //Almacenando en el archivo
  printf("Tiempo de ejecucion usando OMP: %.9f\n", (double)(end - start)/CLOCKS_PER_SEC);

  free(v_a);
  free(v_b);
  free(result);
  return 0;
}
