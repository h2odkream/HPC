#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N = 123
void filling_vector(int *vector, int len){
   for (int i = 0; i < len; i++) {
    vector[i] = ((int)rand() / (int)(RAND_MAX)) ;
  }
}

void add(int *v_1, int *v_2, int *result, int len){
  for (int i = 0; i < len; i++) {
    result[i] = v_1[i] + v_2[i];
  }
}

void print(int *V, int len){
  for (int i = 0; i < len; i++) {
    printf("%.2f ", V[i]);
  }
  printf("\n");
}


int main(){
  int len = N;
  clock_t start, end;

  start = clock();
  //Memoria para los vectores
  int *vector_a = (int*)malloc(len * sizeof(int));
  int *vector_b = (int*)malloc(len * sizeof(int));
  int *result = (int*)malloc(len * sizeof(int));

  //Generando vectores aleatorios
  filling_vector(vector_a, len);
  filling_vector(vector_b, len);

  //Realizando la suma
  add(vector_a, vector_b, result, len);
  end = clock();

  //Almacenando en el archivo

  printf("Tiempo de ejecucuion sin OMP: %.9f\n", (double)(end - start)/CLOCKS_PER_SEC);

  free(vector_a);
  free(vector_b);
  free(result);
  return 0;
}
