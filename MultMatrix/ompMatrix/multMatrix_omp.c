#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define CHUNKSIZE 10
#define N 1200


void filling_matrix(int *matrix, int row, int col){
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      matrix[i * col + j] = (int)rand() / (int)(RAND_MAX);
    }
  }
}

void print(int *matrix, int row, int col){
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%.2f ", matrix[i * col + j]);
    }
    printf("\n");
  }
  printf("\n");
}


void multiply(int *matrix_a, int *matrix_b, int *result, int col_1, int col_2, int row_1, int row_2){

  int counter, i, j , k, chunk = CHUNKSIZE, nthreads, tid;

  if (col_1 != row_2){
    printf("Imposible multiplicar estas matrices\n");
    exit(-1);
  }

  #pragma omp parallel shared(matrix_a, matrix_b, result, chunk, nthreads) private(i, j , k, tid)
  {
    tid = omp_get_thread_num();
    if(tid == 0){
      nthreads = omp_get_num_threads();
      printf("Numero de hilos: %d\n", nthreads);
    }
    printf("Hilo %d iniciando..\n", tid);
    #pragma omp for schedule(dynamic, chunk)
      for (i = 0; i < col_1; ++i){
        for (j = 0; j < row_2; ++j){
          counter = 0;
          for (k = 0; k < row_2; ++k){
            counter += matrix_a[i * col_1 + k] * matrix_b[k * col_2 + j];
          }
          result[i * col_2 + j] = counter;
          //printf("Hilo %d: c[%d] = %d\n", tid, i * col_2 + j, counter);
        }
      }
  }
}


int main() {
  int row_1 = N, row_2 = N, col_1 = N, col_2 = N;
  clock_t start, end;


  int *matrix_1 = (int*)malloc(row_1 * col_1 * sizeof(int));
  int *matrix_2 = (int*)malloc(row_2 * col_2 * sizeof(int));
  int *result = (int*)malloc(row_1 * col_2 * sizeof(int));

  filling_matrix(matrix_1, row_1, col_1);
  filling_matrix(matrix_2, row_2, col_2);

  start = clock();
  multiply(matrix_1, matrix_2, result, col_1, col_2, row_1, row_2);

  end = clock();
  printf("Tiempo de ejecucion usando OMP: %.6f\n", (double)(end - start)/CLOCKS_PER_SEC);

  free(matrix_1);
  free(matrix_2);
  free(result);
  return 0;
}
