#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1200


void fill_matrix(int *M, int row, int col){
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      M[i * col + j] = (int)rand() / (int)(RAND_MAX);
    }
  }
}

void print(int *M, int row, int col){
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      printf("%.2f ", M[i * col + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void multiply(int *matrix_a, int *matrix_b, int *result, int col_1, int col_2, int row_1, int row_2){

  int counter;

  if (col_1 != row_2){
    printf("Imposible multiplicar estas matrices\n");
    exit(-1);
  }

  for (int i = 0; i < col_1; ++i){
    for (int j = 0; j < row_2; ++j){
      counter = 0;
      for (int k = 0; k < row_2; ++k){
        counter += matrix_a[i * col_1 + k] * matrix_b[k * col_2 + j];
      }
      result[i * col_2 + j] = counter;
    }
  }
}

int main() {
  int row_1 = N, row_2 = N, col_1 = N, col_2 = N;
  clock_t start, end;

  int *matrix_1 = (int*)malloc(row_1 * col_1 * sizeof(int));
  int *matrix_2 = (int*)malloc(row_2 * col_2 * sizeof(int));
  int *result = (int*)malloc(row_1 * col_2 * sizeof(int));

  fill_matrix(matrix_1, row_1, col_1);
  fill_matrix(matrix_2, row_2, col_2);

  start = clock();
  multiply(matrix_1, matrix_2, result, col_1, col_2, row_1, row_2);

  end = clock();
  printf("Tiempo de ejecucion sin OMP: %.6f\n", (double)(end - start)/CLOCKS_PER_SEC);

  free(matrix_1);
  free(matrix_2);
  free(result);
  return 0;
}
