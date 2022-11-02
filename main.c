#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void imprimi(double *A, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%.1f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void transpose(double *A, double *B, int n) {
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j*n+i] = A[i*n+j];
        }
    }
}

void multiMat(double *A, double *B, double *C, int n)
{
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double dot  = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B[k*n+j];
            }
            C[i*n+j ] = dot;
        }
    }
}

void multiMat_omp(double *A, double *B, double *C, int n, int threadss)
{
    #pragma omp parallel num_threads(threadss)
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                double dot  = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B[k*n+j];
                }
                C[i*n+j ] = dot;
            }
        }

    }
}

void multiMatTrans(double *A, double *B, double *C, int n)
{
    int i, j, k;
    double *B2;
    B2 = (double*)malloc(sizeof(double)*n*n);
    transpose(B,B2, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double dot  = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B2[j*n+k];
            }
            C[i*n+j ] = dot;
        }
    }
    free(B2);
}

void multiMatTrans_omp(double *A, double *B, double *C, int n, int threadss)
{
    double *B2;
    B2 = (double*)malloc(sizeof(double)*n*n);
    transpose(B,B2, n);
    #pragma omp parallel num_threads(threadss)
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                double dot  = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B2[j*n+k];
                }
                C[i*n+j ] = dot;
            }
        }

    }
    free(B2);
}

int main() {
    int i, n;
    double *A, *B, *C, dtime, speedup;

    n=1500;
    A = (double*)malloc(sizeof(double)*n*n);
    B = (double*)malloc(sizeof(double)*n*n);
    C = (double*)malloc(sizeof(double)*n*n);
    for(i=0; i<n*n; i++) {
            A[i] = 1.0*rand()/RAND_MAX;
            B[i] = 1.0*rand()/RAND_MAX;
    }

    dtime = omp_get_wtime();            // Separa uma thread para pegar o tempo naquele momento
    multiMat(A,B,C, n);
    dtime = omp_get_wtime() - dtime;    // E diminui com o tempo depois de ter terminado a tarefa
    speedup = dtime;
    printf("Multiplicacao sem OpenMp e sem transposicao: %f segundos. Speedup = 1\n", dtime);

    dtime = omp_get_wtime();
    multiMatTrans(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("Multiplicacao sem OpenMp e com transposicao: %f segundos. Speedup = %f\n", dtime, speedup/dtime);

    dtime = omp_get_wtime();
    multiMat_omp(A,B,C, n, 2);
    dtime = omp_get_wtime() - dtime;
    printf("Multiplicacao com OpenMp (2 threads) e sem transposicao: %f segundos. Speedup = %f\n", dtime, speedup/dtime);

    dtime = omp_get_wtime();
    multiMatTrans_omp(A,B,C, n, 2);
    dtime = omp_get_wtime() - dtime;
    printf("Multiplicacao com OpenMp (2 threads) e com transposicao: %f segundos. Speedup = %f\n", dtime, speedup/dtime);

    dtime = omp_get_wtime();
    multiMat_omp(A,B,C, n, 4);
    dtime = omp_get_wtime() - dtime;
    printf("Multiplicacao com OpenMp (4 threads) e sem transposicao: %f segundos. Speedup = %f\n", dtime, speedup/dtime);

    dtime = omp_get_wtime();
    multiMatTrans_omp(A,B,C, n, 4);
    dtime = omp_get_wtime() - dtime;
    printf("Multiplicacao com OpenMp (4 threads) e com transposicao: %f segundos. Speedup = %f\n", dtime, speedup/dtime);

    dtime = omp_get_wtime();
    multiMat_omp(A,B,C, n, 8);
    dtime = omp_get_wtime() - dtime;
    printf("Multiplicacao com OpenMp (8 threads) e sem transposicao: %f segundos. Speedup = %f\n", dtime, speedup/dtime);

    dtime = omp_get_wtime();
    multiMatTrans_omp(A,B,C, n, 8);
    dtime = omp_get_wtime() - dtime;
    printf("Multiplicacao com OpenMp (8 threads) e com transposicao: %f segundos. Speedup = %f\n", dtime, speedup/dtime);

    system("pause");
    return 0;

}
