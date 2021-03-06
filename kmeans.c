#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void init(int K, double** datapointsArray, double** centroidsArray, int* whichClusterArray, int* amountOfPointsInCluster);
int findClosestCluster(double* point, double** centroidArray, int K);
void changeCluster(int i, int newCluster, int* whichClusterArray);
void calcNewCentroids(double** datapointsArray, double** centroidsArray, int* whichClusterArray, int* amountOfPointsInCluster, int N, int K);
void makeCendroidsAndAmountZero(double** centroidsArray,int* amount, int K);
void free_double_pointer(double **array, int arrayLen);

void init(int K, double** datapointsArray, double** centroidsArray, int* whichClusterArray, int* amountOfPointsInCluster) {
    int i, j;

    for (i = 0; i < K; i++) {
        for(j=0; j < K; j++){
            centroidsArray[i][j] = datapointsArray[i][j];
        }
        whichClusterArray[i] = i;
        amountOfPointsInCluster[i] = 1;
    }
}

int findClosestCluster(double* point, double** centroidArray, int K){
    double mindist, sum;
    int newCluster,i,j;
    double* centroidToCheck;

    mindist = -1;
    newCluster = -1;
    for (i = 0; i < K; i++) {
        sum = 0;
        centroidToCheck = centroidArray[i];
        for (j = 0; j < K; j++) {
            sum = sum + ((point[j] - centroidToCheck[j]) * (point[j] - centroidToCheck[j]));
        }
        if (mindist == -1 || sum < mindist) {
            mindist = sum;
            newCluster = i;
        }
    }
    return newCluster;
}

void changeCluster(int i, int newCluster, int* whichClusterArray) {
    whichClusterArray[i] = newCluster;
}

void makeCendroidsAndAmountZero(double** centroidsArray, int* amount, int K) {
    int i, j;
    for (i = 0; i < K; i++) {
        amount[i] = 0;
        for (j = 0; j < K; j++) {
            centroidsArray[i][j] = 0.0;
        }
    }
}

void calcNewCentroids(double** datapointsArray, double** centroidsArray, int* whichClusterArray,
    int* amountOfPointsInCluster, int N, int K) {
    int i, j, newCluster;
    double prevSum, newVal;
    for (i = 0; i < N; i++) {
        newCluster = whichClusterArray[i];
        for (j = 0; j < K; j++) {
            prevSum = centroidsArray[newCluster][j] * amountOfPointsInCluster[newCluster];
            newVal = (prevSum + datapointsArray[i][j]) / (amountOfPointsInCluster[newCluster] + 1);
            centroidsArray[newCluster][j] = newVal;
        }
        amountOfPointsInCluster[newCluster] = amountOfPointsInCluster[newCluster] + 1;
    }
}

void free_double_pointer(double **array, int arrayLen){
    int i;
    for (i=0; i < arrayLen; i++){
        free(array[i]);
    }
    free(array);
}

int kmeans(int K,int N,double** T, double** centroids_mat,int* whichClusterArray) {
    int i, iteration, isChanged, currentCluster, newCluster;
    int* amountOfPointsInCluster;
    double * point;

    amountOfPointsInCluster = (int*)calloc(K, sizeof(int));
    assert(amountOfPointsInCluster && "amountOfPointsArray allocation failed");
    init(K, T, centroids_mat, whichClusterArray, amountOfPointsInCluster);
    isChanged = 1;
    iteration = 0;
    while (isChanged == 1) {
        if (iteration == MAX_ITER_KMEANS) {
            printf("main: max iteration reached\n");
            break;
        }
        iteration = iteration + 1;

        isChanged = 0;
        for (i = 0; i < N; i++) {
            point = T[i];
            currentCluster = whichClusterArray[i];
            newCluster = findClosestCluster(point, centroids_mat, K);  /* find new cluster by minimal norm */
            if (newCluster == -1){
                newCluster = currentCluster;
            }
            if (currentCluster != newCluster) {
                changeCluster(i, newCluster, whichClusterArray);
                isChanged = 1;
            }
        }
        makeCendroidsAndAmountZero(centroids_mat, amountOfPointsInCluster, K);
        calcNewCentroids(T, centroids_mat, whichClusterArray, amountOfPointsInCluster, N, K); /* calc new centroid of new cluster for point[j] */
    }
    return 1; 
}
