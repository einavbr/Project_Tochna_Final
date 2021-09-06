#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void init(int K, int DIM, double** datapointsArray, double** centroidsArray, int* whichClusterArray, int* amountOfPointsInCluster);
int findClosestCluster(double* point, double** centroidArray, int K, int sizeOfPoint);
void changeCluster(int i, int newCluster, int* whichClusterArray);
void calcNewCentroids(double** datapointsArray, double** centroidsArray, int* whichClusterArray, int* amountOfPointsInCluster, int N, int sizeOfPoint);
void makeCendroidsAndAmountZero(double** centroidsArray,int* amount, int K, int pointSize);
void free_double_pointer(double **array, int arrayLen);
void printMatrix(int rows, int cols, double** matrix);

void init(int K, int DIM, double** datapointsArray, double** centroidsArray, int* whichClusterArray, int* amountOfPointsInCluster) {
    
    int i, j;

    for (i = 0; i < K; i++) {
        for(j=0; j< DIM; j++){
            centroidsArray[i][j] = datapointsArray[i][j];
        }
        whichClusterArray[i] = i;
        amountOfPointsInCluster[i] = 1;
    }
    printf("initial centroids:\n");
    printMatrix(K,DIM,centroidsArray);
}

int findClosestCluster(double* point, double** centroidArray, int K, int sizeOfPoint){
    double mindist, sum;
    int newCluster,i,j;
    double* centroidToCheck;

    mindist = -1;
    newCluster = -1;
    for (i = 0; i < K; i++) {
        sum = 0;
        centroidToCheck = centroidArray[i];
        for (j = 0; j < sizeOfPoint; j++) {
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

void makeCendroidsAndAmountZero(double** centroidsArray, int* amount, int K, int sizeOfPoint) {
    int i, j;
    for (i = 0; i < K; i++) {
        amount[i] = 0;
        for (j = 0; j < sizeOfPoint; j++) {
            centroidsArray[i][j] = 0.0;
        }
    }
}

void calcNewCentroids(double** datapointsArray, double** centroidsArray, int* whichClusterArray,
    int* amountOfPointsInCluster, int N, int sizeOfPoint) {
    int i, j, newCluster;
    double prevSum, newVal;
    for (i = 0; i < N; i++) {
        newCluster = whichClusterArray[i];
        for (j = 0; j < sizeOfPoint; j++) {
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

int kmeans(int K,int N, int DIM,double** T, double** centroids_mat,int* whichClusterArray) {
    int i,itermax, iteration, isChanged, currentCluster, newCluster, * amountOfPointsInCluster;
    double * point;

    printf("in kmeans!!\n");
    itermax = 300;

    amountOfPointsInCluster = (int*)calloc(K, sizeof(int));
    assert(amountOfPointsInCluster && "amountOfPointsArray allocation failed");
    
    init(K, DIM, T, centroids_mat, whichClusterArray, amountOfPointsInCluster);
    printf("finished kmeans init\n");
    isChanged = 1;
    iteration = 0;
    while (isChanged == 1) {
        if (iteration == itermax) {
            printf("main: max iteration reached\n");
            break;
        }
        iteration = iteration + 1;

        isChanged = 0;
        for (i = 0; i < N; i++) {
            point = T[i];
            currentCluster = whichClusterArray[i];
            newCluster = findClosestCluster(point, centroids_mat, K, DIM);  /* find new cluster by minimal norm */
            if (newCluster == -1){
                newCluster = currentCluster;
            }
            if (currentCluster != newCluster) {
                changeCluster(i, newCluster, whichClusterArray);
                isChanged = 1;
            }
        }
        makeCendroidsAndAmountZero(centroids_mat, amountOfPointsInCluster, K, DIM);
        calcNewCentroids(T, centroids_mat, whichClusterArray, amountOfPointsInCluster, N, DIM); /* calc new centroid of new cluster for point[j] */
    }
    printf("finished kmeans!!\n");
    return 1; 
}
