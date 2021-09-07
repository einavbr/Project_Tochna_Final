# define PY_SSIZE_T_CLEAN

int POINT_SIZE, N, K, MAX_ITER;

void printMatrixpp(int rows, int cols, double** matrix);
void printMatrixintpp(int rows, int cols, int* matrix);
int findClosestClusterpp(double* point, double** centroidArray);
void changeClusterpp(int i, int newCluster, int* whichClusterArray);
void makeCendroidsAndAmountZeropp(double** centroidsArray, int* amount);
void calcNewCentroidspp(double** datapointsArray, double** centroidsArray, int* whichClusterArray, int* amountOfPointsInCluster);
void free_double_pointerpp(double **array, int arrayLen);
double** kmeanspp(double** points, double** centroids, int K,int N, int POINT_SIZE, int MAX_ITER);

void printMatrixpp(int rows, int cols, double** matrix) {
    int i, j; 
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (j == cols - 1) {
                printf("%.4f", matrix[i][j]);
            }
            else {
                printf("%.4f,", matrix[i][j]);
            }
        }
        printf("\n");
    }
    printf("\n\n");
}

void printMatrixintpp(int rows, int cols, int* matrix) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%d ,", matrix[i]);
        }
        printf("\n");
    }
}

int findClosestClusterpp(double* point, double** centroidArray){
    double mindist, sum;
    int newCluster, i, j;
    double* centroidToCheck;

    mindist = -1;
    newCluster = -1;
    for (i = 0; i < K; i++) {
        sum = 0;
        centroidToCheck = centroidArray[i];
        for (j = 0; j < POINT_SIZE; j++) {
            sum = sum + ((point[j] - centroidToCheck[j]) * (point[j] - centroidToCheck[j]));
        }
        if (mindist == -1 || sum < mindist) {
            mindist = sum;
            newCluster = i;
        }
    }
    return newCluster;
}

void changeClusterpp(int i, int newCluster, int* whichClusterArray) {
    whichClusterArray[i] = newCluster;
}

void makeCendroidsAndAmountZeropp(double** centroidsArray, int* amount) {
    int i, j;
    for (i = 0; i < K; i++) {
        amount[i] = 0;
        for (j = 0; j < POINT_SIZE; j++) {
            centroidsArray[i][j] = 0.0;
        }
    }
}

void calcNewCentroidspp(double** datapointsArray, double** centroidsArray, int* whichClusterArray,
    int* amountOfPointsInCluster) {
    int i, j, newCluster;
    double prevSum, newVal;
    for (i = 0; i < N; i++) {
        newCluster = whichClusterArray[i];
        for (j = 0; j < POINT_SIZE; j++) {
            prevSum = centroidsArray[newCluster][j] * amountOfPointsInCluster[newCluster];
            newVal = (prevSum + datapointsArray[i][j]) / (amountOfPointsInCluster[newCluster] + 1);
            centroidsArray[newCluster][j] = newVal;
        }
        amountOfPointsInCluster[newCluster] = amountOfPointsInCluster[newCluster] + 1;
    }
}

void free_double_pointerpp(double **array, int arrayLen){
    int i;
    for (i=0; i < arrayLen; i++){
        free(array[i]);
    }
    free(array);
}

double** kmeanspp(double** points, double** centroids, int K,int N, int POINT_SIZE, int MAX_ITER) {
    int i, isChanged, iteration, currentCluster, newCluster;
    int *whichClusterArray, *amountOfPointsInCluster;
    double *point;

    K=K;
    N=N;
    POINT_SIZE = POINT_SIZE;
    MAX_ITER = MAX_ITER;
    whichClusterArray = (int*)malloc(N * sizeof(int));
    assert(whichClusterArray && "whichClusterArray allocation failed");
    amountOfPointsInCluster = (int*)calloc(K, sizeof(int));
    assert(amountOfPointsInCluster && "amountOfPointsArray allocation failed");

    /*init whichClusterArray*/
    for (i = 0; i < N; i++) {
        whichClusterArray[i] = -1;
    }

    isChanged = 1;
    iteration = 0;
    while (isChanged == 1) {
        if (iteration == MAX_ITER) {
            printf("main: max iteration reached\n");
            break;
        }
        iteration = iteration + 1;

        isChanged = 0;
        for (i = 0; i < N; i++) {
            /* handeling point i */
            point = points[i];
            currentCluster = whichClusterArray[i];
            newCluster = findClosestClusterpp(point, centroids);  /* find new cluster by minimal norm */
            if (newCluster == -1){
                printf("error in find closest cluster\n");
                return NULL;
            }
            if (currentCluster != newCluster) {
                changeClusterpp(i, newCluster, whichClusterArray);
                isChanged = 1;
            }
        }
        makeCendroidsAndAmountZeropp(centroids, amountOfPointsInCluster);
        calcNewCentroidspp(points, centroids, whichClusterArray, amountOfPointsInCluster); /* calc new centroid of new cluster for point[j] */
    }

    free(whichClusterArray);
    free(amountOfPointsInCluster);

    return centroids;
}



