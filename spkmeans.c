#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include "spkmeans.h"
#include "kmeans.c"
#include "kmeanspp.c"

int N, K, DIM;

double calcEuclideanNorm(double* vector1, double* vector2);
void printMatrix(int rows, int cols, double** matrix);
void printArray(int len, double* matrix);
int printTest(int num);

/** ---------------------------------- GRAPH FUNCTIONS ---------------------------------------- **/

void fillWeightedMat(double** vertices, double** weighted_mat,int N){
    /* given a list of vectors, this fuction should return a matrix of weights 
     * for each pair of vectors
     * NOTE : remember this is a symetric matrix, we only need to compute for i<j
     * and put the result in 2 slots
     */
    int i,j;
    double euclidianNorm, power;

    for(i=0 ; i<N ; i++){
        /* weighted_mat[i][i] = 0.0; */
        for(j=i+1; j<N ; j++){
            euclidianNorm = calcEuclideanNorm(vertices[i],vertices[j]);
            power = -euclidianNorm/2;
            weighted_mat[i][j] = weighted_mat[j][i] = exp(power);
            /* printf("weighted matrix in %d,%d is: %f\n",i,j,exp(power));
            printf("weighted_mat[i][j] is: %f",weighted_mat[i][j]);
            printf("\n"); */
        }
    }
}
    
void fillDiagonalDegreeArray(double** weighted_mat, double* diagonal_degree_array, int N){
     /* 
     * given a list of vectors, this fuction should return the Diagonal Degree Matrix ^ -0.5 
     */
    int i,j;
    for(i=0; i<N; i++){
        for(j=0;j<N;j++){
            /* printf("i,j: %d,%d\n", i,j); 
            printf("w_ij: %f", weighted_mat[i][j]); */
            diagonal_degree_array[i] = diagonal_degree_array[i] + weighted_mat[i][j];
        }
        diagonal_degree_array[i] = pow(diagonal_degree_array[i],-0.5);
    }
}
  
void constructGraph(FILE* file, double** vertices, double** weighted_mat, double* diagonal_degree_array, Graph* graph){
    char* point, *line;
    int i, j;

    fseek(file, 0, SEEK_SET);
    i = 0;
    line = (char*) malloc(1000 * sizeof(char));
    assert(line && ERROR_OCCURED);
    while (fgets(line, 1000, file) != NULL) {
        j = 0;
        point = strtok(line, ",");
        while (point != NULL) {
            vertices[i][j] = atof(point);
            point = strtok(NULL, ",");
            j = j + 1;
        }
        i = i + 1;
    }
    DIM = j;
    N = i;
    graph->size = i;
    graph->dim = j;

    printf("filling weightedMat\n");
    fillWeightedMat(vertices, weighted_mat, N);
    printf("finished filling weightedMat\n");
    printf("filling DiagonalDegreeArray\n");
    fillDiagonalDegreeArray(weighted_mat, diagonal_degree_array, N);
    printf("finished filling DiagonalDegreeArray\n");
    

    graph->vertices = vertices;
    graph->weighted_mat = weighted_mat;
    graph->diagonal_degree_array = diagonal_degree_array;
    printMatrix(N, N, graph->weighted_mat);
    printArray(N, graph->diagonal_degree_array);
    free(line);
}

/** ---------------------------------- PRINTS ---------------------------------------- **/

void printEigens(Eigen** eigens, int n){
    int i;
    
    for (i=0; i < n; i++) {
        printf("eigen value %d is:",i);
        printf("%f, ", eigens[i]->eigenvalue);
        printf("\n");
    }
    for (i=0; i < n; i++) {
        printf("eigen vector %d is:\n",i);
        printArray(N, eigens[i]->eigenvector);
        printf("\n");
    }
}

void printMatrix(int rows, int cols, double** matrix) {
    /** TODO: fix -0.0000 rounding**/
    int i, j; 
    printf("in printMatrix\n");
    if (!matrix){
        printf("matrix is NULL\n");
    }
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (j == cols - 1) {
                printf("%f", matrix[i][j]); 
            }
            else {
                printf("%f,", matrix[i][j]); 
            }
        }
        printf("\n");
    }

    printf("\n\n");
}

void printArray(int len, double* matrix) {
    int i; 
    for (i = 0; i < len; i++) {
         if ( i == len - 1) {
                printf("%.4f", matrix[i]);
        }
        else{
        printf("%.4f,", matrix[i]);
        }
    }
    printf("\n\n");
}

/** ---------------------------------- ALLOCATIONS ---------------------------------------- **/

/* void allocateMatrix(int row, int col, double **matrix){
    int i;
    printf("in allocateMatrix");
    i = 0;
    matrix = (double **)malloc(sizeof(double *) * row);
    for(i = 0; i < col; i++){
        matrix[i] = (double *)malloc(sizeof(double) * col);
    }
} */

double** allocateMatrix(int rows, int cols)
{
    double ** m;
    int r;

    printf("in allocateMatrix\n");
    /* Allocate array of row pointers */
    m = (double**)calloc(rows, sizeof(double*));
    if (!m) return NULL;
    /* printf("finished allocating array of row pointers\n"); */

    /* Allocate block for data */
    m[0] = (double*)calloc(rows * cols, sizeof(double));
    if (!m[0]) {
        free(m);
        return NULL;
    }
    /* printf("finished allocating block for data\n"); */

    /* Assign row pointers */
    for(r = 1; r < rows; r++) {
        m[r] = m[r-1]+cols;
    }
    /* printf("finished Assigning row pointers\n"); */

    return m; 
}

Eigen* allocateEigen(){
    Eigen* eigen;

    eigen = (Eigen*)malloc(sizeof(Eigen));
    assert(eigen && ERROR_OCCURED);
    eigen->eigenvector = (double*)malloc(N * sizeof(double));
    assert(eigen ->eigenvector && ERROR_OCCURED);
    return eigen;
}

/** -------------------------------- FREE -------------------------------------------------- **/

void freeEigensArray (Eigen** freeEigensArray, int N) {
    int i;
    for (i=0 ; i<N ; i++) {
        free(freeEigensArray[i]-> eigenvector);
    }
    free(freeEigensArray);
}

void freeMatrix(double **m){
    if (m) free(m[0]);
    free(m);
}

void freeGraph(Graph* graph){
    free(graph->diagonal_degree_array);
    freeMatrix(graph->weighted_mat);
    freeMatrix(graph->vertices);
    free(graph);
}
/** ---------------------------------- CALCULATIONS ---------------------------------------- **/


int howManyLines(FILE* file) {
    int counterOfLines; /* Line counter (result) */
    char c;  /* To store a character read from file */
    fseek(file, 0, SEEK_SET);
    counterOfLines = 1;  
    /* Extract characters from file and store in character c */
    for (c = getc(file); c != EOF; c = getc(file)){
        if (c == '\n') { /* Increment count if this character is newline */
            counterOfLines = counterOfLines + 1;
        }      
    }
    return counterOfLines;
}

int pointSize(FILE* file) {
    int numOfCoords;
    char c;

    numOfCoords = 1;
    fseek(file, 0, SEEK_SET);
    for (c = getc(file); c != '\n'; c = getc(file))
        if (c == ',') {
            numOfCoords = numOfCoords + 1;
        }
    return numOfCoords;
}

void memcpy_matrix(double** src, double** dest, int rows, int cols) {
    int i, j;
    for (i=0 ; i<rows; i++){
        for (j=0; j<cols; j++){
            dest[i][j] = src[i][j];
        }
    }
}

void transposeSquareMatrix(double** matrix, int N) {
    int i, j;
    double temp;

    for(i=0; i < N; i++) {
        for(j=0; j < i; j++) {
            temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}

double calcEuclideanNorm(double* vector1, double* vector2){
    /*
     * given two points, this function returns the euclidean norm 
     */  
    int i;
    double euclidianNorm;
    euclidianNorm = 0;

    for(i=0; i< DIM; i++){
        euclidianNorm += pow((vector1[i]-vector2[i]),2);
    }
    /* printf("euclidian norm:");
    printf("%f", euclidianNorm);
    printf("\n");
    printf("square of euclidian norm:");
    printf("%f", pow(euclidianNorm, 0.5));
    printf("\n"); */
    return pow(euclidianNorm, 0.5);
}

int eigenComperator(const void *eigen1, const void *eigen2){
    /**
     * This is a compare function which will take 2 elements from the eigen array 
     * and return eigen1->eigenvalue - eigen2->eigenvalue
     */
    Eigen *eigenOne = (Eigen *)eigen1;
    Eigen *eigenTwo = (Eigen *)eigen2;
    if(eigenOne->eigenvalue < eigenTwo->eigenvalue) {
        return 0;
    }
    return 1;
}

double calcOff(double** mat) {
    int i, j;
  /*  double sumTot, sumDiag, powVal;*/
    double sumTot, powVal;
    sumTot = 0;
   /* sumDiag = 0;*/
    for (i = 0; i < N ; i++) {
        for (j = i+1; j < N ; j++) {
            powVal = pow(mat[i][j], 2);
            sumTot += powVal;
        }
    }
    sumTot = sumTot*2;
    return sumTot;
}

int is_diagonal(double** A, double** A_tag){
    /** TODO: calculate the convergence.
     * return True if the result is smaller than epsilon = e^-15
     */
    double offA, offA_tag;

    offA = calcOff(A);
    printf("is_diagonal: offA is: %f\n", offA);
    offA_tag = calcOff(A_tag);
    printf("is_diagonal: offA_tag is: %f\n", offA_tag);
    printf("is_diagonal: offA - offA_tag is: %f\n", offA - offA_tag);
    printf("is_diagonal: epsilon is %.10f\n", EPSILON);
    printf("is_diagonal: return value is: %d\n", offA - offA_tag <= EPSILON);
    return (offA - offA_tag <= EPSILON);
}

int obtainCT(double A_ii, double A_jj, double A_ij, double* c_t) {
    /** given a pivot value, return c and t as explained in the project */
    double theta, c, t;
    int sign;

    printf("obtainCT: A_ij = %f, A_ii = %f, A_jj = %f\n", A_ij, A_ii, A_jj);
    theta = (A_jj - A_ii) / A_ij;
    if (theta >= 0) {
        sign = 1;
    }
    else {
        sign = -1;
    }
    printf("obtainCT: theta = %f, sign = %d\n", theta, sign);
    t = sign / (fabs(theta)+sqrt(pow(theta,2) + 1));
    c = 1 / sqrt(pow(t,2) + 1);
    c_t[0] = c;
    c_t[1] = t;
    return 1;
}

void calcATag(double** A, double** A_tag, int pivot_i, int pivot_j, double c, double s) {
    int r;

    /* memcpy_matrix(A, A_tag, N, N); */
    A_tag[pivot_i][pivot_i] = pow(c,2) * A[pivot_i][pivot_i] + pow(s,2) * A[pivot_j][pivot_j] - 2 * s * c * A[pivot_i][pivot_j];
    A_tag[pivot_j][pivot_j] = pow(c,2) * A[pivot_i][pivot_i] + pow(s,2) * A[pivot_j][pivot_j] + 2 * s * c * A[pivot_i][pivot_j];
    A_tag[pivot_i][pivot_j] = 0;
    A_tag[pivot_j][pivot_i] = 0;
    for (r = 0; r < N; r++) {
        if (r == pivot_i || r == pivot_j) {
            continue;
        }
        A_tag[r][pivot_i] = c * A[r][pivot_i] - s * A[r][pivot_j];
        A_tag[pivot_i][r] = c * A[r][pivot_i] - s * A[r][pivot_j];
        A_tag[r][pivot_j] = c * A[r][pivot_j] + s * A[r][pivot_i];
        A_tag[pivot_j][r] = c * A[r][pivot_j] + s * A[r][pivot_i];
    }
}

void calcV(double** V, double c, double s, int pivot_i, int pivot_j) {
    V[pivot_i][pivot_i] = V[pivot_i][pivot_i] * c;
    V[pivot_j][pivot_j] = V[pivot_j][pivot_j] * c;
    V[pivot_i][pivot_j] = V[pivot_i][pivot_j] * s;
    V[pivot_j][pivot_i] = V[pivot_j][pivot_i] * (-1 * s);
}

void calcU(Eigen** eigensArray, double** U) {
    int i, j;

    for (i = 0; i < K; i++) {
        /** 
         * eigensArray[i]->eigenvector is the next eigenvector in U 
         * copy eigensArray[i]->eigenvector into the kIndex column of U 
         */
        for (j=0; j < N; j++) {
            U[j][i] = eigensArray[i]->eigenvector[j];
        }
    }
} 

void calcT(double **U, double **T){
    int i,j;
    double rowLength;

    for(i=0 ; i<N ; i++){
        rowLength = 0;
        for(j=0 ; j<2*K ; j++){
            if(j<K){
                rowLength = rowLength + pow(U[i][j],2);
            }
            else if(j==K){
                rowLength = pow(rowLength,0.5);
                if(rowLength != 0){
                    T[i][2*K-j] = U[i][2*K-j] / rowLength;
                }
                else{
                    T[i][2*K-j] = 0;
                }
            }
            else{
               if(rowLength != 0){
                    T[i][2*K-j] = U[i][2*K-j] / rowLength;
                }
                else{
                    T[i][2*K-j] = 0;
                } 
            }
        }
    }
}

int runEigengapHeuristic(Eigen** eigensArray) {
    int i, maxI;
    double gap, maxGap;

    maxI = -1;
    maxGap = -1.0;

    for (i=1; i < N/2 ; i++) {
        gap = eigensArray[i]->eigenvalue - eigensArray[i-1]->eigenvalue;
        if (gap > maxGap) {
            maxI = i;
            maxGap = gap;
        }
    }

    return maxI;
}

/** ----------------------------- PYTHON HELPERS ---------------------------------------- **/

Graph* pythonGraphInit(char* k, char* file_name) {
    double** weighted_mat, **vertices, *diagonal_degree_array;
    FILE* file;
    Graph* graph;

    weighted_mat = NULL;
    vertices = NULL;

    printf("In pythonGraphInit");
    printf("\n");

    K = atoi(k);
    printf("K is:%d",K);
    printf("\n");
    if ( K < 0) {
        printf("%s", INVALID_INPUT);
        exit(1);
    } 
    /* NOTE : if k == 0 - use heuristic */

    printf("file_name is:%s",file_name);
    printf("\n");
    if (!file_name) {
        printf("%s", ERROR_OCCURED);
        printf("entered filename is not defined");
        printf("\n");
        exit(1);
    }
    
    file = fopen(file_name, "r");
    N = howManyLines(file);
    DIM = pointSize(file);

    printf("N is:%d",N);
    printf("\n");
    printf("DIM is:%d",DIM);
    printf("\n");

    if (K >= N){
        printf("%s", INVALID_INPUT);
        exit(1);
    }

    /* Create the graph */
    /* allocateMatrix(N, DIM, vertices); */
    vertices = allocateMatrix(N, DIM);
    /* allocateMatrix(N, N, weighted_mat); */
    weighted_mat = allocateMatrix(N, N);
    diagonal_degree_array = (double*)calloc(N, sizeof(double));
    assert(diagonal_degree_array && ERROR_OCCURED);
    graph = (Graph*) malloc(sizeof (Graph));
    assert(graph && ERROR_OCCURED);
    constructGraph(file, vertices, weighted_mat, diagonal_degree_array, graph);
    return graph;
}

/** ---------------------------------- FLOWS ---------------------------------------- **/

void runLnormFlow(Graph* graph, double** laplacian_mat, int print_bool){
    /** TODO: 
     * Calculate and output the Normalized Graph Laplacian as described in 1.1.3.
     * The function should print appropriate output if print == True
     * fill the provided Laplacian matrix 
     */
    int i, j;
    double degrees, weight, toSub;
    printf("in runLnormFlow\n");
    for (i=0; i < N; i++){
        for (j=0; j < N; j++){
            if(i == j){
                laplacian_mat[i][j] = 1.0;
            }
            degrees = graph->diagonal_degree_array[i] * graph->diagonal_degree_array[j];
            weight = graph->weighted_mat[i][j];
            toSub = weight* degrees;
            laplacian_mat[i][j] = laplacian_mat[i][j] - toSub;
        }
    }
    printf("lap mat from runLnormFlow is:\n");
    printMatrix(N,N,laplacian_mat);
    if (print_bool){
        printMatrix(N, N, laplacian_mat);
    }
}

void runJacobiFlow(Graph* graph, double** A, Eigen** eigensArray, int print_bool) {
    /** 
     * Calculate and output the eigenvalues and eigenvectors as described in 1.2.1.
     * 
     * Once done, the values on the diagonal of A are eigenvalues
     * and the columns of V are eigenvectors 
     * 
     * The function should print appropriate output if print == True
     * and the eigensArray should be ORDERED with all eigenvalues and eigenvectors. 
     * Use sortEigens()
     */
    int i, j, pivot_i, pivot_j, iter_num;
    double pivot, c, t, s;
    double* c_t;
    double** A_tag, **V;

    /* first A mat is the laplacian */
    runLnormFlow(graph, A, FALSE);
    /* allocateMatrix(N, N, A_tag); */
    printf("runJacobiFlow: alloc A_tag\n");
    A_tag = allocateMatrix(N, N);
    memcpy_matrix(A, A_tag, N, N);
    /* allocateMatrix(N, N, V); */
    printf("runJacobiFlow: alloc V\n");
    V = allocateMatrix(N, N);
    c_t = (double*)calloc(2,sizeof(double));
    for (i=0 ; i<N ; i++) {
        V[i][i] = 1;
    }
    iter_num = 0;
    do {
        if(iter_num == 100){
            break;
        }
        memcpy_matrix(A_tag, A, N, N);
        printf("runJacobiFlow: start iteration, A:\n");
        printMatrix(N, N, A);
        /* calc pivot */
        printf("runJacobiFlow: calc pivot\n");
        pivot = 0.0;
        pivot_i = -1;
        pivot_j = -1;
        for (i=0; i < N; i++){
            for (j=0; j < N; j++){
                if (i != j){
                    if (fabs(A[i][j]) > fabs(pivot)){
                        printf("runJacobiFlow: fabs is: %f", fabs(A[i][j]));
                        pivot = A[i][j];
                        pivot_i = i;
                        pivot_j = j;
                    }
                }
            }
        }
        printf("runJacobiFlow: pivot is %f\n", pivot);
        printf("runJacobiFlow: pivot_i is %d\n", pivot_i);
        printf("runJacobiFlow: pivot_j is %d\n", pivot_j);
        /* calc c,t,s */
        printf("runJacobiFlow: obtainCT\n");
        obtainCT(A[pivot_i][pivot_i], A[pivot_j][pivot_j], pivot, c_t);
        c = c_t[0];
        t = c_t[1];
        s = c * t;

        /* transform A using "Relation between A and A'" */
        printf("runJacobiFlow: calcATag\n");
        calcATag(A, A_tag, pivot_i, pivot_j,c,s);

        printf("runJacobiFlow: calcV\n");
        calcV(V, c, s, pivot_i, pivot_j);

        iter_num = iter_num +1;

        printf("runJacobiFlow: end of iteration, A is:\n");
        printMatrix(N, N, A);
        printf("runJacobiFlow: end of iteration, A_tag is:\n");
        printMatrix(N, N, A_tag);

    } while (is_diagonal(A, A_tag) == FALSE);
    printf("stopped calculating pivot after %d iterations\n", iter_num);
    transposeSquareMatrix(V, N);
    for (i = 0; i < N ; i++) {
       eigensArray[i] = allocateEigen();
       eigensArray[i]->eigenvalue = A_tag[i][i];
       /*memcpy(eigensArray[i]->eigenvector, V[i], N);*/
       for(j=0;j<N;j++){
           eigensArray[i]->eigenvector[j] = V[i][j];
       }
    }

    freeMatrix(A_tag);
    freeMatrix(V);

    qsort(eigensArray, N, sizeof(Eigen*), eigenComperator);

    if (print_bool){
        printEigens(eigensArray, N);
    }
    printf("Jacobi END\n");
}

void runSpkFlow(Graph* graph, double** laplacian_mat, Eigen** eigensArray, double **centroids_mat,
                int *whichClusterArray, int print_bool){
    /** TODO: Perform full spectral kmeans as described in 1.
     * The function should print appropriate output
     */

    /** Algorithm:
     * 1. runLnormFlow (included in runJacobiFlow)
     * 2. runJacobiFlow
     * 3. if k==0: k = run_eigengap_heuristic(eigenvalues)
     * 4. U = transpose_matrix(eigenvectors[:k])
     * 5. T = renormalize_mat(U)
     * 6. run_kmeanspp(T)
     * 7. Assign points to relevant clusters as described in Algorithm1 of project description
     */
    double** U, **T;

    printf("in runSpkFlow\n");
    runJacobiFlow(graph, laplacian_mat, eigensArray, FALSE);
    printf("back in spkFlow\n");
    printf("laplacian:\n");
    printMatrix(N,N,laplacian_mat);
    printf("eigenArray:\n");
    printEigens(eigensArray,N);

    if (K == 0) {
        K = runEigengapHeuristic(eigensArray);
    }
    /* allocateMatrix(N, K, U); */
    U = allocateMatrix(N, K);
    calcU(eigensArray, U);
    printf("U is:\n");
    printMatrix(N,K,U);
    /* allocateMatrix(N, K, T); */
    T = allocateMatrix(N, K);
    calcT(U,T);
    printf("T is:\n");
    printMatrix(N,K,T);

    kmeans(K,N,DIM,T, centroids_mat, whichClusterArray);
    printf("back in runSPKflow\n");
    printf("centroids are:\n");
    printMatrix(K,DIM,centroids_mat);

    if (print_bool){
        printMatrix(K,DIM,centroids_mat);
    }
}

void runSpkFlowPython(Graph* graph, int *k, double*** T){
    /** TODO: Perform full spectral kmeans as described in 1.
     * The function should print appropriate output
     */

    /** Algorithm:
     * 1. runLnormFlow (included in runJacobiFlow)
     * 2. runJacobiFlow
     * 3. if k==0: k = run_eigengap_heuristic(eigenvalues)
     * 4. U = transpose_matrix(eigenvectors[:k])
     * 5. T = renormalize_mat(U)
     * 6. run_kmeanspp(T)
     * 7. Assign points to relevant clusters as described in Algorithm1 of project description
     */
    double** U, **laplacian_mat;
    Eigen** eigensArray;

    printf("in runSpkFlow\n");

    laplacian_mat = allocateMatrix(N, N);
    assert(laplacian_mat && ERROR_OCCURED);
    eigensArray = (Eigen**)malloc(N * N * sizeof(Eigen*));
    assert(eigensArray && ERROR_OCCURED);

    runJacobiFlow(graph, laplacian_mat, eigensArray, FALSE);
    printf("back in spkFlowPython\n");
    printf("runSpkFlowPython: laplacian:\n");
    printMatrix(N,N,laplacian_mat);
    printf("runSpkFlowPython: eigenArray:\n");
    printEigens(eigensArray,N);

    if (K == 0) {
        K = runEigengapHeuristic(eigensArray);
    }
    *k = K;

    U = allocateMatrix(N, K);
    calcU(eigensArray, U);
    printf("runSpkFlowPython: U is:\n");
    printMatrix(N,K,U);
    /* allocateMatrix(N, K, T); */
    *T = allocateMatrix(N, K);
    calcT(U,*T);
    printf("T is:\n");
    printMatrix(N,K,*T);

    freeMatrix(U);
}


/** MAIN **/
int main(int argc, char* argv[]) {
    
    double** weighted_mat, **vertices, **laplacian_mat, *diagonal_degree_array, **centroids_mat;
    int *whichClusterArray;
    char* file_name, *goal;
    FILE* file;
    Graph* graph;
    Eigen** eigensArray;
    /*
   double** weighted_mat, **vertices, *diagonal_degree_array;
    char* file_name, *goal;
    FILE* file;
    Graph* graph;
    */
    printf("In Main");
    printf("\n");
    assert((argc == 4) && INVALID_INPUT); 

    K = atoi(argv[1]);
    printf("K is:%d",K);
    printf("\n");
    if ( K < 0) {
        printf("%s", INVALID_INPUT);
        exit(1);
    } 
    /* NOTE : if k == 0 - use heuristic */

    file_name = argv[3];
    printf("file_name is:%s",file_name);
    printf("\n");
    if (!file_name) {
        printf("%s", ERROR_OCCURED);
        printf("entered filename is not defined");
        printf("\n");
        exit(1);
    }
    
    file = fopen(file_name, "r");
    N = howManyLines(file);
    DIM = pointSize(file);

    printf("N is:%d",N);
    printf("\n");
    printf("DIM is:%d",DIM);
    printf("\n");

    if (K >= N){
        printf("%s", INVALID_INPUT);
        exit(1);
    }

    goal = argv[2];
    if (!goal) {
        printf("%s", ERROR_OCCURED);
        printf("entered goal is not defiened");
        printf("\n");
        exit(1);
    } 
    printf("goal is:%s",goal);
    printf("\n");

    /* Create the graph */
    vertices = allocateMatrix(N, DIM);
    weighted_mat = allocateMatrix(N, N);
    diagonal_degree_array = (double*)calloc(N, sizeof(double));
    assert(diagonal_degree_array && ERROR_OCCURED);
    graph = (Graph*)malloc(sizeof (Graph));
    assert(graph && ERROR_OCCURED);
    constructGraph(file, vertices, weighted_mat, diagonal_degree_array, graph);

    printf("vertices:");
    printf("\n");
    printMatrix(N, DIM, vertices);
    printf("weighted mat:");
    printf("\n");
    printMatrix(N, N, weighted_mat);
    printf("diagonal array:");
    printf("\n");
    printArray(N, diagonal_degree_array);

    if (!strcmp(goal, "spk")) {
        printf("in spk flow\n");
        laplacian_mat = allocateMatrix(N, N);
        centroids_mat = allocateMatrix(N, DIM);
        whichClusterArray = (int*)calloc(N,sizeof(int));
        assert(whichClusterArray && ERROR_OCCURED);
        eigensArray = (Eigen**)malloc(N * N * sizeof(Eigen*));
        assert(eigensArray && ERROR_OCCURED);
        runSpkFlow(graph, laplacian_mat, eigensArray,centroids_mat, whichClusterArray, TRUE);
        freeMatrix(laplacian_mat);
        freeMatrix(centroids_mat);
        free(whichClusterArray);
        freeEigensArray(eigensArray, N);
    }
    else if (!strcmp(goal, "wam")) {
        printMatrix(N, N, graph->weighted_mat);
    }
    else if (!strcmp(goal, "ddg")) {
        printArray(N, graph->diagonal_degree_array);
    }
    else if (!strcmp(goal, "lnorm")) {
        /* allocateMatrix(N, N, laplacian_mat); */
        laplacian_mat = allocateMatrix(N, N);
        runLnormFlow(graph, laplacian_mat, TRUE);
        freeMatrix(laplacian_mat);
    }
    else if (!strcmp(goal, "jacobi")) {
        /* allocateMatrix(N, N, laplacian_mat); */
        laplacian_mat = allocateMatrix(N, N);
        eigensArray = (Eigen**)malloc(N * sizeof(Eigen*));
        assert(eigensArray && ERROR_OCCURED);
        runJacobiFlow(graph, laplacian_mat, eigensArray, TRUE);
        freeMatrix(laplacian_mat);
        freeEigensArray(eigensArray, N);
    }
    else {
        printf("%s", INVALID_INPUT);
        exit(1);
    }
    
    fclose(file);
    
    freeGraph(graph);

   return 1; 
}


