#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

#define TRUE 1
#define FALSE 0
#define INVALID_INPUT "Invalid Input!"
#define ERROR_OCCURED "An Error Has Occured"
#define MAX_ITER 100
#define EPSILON 0.001

int N, K, DIM;

double calcEuclideanNorm(double* vector1, double* vector2);
void printMatrix(int rows, int cols, double** matrix);
void printArray(int len, double* matrix);

/* ------------------------------ GRAPH IMPLEMENTATION ---------------------------------------------------- */

typedef struct Graph {
    /** A graph contains:
     * vertices - A list of the vertices in the graph
     * weighted_mat - Weighted Adjacency Matrix (represented as a sparse matrix, array implementation), weight = 0 means no edge
     * diagonal_degree_array - The values on the diagonal of the Diagonal Degree Matrix ^ -0.5 
     **/
    double** vertices;
    double** weighted_mat;
    double* diagonal_degree_array;
} Graph;

typedef struct Eigen {
    /* An Eigen is a "tuple" of an eigenvalue and it's corresponding eigenvector */
    double eigenvalue;
    double* eigenvector;
} Eigen;

/** ---------------------------------- GRAPH FUNCTIONS ---------------------------------------- **/

void fillWeightedMat(double** vertices, double** weighted_mat,int N){
    /** TODO : 
     * given a list of vectors, this fuction should return a matrix of weights 
     * for each pair of vectors
     * NOTE : remember this is a symetric matrix, we only need to compute for i<j
     * and put the result in 2 slots
     */
    int i,j;
    double euclidianNorm,power;
    for(i=0 ; i<N ; i++){
        weighted_mat[i][i] = 0;
        for(j=i+1; j<N ; j++){
            euclidianNorm = calcEuclideanNorm(vertices[i],vertices[j]);
            power = -euclidianNorm/2;
            weighted_mat[i][j] = weighted_mat[j][i] = exp(power);
            printf("weighted matrix in %d,%d is: %f",i,j,exp(power));
            printf("\n");
        }
    }
}
    
void fillDiagonalDegreeArray(double** weighted_mat, double* diagonal_degree_array, int N){
     /** TODO : 
     * given a list of vectors, this fuction should return the Diagonal Degree Matrix ^ -0.5 
     */
    int i,j;
    for(i=0; i<N; i++){
        for(j=0;j<N;j++){
            printf("i,j: %d,%d\n", i,j); 
            printf("w_ij: %f", weighted_mat[i][j]);
            diagonal_degree_array[i] = diagonal_degree_array[i] + weighted_mat[i][j];
        }
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

    /** TODO:
     * create fillWeightedMat(vertices, weighted_mat, N, DIM)
     * create fillDiagonalDegreeArray(weighted_mat, diagonal_degree_array, N)
     */
    fillWeightedMat(vertices, weighted_mat, N);
    fillDiagonalDegreeArray(weighted_mat, diagonal_degree_array, N);

    graph->vertices = vertices;
    graph->weighted_mat = weighted_mat;
    graph->diagonal_degree_array = diagonal_degree_array;
    free(line);
}

/** ---------------------------------- PRINTS ---------------------------------------- **/

void printEigens(Eigen** eigens, int n){
    int i;
    
    for (i=0; i < n; i++) {
        printf("%f, ", eigens[i]->eigenvalue);
    }
    for (i=0; i < n; i++) {
        printArray(N, eigens[i]->eigenvector);
    }
}

void printMatrix(int rows, int cols, double** matrix) {
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

double** allocateMatrix(int rows, int cols){
    int i;
    double** matrix;
    
    matrix = (double**)calloc(rows, sizeof(double*));
    for (i=0 ; i< rows; i++){
        matrix[i] = (double*)calloc(cols, sizeof(double));
    }
    return matrix;
}

Eigen* allocateEigen(){
    Eigen* eigen;

    eigen = (Eigen*)malloc(sizeof(Eigen));
    eigen->eigenvector = (double*)malloc(N * sizeof(double));
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

void freeMatrix(double **matrix, int rowsLen){
    int i;
    for (i=0; i < rowsLen; i++){
        free(matrix[i]);
    }
    free(matrix);
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
    int i;
    for (i=0 ; i<rows; i++){
        memcpy(dest[i], src[i], cols);
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
    /** TODO :
     * given two points, this function returns the euclidean norm 
     */  
    int i;
    double euclidianNorm;
    euclidianNorm = 0;
    for(i=0; i< DIM; i++){
        euclidianNorm += pow((vector1[i]-vector2[i]),2);
    }
    printf("euclidian norm:");
    printf("%f", euclidianNorm);
    printf("\n");
    printf("square of euclidian norm:");
    printf("%f", pow(euclidianNorm, 0.5));
    printf("\n");
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
    double sumTot, sumDiag, powVal;

    sumTot = 0;
    sumDiag = 0;
    for (i = 0; i < N ; i++) {
        for (j = 0; j < N ; j++) {
            powVal = pow(mat[i][j], 2);
            if (i == j) {
                sumDiag += powVal;
            }
            sumTot += powVal;
        }
    }
    return (sqrt(sumTot) - sumDiag);
}

bool is_diagonal(double** A, double** A_tag){
    /** TODO: calculate the convergence.
     * return True if the result is smaller than epsilon = 0.001
     */
    double offA, offA_tag;

    offA = calcOff(A);
    offA_tag = calcOff(A_tag);

    return (offA - offA_tag <= EPSILON);
}

double* obtainCT(double A_ii, double A_jj, double A_ij) {
    /** given a pivot value, return c and t as explained in the project */
    double theta, c, t,*res;
    int sign;

    printf("A_ij = %f, A_ii = %f, A_jj = %f", A_ij, A_ii, A_jj);
    theta = (A_jj - A_ii) / A_ij;
    if (theta >= 0) {
        sign = 1;
    }
    else {
        sign = -1;
    }
    printf("theta = %f, sign = %d", theta, sign);
    t = sign / (fabs(theta)+sqrt(pow(theta,2) + 1));
    c = 1 / sqrt(pow(t,2) + 1);
    res = (double*)calloc(2,sizeof(double));
    res[0] = c;
    res[1] = t;
    return res;
}

void calcATag(double** A, double** A_tag, int pivot_i, int pivot_j, double c, double s) {
    int r;

    memcpy_matrix(A, A_tag, N, N);
    A_tag[pivot_i][pivot_i] = pow(c,2) * A[pivot_i][pivot_i] + pow(s,2) * A[pivot_j][pivot_j] - 2 * s * c * A[pivot_i][pivot_j];
    A_tag[pivot_j][pivot_j] = pow(c,2) * A[pivot_i][pivot_i] + pow(s,2) * A[pivot_j][pivot_j] + 2 * s * c * A[pivot_i][pivot_j];
    A_tag[pivot_i][pivot_j] = 0;
    for (r = 0; r < N; r++) {
        if (r == pivot_i || r == pivot_j) {
            continue;
        }
        A_tag[r][pivot_i] = c * A[r][pivot_i] - s * A[r][pivot_j];
        A_tag[r][pivot_j] = c * A[r][pivot_j] + s * A[r][pivot_i];
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

int runEigengapHeuristic(Eigen** eigensArray) {
    int i, maxI;
    double gap, maxGap;

    maxI = -1
    maxGap = -1.0;

    for (i=1; i < N/2 ; i++) {
        gap = eigensArray[i]->eigenvalue - eigensArray[i-1]->eigenvalue;
        if (gap > maxGap) {
            maxI = i;
            maxGap = gap;
        }
    }

    return maxI
}

/** ---------------------------------- FLOWS ---------------------------------------- **/

void runLnormFlow(Graph* graph, double** laplacian_mat, int print_bool){
    /** TODO: 
     * Calculate and output the Normalized Graph Laplacian as described in 1.1.3.
     * The function should print appropriate output if print == True
     * fill the provided Laplacian matrix 
     */
    int i, j;
    double mechane, mone;

    for (i=0; i < N; i++){
        for (j=i; j < N; j++){
            if(i == j){
                laplacian_mat[i][j] = 1.0;
            }
            mechane = graph->diagonal_degree_array[i] * graph->diagonal_degree_array[j];
            mone = graph->weighted_mat[i][j];
            laplacian_mat[i][j] = 1 - (mechane / mone);
            laplacian_mat[j][i] = 1 - (mechane / mone);
        }
    }
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
    int i, j, pivot_i, pivot_j;
    double pivot, c, t, s;
    double* c_t;
    double** A_tag, **V;

    /* first A mat is the laplacian */
    runLnormFlow(graph, A, FALSE);
    A_tag = allocateMatrix(N, N);
    V = allocateMatrix(N, N);
    for (i=0 ; i<N ; i++) {
        V[i][i] = 1;
    }

    do {
        /* calc pivot */
        pivot = 0;
        pivot_i = -1;
        pivot_j = -1;
        for (i=0; i < N; i++){
            for (j=0; j < N; j++){
                if (i != j){
                    if (A[i][j] > pivot){
                        pivot = A[i][j];
                        pivot_i = i;
                        pivot_j = j;
                    }
                }
            }
        }

        /* calc c,t,s */
        c_t = obtainCT(A[pivot_i][pivot_i], A[pivot_j][pivot_j], pivot);
        c = c_t[0];
        t = c_t[1];
        s = c * t;

        /* transform A using "Relation between A and A'" */
        calcATag(A, A_tag, pivot_i, pivot_j,c,s);

        calcV(V, c, s, pivot_i, pivot_j);
        
    } while (!is_diagonal(A,A_tag));

    transposeSquareMatrix(V, N);
    for (i = 0; i < N ; i++) {
       eigensArray[i] = allocateEigen();
       eigensArray[i]->eigenvalue = A_tag[i][i];
       memcpy(eigensArray[i]->eigenvector, V[i], N);
    }

    freeMatrix(A_tag , N);
    freeMatrix(V, N);

    qsort(eigensArray, N, sizeof(Eigen*), eigenComperator);

    if (print_bool){
        printEigens(eigensArray, N);
    }
}

void runSpkFlow(Graph* graph, double** laplacian_mat, Eigen** eigensArray){
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
    double** U;

    runJacobiFlow(graph, laplacian_mat, eigensArray, FALSE);
    if (K == 0) {
        K = runEigengapHeuristic;
    }
    U = allocateMatrix(N, K);
    calcU(eigensArray, U);
}

/** MAIN **/
int main(int argc, char* argv[]) {
    /*
    double** weighted_mat, **vertices, **laplacian_mat, *diagonal_degree_array;
    char* file_name, *goal;
    FILE* file;
    Graph* graph;
    Eigen* eigensArray;
    */
   double** weighted_mat, **vertices, *diagonal_degree_array;
    char* file_name, *goal;
    FILE* file;
    Graph* graph;

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
        printf("entered filename is not defiened");
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
    graph = (Graph*) malloc(sizeof (Graph));
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
/*
    if (!strcmp(goal, "spk")) {
        laplacian_mat = allocateMatrix(N, N);
        eigensArray = (Eigen*)malloc(N * N * sizeof(Eigen));
        runSpkFlow(graph, laplacian_mat, eigensArray);
        freeMatrix(laplacian_mat, N);
        freeEigensArray(eigensArray, N);
    }
    else if (!strcmp(goal, "wam")) {
        printMatrix(N, N, graph->weighted_mat);
    }
    else if (!strcmp(goal, "ddg")) {
        printArray(N, graph->diagonal_degree_array);
    }
    else if (!strcmp(goal, "lnorm")) {
        laplacian_mat = allocateMatrix(N, N);
        runLnormFlow(graph, laplacian_mat, TRUE);
        freeMatrix(laplacian_mat, N);
    }
    else if (!strcmp(goal, "jacobi")) {
        laplacian_mat = allocateMatrix(N, N);
        eigensArray = (Eigen**)malloc(N * sizeof(Eigen*));
        runJacobiFlow(graph, laplacian_mat, eigensArray, TRUE);
        freeMatrix(laplacian_mat, N);
        freeEigensArray(eigensArray, N);
    }
    else {
        printf("%s", INVALID_INPUT);
        exit(1);
    }*/
    
    fclose(file);
    /*
    free(diagonal_degree_array);
    freeMatrix(weighted_mat, N);
    freeMatrix(vertices, N);
    */
   return 1; 
}


