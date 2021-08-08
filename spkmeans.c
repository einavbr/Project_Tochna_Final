#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define TRUE 1
#define FALSE 0
#define INVALID_INPUT "Invalid Input!"
#define ERROR_OCCURED "An Error Has Occured"
#define MAX_ITER 100

int N, K, DIM;

/* ------------------------------ GRAPH IMPLEMENTATION ------------------------------------------------------------ */

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

double calcEuclideanNorm(vector1, vector2);
    /** TODO :
     * given two points, this function returns the euclidean norm 
     */

void fillWeightedMat(vertices, weighted_mat);
    /** TODO : 
     * given a list of vectors, this fuction should return a matrix of weights 
     * for each pair of vectors
     * NOTE : remember this is a symetric matrix, we only need to compute for i<j
     * and put the result in 2 slots
     */

void fillDiagonalDegreeArray(vertices, weighted_mat, diagonal_degree_array);
    /** TODO : 
     * given a list of vectors, this fuction should return the Diagonal Degree Matrix ^ -0.5 
     */

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
            /* point = strtok(NULL, ","); */
            j = j + 1;
        }
        i = i + 1;
    }
    DIM = j;
    N = i;

    /** TODO:
     * create fillWeightedMat(vertices, weighted_mat)
     * create fillDiagonalDegreeArray(vertices, weighted_mat, diagonal_degree_array)
     */

    graph->vertices = vertices;
    graph->weighted_mat = weighted_mat;

    free(line);
}

void runLnormFlow(Graph* graph, double** laplacian_mat, int print_bool) {
    /** TODO: 
     * Calculate and output the Normalized Graph Laplacian as described in 1.1.3.
     * The function should print appropriate output if print == True
     * fill the provided Laplacian matrix 
     */
}

double eigenComperator(Eigen* eigen1, Eigen* eigen2);
    /** TODO: 
     * This is a compare function which will take 2 elements from the eigen array 
     * and return eigen1->eigenvalue - eigen2->eigenvalue
     */

void sortEigens(Eigen* eigensArray);
    /** TODO: 
     * create eigen couples for each eigenvalue on the A matrix's diagonal and it's corresponding
     * eigenvector from the V matrix, and fill the eigenArray with them.
     * Use qsort() to sort the array.
     */

void runJacobiFlow(Graph* graph, double** laplacian_mat, Eigen* eigensArray, int print_bool) {
    /** TODO:
     * Calculate and output the eigenvalues and eigenvectors as described in 1.2.1.
     * 
     * Once done, the values on the diagonal of A are eigenvalues
     * and the columns of V are eigenvectors 
     * 
     * The function should print appropriate output if print == True
     * and the eigensArray should be ORDERED with all eigenvalues and eigenvectors. 
     * Use sortEigens()
     */
}

void runSpkFlow(Graph* graph, double** laplacian_mat, Eigen* eigensArray) {
    /** TODO: Perform full spectral kmeans as described in 1.
     * The function should print appropriate output
     */

    /** Algorithm:
     * 1. runLnormFlow
     * 2. runJacobiFlow
     * 3. if k==0: k = run_eigengap_heuristic(eigenvalues)
     * 4. U = transpose_matrix(eigenvectors[:k])
     * 5. T = renormalize_mat(U)
     * 6. run_kmeanspp(T)
     * 7. Assign points to relevant clusters as described in Algorithm1 of project description
     */
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

double** allocateMatrix(int rows, int cols){
    int i;
    double** matrix;
    
    matrix = (double**)malloc((rows) * sizeof(double*));
    for (i=0 ; i< rows; i++){
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

void freeEigensArray (Eigen* freeEigensArray, int N) {
    int i;
    for (int i=0 ; i<N ; i++) {
        free(freeEigensArray[i].eigenvector);
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

/** MAIN **/
int main(int argc, char* argv[]) {
    double** weighted_mat, **vertices, **laplacian_mat, *diagonal_degree_array;
    char* file_name, *goal;
    FILE* file;
    Graph* graph;
    Eigen* eigensArray;

    assert((argc == 4) && INVALID_INPUT); 

    K = atoi(argv[1]);
    if (!K || K < 0) {
        if (!K) {
            printf("%s", ERROR_OCCURED);
        }
        else {
            printf("%s", INVALID_INPUT);
        }
        exit(1);
    } 
    /* NOTE : if k == 0 - use heuristic */

    file_name = argv[3];
    if (!file_name) {
        printf("%s", ERROR_OCCURED);
        exit(1);
    }

    file = fopen(file_name, "r");
    N = howManyLines(file);
    DIM = pointSize(file);

    if (K >= N){
        printf("%s", INVALID_INPUT);
        exit(1);
    }

    goal = argv[2];
    if (!goal) {
        printf("%s", ERROR_OCCURED);
        exit(1);
    } 

    /* Create the graph */
    vertices = allocateMatrix(N, DIM);
    weighted_mat = allocateMatrix(N, N);
    diagonal_degree_array = (double*)malloc((N) * sizeof(double));
    graph = (Graph*) malloc(sizeof (Graph));
    constructGraph(file, vertices, weighted_mat, diagonal_degree_array, graph);

    printMatrix(N, DIM, vertices);
    printMatrix(N, N, weighted_mat);
    printMatrix(1, N, diagonal_degree_array);

    if (!strcmp(goal, 'spk')) {
        laplacian_mat = allocateMatrix(N, N);
        eigensArray = (Eigen*)malloc(N * N * sizeof(Eigen));
        runSpkFlow(graph, laplacian_mat, eigensArray);
        freeMatrix(laplacian_mat, N);
        freeEigensArray(eigensArray, N);
    }
    else if (!strcmp(goal, 'wam')) {
        printMatrix(N, N, graph->weighted_mat);
    }
    else if (!strcmp(goal, 'ddg')) {
        printMatrix(1, N, graph->diagonal_degree_array);
    }
    else if (!strcmp(goal, 'lnorm')) {
        laplacian_mat = allocateMatrix(N, N);
        runLnormFlow(graph, laplacian_mat, TRUE);
        freeMatrix(laplacian_mat, N);
    }
    else if (!strcmp(goal, 'jacobi')) {
        laplacian_mat = allocateMatrix(N, N);
        eigensArray = (Eigen*)malloc(N * N * sizeof(Eigen));
        runJacobiFlow(graph, laplacian_mat, eigensArray, TRUE);
        freeMatrix(laplacian_mat, N);
        freeEigensArray(eigensArray, N);
    }
    else {
        print("%s", INVALID_INPUT);
        exit(1);
    }
    
    fclose(file);
    free(diagonal_degree_array);
    freeMatrix(weighted_mat, N);
    freeMatrix(vertices, N);
}


