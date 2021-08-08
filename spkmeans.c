#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define INVALID_INPUT "Invalid Input!"
#define ERROR_OCCURED "An Error Has Occured"
#define MAX_ITER 100

int POINT_SIZE, N, K, DIM;

/* ------------------------------ GRAPH IMPLEMENTATION ------------------------------------------------------------ */

typedef struct Graph {
    /** TODO : The nodes are provided from the input therefor no arguments (use FILENAME).
     * A graph contains:
     * vertices - A list of the vertices in the graph
     * weighted_mat - Weighted Adjacency Matrix (represented as a sparse matrix, array implementation), weight = 0 means no edge
     * Droot - Diagonal Degree Matrix ^ -0.5
     **/

    double** vertices;
    double** weighted_mat;
} Graph;

/* counts lines in file */
int howManyLines(FILE* file) {
    int counterOfLines; /* Line counter (result) */
    char c;  /* To store a character read from file */
    fseek(file, 0, SEEK_SET);
    counterOfLines = 0;  
    /* Extract characters from file and store in character c */
    for (c = getc(file); c != EOF; c = getc(file))
        if (c == '\n') /* Increment count if this character is newline */
            counterOfLines = counterOfLines + 1;

    return counterOfLines;
}

double calc_euclidean_norm(double* point1, double* point2){
    int i;
    double sum;

    sum = 0.0;
    for (i=0 ; i<DIM ; i++){
        sum += pow((point1[i] - point2[i]), 2);
    }

    return sqrt(sum);
}

void fillWeightedMat(double** vertices, double** weighted_mat){
    /** TODO : given a list of vertices and a weighted_mat filled with zeros, 
     * this fuction should return a matrix of weights for each pair of nodes.
     * NOTE : remember this is a symetric matrix, we only need to compute for i<j
     * and put the result in 2 slots
     **/
    int i, j, edges_i, max_amount_of_weights;
    double* rows, *columns, *weights, norm;
    
    max_amount_of_weights = (pow(N,2) - N) / 2;
    rows = (double*)malloc(max_amount_of_weights * sizeof(double));
    columns = (double*)malloc(max_amount_of_weights * sizeof(double));
    weights = (double*)malloc(max_amount_of_weights * sizeof(double));

    edges_i = 0;
    for (i=0 ; i<N ; i++){
        for (j=i+1 ; j<N ; j++){
            norm = calc_euclidean_norm(vertices[i], vertices[j]);
            if (norm > 0){
                rows[edges_i] = i;
                columns[edges_i] = j;
                weights[edges_i] = exp(-1 * (norm/2));
            }
            weighted_mat[i][j] = weighted_mat[j][i] = exp(-1 * (norm/2));
        }
    }
}

void constructGraph(FILE* file, double** vertices, double** weighted_mat, Graph* graph){
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

    fillWeightedMat(vertices, weighted_mat);

    graph->vertices = vertices;
    graph->weighted_mat = weighted_mat;

    free(line);
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

void freeMatrix(double **matrix, int rowsLen){
    int i;
    for (i=0; i < rowsLen; i++){
        free(matrix[i]);
    }
    free(matrix);
}

/** MAIN **/
int main(int argc, char* argv[]) {
    double** weighted_mat, **vertices;
    char* file_name, *goal;
    FILE* file;
    Graph* graph;

    printf("reached here 1");

    assert((argc == 4) && INVALID_INPUT); 
    
    printf("reached here 2");

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
    
    printf("reached here 3");

    file_name = argv[3];
    if (!file_name) {
        printf("%s", ERROR_OCCURED);
        exit(1);
    }

    file = fopen( file_name, "r");
    N = howManyLines(file);

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
    printf("reached here 1");
    weighted_mat = (double**)malloc((3) * sizeof(double*));
    vertices = (double**)malloc((N) * sizeof(double*));
    graph = (Graph*) malloc(sizeof (Graph));
    constructGraph(file, vertices, weighted_mat, graph);

    printMatrix(N, DIM, vertices);
    printMatrix(3, (pow(N,2) - N) / 2, weighted_mat);


    fclose(file);
    freeMatrix(weighted_mat, 3);
    freeMatrix(vertices, N);
    
    /*
    try:
        GOAL = sys.argv[2]
    except Exception:
        print(INVALID_INPUT)

    if GOAL == 'spk':
        run_spk_flow()


    # elif GOAL == 'wam':
    #     printMatrix(graph.W)
    # elif GOAL == 'ddg':
    #     printMatrix(graph.D)
    # elif GOAL == 'lnorm':
    #     run_lnorm_flow(True)
    # elif GOAL == 'jacobi':
    #     run_jacobi_flow(graph.W, True)
    # else:
    #     raise Exception(INVALID_INPUT)
    */
}
