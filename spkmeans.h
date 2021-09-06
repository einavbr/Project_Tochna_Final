#ifndef SPKMEANS_H_   /* Include guard */
#define SPKMEANS_H_

#define TRUE 1
#define FALSE 0

/* ------------------------------ GRAPH DECLERATION ---------------------------------------------------- */

typedef struct Graph {
    /** A graph contains:
     * vertices - A list of the vertices in the graph
     * weighted_mat - Weighted Adjacency Matrix (represented as a sparse matrix, array implementation), weight = 0 means no edge
     * diagonal_degree_array - The values on the diagonal of the Diagonal Degree Matrix ^ -0.5 
     **/
    double** vertices;
    double** weighted_mat;
    double* diagonal_degree_array;
    int size;
    int dim;
} Graph;

typedef struct Eigen {
    /* An Eigen is a "tuple" of an eigenvalue and it's corresponding eigenvector */
    double eigenvalue;
    double* eigenvector;
} Eigen;

Graph* pythonGraphInit(char* k, char* file_name);
void freeGraph(Graph* graph);
double** allocateMatrix(int rows, int cols);
void runLnormFlow(Graph* graph, double** laplacian_mat, int print_bool);
void runJacobiFlow(Graph* graph, double** A, Eigen** eigensArray, int print_bool);
void runSpkFlow(Graph* graph, double** laplacian_mat, Eigen** eigensArray, double **centroids_mat, int *whichClusterArray, int kmeanspp_bool, int print_bool);

#endif
