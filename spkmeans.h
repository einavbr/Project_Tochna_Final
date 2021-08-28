#ifndef SPKMEANS_H_   /* Include guard */
#define SPKMEANS_H_

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
} Graph;

typedef struct Eigen {
    /* An Eigen is a "tuple" of an eigenvalue and it's corresponding eigenvector */
    double eigenvalue;
    double* eigenvector;
} Eigen;

int printTest(int num);
Graph* pythonGraphInit(char* k, char* file_name);
void freeGraph(Graph* graph);

#endif
