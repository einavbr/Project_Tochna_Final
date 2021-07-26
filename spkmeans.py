# Python implementation of the normalized spectral clustering algorithm
import sys

INVALID_INPUT = 'Invalid Input!'
ERROR_OCCURED = 'An Error Has Occured'

class Node():
    def __init__(self, point, i):
        # TODO : given a list contining a single point and an index.
        # A node contains:
        # index field
        # point field - an array containing the coordinates of the point 
        pass

class Graph():
    def __init__(self):
        # TODO : The nodes are provided from the input therefor no arguments (see PT1). 
        # A graph contains:
        # size field - the number of nodes in the graph
        # V - A list of the vertexes in the graph
        # W - Weighted Adjacency Matrix (A matrix is a 2D list), weight = 0 means no edge
        # Droot - Diagonal Degree Matrix ^ -0.5
        pass

def get_weighted_adjacency_matrix():
    # TODO : given a list of nodes, this fuction should return a matrix of weights 
    # for each pair of nodes
    # NOTE : remember this is a symetric matrix, we only need to compute for i<j
    # and put the result in 2 slots
    pass

def get_Droot_matrix():
    # TODO : given a list of nodes, this fuction should return the Diagonal Degree Matrix ^ -0.5 
    pass

def calc_euclidean_norm(a, b):
    # TODO : given two points(list of coordinates from relevant the Node field), 
    # this function calculates the euclidean norm 
    pass

def get_frobenius_norm(A):
    # TODO : given a matrix, 
    # this function calculates the euclidean norm 
    pass

# Functions on Matrices:
def print_matrix(A):
    # TODO : print the matrix, take print func from PT1 or PT2
    pass

def multiply_matrices(A , B):
    # TODO : given 2 matrixes (2D lists), perform matrix multiplication A x B 
    pass

def transpose_matrix(A):
    # TODO : given a matrix, transpose it
    pass

def is_diagonal(A):
    # TODO : given a matrix, this funtion returns True if the matrix 
    # is diagonal and False otherwise
    # check using Convergence. Use "calc_frobenius_norm"
    pass

def get_pivot(A):
    # TODO : given a matrix, return the off-diagonal element with the largest absolute value
    pass

def obtain_c_t(pivot):
    # TODO : given a pivot value, return c and t as explained in the project
    pass

def get_rotation_mat(pivot):
    # TODO : given a pivot value:
    # calc c, t. Use "obtain_c_t"
    # calc s
    # create rotation matrix
    pass

def renormalize_mat(A):
    # TODO : perform step 5 of algorithm 1.
    # NOTE : for the mechane we can use calc_euclidean_norm and set b = -a
    pass

def run_kmeanspp():
    # TODO : incorporate the kmeans++ algorithm from PT2 in this code
    # Use the C extension spkmeansmodule.
    pass

def run_eigengap_heuristic(eigenvalues):
    # TODO : given an ORDERED list of eigenvalues, return the max eigengap
    pass

# Flow Functions
def run_lnorm_flow(print):
    # TODO : Calculate and output the Normalized Graph Laplacian as described in 1.1.3.
    # The function should print appropriate output if print == True
    # return a Laplacian matrix 

    # 1. calculate laplacian matrix for GRAPH
    # 2. print laplacian matrix if print == True
    # 3. return laplacian matrix
    pass

def run_jacobi_flow(A, print):
    # TODO : Calculate and output the eigenvalues and eigenvectors as described in 1.2.1.
    # A is the matrix whos eigenvalues and eigenvectors we want
    # The function should print appropriate output if print == True
    # return ORDERED eigenvalues and eigenvectors

    # pivot = get_pivot(graph.W)
    # Build a rotation matrix P. get_rotation_mat(pivot)
    # V = P. V is the multiplication matrix of all the P matrices.
    # Repeat a,b until A is diagonal matrix. Use "is_diagonal":
    #   1. calc new P. Use "get_rotation_mat"
    #   2. A = A'. to calculate A' see "Relation between A and A'"" in the project description
    #   3. V = V x P. Use "multiply_matrices"
    #   4. is A diagonal? If yes - stop. Use "is_diagonal"
    #   5. did we reach 100 iterations? If yes - stop.
    # Once done, the values on the diagonal of A are eigenvalues
    # and the columns of V are eigenvectors
    # This fuction should return an ORDERED list of eigenvalues, 
    # and a matrix with the eigenvetors in the corresponding columns
    pass

def run_spk_flow():
    # TODO : Perform full spectral kmeans as described in 1.
    # The function should print appropriate output

    # 1. get laplacian: laplacian = run_lnorm_flow(False)
    # 2. laplacian's eigenvalues, eigenvectors = run_jacobi_flow(laplacian, False)
    # 3. if k==0: k = run_eigengap_heuristic(eigenvalues)
    # 4. U = transpose_matrix(eigenvectors[:k])
    # 5. T = renormalize_mat(U)
    # 6. run_kmeanspp(T)
    # 7. Assign points to relevant clusters as described in Algorithm1 of project description
    pass

### MAIN ###
try:
    K = int(sys.argv[1])
except Exception:
    print (INVALID_INPUT)
if K < 0:
    raise Exception(INVALID_INPUT)
# NOTE : if k == 0 - use heuristic

try:
    FILE_NAME = sys.argv[3]
except Exception:
    print (INVALID_INPUT)

# create the graph
graph = Graph() 

try:
    GOAL = int(sys.argv[2])
except Exception:
    print(INVALID_INPUT)

if GOAL == 'spk':
    run_spk_flow()
elif GOAL == 'wam':
    print_matrix(graph.W)
elif GOAL == 'ddg':
    print_matrix(graph.D)
elif GOAL == 'lnorm':
    run_lnorm_flow(True)
elif GOAL == 'jacobi':
    run_jacobi_flow(graph.W, True)
else:
    raise Exception(INVALID_INPUT)

