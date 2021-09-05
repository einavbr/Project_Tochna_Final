# Python implementation of the normalized spectral clustering algorithm
import sys
import numpy as np
import pandas as pd
import math

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
        # TODO : The nodes are provided from the input therefor no arguments (use FILENAME). 
        # A graph contains:
        # dimensions - the number of coordinates in a single point
        # size field - the number of nodes in the graph
        # V - A list of the vertexes in the graph
        # W - Weighted Adjacency Matrix (A matrix is a 2D list), weight = 0 means no edge
        # Droot - Diagonal Degree Matrix ^ -0.5
        data_frame = pd.read_csv(FILE_NAME)
        self.vertexes = data_frame.to_numpy()
        self.dimensions = len(data_frame.columns)
        self.size = len(data_frame.index)
        self.weighted_mat = get_weighted_adjacency_matrix(self.size, self.vertexes)
        self.diag_degree_root_mat = get_Droot_matrix(self.size, self.weighted_mat)

def get_weighted_adjacency_matrix(n, vertexes):
    # TODO : given a list of nodes, this fuction should return a matrix of weights 
    # for each pair of nodes
    # NOTE : remember this is a symetric matrix, we only need to compute for i<j
    # and put the result in 2 slots
    weighted_mat = np.zeros([n, n], dtype = np.double)
    for i in range(n):
        for j in range(i+1, n):
            norm = calc_euclidean_norm(vertexes[i], vertexes[j])
            weighted_mat[i][j] = weighted_mat[j][i] = math.exp(-1 * (norm/2))
    return weighted_mat

def get_Droot_matrix(n, weighted_mat):
    # TODO : given a list of nodes, this fuction should return the Diagonal Degree Matrix ^ -0.5 
    diag_degree_root_mat = np.zeros([n, n], dtype = np.double)
    for i in range(n):
        for z in range(n):
            diag_degree_root_mat[i][i] += weighted_mat[i][z]
    for i in range(n):
        diag_degree_root_mat[i][i] = 1 / math.sqrt(diag_degree_root_mat[i][i])
    return diag_degree_root_mat

def calc_euclidean_norm(a, b):
    # TODO : given two points(list of coordinates from relevant the Node field), 
    # this function calculates the euclidean norm 
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2
    return math.sqrt(sum)

def get_frobenius_norm(A):
    # TODO : given a matrix, 
    # this function calculates the euclidean norm 
    pass

# Functions on Matrices:
def print_matrix(A, message = ''):
    # TODO : print the matrix, take print func from PT1 or PT2
    print(message)
    np.set_printoptions(precision=4, suppress=True)
    print(A)
    print('\n')

def multiply_matrices(A , B):
    # TODO : given 2 matrixes (2D np lists), perform matrix multiplication A x B 
    return np.matmul(A, B) 

def transpose_matrix(A):
    # TODO : given a matrix, transpose it
    return A.transpose

def is_diagonal(A):
    # TODO : given a matrix, this funtion returns True if the matrix 
    # is diagonal and False otherwise
    # check using Convergence. Use "calc_frobenius_norm"
    pass

def get_pivot(A):
    # TODO : given a matrix, return the off-diagonal element with the largest absolute value
    max = 0
    max_i = -1
    max_j = -1
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i != j:
                if A[i][j] > max:
                    max = A[i][j]
                    max_i = i
                    max_j = j
    return max_i, max_j, max

def obtain_c_t(A_ii, A_jj, A_ij):
    # TODO : given a pivot value, return c and t as explained in the project
    print(f'A_ij = {A_ij}, A_ii = {A_ii}, A_jj = {A_jj}')
    theta = (A_jj - A_ii) / A_ij
    sign = np.sign(theta)
    if sign == 0: 
        sign = 1
    print(f'theta = {theta}, sign = {sign}')
    t = sign / (abs(theta)+math.sqrt(theta**2 + 1))
    c = 1 / math.sqrt(t**2 + 1)
    return c, t

def get_rotation_mat(pivot_i, pivot_j, A_ij, A):
    print(f'pivot_i = {pivot_i}, pivot_j = {pivot_j}')
    c, t = obtain_c_t(A[pivot_i][pivot_i], A[pivot_j][pivot_j], A_ij)
    s = c * t
    print(f'c = {c}, t = {t}, s = {s}')
    rotation_mat = np.zeros(A.shape, dtype = np.double)
    for i in range(len(A)):
        if i == pivot_i or i == pivot_j:
            rotation_mat[i][i] = c
        else:
            rotation_mat[i][i] = 1
    rotation_mat[pivot_i][pivot_j] = s
    rotation_mat[pivot_j][pivot_i] = -s
    return rotation_mat


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
    n = graph.size
    id_mat = np.zeros([n, n], dtype = np.double)
    for i in range(n):
        id_mat[i][i] = np.double(1)

    temp_mat_1 = multiply_matrices(graph.diag_degree_root_mat, graph.weighted_mat)
    temp_mat_2 = multiply_matrices(temp_mat_1, graph.diag_degree_root_mat)
    laplacian_mat = np.zeros([n, n], dtype = np.double)
    for i in range(n):
        for j in range(n):
            laplacian_mat[i][j] = id_mat[i][j] - temp_mat_2[i][j]
    
    if print:
        print_matrix(laplacian_mat, 'laplacian matrix:')
    
    return laplacian_mat

def run_jacobi_flow(A, print):
    # TODO : Calculate and output the eigenvalues and eigenvectors as described in 1.2.1.
    # A is the matrix whos eigenvalues and eigenvectors we want
    # The function should print appropriate output if print == True
    # return ORDERED eigenvalues and eigenvectors
    iterations = 0
    print_matrix(A, 'laplacian mat in jacobi flow')
    P = np.ones(A.shape)
    V = P  # V is the multiplication matrix of all the P matrices.
    
    # Repeat a,b until A is diagonal matrix
    while not is_diagonal(A):
        if iterations >= 100:
            break
    
        #   1. calc new P. Use "get_rotation_mat"
        pivot_i, pivot_j, pivot = get_pivot(A)
        P = get_rotation_mat(pivot_i, pivot_j, pivot, A)
        print_matrix(P, 'rotation matrix')
        
        #   2. A = A'
        temp_mat = multiply_matrices(transpose_matrix(P), A)
        A = multiply_matrices(temp_mat, P)
    
        #   3. V = V x P
        V = multiply_matrices(V, P)
    
    eigenvalues = {}
    for i in range(len(A)):
        eigenvalues.append(A[i][i])
    
    # Once done, the values on the diagonal of A are eigenvalues
    # and the columns of V are eigenvectors
    # This fuction should return an ORDERED list of eigenvalues, 
    # and a matrix with the eigenvetors in the corresponding columns

def run_spk_flow():
    # TODO : Perform full spectral kmeans as described in 1.
    # The function should print appropriate output

    laplacian = run_lnorm_flow(True)
    run_jacobi_flow(laplacian, False)
    # eigenvalues, eigenvectors = run_jacobi_flow(laplacian, False)
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

print_matrix(graph.weighted_mat)
print_matrix(graph.diag_degree_root_mat)

try:
    GOAL = sys.argv[2]
except Exception:
    print(INVALID_INPUT)

if GOAL == 'spk':
    run_spk_flow()


# elif GOAL == 'wam':
#     print_matrix(graph.W)
# elif GOAL == 'ddg':
#     print_matrix(graph.D)
# elif GOAL == 'lnorm':
#     run_lnorm_flow(True)
# elif GOAL == 'jacobi':
#     run_jacobi_flow(graph.W, True)
# else:
#     raise Exception(INVALID_INPUT)

