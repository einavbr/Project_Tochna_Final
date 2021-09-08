from numpy.lib.twodim_base import diag
import spkmeansmodule as spkmeans
import numpy as np
import sys
import math

INVALID_INPUT = 'Invalid Input!'
ERROR_OCCURED = 'An Error Has Occured'
MAX_ITER_KMEANS = 300

# Functions on Matrices:
def print_matrix(A, message = ''):
    # TODO : print the matrix, take print func from PT1 or PT2
    print(message)
    np.set_printoptions(precision=4, suppress=True)
    print(A)
    print('\n')

#finding centroids indexes 
def calc_centroids_indexes(k, all_points, point_size, N, max_iter):
    Z = 1
    point_index = 0
    iternum = 0
    np.random.seed(0)
    indexes = [0 for i in range(k)]
    indexes[0] = np.random.choice(N)
    while (Z < k):
        allProbs = [0 for i in range(N)]
        while (point_index < N):
            currpoint = all_points[point_index]
            distances = math.inf
            for centroid_index in range(Z):
                currcentroid = all_points[indexes[centroid_index]]
                D = 0.0
                for c in range(point_size):
                    D += (currpoint[c]-currcentroid[c])**2
                if D < distances:
                    distances = D
            allProbs[point_index] = distances
            point_index +=1
        sumofProbs = sum(allProbs)
        for i in range(N):
            allProbs[i] = allProbs[i]/sumofProbs
        indexes[Z] = np.random.choice(N, 1, p=allProbs)[0]
        Z+=1
        point_index = 0
        iternum +=1
        if (iternum > max_iter):
            print ("max iteration numer reached")
            break
    return indexes

#finding centroids from indexes
def indexes_to_centroids(all_points, Centroids_Indexes,k):
    res = []
    for index in Centroids_Indexes:
        res.append(all_points[index])
    return res

# MAIN
try:
    K = int(sys.argv[1])
    print(f'k in python is: {K}')
except Exception:
    print (INVALID_INPUT)
if K < 0:
    raise Exception(INVALID_INPUT)
# NOTE : if k == 0 - use heuristic

try:
    FILE_NAME = sys.argv[3]
    print(f'file_name in python is: {FILE_NAME}')
except Exception:
    print (INVALID_INPUT)

try:
    GOAL = sys.argv[2]
except Exception:
    print(INVALID_INPUT)

if GOAL == 'wam':
    wieghted_mat = spkmeans.pythonRunWamFlow(str(K), FILE_NAME)
    print('print wam in python')
    print_matrix(np.array(wieghted_mat))
elif GOAL == 'ddg':
    diag_degree_arr = spkmeans.pythonRunDdgFlow(str(K), FILE_NAME)
    print('print ddg in python')
    N = len(diag_degree_arr)
    diag_degree_mat = np.zeros((N, N))
    for i in range(N):
        diag_degree_mat[i][i] = diag_degree_arr[i]
    print_matrix(diag_degree_mat)
elif GOAL == 'lnorm':
    laplacian_mat = spkmeans.pythonRunLnormFlow(str(K), FILE_NAME)
    print('print lnorm in python')
    print_matrix(np.array(laplacian_mat))
elif GOAL == 'jacobi':
    jacobi_mat = spkmeans.pythonRunJacobiFlow(str(K), FILE_NAME)
    print_matrix(np.array(jacobi_mat))
elif GOAL == 'spk':
    K, DIM, N, T = spkmeans.pythonRunSpkFlow(str(K), FILE_NAME)
    print(K)
    print(T)
    centroids_indexes = calc_centroids_indexes(K, T, DIM, N, MAX_ITER_KMEANS)
    centroids = indexes_to_centroids(T, centroids_indexes, K)
    print(f'centroids in python is: {centroids}')
    centroids_array = spkmeans.pythonRunkmeanspp(T, centroids, K, N, DIM)
    print_matrix(np.array(centroids_array))
