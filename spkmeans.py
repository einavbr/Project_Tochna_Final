from numpy.lib.twodim_base import diag
import spkmeansmodule as spkmeans
import numpy as np
import sys

INVALID_INPUT = 'Invalid Input!'
ERROR_OCCURED = 'An Error Has Occured'

# Functions on Matrices:
def print_matrix(A, message = ''):
    # TODO : print the matrix, take print func from PT1 or PT2
    print(message)
    np.set_printoptions(precision=4, suppress=True)
    print(A)
    print('\n')

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
    centroids_array = spkmeans.pythonRunSpkFlow(str(K), FILE_NAME)
    print_matrix(np.array(centroids_array))

