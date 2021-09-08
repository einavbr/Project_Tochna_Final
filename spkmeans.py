from numpy.lib.twodim_base import diag
import spkmeansmodule as spkmeans
import numpy as np
import sys
import math

INVALID_INPUT = 'Invalid Input!'
ERROR_OCCURED = 'An Error Has Occured'
MAX_ITER_KMEANS = 300

def print_list(lst):
    str_list = str(lst)
    print(str_list.replace(' ','').replace('[','').replace(']',''))

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
def main():
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

    try:
        GOAL = sys.argv[2]
    except Exception:
        print(INVALID_INPUT)

    if GOAL == 'wam':
        spkmeans.pythonRunWamFlow(str(K), FILE_NAME)
    elif GOAL == 'ddg':
        spkmeans.pythonRunDdgFlow(str(K), FILE_NAME)
    elif GOAL == 'lnorm':
        spkmeans.pythonRunLnormFlow(str(K), FILE_NAME)
    elif GOAL == 'jacobi':
        spkmeans.pythonRunJacobiFlow(str(K), FILE_NAME)
    elif GOAL == 'spk':
        K, DIM, N, T = spkmeans.pythonRunSpkFlow(str(K), FILE_NAME)
        centroids_indexes = calc_centroids_indexes(K, T, DIM, N, MAX_ITER_KMEANS)
        print_list(centroids_indexes)
        centroids = indexes_to_centroids(T, centroids_indexes, K)
        spkmeans.pythonRunkmeanspp(T, centroids, K, N, DIM)

# call main
main()