# This file should conatain a test func for each func in spkmeans.py
import sys
import numpy as np
import spkmeans

def test_multiply_matrices():
    A = np.array([[1,2,3],[4,5,6]])
    B = np.array([[3,1],[5,6],[7,8]])
    product = spkmeans.multiply_matrices(A, B)
    np.set_printoptions(precision=4, suppress=True)
    print(product)
    print('\n')

def test_get_pivot():
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(spkmeans.get_pivot(A))

    
# test_multiply_matrices()
test_get_pivot()