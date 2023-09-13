#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jul 26 23:08:55 2023

@author: kevinr
"""

# 4. Numpy
# 4.1 Create a 2x3 matrix A with random integers between 0 and 10, print it

import numpy as np

def create_random_matrix(r,c,min_val,max_val):
    return np.random.randint(min_val,max_val+1, size = (r,c))

def print_matrix(matrix):
    for row in matrix:
        print(row)
        
def main():
    
    # Create Matrix A
    Arow = 2
    Acol = 3
    Amin_value = 0
    Amax_value = 10
    
    matrix_A = create_random_matrix(Arow,Acol,Amin_value,Amax_value)
    print("Matrix A: ")
    print_matrix(matrix_A)
    
    # Create Matrix B
    Brow = 3
    Bcol = 4
    Bmin_value = 0
    Bmax_value = 3
    
    matrix_B = create_random_matrix(Brow,Bcol,Bmin_value,Bmax_value)
    print("\nMatrix B: ")
    print_matrix(matrix_B)
    
    # Calculate A's transpose and print it
    matrix_A_transpose = np.transpose(matrix_A)
    print("\nMatrix A's transpose: ")
    print_matrix(matrix_A_transpose)
    
    # Calculate the dot product of A and B
    matrix_dot_product = np.dot(matrix_A, matrix_B)
    print("\nDot product of A and B: ")
    print_matrix(matrix_dot_product)
    
    
    
if __name__ == "__main__":
    main()