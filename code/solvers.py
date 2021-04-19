'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye ,csr_matrix
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt




def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    N = A.shape[1]
    x = np.zeros((N, ))
    x = inv(A.T@A)@A.T@b
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)

    lu_obj = splu(A.T@A,permc_spec='NATURAL')
    U = lu_obj.U
    x = lu_obj.solve(A.T@b)
    
    return x, U

def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)

    lu_obj = splu(A.T@A,permc_spec='COLAMD')
    U = lu_obj.U
    x = lu_obj.solve(A.T@b)

    return x, U

def custom_lu(A,b):
    N = A.shape[1]
    # print(np.shape(A))
    x = np.zeros((N, ))
    U = eye(N)
    
    # lu_obj = splu(A.T@A,permc_spec='COLAMD')
    lu_obj = splu(A.T@A,permc_spec='NATURAL')

    L = lu_obj.L
    U = lu_obj.U
    perm_r = lu_obj.perm_r
    perm_c = lu_obj.perm_c
    
    b_new = A.T@b
    if perm_r is not None:
        print("row permutated")
        b_old = b_new.copy()
        for old_index, new_index in enumerate(perm_r):
            b_new[new_index] = b_old[old_index]

    try:            # unit_diagonal only for version 1.4 and newer
        y = spsolve_triangular(L.tocsr(), b_new, lower=True, unit_diagonal=True)
    except TypeError:
        y = spsolve_triangular(L.tocsr(), b_new, lower=True)
    x = spsolve_triangular(U.tocsr(), y, lower=False)
    if perm_c is None:
        return x , U
    print("col permutated")
    return x[perm_c] , U



def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)

    Z,R,E,rank = rz(A,b,permc_spec = 'NATURAL')
    # print(rank)
    x = spsolve_triangular(csr_matrix(R),Z,lower=False)

    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2,
    #       with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)

    Z,R,E,rank = rz(A,b,permc_spec = 'COLAMD')
    # print(rank)
    R = R.tocsr()
    Z = Z.flatten()
    x = permutation_vector_to_matrix(E) @ spsolve_triangular(R, Z, lower=False)
    # x = spsolve_triangular(csr_matrix(R@permutation_vector_to_matrix(E).T),Z,lower=False)

    return x, R


def solve(A, b, method='default'):
    '''
    param A (M, N) Jacobian matirx
    param b (M, 1) residual vector
    return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
        'custom_lu': custom_lu,
    }

    return fn_map[method](A, b)
