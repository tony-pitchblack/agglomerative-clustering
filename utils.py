from math import sqrt
import numpy as np


def _single(d_xi, d_yi, d_xy,
            size_x, size_y, size_i):
    return min(d_xi, d_yi)


def _complete(d_xi, d_yi, d_xy,
              size_x, size_y, size_i):
    return max(d_xi, d_yi)


def _average(d_xi, d_yi, d_xy,
             size_x, size_y, size_i):
    return (size_x * d_xi + size_y * d_yi) / (size_x + size_y)


def _ward(d_xi, d_yi, d_xy,
          size_x, size_y, size_i):
    t = 1.0 / (size_x + size_y + size_i)
    return sqrt((size_i + size_x) * t * d_xi * d_xi +
                (size_i + size_y) * t * d_yi * d_yi -
                size_i * t * d_xy * d_xy)


linkage_methods = {
    'single': _single,
    'complete': _complete,
    'average': _average,
    'ward': _ward
}


def _euclid(x, y):
    ss = ((x - y) ** 2).sum()
    return np.sqrt(ss)


metric_functions = {
    'euclidean': _euclid,
    'precomputed': None
}


def compute_distances(X, metric_fn):
    """Compute lower triangular distance matrix"""
    N = X.shape[0]
    dist_matrix = np.full((N, N), fill_value=np.inf)
    for i in range(N):
        for j in range(i):
            dist = metric_fn(X[i], X[j])
            dist_matrix[i, j] = dist
    return dist_matrix


def tril_idx(i, j):
    """Convert i, j to indices suitable for lower triangular matrix"""
    if i < j:
        return j, i
    else:
        return i, j
