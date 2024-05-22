import pandas as pd
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class AgglomerativeClustering():
    def __init__(self, n_clusters=2, metric='euclidean', linkage='single'):
        self.n_clusters = n_clusters

        metric_choices = {
            'euclidean': lambda a, b: np.sqrt(((a-b) ** 2).sum()),
            'precomputed': None
        }
        try:
            metric_func = metric_choices[metric]
        except KeyError as e:
            raise ValueError(
                f"Unknown metric."
            ) from e
        self._metric_func = metric_func

        def ward_join(p, q, D):
            N_p = np.count_nonzero(self.labels_ == p)
            N_q = np.count_nonzero(self.labels_ == q)
            D_t = np.ndarray((D.shape[0]))  # new array
            assert D[p][q] == np.min(D)

            non_masked = np.nonzero(~(D.mask[p] | D.mask[q]))
            for r in list(non_masked[0]):
                N_r = np.count_nonzero(self.labels_ == r)
                D_t[r] = np.sqrt(((N_r + N_p) * D[p][r] ** 2
                                  + (N_r + N_q) * D[q][r] ** 2
                                  - N_r * D[p][q] ** 2) / (N_p + N_q + N_r))
                pass

            return (D_t)

        linkage_choices = {
            'single': lambda i, j, D: np.minimum(D[i], D[j]),
            'average': lambda i, j, D: (D[i]+D[j])/2,
            'complete': lambda i, j, D: np.maximum(D[i], D[j]),
            'ward': ward_join
        }
        try:
            join_func = linkage_choices[linkage]
        except KeyError as e:
            raise ValueError(
                f"Unknown linkage. Should be one of {linkage_choices.keys()}"
            ) from e
        if linkage == 'ward' and metric != 'euclidean':
            raise ValueError(
                f"Only euclidean disance is availible for ward's method."
            )
        self._join_func = join_func
        self._linkage = linkage

        self.distance_matrix = None
        self.children_ = None

    def __compute_distance_matrix(self, X):
        N = X.shape[0]
        matrix = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                dist = self._metric_func(X[i], X[j])
                matrix[i, j] = dist
        self.distance_matrix = matrix

    def fit(self, X):
        if type(X) == pd.core.frame.DataFrame:
            X = X.to_numpy()

        if self._metric_func != None:
            self.__compute_distance_matrix(X)
        else:
            self.distance_matrix = X

        D = self.distance_matrix.copy()
        N = D.shape[0]

        agglomerative_schedule = []

        elem_count = np.zeros((N, 1), dtype='int')
        clust_map = np.arange(N, dtype=int)

        self.children_ = np.ndarray((N-1, 2), dtype='int')
        self.distances_ = np.ndarray((N-1))
        self.labels_ = np.arange(N)

        mask = np.zeros_like(D)
        D_masked = ma.masked_array(D, mask)
        D_masked.fill_value = np.inf
        np.fill_diagonal(D_masked.mask, 1)

        cluster_num = N
        for k in range(N-1):
            ind_min_flat = np.argmin(D_masked)
            ind_min = np.unravel_index(ind_min_flat, D_masked.shape)
            i, j = ind_min

            new_cluster = self._join_func(i, j, D_masked)
            elem_count[i] += elem_count[j]
            elem_count[j] = 0

            agglomerative_schedule.append((i+1, j+1,
                                           D_masked[ind_min].copy(), elem_count[i]))
            self.distances_[k] = D_masked[ind_min]

            self.children_[cluster_num - N, :] = [clust_map[i], clust_map[j]]
            clust_map[i] = cluster_num
            cluster_num += 1

            D_masked[:, [j]] = ma.masked
            D_masked[[j], :] = ma.masked
            D_masked.data[:, [i]] = np.atleast_2d(new_cluster).T
            D_masked.data[[i], :] = new_cluster

            cluster_count = N - k - 1
            if cluster_count == self.n_clusters:
                self.labels_ = np.where(self.labels_ == j, i, j)

        self.agglomerative_schedule = agglomerative_schedule

        return self
