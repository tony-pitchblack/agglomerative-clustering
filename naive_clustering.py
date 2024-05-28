import numpy as np
from numpy import ma
import pandas as pd

from utils import linkage_methods, metric_functions
from utils import compute_distances, tril_idx


class NaiveClustering():
    def __init__(self, n_clusters=2, metric='euclidean', linkage='average'):
        self.n_clusters = n_clusters

        try:
            metric_func = metric_functions[metric]
        except KeyError as e:
            raise ValueError(
                f"Unknown metric."
            ) from e

        try:
            join_func = linkage_methods[linkage]
        except KeyError as e:
            raise ValueError(
                f"Unknown linkage. Should be one of {linkage_methods.keys()}"
            ) from e
        if linkage == 'ward' and metric != 'euclidean':
            raise ValueError(
                f"Only euclidean disance is availible for ward's method."
            )

        self._metric_func = metric_func
        self._join_func = join_func

        self.distances_ = None
        self.children_ = None

    def fit(self, X):
        if type(X) == pd.core.frame.DataFrame:
            X = X.to_numpy()

        if self._metric_func != None:
            D = compute_distances(X, self._metric_func)
        else:
            D = X

        N = D.shape[0]
        size = np.ones(N)

        self.children_ = np.ndarray((N-1, 2), dtype='int')
        self.distances_ = np.ndarray((N-1))
        self.labels_ = np.arange(N)

        agglomerative_schedule = []
        idx_to_label = np.arange(N, dtype=int)

        mask = np.zeros_like(D)
        D_masked = ma.masked_array(D, mask)
        D_masked.fill_value = np.inf
        np.fill_diagonal(D_masked.mask, 1)

        new_node_label = N
        for k in range(N-1):
            ind_min_flat = np.argmin(D_masked)
            ind_min = np.unravel_index(ind_min_flat, D_masked.shape)
            y, x = ind_min

            if x > y:
                x, y = y, x

            agglomerative_schedule.append(
                (y+1, x+1, D_masked[ind_min], size[y]))
            self.distances_[k] = D_masked[ind_min]

            self.children_[new_node_label - N,
                           :] = [idx_to_label[x], idx_to_label[y]]
            idx_to_label[y] = new_node_label
            new_node_label += 1

            size[y] += size[x]
            size[x] = 0

            for z in range(N):
                if size[z] == 0 or z == y:
                    continue
                # D[z, y] = self._join_func(
                D_masked[tril_idx(z, y)] = self._join_func(
                    D[tril_idx(z, x)], D[tril_idx(z, y)], D[tril_idx(x, y)],
                    size[x], size[y], size[z]
                )

            D_masked[:, [x]] = ma.masked
            D_masked[[x], :] = ma.masked

            cluster_count = N - k - 1
            if cluster_count == self.n_clusters:
                self.labels_ = np.where(self.labels_ == x, y, x)

        self.agglomerative_schedule = agglomerative_schedule

        return self
