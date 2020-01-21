import numpy as np
import os
import torch
import datetime
import csv
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix

now = lambda: datetime.datetime.now()


def create_dir_if_not_exist(dirpath, verbose=True):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        if verbose:
            print('created dir at', dirpath)


class Logger(object):
    def __init__(self, logfilepath, cols):
        self.logfilepath = logfilepath
        self.cols = cols
        self.base_cols = ['date']

        self.add2csv(self.logfilepath, self.base_cols + self.cols)

    def add2csv(self, csvpath, fields):
        with open(csvpath, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def assert_row(self, row_dict):
        a = set(self.cols)
        b = set(row_dict.keys())
        if a != b:
            raise Exception('No such keys:', b - a)

    def log(self, row_dict, verbose=False):
        self.assert_row(row_dict)
        fields = [now()] + [row_dict[k] for k in self.cols]
        self.add2csv(self.logfilepath, fields)

        if verbose:
            print('logged', datetime.datetime.now(), 'gpu (%s)' % (os.environ['CUDA_VISIBLE_DEVICES']), row_dict)


def knn_adj(points, k=5):
    n = points.shape[0]
    dists = cdist(points, points)
    idx = np.argpartition(dists, range(k), axis=0)[1:k]
    A = -coo_matrix((np.ones(n*(k-1)), (np.arange(n).repeat(k-1), np.hstack(idx.transpose()))), shape=(n, n))
    A = A + A.transpose()
    A = coo_matrix((np.ones(np.nonzero(A)[0].shape[0]), (np.nonzero(A)[0], np.nonzero(A)[1])), shape=(n, n))
    A += coo_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
    A = A.toarray()
    A0 = A.sum(0)
    A = A/np.sqrt(A.sum(1))
    A = A / np.sqrt(A0)
    return A.astype(np.float)


def make_batch_of_adj_matrices(points, k=5):
    A = knn_adj(points[0], k)
    A = torch.from_numpy(A).float().unsqueeze(0)
    for i in range(1, points.shape[0]):
        B = torch.from_numpy(knn_adj(points[i], k)).float().unsqueeze(0)
        A = torch.cat((A, B), dim=0)
    return A

