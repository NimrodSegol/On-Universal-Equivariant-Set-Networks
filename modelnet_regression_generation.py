from Modelnet_data.MODELNETloader import ModelNet40Cls
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
import numpy as np
import utils as utils
from scipy.sparse.linalg import eigsh
import sys
import os


def eigenvector(points):
    n = points.shape[0]
    k = 10
    dists = cdist(points, points)
    idx = np.argpartition(dists, range(k), axis=0)[1:k]
    A = -coo_matrix((np.ones(n*(k-1)), (np.arange(n).repeat(k-1), np.hstack(idx.transpose()))), shape=(n, n))
    A = A + A.transpose()
    A = -coo_matrix((np.ones(np.nonzero(A)[0].shape[0]), (np.nonzero(A)[0], np.nonzero(A)[1])), shape=(n, n))
    A -= coo_matrix((A@np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
    eigs, vecs =eigsh(A, k=4, which='SM')
    for i, eig in enumerate(eigs):
        if eig > 1e-5:
            break
    v = vecs[:, i]
    n = np.sqrt((v**2).sum())
    v = v/n
    v = np.abs(v)
    return v


def generate_regression_dataset(base_dir, MODELNET_FOLDER='modelnet40_ply_hdf5_2048'):
    n= 512
    dataset = ModelNet40Cls(num_points=n, folder=MODELNET_FOLDER,
                            base_dir=base_dir,
                            aug_transforms=None, train=True, download=True)
    labels = np.zeros([len(dataset), n])
    matrices = np.zeros([len(dataset), n, n])
    for i, P in enumerate(dataset.points):
        labels[i, :] = eigenvector(P[:n])
        matrices[i] = utils.knn_adj(P[:n], k=10)
        if i % 10 == 0:
            print(i)
    dataset_path = os.path.join(base_dir, 'modelnet_regression_dataset')
    np.save(os.path.join(dataset_path, 'points_clouds'), dataset.points[:, :n, :])
    np.save(os.path.join(dataset_path, 'labels'), labels)
    np.save(os.path.join(dataset_path, 'math'), matrices)

    test_dataset = ModelNet40Cls(num_points=n, folder=MODELNET_FOLDER, base_dir=base_dir,
                                 aug_transforms=None, train=False)
    test_labels = np.zeros([len(test_dataset), n])
    test_matrices = np.zeros([len(test_dataset), n, n])
    for i, P in enumerate(test_dataset.points):
        test_labels[i, :] = eigenvector(P[:n])
        test_matrices[i] = utils.knn_adj(P[:n], k=10)
        if i % 10 == 0:
            print(i)
    np.save(os.path.join(dataset_path, 'test_points_clouds'), test_dataset.points[:, :n, :])
    np.save(os.path.join(dataset_path, 'test_labels'), test_labels)
    np.save(os.path.join(dataset_path, 'test_mats'), test_matrices)


if __name__ == "__main__":
    base_dir = sys.argv[1]
    generate_regression_dataset(base_dir)
