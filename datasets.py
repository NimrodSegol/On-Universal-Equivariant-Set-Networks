import torch
import torch.utils.data as data
import numpy as np
import utils as utils


class knapsack_dataset(data.Dataset):
    def __init__(self, path, train=True, graph=False):
        super().__init__()
        self.graph = graph
        if train:
            filename = path + '/dataset_4feat.npy'
            self.mats = np.load(path + '/dataset_4feat_mat.npy')
        else:
            filename = path + '/test_dataset_4feat.npy'
            self.mats = np.load(path + '/test_dataset_4feat_mat.npy')

        x = np.load(filename)
        num_features=4
        self.values = x[:, :, :num_features].astype(np.float32)
        self.values = (self.values - self.values.min()) / (self.values.max() - self.values.min())
        self.labels = x[:, :, num_features].astype(np.long)

    def __getitem__(self, index):
        if self.graph:
            return torch.from_numpy(self.values[index, :, :]), torch.from_numpy(self.mats[index]),\
                   torch.from_numpy(self.labels[index, :])
        else:
            return torch.from_numpy(self.values[index, :, :]), torch.from_numpy(self.labels[index, :])

    def __len__(self):
        return self.values.shape[0]


class modelnet_regression_dataset(data.Dataset):
    def __init__(self, path, num_points=512, train=True, graph=False):
        super().__init__()
        self.num_points = num_points
        if train:
            self.point_clouds = np.load(path+'/point_clouds.npy')
            self.labels = np.load(path+'/labels.npy')
            self.mats = np.load(path+'/mats.npy')
        else:
            self.point_clouds = np.load(path + '/test_point_clouds.npy')
            self.labels = np.load(path + '/test_labels.npy')
            self.mats = np.load(path + '/test_mats.npy')
        self.graph = graph

    def __getitem__(self, idx):

        current_points = torch.from_numpy(self.point_clouds[idx, :])
        label = torch.from_numpy(self.labels[idx, :]).float()
        mats = torch.from_numpy(self.mats[idx])
        if self.graph:
            return current_points, mats, label
        else:
            return current_points, label

    def __len__(self):
        return self.point_clouds.shape[0]


class Equidataset(data.Dataset):
    def __init__(self, x, y, graph=False, k=10):
        super().__init__()
        self.x = x
        self.y = y
        self.graph = graph
        self.mats = utils.make_batch_of_adj_matrices(x, k)

    def __getitem__(self, idx):
        if self.graph:
            return self.x[idx], self.mats[idx], self.y[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]

