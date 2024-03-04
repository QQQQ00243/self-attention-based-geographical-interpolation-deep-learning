import os
import torch
import numpy as np

from collections import OrderedDict

from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
    

class Site:
    def __init__(self, path: str, n_bhs: int):
        assert (n_bhs > 0), "number of boreholes must be greater than 0."
        self.path = path
        self.n_bhs = n_bhs
        data = self.load_data(path)
        self.data = data
        self.train_js = []

    @staticmethod
    def load_data(path: str):
        if path.endswith("csv"):
            data = np.genfromtxt(path, delimiter=',') - 1
        elif path.endswith("npy"):
            data = np.load(path)
        return data
    
    def train_test_split(self, normalize, return_tensor=False):
        data = self.data
        js = list(range(data.shape[1])) # indices on horizontal axis of the site
        
        if self.n_bhs == 1:
            train_js = [len(js)//2]
        else:
            train_js = [js[0]] 
            step = (js[-1] - js[0]) // (self.n_bhs - 1)
            for i in range(1, self.n_bhs - 1):
                element = js[0] + (i * step)
                train_js.append(element)
            train_js.append(js[-1])

            '''
            spacing = (len(js) - 2) // (self.n_bhs - 1) # spacing of boreholes
            # train_js = js[spacing//2::spacing] # horizontal indices for training
            train_js = js[1:len(js)-spacing:spacing]
            train_js.append(len(js) - 2)
            '''
        self.train_js = train_js
        pos_train = np.empty((len(train_js)*data.shape[0], 2))
        cls_train = np.empty((pos_train.shape[0]))
        pos_test = np.empty(((data.shape[1] - len(train_js))*data.shape[0], 2))
        cls_test = np.empty((pos_test.shape[0]))

        train_i, test_i = 0, 0
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):
                if j in train_js:
                    pos_train[train_i, 0] = i
                    pos_train[train_i, 1] = j
                    cls_train[train_i] = data[i, j]
                    train_i += 1
                else:
                    pos_test[test_i, 0] = i
                    pos_test[test_i, 1] = j
                    cls_test[test_i] = data[i, j]
                    test_i += 1
        
        # normalize coordinates
        if normalize:
            pos_train = pos_train / np.array(data.shape)
            pos_test = pos_test / np.array(data.shape)

        # cast numpy array to tensor if `return_tensor` is True
        if return_tensor:
            pos_train = torch.tensor(pos_train).float()
            cls_train = torch.tensor(cls_train).long()
            pos_test = torch.tensor(pos_test).float()
            cls_test = torch.tensor(cls_test).long()

        return pos_train, cls_train, pos_test, cls_test


class SiteDataset:
    def __init__(
            self, 
            path: str, 
            n_bhs: int, 
            n_cls: int, 
            train: bool=True, 
            shuffle_site=True, 
            normalize=True
        ):
        self.site = Site(path=path, n_bhs=n_bhs)
        self.n_cls = n_cls
        self.pos_train, self.cls_train, self.pos_test, self.cls_test \
            = self.site.train_test_split(normalize=normalize, return_tensor=True)

        self.train = train
        self.shuffle_site = shuffle_site
        self.normalize = normalize
        
    def __len__(self):
        return self.pos_train.size(0) if self.train else self.pos_test.size(0)
    
    def __getitem__(self, idx):
        if self.train:
            # drop trg item in site for training
            pos_train = torch.concat([self.pos_train[:idx], self.pos_train[idx+1:]], dim=0)
            cls_train = torch.concat([self.cls_train[:idx], self.cls_train[idx+1:]], dim=0)
        else:
            pos_train = torch.clone(self.pos_train)
            cls_train = torch.clone(self.cls_train)

        # shuffle items if `self.shuffle_site` is True
        if self.shuffle_site:
            mask = torch.randperm(pos_train.size(0))
        else:
            mask = torch.arange(0, pos_train.size(0), 1)
        pos_train = pos_train[mask]
        cls_train = cls_train[mask]

        if self.train:
            return {
                "net_input": {
                    "site_coords": pos_train, 
                    "site_cls": cls_train,
                    "trg_coord": self.pos_train[idx],
                },
                "target": self.cls_train[idx],
            }
        else:
            return {
                "net_input": {
                    "site_coords": pos_train, 
                    "site_cls": cls_train,
                    "trg_coord": self.pos_test[idx],
                },
                "target": self.cls_test[idx],
            }


class BaseDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    

class KeyDataset(BaseDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][self.key]
     

class DistanceDataset(BaseDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __getitem__(self, idx):
        site_coords, site_cls, trg_coord, trg_cls = self.dataset[idx]
        coords = torch.cat([trg_coord.unsqueeze(0), site_coords], dim=0).numpy()
        dist = distance_matrix(coords, coords).astype(np.float32)
        return torch.from_numpy(dist)


class EdgeTypeDataset(BaseDataset):
    def __init__(self, dataset: SiteDataset):
        self.dataset = dataset
        self.n_cls = self.dataset.n_cls + 1

    def __getitem__(self, idx):
        site_coords, site_cls, trg_coord, trg_cls = self.dataset[idx]

        # The site has `n_cls` classes and the class is encoded as 0, 1, ..., n_cls-1.
        # Since the class of target is unknown during training, the trg_cls is set to -1
        trg_cls = torch.tensor([-1])
        # To encode the types of edge, the encoding of class must start from 0, so 1 is added to cls
        cls = torch.cat([trg_cls, site_cls]) + 1

        # Edge type encoding = class1 * n_cls + class2
        edge_type = cls.view(-1, 1) * self.n_cls + cls.view(1, -1)
        return edge_type
    

class SiteDatasetWithPair:
    def __init__(
            self, 
            path: str, 
            n_bhs: int, 
            n_cls: int, 
            train: bool=True, 
            shuffle_site=True, 
            normalize=True
        ):
        self.site = Site(path=path, n_bhs=n_bhs)
        self.n_cls = n_cls
        self.pos_train, self.cls_train, self.pos_test, self.cls_test \
            = self.site.train_test_split(normalize=normalize, return_tensor=True)

        self.train = train
        self.shuffle_site = shuffle_site
        self.normalize = normalize
        
    def __len__(self):
        return self.pos_train.size(0) if self.train else self.pos_test.size(0)
    
    def __getitem__(self, idx):
        # ------- drop trg item in site for training ------- #
        if self.train:
            pos_train = torch.concat([self.pos_train[:idx], self.pos_train[idx+1:]], dim=0)
            cls_train = torch.concat([self.cls_train[:idx], self.cls_train[idx+1:]], dim=0)
        else:
            pos_train = torch.clone(self.pos_train)
            cls_train = torch.clone(self.cls_train)

        # ------- shuffle items if `self.shuffle_site` is True ------- #
        if self.shuffle_site:
            mask = torch.randperm(pos_train.size(0))
        else:
            mask = torch.arange(0, pos_train.size(0), 1)
        pos_train = pos_train[mask]
        cls_train = cls_train[mask]

        trg_coord = self.pos_train[idx] if self.train else self.pos_test[idx]

        # ------- compute distance matrix ------- #
        coords = torch.concat((trg_coord.unsqueeze(0), pos_train), dim=0).numpy()
        distance = distance_matrix(coords, coords).astype(np.float32)
        distance = torch.tensor(distance)

        # ------- compute edge type matrix ------- #
        # The site has `n_cls` classes and the class is encoded as 0, 1, ..., n_cls-1.
        # Since the class of target is unknown during training, the trg_cls is set to -1
        trg_cls = torch.tensor([-1])
        # To encode the types of edge, the encoding of class must start from 0, so 1 is added to cls
        cls = torch.cat([trg_cls, cls_train]) + 1
        # Edge type encoding = class1 * n_cls + class2
        edge_type = cls.view(-1, 1) * self.n_cls + cls.view(1, -1)

        if self.train:
            return {
                "net_input": {
                    "site_coords": pos_train, 
                    "site_cls": cls_train,
                    "trg_coord": trg_coord,
                    "distance": distance,
                    "edge_type": edge_type,
                },
                "target": self.cls_train[idx],
            }
        else:
            return {
                "net_input": {
                    "site_coords": pos_train, 
                    "site_cls": cls_train,
                    "trg_coord": trg_coord,
                    "distance": distance,
                    "edge_type": edge_type,
                },
                "target": self.cls_test[idx],
            }



if __name__ == "__main__":
    dataset = SiteDatasetWithPair(
        path="/nvme/qiyahang/GIT/dataset/TI_reduced_final.csv",
        n_bhs=6,
        n_cls=4,
        train=True,
    )
    for i in dataset:
        print(i)
        break
