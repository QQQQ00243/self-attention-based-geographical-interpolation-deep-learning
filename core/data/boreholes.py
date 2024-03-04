import os
import torch
import numpy as np
import pandas as pd

from typing import List


class Boreholes:
    def __init__(self, path: str, sub_depth=True):
        self.raw = None
        self.sub_depth = sub_depth
        self._process_raw(path, sub_depth)
        self._get_bhs_dict()

    @property
    def lower_bound_norm(self):
        return np.array([self.raw["x"].min(), self.raw["y"].min()])
    
    @property
    def upper_bound_norm(self):
        return np.array([self.raw["x"].max(), self.raw["y"].max()])
    
    @property
    def size_norm(self):
        return self.upper_bound_norm - self.lower_bound_norm
    
    @property
    def vertices(self):
        v = np.array(list(set(zip(self.raw["X(DISTANCE)"], self.raw["WaterDepth"]))))
        v = v[np.argsort(v[:, 0])]
        return v

    def _normalize_coords(self, coords):
        return (coords - self.lower_bound_norm) / self.size_norm

    def _process_raw(self, path: str, sub_depth: bool):
        raw = pd.read_csv(path)
        raw["x"] = raw["X(DISTANCE)"]
        if sub_depth:
            raw["y"] = raw["Z(DEPTH)"] - raw["WaterDepth"]
        else:
            raw["y"] = raw["Z(DEPTH)"]
        raw["label"] = raw["If_hydrate(1-NO;2-YES)"] - 1
        self.raw = raw
        return self.raw
    
    def _get_bhs_dict(self):
        bhs_dict = {}
        # Iterate over raw grouped by "Site"
        for site, group in self.raw.groupby('Site'):
            site_dict = {"coords": None, "labels": None, "depth": None}

            # check whether "WaterDepth" is the same in each site
            assert len(set(group["WaterDepth"].values)) == 1
            site_dict["depth"] = set(group["WaterDepth"].values).pop()
            
            # Extract coordinates and labels as NumPy arrays
            coords = group[["x", "y"]].values
            labels = group["label"].values
            
            # Assign the NumPy arrays to the nested dictionary
            site_dict["coords"] = coords
            site_dict["labels"] = labels
            
            # Add the nested dictionary to the main dictionary
            bhs_dict[site] = site_dict

        self.bhs_dict = bhs_dict
        return self.bhs_dict
        
    def split_train_valid(
        self, 
        valid_size: float, 
        sites: List[str], 
        normalize: bool=True, 
        return_tensor: bool=True
    ):
        coords = np.concatenate([self.bhs_dict[site]["coords"] for site in sites], axis=0)
        labels = np.concatenate([self.bhs_dict[site]["labels"] for site in sites], axis=0)

        # get mask for train-valid splitting
        N = len(labels)
        mask = np.arange(N)
        np.random.shuffle(mask)
        n_trains = int(N*(1-valid_size))

        # train-valid split
        coords_train = coords[mask[:n_trains]]
        labels_train = labels[mask[:n_trains]]

        coords_valid = coords[mask[n_trains:]]
        labels_valid = labels[mask[n_trains:]]

        # normalize coords if `normalize` is True
        if normalize:
            coords_train = self._normalize_coords(coords_train)
            coords_valid = self._normalize_coords(coords_valid)

        # convert numpy array to torch tensor if `return_tensor` is True
        if return_tensor:
            coords_train = torch.tensor(coords_train).float()
            labels_train = torch.tensor(labels_train).long()
            coords_valid = torch.tensor(coords_valid).float()
            labels_valid = torch.tensor(labels_valid).long()

        self.train_valid_split = {
            "train": {"coords": coords_train, "labels": labels_train},
            "valid": {"coords": coords_valid, "labels": labels_valid},
        }
        return self.train_valid_split
    
    def get_test(self, sites: List[str], normalize: bool=True, return_tensor: bool=True):
        coords = np.concatenate([self.bhs_dict[site]["coords"] for site in sites], axis=0)
        labels = np.concatenate([self.bhs_dict[site]["labels"] for site in sites], axis=0)

        # normalize coords if `normalize` is True
        if normalize:
            coords = self._normalize_coords(coords)

        if return_tensor:
            coords = torch.tensor(coords).float()
            labels = torch.tensor(labels).long()

        self.data_test = {"coords": coords, "labels": labels}
        return self.data_test
    
    def generate_grids(self, n_grids, lower_bound, upper_bound, normalize=True, return_tensor=True):
        # site is 2 dimensional
        assert (len(n_grids) == 2), "`n_grids` must be given in the form of (n_grids_x, n_grids_y)."
        self.n_grids = np.array(n_grids)
        self.lower_bound_test = np.array(lower_bound)
        self.upper_bound_test = np.array(upper_bound)

        # Compute the grid size in each dimension
        size_test = self.upper_bound_test - self.lower_bound_test
        self.grid_size = size_test / self.n_grids

        # compute vector along each axis
        xv = np.linspace(lower_bound[0] + self.grid_size[0]/2,  self.upper_bound_test[0] - self.grid_size[0]/2,  n_grids[0])
        yv = np.linspace(lower_bound[1] + self.grid_size[1]/2,  self.upper_bound_test[1] - self.grid_size[1]/2,  n_grids[1])

        # generate mesh grid
        xx, yy = np.meshgrid(xv, yv)

        grids_df = pd.DataFrame(np.stack([xx.flatten(), yy.flatten()], axis=-1), columns=["x", "y"])

        grids_df["depth"] = np.interp(grids_df["x"], self.vertices[:, 0], self.vertices[:, 1])

        grids_df = grids_df[grids_df["y"] >= grids_df["depth"]]

        if self.sub_depth:
            grids_df["y"] = grids_df["y"] - grids_df["depth"]

        coords_grids = grids_df[["x", "y"]].to_numpy()

        # normalize coords if `normalize` is True
        if normalize:
            coords_grids = self._normalize_coords(coords_grids)

        # In this example label at test grids is not known
        # A vector of zeros is created to align with training and validating case
        labels = np.zeros(coords_grids.shape[0])*(-1)

        # convert numpy array to torch tensor if `return_tensor` is True
        if return_tensor:
            coords_grids = torch.tensor(coords_grids).float()

        self.data_grids = {"coords": coords_grids, "labels": labels}
        return self.data_grids
        

class TransformerDataset:
    def __init__(self, data_train, shuffle_site, train, data_test=None):
        if not train:
            assert data_test is not None
            self.coords_test = data_test["coords"]
            self.labels_test = data_test["labels"]
        self.coords_train = data_train["coords"]
        self.labels_train = data_train["labels"]
        # during training, the target point is selected from sampled points,
        # so the label at target point must be dropped, otherwise the model just do "copy" work
        self.train = train
        self.shuffle_site = shuffle_site
    
    def __len__(self):
        if self.train:
            return self.coords_train.size(0)
        else:
            return self.coords_test.size(0)
    
    def __getitem__(self, idx):
        # drop target item when training
        if self.train:
            coords = torch.concat([self.coords_train[:idx], self.coords_train[idx+1:]], dim=0)
            labels = torch.concat([self.labels_train[:idx], self.labels_train[idx+1:]], dim=0)
        else:
            coords = torch.clone(self.coords_train)
            labels = torch.clone(self.labels_train)

        # shuffle site items if `self.shuffle_site` is True
        if self.shuffle_site:
            mask = torch.randperm(coords.size(0))
            coords = coords[mask]
            labels = labels[mask]

        if self.train:
            trg_coord = self.coords_train[idx]
            trg_label = self.labels_train[idx]
        else:
            trg_coord = self.coords_test[idx]
            trg_label = self.labels_test[idx]

        return {
                "net_input": {
                    "site_coords": coords, 
                    "site_cls": labels,
                    "trg_coord": trg_coord,
                },
                "target": trg_label,
            }
