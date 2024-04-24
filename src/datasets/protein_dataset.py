import os
import sys
import os.path as osp
import warnings
import pathlib
from torch_geometric.data.data import BaseData
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import sigmoid
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data.dataset import Dataset, _repr, files_exist
from operator import itemgetter
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# TODO(jiahang): train-valid-test split not implemented yet

class ProteinDataset(InMemoryDataset):
    def __init__(self, root, norm, eps, diffusion_steps, 
                 dataset_name='protein', transform=None, pre_transform=None, pre_filter=None, 
                 force_reload=False):
        self.dataset_name = dataset_name
        self.network_path = root
        self.force_reload = force_reload
        self.eps = eps
        self.norm = norm
        self.diffusion_steps = diffusion_steps
        assert self.norm in ['maxmim_norm', 'eigen_norm', 'ND_norm', 'normal']
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.edge_prob = torch.load(self.processed_paths[1])
        if self.norm == 'ND_norm':
            eig = torch.load(self.processed_paths[2])
            self.eigval_pow_cumsum, self.eigvec = eig['eigval_pow_cumsum'], eig['eigvec']

    @property
    def raw_dir(self):
        return self.network_path
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return self.raw_dir
    
    @property
    def processed_file_names(self):
        if self.norm == 'ND_norm':
            return [f'network_all_{self.norm}_{self.diffusion_steps}.pt', 
                    f'edge_prob_{self.norm}_{self.diffusion_steps}.pt',
                    f'eig_{self.diffusion_steps}.pt'
                ]
        return [f'network_all_{self.norm}.pt', f'edge_prob_{self.norm}.pt']

    def _process(self):
        # taken from PyG original implementations
        f = osp.join(self.processed_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first")

        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, make sure to "
                "delete '{self.processed_dir}' first")

        if not self.force_reload and files_exist(self.processed_paths):  # pragma: no cover
            return
        
        if self.force_reload:
            print('Force re-processing...', file=sys.stderr)

        if self.log and 'pytest' not in sys.modules:
            print('Processing...', file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)
    
    def process(self):
        files = os.listdir(self.raw_dir)
        data_list = []
        eigval_pow_cumsum_list, eigvec_list = [], []
        edge_prob_list = []
        for filename in tqdm(files):
            if filename.endswith('.mat') or filename.endswith('.pt') or'_ND_' in filename \
                or ('DI_' not in filename and 'MI_' not in filename):
                continue
            adj_path = os.path.join(self.raw_dir, filename)
            adj = torch.tensor(np.load(adj_path))
            if self.norm == 'maxmin_norm':
                adj = self.maxmin_norm(adj)
            elif self.norm == 'eigen_norm':
                adj = self.eigen_norm(adj)
            elif self.norm in ['ND_norm']:
                eigval_pow_cumsum, eigvec, adj = self.ND_norm(adj)
                eigval_pow_cumsum_list.append(eigval_pow_cumsum)
                eigvec_list.append(eigvec)
                adj = sigmoid(adj)
            elif self.norm == 'normal':
                pass
            edge_index, edge_prob = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[(edge_prob > 0.5) , 1] = 1
            edge_attr[(edge_prob > 0.5) , 0] = 0
            edge_attr[~(edge_prob > 0.5) , 1] = 0
            edge_attr[~(edge_prob > 0.5) , 0] = 1
            
            num_nodes = adj.shape[0]
            X = torch.ones(num_nodes, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float() # TODO(jiahang): what's this?
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                                y=y, n_nodes=num_nodes)

            data_list.append(data)
            edge_prob_list.append(adj.flatten())

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(edge_prob_list, self.processed_paths[1])
        if self.norm == 'ND_norm':
            torch.save({
                'eigval_pow_cumsum': eigval_pow_cumsum_list,
                'eigvec': eigvec_list
            }, self.processed_paths[2])
        

    def maxmin_norm(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data

    def eigen_norm(self, data):
        eigval, _ = torch.linalg.eigh(data)
        data = data / (eigval.abs().max() + self.eps)
        return data

    def ND_norm(self, data):
        data = data - data.mean()
        data = self.eigen_norm(data)
        assert (data == torch.transpose(data, 0, 1)).all(), "input data is not symmetric"
        eigval, eigvec = torch.linalg.eigh(data)
        eigval_pow = torch.stack([eigval ** i for i in range(1, self.diffusion_steps + 1)])
        eigval_pow_cumsum = torch.cumsum(eigval_pow, dim=0)
        return eigval_pow_cumsum, eigvec, data

    def get_info(self, idx):
        if isinstance(idx, list):
            eigval_pow_cumsum = itemgetter(*idx)(self.eigval_pow_cumsum)
            eigvec = itemgetter(*idx)(self.eigvec)
            edge_prob = itemgetter(*idx)(self.edge_prob)
        else:
            eigval_pow_cumsum = self.eigval_pow_cumsum[idx]
            eigvec = self.eigvec[idx]
            edge_prob = self.edge_prob[idx]

        return eigval_pow_cumsum, eigvec, edge_prob

    def __getitem__(self, idx):
        eigval_pow_cumsum, eigvec, edge_prob = self.get_info(idx)
        res = {
            'g': self.get(idx),
            'eigval_pow_cumsum': eigval_pow_cumsum,
            'eigvec': eigvec,
            'edge_prob': edge_prob
        }
        return res
    
class ProteinDataModule(AbstractDataModule):
    # TODO(jiahang): getitem in this class and its parent class is of no use?
    def __init__(self, cfg):
        self.cfg = cfg
        self.network_dir = cfg.dataset.network_dir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        network_path = os.path.join(base_path, self.network_dir)

        config = {
            'root': network_path, 'norm': self.cfg.dataset.norm, 'eps': self.cfg.dataset.eps,
            'diffusion_steps': self.cfg.dataset.diffusion_steps, 'force_reload': self.cfg.dataset.force_reload
        }

        datasets = {'train': ProteinDataset(**config), 
                    'val': ProteinDataset(**config), 
                    'test': ProteinDataset(**config)
                }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]

    

class ProteinDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = datamodule.train_dataset.dataset_name
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        # self.edge_types = self.datamodule.edge_counts()
        self.edge_types = torch.tensor([1]) # no use in our case
        super().complete_infos(self.n_nodes, self.node_types)

