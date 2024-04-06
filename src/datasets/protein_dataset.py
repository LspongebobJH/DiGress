import os
import sys
import os.path as osp
import warnings
import pathlib
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data.dataset import _repr, files_exist

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# TODO(jiahang): train-valid-test split not implemented yet

class ProteinDataset(InMemoryDataset):
    def __init__(self, root, 
                 dataset_name='protein', transform=None, pre_transform=None, pre_filter=None, 
                 force_reload=False):
        self.dataset_name = dataset_name
        self.network_path = root
        self.force_reload = force_reload
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        return [f'network_all.pt']

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
        for filename in tqdm(files):
            if filename.endswith('.mat') or filename.endswith('.pt') or'_ND_' in filename \
                or ('DI_' not in filename and 'MI_' not in filename):
                continue
            adj_path = os.path.join(self.raw_dir, filename)
            adj = torch.tensor(np.load(adj_path))
            adj = self.maxmin_norm(adj)
            edge_index, edge_prob = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = edge_prob
            edge_attr[:, 0] = 1. - edge_prob
            num_nodes = adj.shape[0]
            X = torch.ones(num_nodes, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float() # TODO(jiahang): what's this?
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                                y=y, n_nodes=num_nodes)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def maxmin_norm(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data


class ProteinDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.network_dir = cfg.dataset.network_dir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        network_path = os.path.join(base_path, self.network_dir)

        datasets = {'train': ProteinDataset(root=network_path, 
                                            force_reload=self.cfg.dataset.force_reload), 
                    'val': ProteinDataset(root=network_path, 
                                          force_reload=self.cfg.dataset.force_reload), 
                    'test': ProteinDataset(root=network_path, 
                                            force_reload=self.cfg.dataset.force_reload)
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
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

