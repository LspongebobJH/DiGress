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
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data.dataset import Dataset, _repr, files_exist
from torch_geometric.utils import to_dense_adj
from operator import itemgetter
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.utils import slope_sigmoid
from torch.utils.data import Dataset

# TODO(jiahang): train-valid-test split not implemented yet

class ProteinDataset(InMemoryDataset):
    def __init__(self, root, norm, eps, diffusion_steps, slope=1,
                 dataset_name='protein', transform=None, pre_transform=None, pre_filter=None, 
                 force_reload=False):
        self.dataset_name = dataset_name
        self.network_path = root
        self.force_reload = force_reload
        self.eps = eps
        self.norm = norm
        self.diffusion_steps = diffusion_steps
        self.slope = slope
        assert self.norm in ['maxmim_norm', 'eigen_norm', 'ND_norm', 'normal']
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        eig = torch.load(self.processed_paths[1])
        self.eigval_pow_cumsum, self.eigvec = eig['eigval_pow_cumsum'], eig['eigvec']
        self.idx = torch.load(self.processed_paths[2])

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
        return [f'network_all_{self.eps}.pt', f'eig_{self.eps}.pt', f'idx_{self.eps}.pt']

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
        data_list, idx_list = [], []
        eigval_pow_cumsum_list, eigvec_list = [], []
        for filename in tqdm(files):
            if filename.endswith('.mat') or filename.endswith('.pt') or'_ND_' in filename \
                or ('DI_' not in filename and 'MI_' not in filename):
                continue
            idx = filename.split('I_')[1].split('.npy')[0]
            idx_list.append(idx)
            adj_path = os.path.join(self.raw_dir, filename)
            adj = torch.tensor(np.load(adj_path))
            adj = self.normalize_data(adj)
            eigval, eigvec = torch.linalg.eigh(adj) # this eigval is for cumsum/cumprod compute
            eigval_pow = torch.stack([eigval ** i for i in range(1, self.diffusion_steps + 1)])
            eigval_pow_cumsum = torch.cumsum(eigval_pow, dim=0)
            eigval_pow_cumsum_list.append(eigval_pow_cumsum)
            eigvec_list.append(eigvec)
            edge_index, edge_prob = torch_geometric.utils.dense_to_sparse(adj)
            num_nodes = adj.shape[0]
            X = torch.ones(num_nodes, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float() # TODO(jiahang): what's this?
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_prob,
                                                y=y, n_nodes=num_nodes)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save({
            'eigval_pow_cumsum': eigval_pow_cumsum_list,
            'eigvec': eigvec_list
        }, self.processed_paths[1])
        torch.save(idx_list, self.processed_paths[2])

    def normalize_data(self, adj):
        # This function name is to avoid name collision with the default one
        if self.norm in ['maxmin_norm', 'eigen_norm', 'normal']:
            raise Exception(f"{self.norm} not supported!")
        elif self.norm in ['ND_norm']:
            adj = self.ND_norm(adj)
        return adj

    def __getitem__(self, idx):
        g = self.get(idx)
        g.edge_attr = slope_sigmoid(g.edge_attr, self.slope)
        eigval_pow_cumsum, eigvec = self.get_eig(idx)
        g_idx = self.idx[idx]
        
        res = {
            'g': g,
            'eigval_pow_cumsum': eigval_pow_cumsum,
            'eigvec': eigvec,
            'g_idx': g_idx
        }
        return res

    def ND_norm(self, data):
        data = (data + self.eps).log2()
        data = data - data.mean()
        eigval = torch.linalg.eigvals(data).abs().max() # this eigval is for norm
        data = data / (eigval + self.eps)
        assert (data == torch.transpose(data, 0, 1)).all(), "input data is not symmetric"
        return data

    def get_eig(self, idx):
        if isinstance(idx, list):
            eigval_pow_cumsum = itemgetter(*idx)(self.eigval_pow_cumsum)
            eigvec = itemgetter(*idx)(self.eigvec)
        else:
            eigval_pow_cumsum = self.eigval_pow_cumsum[idx]
            eigvec = self.eigvec[idx]

        return eigval_pow_cumsum, eigvec
    
    def maxmin_norm(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data

    def eigen_norm(self, data, eigval):    
        data = data / (eigval.abs().max() + self.eps)
        return data

class ProteinGoldenDataset(InMemoryDataset):
    def __init__(self, root, force_reload=False):
        self.network_path = root
        self.force_reload = force_reload
        super().__init__(root = root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.idx = torch.load(self.processed_paths[1])

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
        return [f'golden_all.pt', 'golden_idx.pt']
    
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
        data_list, idx_list = [], []
        for filename in tqdm(files):
            if not ('contact' in filename and '.mat' not in filename and '.pt' not in filename):
                continue

            # get index
            idx = filename.split('contact_')[1].split('.npy')[0]
            idx_list.append(idx)
            adj_path = os.path.join(self.raw_dir, filename)
            adj = torch.tensor(np.load(adj_path))
            edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj)
            num_nodes = adj.shape[0]
            X = torch.ones(num_nodes, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float() # TODO(jiahang): what's this?
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                                y=y, n_nodes=num_nodes)

            data_list.append(data)


        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(idx_list, self.processed_paths[1])

    def __getitem__(self, idx):
        g = self.get(idx)
        g_idx = self.idx[idx]
        
        res = {
            'golden_g': g,
            'golden_idx': g_idx,
        }
        return res

class ProteinInferDataset(Dataset):
    def __init__(self, **kwargs):
        self.data = ProteinDataset(**kwargs)
        self.golden = ProteinGoldenDataset(kwargs['root'], kwargs['force_reload'])
        super().__init__()

    def __getitem__(self, idx):
        res = {}
        cur_data = self.data[idx]
        g_idx = cur_data['g_idx']
        golden_idx_list = self.golden.idx
        golden_pos = golden_idx_list.index(g_idx)
        res.update(cur_data)
        res.update(self.golden[golden_pos])
        return res

    def __len__(self):
        return len(self.data)

    
class ProteinDataModule(AbstractDataModule):
    # TODO(jiahang): getitem in this class and its parent class is of no use?
    def __init__(self, cfg):
        self.cfg = cfg
        self.network_dir = cfg.dataset.network_dir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        network_path = os.path.join(base_path, self.network_dir)

        config = {
            'root': network_path, 'norm': self.cfg.dataset.norm, 'eps': self.cfg.dataset.eps,
            'diffusion_steps': self.cfg.dataset.diffusion_steps, 'force_reload': self.cfg.dataset.force_reload,
            'slope': self.cfg.dataset.slope
        }

        datasets = {'train': ProteinDataset(**config), 
                    'val': ProteinDataset(**config), 
                    'test': ProteinDataset(**config) if not cfg.general.infer else \
                        ProteinInferDataset(**config)
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

