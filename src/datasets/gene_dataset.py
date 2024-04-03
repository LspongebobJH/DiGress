import os
import pathlib
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class GeneDataset(InMemoryDataset):
    def __init__(self, dataset_name=None, split=None, root=None, transform=None, pre_transform=None, pre_filter=None):
        self.gene_expr_path = 'data/dream5/dream5_data'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return 'data/ND-code-datasets/Application1-gene-regulatory-network/networks'
    
    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    @property
    def processed_dir(self):
        return self.raw_dir
    
    @property
    def processed_file_names(self):
        return ['network_all.pt']
    
    def process(self):
        files = os.listdir(self.raw_dir)
        data_list = []
        for filename in tqdm(files):
            if filename.endswith('.mat') or '_ND_' in filename or filename == 'network_all.pt':
                continue
            net_idx = int(re.search(r'network_(\d+)_', filename).group(1))
            gene_expr_path = os.path.join(self.gene_expr_path, f'net{net_idx}_expression_data.tsv')
            gene_expr = torch.tensor(pd.read_csv(gene_expr_path, sep = '\t').T.to_numpy())
            adj_path = os.path.join(self.raw_dir, filename)
            adj = torch.tensor(np.load(adj_path)) # TODO(jiahang): asymmetric, directed graph, re-consider it
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1 # NOTE(jiahang): link prediction
            num_nodes = gene_expr.shape[0]
            y = torch.zeros([1, 0]).float() # TODO(jiahang): what's this?
            data = torch_geometric.data.Data(x=gene_expr, edge_index=edge_index, edge_attr=edge_attr,
                                                y=y, n_nodes=num_nodes)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0]) # TODO(jiahang): problem!


class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

