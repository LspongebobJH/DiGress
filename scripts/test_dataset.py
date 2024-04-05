from src.datasets.gene_dataset import GeneDataModule, GeneDatasetInfos
from src.datasets.spectre_dataset import SpectreGraphDataModule
from src.datasets.protein_dataset import ProteinDataModule, ProteinDatasetInfos
import hydra
from omegaconf import DictConfig

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    '''Gene Dataset'''
    # datamodule = GeneDataModule(cfg)
    # dataset_infos = GeneDatasetInfos(datamodule, cfg['dataset'])

    '''Spectre Dataset'''
    # datamodule = SpectreGraphDataModule(cfg)

    '''Protein dataset'''
    datamodule = ProteinDataModule(cfg)
    dataset_infors = ProteinDatasetInfos(datamodule, cfg['dataset'])

    pass 

main()