from src.datasets.gene_dataset import GeneDataModule, GeneDatasetInfos
import hydra
from omegaconf import DictConfig

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    datamodule = GeneDataModule(cfg)
    dataset_infos = GeneDatasetInfos(datamodule, cfg['dataset'])

    pass 

main()