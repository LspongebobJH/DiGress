import torch.nn as nn

class DummySamplingMetrics(nn.Module):
    # NOTE(jiahang): adapted from SpectreSamplingMetrics
    def __init__(self, datamodule=None, compute_emd=None, metrics_list=None):
        super().__init__()
        pass

    def forward(self, generated_graphs: list, name, current_epoch, val_counter, local_rank, test=False):
        pass

    def reset(self):
        pass