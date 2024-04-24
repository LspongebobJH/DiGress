import torch
from torch import Tensor
from torchmetrics import Metric, Accuracy, AUROC, Precision, Recall
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy

from src import utils
from src.diffusion import diffusion_utils

class LPMetric:
    full_state_update = False
    def __init__(self, stage, num_steps, pos_e_w):
        super().__init__()
        self.stage = stage

        # the auroc only computes on the P(G0|G1) and P(G0|G1).
        ## basically G0 == G0. We check both for sanity check.
        self.auroc = {
            'Gs_Gt': AUROC('binary'),
            'G0': AUROC('binary')
        }
        self.acc = {
            'Gs_Gt': Accuracy('binary', multidim_average='samplewise'),
            'G0': Accuracy('binary', multidim_average='samplewise')
        }
        self.prec = {
            'Gs_Gt': Precision('binary', multidim_average='samplewise'),
            'G0': Precision('binary', multidim_average='samplewise')
        }
        self.rec = {
            'Gs_Gt': Recall('binary', multidim_average='samplewise'),
            'G0': Recall('binary', multidim_average='samplewise')
        }
        self.ce = {
            'Gs_Gt': CrossEntropy(num_steps, pos_e_w),
            'G0': CrossEntropy(num_steps, pos_e_w)
        }

    def update(self, G0, chain_E_Gs_Gt, chain_E_Gs_X, mask_E):
        # NOTE(jiahang): at the end of each valid and test batch
        num_steps = chain_E_Gs_Gt.shape[0]
        chain_E_Gs_X_lbls = (chain_E_Gs_X > 0.5).int()
        G0 = G0.expand(num_steps, -1)
        G0_lbls = (G0 > 0.5).int()
        # G0_lbls = 
        # true_labels, true_logits = true_labels.expand(num_steps, -1), true_logits.expand(num_steps, -1)

        # NOTE(jiahang): we only compute the auroc of 0-th step since auroc not support samplewise.
        self.auroc['Gs_Gt'].update(chain_E_Gs_Gt[-1, :], chain_E_Gs_X[-1, :])
        self.auroc['G0'].update(chain_E_Gs_X[-1, :], G0[-1, :])

        self.acc['Gs_Gt'].update(chain_E_Gs_Gt, chain_E_Gs_X_lbls)
        self.acc['G0'].update(chain_E_Gs_X, G0_lbls)

        self.prec['Gs_Gt'].update(chain_E_Gs_Gt, chain_E_Gs_X_lbls)
        self.prec['G0'].update(chain_E_Gs_X, G0_lbls)

        self.rec['Gs_Gt'].update(chain_E_Gs_Gt, chain_E_Gs_X_lbls)
        self.rec['G0'].update(chain_E_Gs_X, G0_lbls)

        self.ce['Gs_Gt'].update(chain_E_Gs_Gt, chain_E_Gs_X)
        self.ce['G0'].update(chain_E_Gs_X, G0)

    def compute(self):
        # NOTE(jiahang): at the end of each valid and test epoch. test has only one epoch.
        chain_metrics = {}

        chain_metrics.update({
            'Gs_Gt': {
                # 'auroc': self.auroc['Gs_Gt'].compute().cpu(),
                'acc': self.acc['Gs_Gt'].compute().cpu(),
                'prec': self.prec['Gs_Gt'].compute().cpu(),
                'rec': self.rec['Gs_Gt'].compute().cpu(),
                'ce': self.ce['Gs_Gt'].compute().cpu()
            }, 
            'G0': {
                # 'auroc': self.auroc['G0'].compute().cpu(),
                'acc': self.acc['G0'].compute().cpu(),
                'prec': self.prec['G0'].compute().cpu(),
                'rec': self.rec['G0'].compute().cpu(),
                'ce': self.ce['G0'].compute().cpu()
            }
        })

        return chain_metrics

    def compute_auroc(self):
        return self.auroc['Gs_Gt'].compute().item(), self.auroc['G0'].compute().item()

    def reset(self):
        for metric in [self.auroc, self.acc, self.prec, self.rec, self.ce]:
            for item in metric.values():
                item.reset()

class CrossEntropy(Metric):
    # This class can only be used in LPMetric!
    # Please refer to src/metrics/abstract_metrics.py - CrossEntropyMetric for using
    # cross entropy as training loss
    def __init__(self, num_steps, pos_e_w):
        super().__init__()
        # NOTE(jiahang): weight of positive edge samples, only avaiable to edge loss
        ## the actual weight of positive edges = self.pos_w * lambda
        ## 
        self.pos_e_w = pos_e_w # no special weight
        self.add_state('total_ce', default=torch.zeros(num_steps).cuda(), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        weight = torch.ones_like(target).cuda()
        weight[target >= 0.5] = self.pos_e_w
        output = - (target * torch.log2(preds + 1e-5) * weight).sum(-1)

        self.total_ce += output
        self.total_samples += target.shape[1]

    def compute(self):
        return self.total_ce / self.total_samples