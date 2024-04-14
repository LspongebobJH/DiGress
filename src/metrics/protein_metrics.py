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

        # the auroc only computes on the P(G0|G1) and P(X|G1).
        ## basically G0 == X. We check both for sanity check.
        self.auroc = {
            'Gt_1_Gt': AUROC('binary'),
            'X_Gt': AUROC('binary')
        }
        self.acc = {
            'Gt_1_Gt': Accuracy('binary', multidim_average='samplewise'),
            'X_Gt': Accuracy('binary', multidim_average='samplewise')
        }
        self.prec = {
            'Gt_1_Gt': Precision('binary', multidim_average='samplewise'),
            'X_Gt': Precision('binary', multidim_average='samplewise')
        }
        self.rec = {
            'Gt_1_Gt': Recall('binary', multidim_average='samplewise'),
            'X_Gt': Recall('binary', multidim_average='samplewise')
        }
        self.ce = {
            'Gt_1_Gt': CrossEntropy(num_steps, pos_e_w),
            'X_Gt': CrossEntropy(num_steps, pos_e_w)
        }

    def update(self, true_labels, true_logits, chain_E_Gt_1_Gt, chain_E_X_Gt):
        # NOTE(jiahang): at the end of each valid and test batch
        num_steps = chain_E_Gt_1_Gt.shape[0]
        true_labels, true_logits = true_labels.expand(num_steps, -1), true_logits.expand(num_steps, -1)

        self.auroc['Gt_1_Gt'].update(chain_E_Gt_1_Gt[-1, :], true_labels[-1, :])
        self.auroc['X_Gt'].update(chain_E_X_Gt[-1, :], true_labels[-1, :])

        self.acc['Gt_1_Gt'].update(chain_E_Gt_1_Gt, true_labels)
        self.acc['X_Gt'].update(chain_E_X_Gt, true_labels)

        self.prec['Gt_1_Gt'].update(chain_E_Gt_1_Gt, true_labels)
        self.prec['X_Gt'].update(chain_E_X_Gt, true_labels)

        self.rec['Gt_1_Gt'].update(chain_E_Gt_1_Gt, true_labels)
        self.rec['X_Gt'].update(chain_E_X_Gt, true_labels)

        self.ce['Gt_1_Gt'].update(chain_E_Gt_1_Gt, true_logits)
        self.ce['X_Gt'].update(chain_E_X_Gt, true_logits)

    def compute(self):
        # NOTE(jiahang): at the end of each valid and test epoch. test has only one epoch.
        chain_metrics = {}

        chain_metrics.update({
            'Gt_1_Gt': {
                # 'auroc': self.auroc['Gt_1_Gt'].compute().cpu(),
                'acc': self.acc['Gt_1_Gt'].compute().cpu(),
                'prec': self.prec['Gt_1_Gt'].compute().cpu(),
                'rec': self.rec['Gt_1_Gt'].compute().cpu(),
                'ce': self.ce['Gt_1_Gt'].compute().cpu()
            }, 
            'X_Gt': {
                # 'auroc': self.auroc['X_Gt'].compute().cpu(),
                'acc': self.acc['X_Gt'].compute().cpu(),
                'prec': self.prec['X_Gt'].compute().cpu(),
                'rec': self.rec['X_Gt'].compute().cpu(),
                'ce': self.ce['X_Gt'].compute().cpu()
            }
        })

        return chain_metrics

    def compute_auroc(self):
        return self.auroc['Gt_1_Gt'].compute().item(), self.auroc['X_Gt'].compute().item()

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