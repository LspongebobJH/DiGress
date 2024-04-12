import torch
from torchmetrics import Metric, Accuracy, AUROC, Precision, Recall
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy

from src import utils
from src.diffusion import diffusion_utils

class LPMetric(Metric):
    full_state_update = False
    def __init__(self, stage):
        super().__init__()
        self.stage = stage

        self.add_state('true_labels', default=[])
        self.add_state('true_logits', default=[])
        self.add_state('chain_Es_Gt_1_Gt', default=[])
        self.add_state('chain_Es_X_Gt', default=[])

        self.auroc = AUROC('binary')
        self.acc = Accuracy('binary')
        self.prec = Precision('binary')
        self.rec = Recall('binary')

    def update(self, true_label, true_logits, chain_E_Gt_1_Gt, chain_E_X_Gt):
        # NOTE(jiahang): at the end of each valid and test batch
        self.true_labels.append(true_label)
        self.true_logits.append(true_logits)
        self.chain_Es_Gt_1_Gt.append(chain_E_Gt_1_Gt)
        self.chain_Es_X_Gt.append(chain_E_X_Gt)

    def compute(self):
        # NOTE(jiahang): at the end of each valid and test epoch. test has only one epoch.
        label = torch.concat(self.true_labels, dim=0) # NOTE(jiahang): too sparse!!! learn all 0 edges, imbalanced problem
        true_logits = torch.concat(self.true_logits, dim=0)
        chain_E_Gt_1_Gt = torch.concat(self.chain_Es_Gt_1_Gt, dim=1)
        chain_E_X_Gt = torch.concat(self.chain_Es_X_Gt, dim=1)

        chain_auroc = [
            self.auroc(_chain_E, label).item() for _chain_E in chain_E_Gt_1_Gt
        ]
        chain_acc = [
            self.acc(_chain_E, label).item() for _chain_E in chain_E_Gt_1_Gt
        ]
        chain_prec = [
            self.prec(_chain_E, label).item() for _chain_E in chain_E_Gt_1_Gt
        ]
        chain_rec = [
            self.rec(_chain_E, label).item() for _chain_E in chain_E_Gt_1_Gt
        ]
        chain_ce = [
            binary_cross_entropy(_chain_E, true_logits).item() for _chain_E in chain_E_Gt_1_Gt
        ]

        chain_metrics_Gt_1_Gt = {
            'chain_auroc': chain_auroc,
            'chain_acc': chain_acc,
            'chain_prec': chain_prec,
            'chain_rec': chain_rec,
            'chain_ce': chain_ce
        }

        chain_auroc = [
            self.auroc(_chain_E, label).item() for _chain_E in chain_E_X_Gt
        ]
        chain_acc = [
            self.acc(_chain_E, label).item() for _chain_E in chain_E_X_Gt
        ]
        chain_prec = [
            self.prec(_chain_E, label).item() for _chain_E in chain_E_X_Gt
        ]
        chain_rec = [
            self.rec(_chain_E, label).item() for _chain_E in chain_E_X_Gt
        ]
        chain_ce = [
            binary_cross_entropy(_chain_E, true_logits).item() for _chain_E in chain_E_X_Gt
        ]

        chain_metrics_X_Gt = {
            'chain_auroc': chain_auroc,
            'chain_acc': chain_acc,
            'chain_prec': chain_prec,
            'chain_rec': chain_rec,
            'chain_ce': chain_ce
        }

        return chain_metrics_Gt_1_Gt, chain_metrics_X_Gt