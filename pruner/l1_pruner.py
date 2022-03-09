import torch
import torch.nn as nn
import copy
import time
import numpy as np
import torch.optim as optim
from .meta_pruner import MetaPruner
from utils import Timer

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, passer):
        super(Pruner, self).__init__(model, args, logger, passer)

    def prune(self):
        self._get_kept_wg_L1()
        self._prune_and_build_new_model()
        return self.model
            
    def _save_model(self, model, optimizer, acc1=0, acc5=0, mark=''):
        state = {'iter': self.total_iter,
                'arch': self.args.arch,
                'model': model,
                'state_dict': model.state_dict(),
                'acc1': acc1,
                'acc5': acc5,
                'optimizer': optimizer.state_dict(),
                'ExpID': self.logger.ExpID,
        }
        self.save(state, is_best=False, mark=mark)