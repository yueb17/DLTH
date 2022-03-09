import torch
import torch.nn as nn
import copy
import time
import numpy as np
import torch.optim as optim
from .meta_pruner import MetaPruner
from utils import PresetLRScheduler, Timer
from pdb import set_trace as st

class Pruner(MetaPruner):
    def __init__(self, model, args, logger, passer):
        super(Pruner, self).__init__(model, args, logger, passer)
        self.pr_backup = {}
        for k, v in self.pr.items():
            self.pr_backup[k] = v


    def _update_pr(self, cycle):
        '''update layer pruning ratio in iterative pruning
        '''
        for layer, pr in self.pr_backup.items():
            pr_each_time_to_current = 1 - (1 - pr) ** (1. / self.args.num_cycles)
            pr_each_time = pr_each_time_to_current * ( (1-pr_each_time_to_current) ** (cycle-1) )
            self.pr[layer] = pr_each_time if self.args.wg in ['filter', 'channel'] else pr_each_time + self.pr[layer]



    def _apply_mask_forward(self):
        assert hasattr(self, 'mask') and len(self.mask.keys()) > 0
        for name, m in self.model.named_modules():
            if name in self.mask:
                m.weight.data.mul_(self.mask[name])

    def _finetune(self, cycle):
        lr_scheduler = PresetLRScheduler(self.args.lr_ft_mini)
        optimizer = optim.SGD(self.model.parameters(), 
                                lr=0, # placeholder, this will be updated later
                                momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)
        
        best_acc1, best_acc1_epoch = 0, 0
        timer = Timer(self.args.epochs_mini)
        for epoch in range(self.args.epochs_mini):
            lr = lr_scheduler(optimizer, epoch)
            self.logprint(f'[Subprune #{cycle} Finetune] Epoch {epoch} Set LR = {lr}')
            for ix, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                self.model.train()
                y_ = self.model(inputs)
                loss = self.criterion(y_, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

               	mask_forward = not (self.args.LTH_Iter and cycle == 1)
                if self.args.method and self.args.wg == 'weight' and mask_forward:
                    self._apply_mask_forward()

                if ix % self.args.print_interval == 0:
                    self.logprint(f'[Subprune #{cycle} Finetune] Epoch {epoch} Step {ix} loss {loss:.4f}')
            # test
            acc1, *_ = self.test(self.model)
            if acc1 > best_acc1:
                best_acc1 = acc1
                best_acc1_epoch = epoch
            self.accprint(f'[Subprune #{cycle} Finetune] Epoch {epoch} Acc1 {acc1:.4f} (Best_Acc1 {best_acc1:.4f} @ Best_Acc1_Epoch {best_acc1_epoch}) LR {lr}')
            self.logprint(f'predicted finish time: {timer()}')
    
    def prune(self):
        # clear existing pr
        for layer in self.pr:
            self.pr[layer] = 0

        if self.args.LTH_Iter:
        	self.random_initialized_model_backup = copy.deepcopy(self.model) # load random model for LTH

        	for cycle in range(1, self.args.num_cycles + 1):
        		self.logprint(f'==> Start subtraining #{cycle}')
        		self._finetune(cycle) # get subtrained model
        		self._update_pr(cycle) # get pr
        		self._get_kept_wg_L1() # from pr, get self.pruned_wg
        		self._prune_and_build_new_model() # from self.pruned_wg, get mask for wg = weight
        		self.model = copy.deepcopy(self.random_initialized_model_backup) # reset model 
        		self._apply_mask_forward() # mask once
        	return self.model

        else:
	        for cycle in range(1, self.args.num_cycles + 1):
	            self.logprint(f'==> Start subprune #{cycle}')
	            self._update_pr(cycle)
	            self._get_kept_wg_L1()
	            self._prune_and_build_new_model()
	            if cycle < self.args.num_cycles:
	                self._finetune(cycle) # there is a big finetuning after the last pruning, so do not finetune here
	        
	        return self.model