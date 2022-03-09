
from collections import OrderedDict
import torch.nn as nn
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class FeatureAnalyzer():
    def __init__(self, model, data_loader, criterion, print=print):
        self.feat_mean = OrderedDict()
        self.grad_mean = OrderedDict()
        self.data_loader = data_loader
        self.criterion = criterion
        self.print = print
        self.layer_names = {}
        for name, module in model.named_modules():
            self.layer_names[module] = name
        self.register_hooks(model)
        self.analyze_feat(model)
        self.rm_hooks(model)

    def register_hooks(self, model):
        def forward_hook(m, i, o):
            name = self.layer_names[m]
            if name not in self.feat_mean:
                self.feat_mean[name] = AverageMeter(name)
            self.feat_mean[name].update(o.abs().mean().item(), o.size(0))
        
        def backward_hook(m, grad_i, grad_o):
            name = self.layer_names[m]
            if name not in self.grad_mean:
                self.grad_mean[name] = AverageMeter(name)
            assert len(grad_o) == 1
            self.grad_mean[name].update(grad_o[0].abs().mean().item(), grad_o[0].size(0))

        for _, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                 module.register_forward_hook(forward_hook)
                 module.register_backward_hook(backward_hook)
    
    def rm_hooks(self, model):
         for _, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                 module._forward_hooks = OrderedDict()
                 module._backward_hooks = OrderedDict()
    
    def analyze_feat(self, model):
        # forward to activate hooks
        for i, (images, target) in enumerate(self.data_loader):
            images, target = images.cuda(), target.cuda()
            output = model(images)
            loss = self.criterion(output, target)
            loss.backward()
        
        max_key_len = np.max([len(k) for k in self.feat_mean.keys()])
        for k, v in self.feat_mean.items():
            grad = self.grad_mean[k]
            self.print(f'{k.rjust(max_key_len)} -- feat_mean {v.avg:.4f} grad_mean {grad.avg:.10f}')