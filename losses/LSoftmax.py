import torch
import torch.nn as nn
import math
from scipy.special import binom

def LSoftmax(margin,input_dim,output_dim):
    #https://github.com/jihunchoi/lsoftmax-pytorch/blob/master/lsoftmax.py
    return LSoftmaxLoss(margin=margin, input_dim=input_dim, output_dim=output_dim)
class LSoftmaxLoss(nn.Module):

    def __init__(self, input_dim, output_dim, margin):
        super(LSoftmaxLoss, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim)).cuda()

        self.divisor = math.pi / self.margin
        self.coeffs = binom(self.margin, range(0, self.margin + 1, 2))
        self.cos_exps = range(self.margin, -1, -2)
        self.sin_sq_exps = range(len(self.cos_exps))
        self.signs = [1]
        self.reset_parameters()
        self.criterion = nn.CrossEntropyLoss()
        for i in range(1, len(self.sin_sq_exps)):
            self.signs.append(self.signs[-1] * -1)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k
        
    def forward(self, input, target=None):
        assert target is not None
        logit = input.matmul(self.weight)
        batch_size = logit.size(0)
        
        logit_target = logit[range(batch_size), target]
        weight_target_norm = self.weight[:, target].norm(p=2, dim=0)
        input_norm = input.norm(p=2, dim=1)
        # norm_target_prod: (batch_size,)
        norm_target_prod = weight_target_norm * input_norm
        # cos_target: (batch_size,)
        cos_target = logit_target / (norm_target_prod + 1e-10)
        sin_sq_target = 1 - cos_target**2

        num_ns = self.margin//2 + 1
        # coeffs, cos_powers, sin_sq_powers, signs: (num_ns,)
        coeffs = input.data.new(self.coeffs)
        cos_exps = input.data.new(self.cos_exps)
        sin_sq_exps = input.data.new(self.sin_sq_exps)
        signs = input.data.new(self.signs)

        cos_terms = cos_target.unsqueeze(1) ** cos_exps.unsqueeze(0)
        sin_sq_terms = (sin_sq_target.unsqueeze(1)
                        ** sin_sq_exps.unsqueeze(0))

        cosm_terms = (signs.unsqueeze(0) * coeffs.unsqueeze(0)
                      * cos_terms * sin_sq_terms)
        cosm = cosm_terms.sum(1)
        k = self.find_k(cos_target)

        ls_target = norm_target_prod * (((-1)**k * cosm) - 2*k)
        logit[range(batch_size), target] = ls_target
        return self.criterion(logit,target)   