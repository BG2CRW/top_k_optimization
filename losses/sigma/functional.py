import torch
import torch.autograd as ag

from losses.sigma.polynomial.sp import hard_topk,log_sum_exp, LogSumExp,straight_forward_log_sum_exp
from losses.sigma.logarithm import LogTensor
import numpy as np

def Topk_Hard_SVM():
    def fun(x,k):
        res=hard_topk(x,k)
        return res
    return fun

def Topk_Smooth_SVM(tau):    
    def fun(x,k):
        if k ==1 or k==2:
            if k==1:
                loss = tau*log_sum_exp(x / tau)
                return loss
            if k==2:
                return hard_topk(x,2)

        elif x.size()[1]-k==1:
            loss = x.sum(1)+tau*log_sum_exp(-x / tau)
            return loss
        elif x.size()[1]-k==0:
            loss = x.sum(1)
            return loss
        else:
            lsp = LogSumExp(k)
            x.div_(tau)
            res1 = lsp(x)
            term_1 = res1[1]  #sig a k
            term_1 = LogTensor(term_1)
            loss = tau * term_1.torch()
            return loss        
        '''
        if x.size()[1]-k==2 or k==2:
            new_lsp=LogSumExp_new(k)
            res = tau * new_lsp(x / tau)
            #print(res)
            res=hard_topk(x,k)
            #print(res)
            lsp = LogSumExp(k)
            x.div_(tau)
            res1 = lsp(x)
            term_1 = res1[1]  #sig a k
            term_1 = LogTensor(term_1)
            loss = tau * term_1.torch()
            #print(loss)
            return loss
        '''
    return fun
