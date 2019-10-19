import torch
import torch.nn as nn
from torch.autograd import Function

from losses.sigma.polynomial.divide_conquer import divide_and_conquer
from losses.sigma.polynomial.multiplication import Multiplication
from losses.sigma.polynomial.grad import d_logS_d_expX
from itertools import combinations
def CNK(n,k):
    combins = [c for c in  combinations(range(n), k)]
    return combins

class LogSumExp(nn.Module):
    def __init__(self, k, p=None, thresh=1e-5):
        super(LogSumExp, self).__init__()
        self.k = k
        self.p = int(1 + 0.2 * k) if p is None else p
        self.mul = Multiplication(self.k + self.p - 1)
        self.thresh = thresh

        self.register_buffer('grad_k', torch.Tensor(0))
        self.register_buffer('grad_km1', torch.Tensor(0))

        self.buffers = (self.grad_km1, self.grad_k)

    def forward(self, x):
        f = LogSumExp_F(self.k, self.p, self.thresh, self.mul, self.buffers)
        return f(x)


class LogSumExp_F(Function):
    def __init__(self, k, p, thresh, mul, buffers):
        self.k = k
        self.p = p
        self.mul = mul
        self.thresh = thresh

        # unpack buffers
        self.grad_km1, self.grad_k = buffers

    def forward(self, x):
        """
        Returns a matrix of size (2, n_samples) with sigma_{k-1} and sigma_{k}
        for each sample of the mini-batch.
        """
        self.save_for_backward(x)

        # number of samples and number of coefficients to compute
        n_s = x.size(0)
        kp = self.k + self.p - 1

        assert kp <= x.size(1)

        # clone to allow in-place operations
        x = x.clone()

        # pre-compute normalization
        x_summed = x.sum(1)

        # invert in log-space
        x.t_().mul_(-1)
        #print(x)
        # initialize polynomials (in log-space)
        x = [x, x.clone().fill_(0)]
        #print(x)
        # polynomial multiplications
        log_res = divide_and_conquer(x, kp, mul=self.mul)

        # re-normalize
        coeff = log_res + x_summed[None, :]

        # avoid broadcasting issues (in particular if n_s = 1)
        coeff = coeff.view(kp + 1, n_s)

        # save all coeff for backward
        self.saved_coeff = coeff
        #print(coeff)
        res=coeff[self.k - 1: self.k + 1]
        #print(res)
        #print(res[1])

        return coeff[self.k - 1: self.k + 1]

    def backward(self, grad_sk):
        """
        Compute backward pass of LogSumExp.
        Python variables with an upper case first letter are in
        log-space, other are in standard space.
        """

        # tensors from forward pass
        X, = self.saved_tensors
        S = self.saved_coeff

        # extend to shape (self.k + 1, n_samples, n_classes) for backward
        S = S.unsqueeze(2).expand(S.size(0), X.size(0), X.size(1))

        # compute gradients for coeff of degree k and k - 1
        self.grad_km1 = d_logS_d_expX(S, X, self.k - 1, self.p, self.grad_km1, self.thresh)
        self.grad_k = d_logS_d_expX(S, X, self.k, self.p, self.grad_k, self.thresh)

        # chain rule: combine with incoming gradients (broadcast to all classes on third dim)
        grad_x = grad_sk[0, :, None] * self.grad_km1 + grad_sk[1, :, None] * self.grad_k
        return grad_x


def log_sum_exp(x):
    """
    Compute log(sum(exp(x), 1)) in a numerically stable way.
    Assumes x is 2d.
    """
    max_score, _ = x.max(1)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score[:, None]), 1))

def straight_forward_log_sum_exp(x,k):  
    combins=CNK(x.size()[1],k)
    new_x=torch.zeros(x.size()[0],len(combins)).cuda()
    for i in range(len(combins)):
        for j in range(k):
            new_x[:,i]+=x[:,combins[i][j]]
    return log_sum_exp(new_x)

def hard_topk(x,k):
    max_1, _ = x.topk(k, dim=1)
    max_1 = max_1.sum(1)+2.0/40 * (1 + (40 * (max_1 - 0.5)).exp().sum(1)).log()
    return max_1


class LogSumExp_new(Function):
    def __init__(self, k):
        self.k = k
    def forward(self,x):
        M=x.sum(1)
        n=x.size()[1]
        batch_size=x.size()[0]

        tbl=torch.ones(batch_size,n,self.k+1).cuda()
        tbl=tbl*(-100)
        
        for m in range(batch_size):
            tbl[m][0][0]=M[m]-x[m][0]
            tbl[m][0][1]=M[m]
            for i in range(n-1):
                tbl[m][i+1][0]=tbl[m][i][0]-x[m][i+1]
                for k in range(self.k):
                    temp=torch.max(tbl[m][i][k],tbl[m][i][k+1]-x[m][i+1])
                    #print(temp)
                    tbl[m][i+1][k+1]=temp+torch.log(torch.exp(tbl[m][i][k]-temp)+torch.exp(tbl[m][i][k+1]-x[m][i+1]-temp))
        self.save_for_backward(x,tbl)
        #print(tbl)
        return tbl[:,n-1,self.k]

    def backward(self,loss):
        x,tbl = self.saved_tensors
        n=x.size()[1]
        batch_size=x.size()[0]

        tbl_gradient=torch.zeros(batch_size,self.k,n).cuda()
        for m in range(batch_size):
            for i in range(n):
                tbl_gradient[m][0][i]=1
                for j in range(self.k-1):
                    tbl_gradient[m][j+1][i]=torch.exp(tbl[m][n-1][j+1])-x[m][i]*tbl_gradient[m][j][i]

        gradient=tbl_gradient[:,self.k-1,:]
        #print(gradient)

        return  gradient
