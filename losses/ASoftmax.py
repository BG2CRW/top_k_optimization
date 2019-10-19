import torch
import torch.nn as nn
import math

def ASoftmax(margin,input_dim,output_dim):
    #https://github.com/clcarwin/sphereface_pytorch/blob/master/net_sphere.py
    return ASoftmaxLoss(margin=margin,input_dim=input_dim,output_dim=output_dim)

class ASoftmaxLoss(nn.Module):
    def __init__(self, input_dim, output_dim, margin = 4, phiflag=True, gamma=0):
        super(ASoftmaxLoss, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.Tensor(input_dim,output_dim)).cuda()
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.margin = margin
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.reset_parameters()

    def myphi(self,x,m):
        x = x * m
        return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
                x**8/math.factorial(8) - x**9/math.factorial(9)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def angle_linear_layer(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=input_dim Classnum=output_dim

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.margin](cos_theta)
            theta = cos_theta.data.acos()
            k = (self.margin*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = self.myphi(theta,self.margin)
            phi_theta = phi_theta.clamp(-1*self.margin,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta =  self.angle_linear_layer(input)
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = torch.nn.functional.log_softmax(output,dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss