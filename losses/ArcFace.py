import torch
import torch.nn as nn
import math


def ArcFace(margin,input_dim,output_dim):
    #https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    return  ArcFaceLoss(input_dim=input_dim, output_dim=output_dim,margin=margin)
class ArcFaceLoss(nn.Module):

    def __init__(self,input_dim,output_dim,margin=0.5,easy_margin=True):
        super().__init__()
        self.m = margin
        self.s = 30
        self.weight = nn.Parameter(torch.FloatTensor(output_dim, input_dim)).cuda()
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin=easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target=None):  #input [batch, embed_size],weight [embed_size, class_number]
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        return self.criterion(output,target)
    