import torch
import torch.nn as nn
import numpy as np

def Correlation(margin):#angular mc
    #https://github.com/ronekko/deep_metric_learning/blob/master/lib/functions/angular_loss.py
    return CorrelationLoss(margin=margin)

class CorrelationLoss(nn.Module):

    def __init__(self,margin):
        super(CorrelationLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding, label):
        batch_size = embedding.size(0)
        width = embedding.size(1)
        miu = embedding.mean(0)  #[1*width]
        embedding_centre=embedding-miu
        a=embedding_centre.unsqueeze(2)
        b=embedding_centre.unsqueeze(1)
        outer_product = a*b
        mean=outer_product.mean(0)
        loss=(mean-torch.eye(width,width).cuda()).norm()
        return  loss