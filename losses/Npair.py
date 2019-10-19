import torch
import torch.nn as nn
import numpy as np

def NPair(margin):
    #https://github.com/ronekko/deep_metric_learning/blob/master/lib/functions/n_pair_mc_loss.py
    return NPairLoss(margin=margin)

class NPairLoss(nn.Module):

    def __init__(self,margin):
        super(NPairLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.margin = margin
        self.l2_reg=0.02

    def cross_entropy(self,target,logits):
        return torch.mean(torch.sum(- target * nn.functional.log_softmax(logits, -1), -1))
    '''
    def forward(self, embedding, label):
        batch_size=embedding.size(0)
        embed_anchor=embedding[0:batch_size//2,:]
        embed_pos=embedding[batch_size//2:batch_size,:]

        embed_anchor_norm = embed_anchor.norm(dim=1)
        embed_pos_norm = embed_pos.norm(dim=1)

        simliarity_matrix = embed_anchor.mm(embed_pos.transpose(0, 1))
        print(simliarity_matrix)
        N=embed_anchor.size()[0]
        target = torch.from_numpy(np.array([i for i in range(N)])).cuda()
        print(target)
        l2loss = (embed_anchor_norm.sum()+embed_pos_norm.sum())/N
        celoss=self.criterion(simliarity_matrix,target)
        return celoss+l2loss*0.003
    '''

    def forward(self, embedding, target):
        batch_size=embedding.size(0)
        anchor=embedding[0:batch_size//2,:]
        positive=embedding[batch_size//2:batch_size,:]
        batch_size = anchor.size(0)
        target = target[0:batch_size].cuda()
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))-target*0.3
        loss_ce = self.cross_entropy(target,logit)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size
        loss = loss_ce+self.l2_reg*0.25*l2_loss
        return loss,0,0

