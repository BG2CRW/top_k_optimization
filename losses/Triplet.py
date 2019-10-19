import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def Triplet(margin):
    return TripletLoss(margin=margin)

class TripletLoss(nn.Module):

    def __init__(self,margin):
        super(TripletLoss, self).__init__()
        self.beta = 40
        self.margin = margin
        self.alpha = 40
        self.sampling_method = 1  #0:None  1:semi-hard   2:hard-mining  3:distance weighted
        method=["None","semi-hard","hard-mining","distances"]
        print(method[self.sampling_method])
        self.triplet=nn.TripletMarginLoss(margin=margin, p=2)
        self.dist = lambda x, y : torch.pow(torch.nn.PairwiseDistance(eps = 1e-16)(x, y),2)
        self.triplets = 0

    def forward(self, embedding, label):
        n = embedding.size(0)
        sim_mat = torch.matmul(embedding, embedding.t())
        label = label.cuda()
        # split the positive and negative pairs
        eyes_ = torch.eye(n, n).cuda()
        
        pos_mask = label.expand(n, n).eq(label.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        num_instances = len(pos_sim)//n + 1
        num_neg_instances = n - num_instances

        pos_sim = pos_sim.resize(len(pos_sim)//(num_instances-1), num_instances-1)
        neg_sim = neg_sim.resize(len(neg_sim) // num_neg_instances, num_neg_instances)

        #  clear way to compute the loss first
        loss = list()
        base = 0.5
        for i, pos_pair_ in enumerate(pos_sim):
            #print(i,pos_pair_.size())
            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_sim[i])[0]
            #print(pos_pair_.size(),neg_pair_.size())
            if self.sampling_method==3:#distance weighted
                neg_pair = torch.masked_select(neg_pair_, beta - d_an + self.margin)
                pos_pair = torch.masked_select(pos_pair_, d_ap - beta + self.margin )
            if self.sampling_method==2:#hard-mining
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_[0] - self.margin)
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ < neg_pair_[-1] + self.margin)
            if self.sampling_method==1:#semi-hard
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ < pos_pair_[-1])
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ > neg_pair_[0])
            if self.sampling_method==0:#none
                pos_pair = pos_pair_
                neg_pair = neg_pair_
            #print(pos_pair.size(),neg_pair.size())
            self.triplets+=neg_pair.size(0)//2

            if len(pos_pair) > 0:
                pos_loss = torch.mean(1 - pos_pair)  
                #pos_loss =   2.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base)))) +  torch.mean(1 - pos_pair)    
            else:
                pos_loss = 0*torch.mean(1 - pos_pair_)
            
            if len(neg_pair)>0:
                #neg_loss = torch.mean(neg_pair)
                neg_loss = 2.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))  +  torch.mean(neg_pair)
            else:
                neg_loss = 0*torch.mean(neg_pair_)

            loss.append(pos_loss + neg_loss)
        #print(self.triplets)
        loss = sum(loss)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return  loss,self.triplets,0

  