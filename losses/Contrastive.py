import torch
import torch.nn as nn

def Contrastive(margin):
    return ContrastiveLoss(margin=margin)

class ContrastiveLoss(nn.Module):

    def __init__(self,margin,eps=1e-12):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6
        self.hard_mining = True
    def forward(self, embedding, label):
        '''
        #same is 1, diff is 0
        same=1
        cnt1 = embedding1.size(0)
        dist_sqr=torch.sum(torch.pow(embedding1-embedding2,2), 1)
        dist=torch.sqrt(dist_sqr)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        #print(dist)
        loss1 = same * dist_sqr + (1 - same) * torch.pow(dist, 2)    
        loss1 = torch.sum(loss1) / 2.0 / cnt1


        diff=1-same
        cnt2 = (cnt1-1)*(cnt1-1)
        dist_sqr2 = torch.zeros(0,1).cuda()
        for i in range(cnt1):
            for j in range(2):
                if j!=i:
                    temp=torch.sum(torch.pow(embedding1[i]-embedding2[j],2)).reshape(1,1)
                    dist_sqr2=torch.cat([dist_sqr2,temp],0)
        dist2=torch.sqrt(dist_sqr2)
        mdist2 = self.margin - dist2
        dist2 = torch.clamp(mdist2, min=0.0)
        loss2 = diff * dist_sqr2 + (1 - diff) * torch.pow(dist2, 2)    
        loss2 = torch.sum(loss2) / 2.0 / cnt2
        return loss1+loss2
        '''


        n = embedding.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(embedding, embedding.t())
        label = label
        loss = list()
        c = 0

        for i in range(n):
            #print(label)
            #print(label[i])
            #print(label==label[i])
            pos_pair_ = torch.masked_select(sim_mat[i], label==label[i])
            #print(pos_pair_)
            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            #print(pos_pair_)
            neg_pair_ = torch.masked_select(sim_mat[i], label!=label[i])
            #print(neg_pair_)

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]
            #print(neg_pair_)
            if self.hard_mining:
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)
            else:
                neg_pair = neg_pair_
            #print(neg_pair)
            neg_loss = 0

            pos_loss = torch.sum(-pos_pair_+1) 
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
            loss.append(pos_loss + neg_loss)

        loss=sum(loss)/n
        prec = float(c)/n
        neg_d = torch.mean(neg_pair_).item()
        pos_d = torch.mean(pos_pair_).item()
        return loss,0,0

        #loss = label * dist_sqr + (1 - label) * torch.pow(dist, 2) 