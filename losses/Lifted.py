import torch
import torch.nn as nn

def Lifted(margin):
    #https://gist.github.com/bkj/565c5e145786cfd362cffdbd8c089cf4
    return LiftedLoss(margin=margin)

class LiftedLoss(nn.Module):

    def __init__(self,margin):
        super(LiftedLoss, self).__init__()
        self.margin = margin
        self.tau = 0.1
        self.hard_mining = True

    '''
    def forward(self, embedding, label):
        loss = 0
        counter = 0
        bsz = embedding.size(0)
        mag = (embedding ** 2).sum(1).expand(bsz, bsz)
        sim = embedding.mm(embedding.transpose(0, 1))  #simliarity inner product
        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()
        for i in range(bsz):
            t_i = label[i]
            for j in range(i + 1, bsz):
                t_j = label[j]
                if t_i == t_j:
                    # Negative component
                    l_ni = (self.margin - dist[i][label != t_i]).exp().sum()
                    l_nj = (self.margin - dist[j][label != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()
                    # Positive component
                    l_p  = dist[i,j]
                    loss += torch.nn.functional.relu(l_n + l_p) ** 2  #max(x,0)
                    counter += 1
        return loss / (2 * counter)  
    '''  

    def forward(self, embedding, label):
        n = embedding.size(0)
        sim_mat = torch.matmul(embedding, embedding.t())
        label = label
        loss = list()
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], label==label[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 0.999)
            neg_pair_ = torch.masked_select(sim_mat[i], label!=label[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            if self.hard_mining:
                try:
                    neg_pair = torch.masked_select(neg_pair_, neg_pair_ + 0.1 > pos_pair_[0])
                    pos_pair = torch.masked_select(pos_pair_, pos_pair_ - 0.1 <  neg_pair_[-1])
                except IndexError:
                    continue
                    #import pdb;pdb.set_trace()
                
                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    c += 1
                    continue 
            else:  
                pos_pair = pos_pair_
                neg_pair = neg_pair_ 

            pos_loss = 2.0*self.tau * torch.log(torch.sum(torch.exp(-pos_pair/self.tau)))
            neg_loss = 2.0*self.tau * torch.log(torch.sum(torch.exp(neg_pair/self.tau)))


            if len(neg_pair) == 0:
                c += 1
                continue

            loss.append(pos_loss + neg_loss)

        loss = sum(loss)/n
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return  loss,0,0