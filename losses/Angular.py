import torch
import torch.nn as nn
import numpy as np

def Angular(margin):#angular mc
    #https://github.com/ronekko/deep_metric_learning/blob/master/lib/functions/angular_loss.py
    return AngularLoss(margin=margin)

class AngularLoss(nn.Module):

    def __init__(self,margin):
        super(AngularLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding, label):
        batch_size=embedding.size(0)
        embed_anchor=embedding[0:batch_size//2,:]
        embed_pos=embedding[batch_size//2:batch_size,:]
        alpha = np.deg2rad(self.margin)
        sq_tan_alpha = torch.tan(torch.from_numpy(alpha.reshape(1,1)).float().cuda()[0][0]) ** 2
        n_pairs = embed_anchor.size()[0]
        # first and second term of f_{a,p,n}
        term1 = 4  * sq_tan_alpha * (embed_anchor + embed_pos).mm(embed_pos.transpose(0, 1))   
        term2 = 2 * (1 + sq_tan_alpha) * (embed_pos*embed_anchor).sum(dim=1)
        f_apn = term1-term2.repeat(n_pairs,1)
        mask = torch.ones(n_pairs,n_pairs)-torch.eye(n_pairs,n_pairs)
        f_apn = f_apn*mask.cuda()
        loss = f_apn.exp().sum(dim=1).log().mean()

        return  loss,0,0