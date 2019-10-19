import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def DistanceWeighted(margin):
    return DistanceWeighted(margin=margin)

class DistanceWeighted(nn.Module):

    def __init__(self,margin=0.2, nb_classes=0):
        super(DistanceWeighted, self).__init__()
        self.cutoff = 0.4

        # We sample only from negatives that induce a non-zero loss.
        # These are negatives with a distance < nonzero_loss_cutoff.
        # With a margin-based loss, nonzero_loss_cutoff == margin + beta.
        self.nonzero_loss_cutoff = 0.3
        self._margin = margin
        self._nu = 0.0
        self.beta = 1.2
        self.trainable_beta = nn.Parameter(torch.FloatTensor(nb_classes,))
        self.triplets = 0
        nn.init.constant(self.trainable_beta,self.beta)

    def get_distance(self,x):
        """Helper function for margin-based loss. Return a distance matrix given a matrix."""
        n = x.size(0)
        square = (x**2.0).sum(1).view(1,-1)
        distance_square = square + square.t() - 2.0*torch.matmul(x, x.t())
        res=(distance_square+torch.eye(n).cuda()).sqrt()
        # Adding identity to make sqrt work.
        return res

    def sample_layer(self, x,batch_k):
        k = batch_k
        distance = self.get_distance(x)
        n, d = x.size(0),x.size(1)   
        # Cut off to avoid high variance.
        distance=distance.clamp(min=self.cutoff)
        
        # Subtract max(log(distance)) for stability.
        log_weights = ((2.0 - float(d)) * distance.log() - (float(d - 3) / 2) * (1.0 - 0.25 * (distance ** 2.0)).log())
        weights = (log_weights - log_weights.max()).exp()
        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = np.ones(weights.shape)
        for i in range(0, n, k):
            mask[i:i+k, i:i+k] = 0
        
        weights = weights * torch.from_numpy(mask).cuda().float()*((distance < self.nonzero_loss_cutoff).float())
        weights = weights / weights.sum(1).view(-1,1)

        a_indices = []
        p_indices = []
        n_indices = []
        
        np_weights = weights.clone().cpu().detach().numpy()
        for i in range(n):
            block_idx = i // k
            try:
                n_indices += np.random.choice(n, k-1, p=np_weights[i]).tolist()
            except:
                n_indices += np.random.choice(n, k-1).tolist()
            for j in range(block_idx * k, (block_idx + 1) * k):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)
        return a_indices, x[a_indices], x[p_indices], x[n_indices], x

    def forward(self, embedding, label):
        n = embedding.size(0)
        batch_k=0
        while True:
            if label[batch_k]==label[batch_k+1]:
                batch_k+=1
            else:
                break
        batch_k+=1
        a_indices,anchors,positives,negatives,_=self.sample_layer(embedding,batch_k)

        beta = self.trainable_beta[a_indices]
        beta_reg_loss = beta.sum()* self._nu

        d_ap = (((positives - anchors)**2).sum(1)+1e-8).sqrt()
        d_an = (((negatives - anchors)**2).sum(1)+1e-8).sqrt()
        
        pos_loss = (d_ap - beta + self._margin).clamp(min=0.0)
        neg_loss = (beta - d_an + self._margin).clamp(min=0.0)
        pair_cnt = ((pos_loss > 0.0) + (neg_loss > 0.0)).clamp(max=1).sum().float()
        self.triplets=pair_cnt
        # Normalize based on the number of pairs.
        loss = ((pos_loss + neg_loss).sum()+ beta_reg_loss) / pair_cnt  
        return loss,self.triplets,0