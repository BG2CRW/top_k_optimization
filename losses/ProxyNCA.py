import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import sklearn.preprocessing

def ProxyNCA(sz_embed, nb_classes):
    #https://github.com/dichotomies/proxy-nca/blob/master/proxynca.py
    return ProxyNCALoss(sz_embed=sz_embed, nb_classes=nb_classes)
class ProxyNCALoss(nn.Module):
    def __init__(self, nb_classes, sz_embed, smoothing_const = 0.0):
        super(ProxyNCALoss,self).__init__()
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.smoothing_const = smoothing_const
    
    def binarize_and_smooth_labels(self, T, nb_classes, smoothing_const = 0.1):
        
        T = T.cpu().numpy()
        T = sklearn.preprocessing.label_binarize(
            T, classes = range(0, nb_classes)
        )
        T = T * (1 - smoothing_const)
        T[T == 0] = smoothing_const / (nb_classes - 1)
        T = torch.FloatTensor(T).cuda()
        return T


    def pairwise_distance(self, a, squared=False):
        """Computes the pairwise distance matrix with numerical stability."""
        pairwise_distances_squared = torch.add(
            a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
            torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
        ) - 2 * (
            torch.mm(a, torch.t(a))
        )

        # Deal with numerical inaccuracies. Set small negatives to zero.
        pairwise_distances_squared = torch.clamp(
            pairwise_distances_squared, min=0.0
        )

        # Get the mask where the zero distances are at.
        error_mask = torch.le(pairwise_distances_squared, 0.0)

        # Optionally take the sqrt.
        if squared:
            pairwise_distances = pairwise_distances_squared
        else:
            pairwise_distances = torch.sqrt(
                pairwise_distances_squared + error_mask.float() * 1e-16
            )

        # Undo conditionally adding 1e-16.
        pairwise_distances = torch.mul(
            pairwise_distances,
            (error_mask == False).float()
        )

        # Explicitly set diagonals to zero.
        mask_offdiagonals = 1 - torch.eye(
            *pairwise_distances.size(),
            device=pairwise_distances.device
        )
        pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

        return pairwise_distances


    def forward(self, X, T):
        
        P = self.proxies
        #print(P)
        P = 3 * F.normalize(P, p = 2, dim = -1)
        X = 3 * F.normalize(X, p = 2, dim = -1)
        D = self.pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared = True
        )[:X.size()[0], X.size()[0]:]

        T = self.binarize_and_smooth_labels(
            T = T, nb_classes = len(P), smoothing_const = self.smoothing_const
        )

        # cross entropy with distances as logits, one hot labels
        # note that compared to proxy nca, positive not excluded in denominator
        loss = torch.sum(- T * F.log_softmax(D, -1), -1)

        return loss.mean(),0,0
'''
class ProxyNCALoss(nn.Module):

    def __init__(self, sz_embed, nb_classes):
        super(ProxyNCALoss,self).__init__()
        self.sz_embed = sz_embed
        self.nb_classes = nb_classes
        self.proxies = torch.nn.Embedding(self.nb_classes, sz_embed).cuda()
        torch.nn.init.xavier_uniform_(self.proxies.weight)
        self.dist = lambda x, y : torch.pow(torch.nn.PairwiseDistance(eps = 1e-16)(x, y),2)

    def nca(self, xs, ys, i):
        # NOTE possibly something wrong with labels/classes ...
        # especially, if trained on labels in range 100 to 200 ... 
        # then y = ys[i] can be for example 105, but Z has 0 to 100
        # therefore, all proxies become negativ!
        x = xs[i] # embedding of sample i, produced by embedded bninception
        y = ys[i].long() # label of sample i
        # for Z: of all labels, select those unequal to label y
        Z = torch.masked_select( 
            torch.autograd.Variable(
                torch.arange(0, self.nb_classes).long()
            ).cuda(),
            torch.autograd.Variable(
                torch.arange(0, self.nb_classes).long()
            ).cuda() != y
        ).long()

        # all classes/proxies except of y
        assert Z.size(0) == self.nb_classes - 1 
        
        # with proxies embedding, select proxy i for target, p(ys)[i] <=> p(y)
        p_dist = torch.exp(
            - self.dist(
                torch.nn.functional.normalize(
                    self.proxies(y), # [1, batch_size], normalize along dim = 1 (batch_size)
                    dim = 0
                ),
                x.unsqueeze(0)
            )
        )      
        n_dist = torch.exp(
            - self.dist(
                torch.nn.functional.normalize(
                    self.proxies(Z), # [nb_classes - 1, batch_size]
                    dim = 1
                ),
                x.expand(Z.size(0), x.size(0)) # [nb_classes - 1, batch_size]
            )
        )
        return -torch.log(p_dist / torch.sum(n_dist))
    def forward(self, xs, ys):
        sz_batch=ys.size()[0]
        return torch.mean(
            torch.stack(
                [self.nca(xs, ys, i) for i in range(sz_batch)]
            )
        ),0,0
'''
