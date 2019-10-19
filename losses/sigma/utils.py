import math
import torch
import torch.autograd as ag
def split(x, y):#when N+>K split into all score and only n_pos score
    x_1 = x
    mask=torch.zeros([x.size()[0],int(y.sum(1)[0])],requires_grad=False)
    for i in range(y.size()[0]):
        k=0
        for j in range(y.size()[1]):     
            if y[i][j]!=0:
                mask[i][k]=j
                k+=1
        x_2 = x_1.gather(1, mask.cuda().long())
    return x_1, x_2

def split_2(x, y):#when N+<K split
    x_1 = x
    mask=torch.zeros([x.size()[0],x.size()[1]-int(y.sum(1)[0])],requires_grad=False)
    for i in range(y.size()[0]):
        k=0
        for j in range(y.size()[1]):     
            if y[i][j]==0:
                mask[i][k]=j
                k+=1
        x_2 = x_1.gather(1, mask.cuda().long())

    mask1=torch.zeros([x.size()[0],int(y.sum(1)[0])],requires_grad=False)
    for i in range(y.size()[0]):
        k=0
        for j in range(y.size()[1]):     
            if y[i][j]!=0:
                mask1[i][k]=j
                k+=1
        x_3 = x_1.gather(1, mask1.cuda().long())
    return x_1, x_2, x_3

def detect_large(x, k, tau, thresh):
    if x.size(0)>k+1:
        top, _ = x.topk(k + 1, 1)
        # switch to hard top-k if (k+1)-largest element is much smaller
        # than k-largest element
        hard = torch.ge(top[:, k - 1] - top[:, k], k * tau * math.log(thresh)).detach()
        smooth = hard.eq(0)
    else:
        hard=torch.ones(x.size(0)).cuda().long()
        smooth = hard.eq(0).cuda().long()
    return smooth, hard


def split_KN_pos(x, y, K):#split into K>N+ and K<=N+
    x1=torch.tensor([]).cuda()#N+>K
    y1=torch.tensor([]).cuda()#N+>K
    x2=torch.tensor([]).cuda()#N+<K
    y2=torch.tensor([]).cuda()#N+<K
    for i in range(y.size()[0]):
        if y[i].sum()>=K:
            x1=torch.cat([x1,x[i].view(1,-1)],0)
            y1=torch.cat([y1,y[i].view(1,-1)],0)
        if y[i].sum()<K:
            x2=torch.cat([x2,x[i].view(1,-1)],0)
            y2=torch.cat([y2,y[i].view(1,-1)],0)

    return x1, y1, x2, y2