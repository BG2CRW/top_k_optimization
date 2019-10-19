import torch
import torch.nn as nn
import losses.sigma.functional as F
from losses.sigma.utils import detect_large, split, split_2, split_KN_pos

def SmoothPrec_at_K(n_input, margin, tau, k, batch_size, n_pos):
    return SmoothPrec_at_K_Loss(n_input=n_input, margin=margin, tau=tau, k=k, batch_size=batch_size,n_pos=n_pos)

class SmoothPrec_at_K_Loss(nn.Module):

    def __init__(self, n_input, margin, tau, k, batch_size, n_pos):
        super(SmoothPrec_at_K_Loss, self).__init__()
        self.k = k
        self.n_pos = 0
        self.tau = tau
        self.thresh = 1e2
        self.margin = margin
        self.N = n_input
        self.batch_size = batch_size
        self.get_losses()
        self.sparsity = 0

    def get_grad(self,g):
        res = (1-g.eq(0)).sum().float()/(g.size(0)*(g.size(1)))
        self.sparsity = res
        
    def forward(self, x, y):
        embd_size = x.size()[1]
        batch_size= self.batch_size
        y=y.view(batch_size,-1)
        head=y[:,0]
        y_=torch.zeros(0,y.size(1)-1)
        for j in range(batch_size):
            categories=-1
            for i in range(y.size(1)):
                if y[j][i]==head[j]:
                    categories+=1
            temp_y=torch.cat([torch.ones(categories),torch.zeros(y.size(1)-1-categories)],0).reshape(1,y.size(1)-1)
            y_=torch.cat([y_,temp_y],0)
        y=y_.cuda()
        n_pos=[]
        for i in range(batch_size):
            n_pos.append(int(y[i].sum()+1))
        score = torch.zeros(0,self.N+1).cuda()
        y_= torch.zeros(0,self.N).cuda()
        mask = torch.zeros(0,self.N).cuda()
        for i in range(batch_size):
            for j in range(n_pos[i]):      
                temp=torch.mm(x[j+i*(self.N+1),:].view(-1,embd_size),x[0+i*(self.N+1):self.N+1+i*(self.N+1),:].t())/torch.sqrt((x[j+i*(self.N+1),:].view(-1,embd_size)**2).sum())/(x[i*(self.N+1):i*(self.N+1)+1+self.N,:]**2).sum(1)
                score=torch.cat([score,temp],0)  
                mask=torch.cat([mask,self.get_1d_array(self.N,j).view(-1,self.N)],0)
                y_=torch.cat([y_,y[i].view(-1,self.N)],0)                      

        score = score.gather(1, mask.cuda().long())
        score_hat = score.add(self.margin*(1-y_))

        _,index = score.topk(self.k, dim=1)
        err_pos=2*torch.gt(index,self.n_pos-1).sum().float()/(score.size(0)*(score.size(1)))
        score.register_hook(lambda g:self.get_grad(g))

        group_x=[]
        group_y_=[]
        pointer=0
        for i in range(batch_size):
            group_x.append(score_hat[pointer:pointer+n_pos[i],:])
            group_y_.append(y_[pointer:pointer+n_pos[i],:])
            pointer+=n_pos[i]

        loss1=0
        loss2=0
        for i in range(batch_size):        
            x1,y1,x2,y2 = split_KN_pos(group_x[i], group_y_[i], self.k)
            if x1.size()[0]:
                loss1 = self.N_posBiggerThanK(x1, y1,self.k) #N+>=K
            if x2.size()[0]:
                loss2 = self.N_posSmallerThanK(x2, y2, self.k) #N+<K

        return loss1 + loss2, self.sparsity, err_pos

    def get_1d_array(self,N,i):
        mask = torch.zeros(N,requires_grad=False).cuda()
        j = 0
        counter = 0
        while counter!=N:
            if j==i:
                j+=1
            mask[counter]=j
            counter+=1
            j+=1
        return mask

    def N_posSmallerThanK(self, x, y_, k):
        n_pos=y_[0].sum()
        #x_1:x,  x_2:have no ground truth  x_3:only have ground truth
        x_1, x_2, x_3 = split_2(x, y_)  
        term1 = 0
        term2 = x_3.sum() / x_3.size(0)

        term1 += self.F_sk(x_1,k).sum() / x_1.size(0)
        
        temp=int((k-n_pos).cpu().numpy())
        term2 += self.F_sk(x_2,temp).sum() / x_2.size(0)  
        return term1 - term2

    def N_posBiggerThanK(self,x, y_,k):
        x_1, x_2 = split(x, y_)# add margin x1 all score ,x2 n_pos score
        smooth_1, hard_1 = detect_large(x_1, k, self.tau, self.thresh)
        smooth_2, hard_2 = detect_large(x_2, k, self.tau, self.thresh)
        loss1 = 0
        loss2 = 0

        if smooth_1.data.sum():
            x_s1 = x_1[smooth_1]
            x_s1 = x_s1.view(-1, x_1.size(1))
            if self.tau!=0:
                loss1 += self.F_sk(x_s1,k).sum() / x_1.size(0)
            else:
                loss1 += self.F_hk(x_s1,k).sum() / x_1.size(0)              
         
        if hard_1.data.sum():
            x_h1 = x_1[hard_1]
            x_h1 = x_h1.view(-1, x_1.size(1))
            loss1 += self.F_hk(x_h1,self.k).sum() / x_1.size(0)

        if smooth_2.data.sum():
            x_s2 = x_2[smooth_2]
            x_s2 = x_s2.view(-1, x_2.size(1))

            if self.tau!=0:
                if x_s2.size()[0]-k<=2:
                    loss2 += self.F_hk(x_s2,k).sum() / x_2.size(0)
                else:
                    loss2 += self.F_sk(x_s2,k).sum() / x_2.size(0)
            else:
                loss2 += self.F_hk(x_s2,k).sum() / x_2.size(0)
                
        
        if hard_2.data.sum():
            x_h2 = x_2[hard_2]
            x_h2 = x_h2.view(-1, x_2.size(1))
            loss2 += self.F_hk(x_h2,k).sum() / x_2.size(0)

        return loss1 - loss2

    def get_losses(self):
        self.F_hk = F.Topk_Hard_SVM()
        self.F_sk = F.Topk_Smooth_SVM(tau=self.tau)