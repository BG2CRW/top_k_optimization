import torch
import torch.nn as nn
import losses.sigma.functional as F
from losses.sigma.utils import detect_large, split, split_2, split_KN_pos
import numpy as np

def Another_P_at_K(n_input, margin, tau, k, n_pos):
	return Another_P_at_K_Loss(n_input=n_input, margin=margin, tau=tau, k=k,n_pos=n_pos)

class Another_P_at_K_Loss(nn.Module):

	def __init__(self, n_input, margin, tau, k, n_pos):
		super(Another_P_at_K_Loss, self).__init__()
		self.k = k
		self.n_pos = 0
		self.tau = tau
		self.thresh = 1e2
		self.margin = margin
		self.N = n_input
		self.get_losses()
		self.sparsity = 0
		self.err_pos = 0

	def get_grad(self,g):
		res = (1-g.eq(0)).sum().float()/(g.size(0)*(g.size(1)))
		self.sparsity = res
	
	def forward(self, embedding, label):
		embd_size = embedding.size(1)
		batch_size = embedding.size(0)

		sample_list=[]
		n_pos_temp=[]
		categories=-1
		for i in range(batch_size):
			if label[i] not in sample_list:
				sample_list.append(label[i])
				categories+=1
				n_pos_temp.append(1)
			else:
				n_pos_temp[categories]+=1
		n_pos=torch.from_numpy(np.array(n_pos_temp))
		# Compute similarity matrix
		score = torch.matmul(embedding, embedding.t()).cuda()
		eyes_ = torch.eye(batch_size, batch_size).cuda().byte()
		y_=label.expand(batch_size, batch_size).eq(label.expand(batch_size, batch_size).t()).float()
		score = torch.masked_select(score, 1-eyes_).view(batch_size,-1)
		y_ = torch.masked_select(y_, 1-eyes_).view(batch_size,-1)
		score_hat = score.add(self.margin*(1-y_))

		'''
		_,index = score.topk(self.k, dim=1)
		self.err_pos=2*torch.gt(index,self.n_pos-1).sum().float()/(score.size(0)*(score.size(1)))
		score.register_hook(lambda g:self.get_grad(g))
		'''
		group_x=[]
		group_y_=[]
		pointer=0
		for i in range(len(n_pos)):
			if n_pos[i]!=1:
				group_x.append(score_hat[pointer:pointer+int(n_pos[i]),:])
				group_y_.append(y_[pointer:pointer+int(n_pos[i]),:])
				pointer+=int(n_pos[i])

				_,index = group_x[i].topk(self.k, dim=1)
				for j in range(index.size(0)):
					predict_topk = group_y_[i][j][index[j]]
					real_gt_num = min(self.k , n_pos[i]-1)
					self.err_pos+=real_gt_num-predict_topk.sum()

		loss1=0
		loss2=0
		
		for i in range(len(group_x)):#split k>n+ and k<n+
			x1,y1,x2,y2 = split_KN_pos(group_x[i], group_y_[i], self.k)
			if x1.size()[0]:
				loss1 += self.N_posBiggerThanK(x1, y1,self.k) #N+>K
			if x2.size()[0]:
				loss2 += self.N_posSmallerThanK(x2, y2, self.k) #N+<K


		miu = embedding.mean(0)  #[1*width]
		embedding_centre=embedding-miu
		a=embedding_centre.unsqueeze(2)
		b=embedding_centre.unsqueeze(1)
		outer_product = a*b
		mean=outer_product.mean(0)
		loss3=(mean-torch.eye(embd_size,embd_size).cuda()).norm()

		return loss1 + loss2 + 0.1*loss3, self.err_pos, self.sparsity

	def N_posSmallerThanK(self, x, y_, k):
		n_pos=y_[0].sum()
		#x_1:x,  x_2:have no ground truth  x_3:only have ground truth
		x_1, x_2, x_3 = split_2(x, y_)  
		term2 = x_3.sum() / x_3.size(0)
		if self.tau!=0:
			term1 = self.F_sk(x_1,k).sum() / x_1.size(0)
			term2 += self.F_sk(x_2,int(k-n_pos)).sum() / x_2.size(0)
		else:
			term1 = self.F_hk(x_1,k).sum() / x_1.size(0)
			term2 += self.F_hk(x_2,int(k-n_pos)).sum() / x_2.size(0)
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