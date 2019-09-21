# -*- coding: utf-8 -*-  
import sklearn.metrics.pairwise
import numpy as np
import torch
from datetime import datetime
import sklearn.cluster
import sklearn.metrics.cluster
from sklearn.metrics import f1_score
from tqdm import tqdm
from itertools import combinations
import os
def predict_batchwise(model, dataloader):
	with torch.no_grad():
		X, Y, Z = zip(*[
			[x, y, z] for X, Y, Z in dataloader
				for x, y, z in zip(
					model(X.cuda()).cpu(), 
					Y, Z
				)
		])
	return torch.stack(X), torch.stack(Y), Z

def assign_by_dist_at_k(X, T, k ,metric):
	""" 
	X : [nb_samples x nb_features], e.g. 3000 x 64 (embeddings)
	T : [nb_samples] (target labels)
	k : for each sample, assign target labels of k nearest points
	"""
	distances = sklearn.metrics.pairwise.pairwise_distances(X,metric=metric)
	#
	# get nearest points
	indices = np.argpartition(distances, [i for i in range(k+1)],axis=1)[:,1:k+1]
	label_sort = np.array([[T[i] for i in ii] for ii in indices])
	Y_=torch.eq(torch.from_numpy(label_sort),T.view(-1,1))
	return Y_

def assign_by_dist_at_k_with_path(X, T, path,k,metric):
	""" 
	X : [nb_samples x nb_features], e.g. 3000 x 64 (embeddings)
	T : [nb_samples] (target labels)
	k : for each sample, assign target labels of k nearest points
	"""
	distances = sklearn.metrics.pairwise.pairwise_distances(X,metric=metric)
	#
	# get nearest points
	indices = np.argpartition(distances, [i for i in range(k+1)],axis=1)[:,0:k+1]
	#import pdb;pdb.set_trace()
	label_sort = np.array([[T[i] for i in ii] for ii in indices])
	path_sort = np.array([[path[i] for i in ii] for ii in indices])
	Y_=torch.eq(torch.from_numpy(label_sort),T.view(-1,1))
	output = open("online_preck"+'.txt', 'a')
	for i in range(path_sort.shape[0]):
		output.write(str(Y_[i]))
		output.write(str(path_sort[i]))
	output.close()
	return Y_

def calc_recall_at_k_(Y, k):
	"""
	Y : [nb_samples x k] (k predicted labels/neighbours)
	"""
	s=0
	for i in range(Y.size(0)):
		if 1 in Y[i,:][:k]:
			s += 1
	return s / (1. * Y.size(0))

def calc_prec_at_k_(Y, k):
	"""
	Y : [nb_samples x k] (k predicted labels/neighbours)
	"""
	s = 0
	for i in range(Y.size(0)):
		s+=Y[i,:][:k].sum().numpy()
	return s / (1.*Y.size(0)*k)

def calc_mAP(Y_):
	mAP = 0
	for i in tqdm(range(Y_.size(0))):
		cnt_T = 0
		AP = 0
		for j in range(Y_.size(1)):
			if Y_[i][j]==1:
				cnt_T+=1
				AP+=cnt_T/(j+1)
		if cnt_T !=0:		
			AP = AP/cnt_T
		else:
			AP=0
		mAP += AP
	mAP = mAP/Y_.size(0)
	return mAP

def calc_PR(Y_):
	inter_cnt = 100
	interPR=np.zeros((Y_.size(0),inter_cnt+1))
	for i in range(Y_.size(0)):
		cnt_T = 0
		PR_sub=[]
		for j in range(Y_.size(1)):
			if Y_[i][j]==1:
				cnt_T+=1
				PR_sub.append(cnt_T/(j+1))
		PR_sub=np.array(PR_sub)
		real_x=[(k+1)/cnt_T for k in range(cnt_T)]
		inter_x=[k/inter_cnt for k in range(inter_cnt+1)]
		inter_prec=np.interp(inter_x, real_x, PR_sub, PR_sub[0], 99)
		interPR[i]=inter_prec
	PR=np.mean(interPR, axis=0)
	return PR

def calc_ROC(Y_):
	true_cnt = Y_.sum(1)
	inter_cnt = 100
	interROC=np.zeros((Y_.size(0),inter_cnt+1))
	for i in range(Y_.size(0)):
		cnt_T = 0
		cnt_F = 0
		ROC_x=[]#FPR
		ROC_y=[]#TPR
		for j in range(Y_.size(1)):
			if Y_[i][j]==0:
				cnt_F+=1
				ROC_x.append(cnt_F/((Y_.size(1)-true_cnt[i]).numpy()*1.))
				ROC_y.append(Y_[i][0:j].sum().numpy()/(1.*true_cnt[i].numpy()))
		ROC_x=np.array(ROC_x)
		ROC_y=np.array(ROC_y)
		inter_x=[k/inter_cnt for k in range(inter_cnt+1)]
		inter_y=np.interp(inter_x, ROC_x, ROC_y, ROC_y[0], 99)
		interROC[i]=inter_y
	ROC=np.mean(interROC, axis=0)
	return ROC

def calc_mAP_PR_ROC_with_interp(Y_):
	true_cnt = Y_.sum(1)
	inter_cnt = 20
	interPR=np.zeros((Y_.size(0),inter_cnt+1))
	interROC=np.zeros((Y_.size(0),inter_cnt+1))
	mAP = 0
	for i in tqdm(range(Y_.size(0))):
		cnt_T = 0
		cnt_F = 0
		ROC_x=[]#FPR
		ROC_y=[]#TPR
		PR_sub=[]
		AP = 0
		for j in range(Y_.size(1)):
			#ROC
			if Y_[i][j]==0:
				cnt_F+=1
				ROC_x.append(cnt_F/((Y_.size(1)-true_cnt[i]).numpy()*1.))
				ROC_y.append(Y_[i][0:j].sum().numpy()/(1.*true_cnt[i].numpy()))
			#PR and mAP
			if Y_[i][j]==1:
				cnt_T+=1
				PR_sub.append(cnt_T/(j+1))
				AP+=cnt_T/(j+1)
		#mAP
		if cnt_T !=0:		
			AP = AP/cnt_T
		else:
			AP=0
		mAP += AP
		#PR
		PR_sub=np.array(PR_sub)
		real_x=[(k+1)/cnt_T for k in range(cnt_T)]
		inter_x=[k/inter_cnt for k in range(inter_cnt+1)]
		inter_prec=np.interp(inter_x, real_x, PR_sub, PR_sub[0], 99)
		interPR[i]=inter_prec
		#ROC
		ROC_x=np.array(ROC_x)
		ROC_y=np.array(ROC_y)
		inter_x=[k/inter_cnt for k in range(inter_cnt+1)]
		inter_y=np.interp(inter_x, ROC_x, ROC_y, ROC_y[0], 99)
		interROC[i]=inter_y
	
	mAP = mAP/Y_.size(0)
	PR=np.mean(interPR, axis=0)
	ROC=np.mean(interROC, axis=0)
	
	return mAP, PR, ROC


def calc_mAP_PR_ROC_without_interp(Y_):
	true_cnt = Y_.sum(1)
	all_sample_num = Y_.size(0)-1
	inter_cnt = 10
	PR=np.zeros((inter_cnt+1))
	ROC=np.zeros((inter_cnt+1))
	x=np.array([i/inter_cnt for i in range(inter_cnt+1)])
	mAP = 0
	for i in tqdm(range(Y_.size(0))):
		cnt_T = 0
		cnt_F = 0
		PR_sub=[]
		AP = 0
		roc_pointer=0
		pr_pointer=0
		for j in range(Y_.size(1)):
			#ROC
			if Y_[i][j]==0:
				cnt_F+=1
				ROC_x_temp=cnt_F/((Y_.size(1)-true_cnt[i]).numpy()*1.)
				ROC_y_temp=Y_[i][0:j].sum().numpy()/(1.*true_cnt[i].numpy())
				if ROC_x_temp>=x[roc_pointer+1]:
					if roc_pointer+1!=inter_cnt:
						ROC[roc_pointer+1]+=ROC_y_temp/(all_sample_num)
					roc_pointer=roc_pointer+1
				if ROC_x_temp==1:
					ROC[-1]+=ROC_y_temp/(all_sample_num)
			#PR	
			if Y_[i][j]==1:
				cnt_T+=1
				PR_x_temp=cnt_T/true_cnt.numpy()[i]#recall
				PR_y_temp=cnt_T/(j+1)#prec
				if cnt_T==1:
					PR[0]+=PR_y_temp/(all_sample_num)
				if PR_x_temp>=x[pr_pointer+1]:
					if pr_pointer+1!=inter_cnt:
						PR[pr_pointer+1]+=PR_y_temp/(all_sample_num)
					pr_pointer+=1
				if PR_x_temp==1:
					PR[-1]+=PR_y_temp/(all_sample_num)
				AP+=cnt_T/(j+1)
		#mAP
		if cnt_T !=0:		
			AP = AP/cnt_T
		else:
			AP=0
		mAP += AP
	
	mAP = mAP/Y_.size(0)	
	return mAP, PR, ROC

def calc_mAP_PR_ROC_for_part_sort(Y_,Y_real_sum):
	true_cnt = Y_real_sum
	all_sample_num = Y_.size(0)-1
	inter_cnt = 10
	PR=np.zeros((inter_cnt+1))
	ROC=np.zeros((inter_cnt+1))
	x_roc=np.array([0,1e-5,2e-5,5e-5,8e-5,1e-4,5e-4,1e-3,5e-3,1e-2,1])
	x_pr=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
	mAP = 0
	for i in tqdm(range(Y_.size(0))):
		cnt_T = 0
		cnt_F = 0
		PR_sub=[]
		AP = 0
		roc_pointer=0
		pr_pointer=0
		PR_one_slice=np.zeros((inter_cnt+1))
		for j in range(Y_.size(1)):
			
			#ROC
			if Y_[i][j]==0:
				cnt_F+=1
				ROC_x_temp=cnt_F/((all_sample_num-true_cnt[i]).numpy()*1.)
				ROC_y_temp=Y_[i][0:j].sum().numpy()/(1.*true_cnt[i].numpy())
				#print(ROC_x_temp,ROC_y_temp)
				if ROC_x_temp>=x_roc[roc_pointer+1]:
					if roc_pointer+1!=inter_cnt:
						ROC[roc_pointer+1]+=ROC_y_temp/(all_sample_num*1.)
					roc_pointer+=1
				if j==Y_.size(1)-1:
					while roc_pointer!=9:
						ROC[roc_pointer+1]+=ROC_y_temp/(all_sample_num*1.)
						roc_pointer+=1
					ROC[-1]+=1./(all_sample_num)
			#PR	
			if Y_[i][j]==1:
				cnt_T+=1
				PR_x_temp=cnt_T/true_cnt.numpy()[i]#recall
				PR_y_temp=cnt_T/(j+1)#prec
				for k in range(PR_one_slice.shape[0]):
					if PR_x_temp<=x_pr[k]:
						PR_one_slice[k]=PR_y_temp
						break
				AP+=cnt_T/(j+1)
		old_num = PR_one_slice[-1]
		for i in range(PR_one_slice.shape[0]):
			if PR_one_slice[10-i]==0:
				PR_one_slice[10-i]=old_num
			else:
				old_num=PR_one_slice[10-i]
		PR+=PR_one_slice/(all_sample_num*1.)
		#mAP
		if cnt_T !=0:		
			AP = AP/cnt_T
		else:
			AP=0
		mAP += AP
	
	mAP = mAP/Y_.size(0)	
	return mAP, PR, ROC

def calc_nmi_F1(nb_classes,X,T):
	nmi,F1=0,0
	kmeans=sklearn.cluster.KMeans(n_clusters=nb_classes, n_jobs=-1).fit(X).labels_
	nmi = sklearn.metrics.cluster.normalized_mutual_info_score(kmeans, T)
	kmeans_part_cnt=np.zeros(nb_classes,dtype=int)
	label_part_cnt=np.zeros(nb_classes,dtype=int)
	kmeans_index=np.zeros((nb_classes,50000),dtype=int)
	label_index=np.zeros((nb_classes,50000),dtype=int)
	T=T-T.min()
	kmeans=kmeans-np.min(kmeans)
	print(kmeans.shape)
	for i in range(kmeans.shape[0]):
		kmeans_index[kmeans[i]][int(kmeans_part_cnt[kmeans[i]])]=i
		kmeans_part_cnt[kmeans[i]]+=1
		label_index[T[i]][int(label_part_cnt[T[i]])]=i
		label_part_cnt[T[i]]+=1

	intersection=0
	len_kmeans_pair = 0
	len_gt_pair = 0
	for i in tqdm(range(nb_classes)):
		kmeans_combins = [c for c in  combinations(kmeans_index[i][:int(kmeans_part_cnt[i])].tolist(), 2)]
		len_kmeans_pair+=kmeans_part_cnt[i]*(kmeans_part_cnt[i]-1)//2
		len_gt_pair+=label_part_cnt[i]*(label_part_cnt[i]-1)//2
		for j in range(len(kmeans_combins)):
			if T[kmeans_combins[j][0]]==T[kmeans_combins[j][1]]:
				intersection+=1
	F1=(2.*intersection)/(len_kmeans_pair+len_gt_pair)
	return nmi,F1

def inference(net,test_loader):
	net_is_training = net.training
	net.eval()
	# calculate embeddings with model, also get labels (non-batch-wise) 
	X, T, img_path = predict_batchwise(net, test_loader)
	net.train(net_is_training) # revert to previous training state
	return X, T, img_path
def test_snapshop(X, T, metric ,test_k):
	X_query=X[0:2000,:]
	T_query=T[0:2000]
	X_db=X[2000:T.size(0),:]
	T_db=T[2000:T.size(0)]
	distances = sklearn.metrics.pairwise.pairwise_distances(X_query,X_db,metric=metric)
	indices = np.argpartition(distances, kth=[i for i in range(test_k)],axis=1)[:,0:test_k]
	label_sort = np.array([[T[i] for i in ii] for ii in indices])
	Y_=torch.eq(torch.from_numpy(label_sort),T_query.view(-1,1))
	recall = []
	for k in [1, 3, 5]:
		r_at_k = calc_recall_at_k_(Y_, k)
		recall.append(r_at_k)
	return recall

def get_some_case(X, T, path, test_k, metric):
	Y_ = assign_by_dist_at_k_with_path(X, T, path,test_k,metric)


def test(X, T,test_k,metric):	
	# calculate recall @ 1, 3, 5, 10
	recall = []
	prec = []

	Y_ = assign_by_dist_at_k(X, T, test_k,metric)
	for k in [1, 3, 5,10]:
		r_at_k = calc_recall_at_k_(Y_, k)
		recall.append(r_at_k*100)
		p_at_k = calc_prec_at_k_(Y_, k)
		prec.append(p_at_k*100)

	return prec,recall

def test_some_scores(X, T, path,metric,nb_classes,name):
	nmi,mAP,PR,ROC,F1=0,0,0,0,0
	#compute Y_
	if not os.path.exists(name+'Y_.npz'):
		if 'Online' in path[0]:
			Y_ = assign_by_dist_at_k(X, T, min(500,T.size(0)-1),metric)
		else:
			Y_ = assign_by_dist_at_k(X, T, T.size(0)-1,metric)
		Y_=Y_.numpy()
		np.savez(name+'Y_',Y_)
	else:
		npzfile=np.load(name+'Y_.npz')
		Y_ = npzfile['arr_0']
	Y_ = torch.from_numpy(Y_)
	#mAP=calc_mAP(Y_)

	#compute mAP&PR&ROC
	if 'Online' not in path[0]:
		mAP,PR,ROC=calc_mAP_PR_ROC_without_interp(Y_)
	else:
		T=T-T.min()
		T_index=torch.zeros(nb_classes)
		for i in range(T.size(0)):
			T_index[T[i]]+=1
		Y_real_sum=torch.zeros(T.size(0))
		for i in range(T.size(0)):
			Y_real_sum[i]=T_index[T[i]]

		mAP,PR,ROC=calc_mAP_PR_ROC_for_part_sort(Y_,Y_real_sum)

	#compute NMI&F1
	if 'Online' in path[0]:
		nb_classes=12
		for i in range(len(path)):
			if 'toaster' in path[i]:
				T[i]=0
			if 'table' in path[i]:
				T[i]=1
			if 'stapler' in path[i]:
				T[i]=2
			if 'sofa' in path[i]:
				T[i]=3
			if 'mug' in path[i]:
				T[i]=4
			if 'lamp' in path[i]:
				T[i]=5
			if 'kettle' in path[i]:
				T[i]=6
			if 'fan' in path[i]:
				T[i]=7
			if 'coffee' in path[i]:
				T[i]=8
			if 'chair' in path[i]:
				T[i]=9
			if 'cabinet' in path[i]:
				T[i]=10
			if 'bicycle' in path[i]:
				T[i]=11
	nmi,F1 = calc_nmi_F1(nb_classes,X,T)
	
	return nmi,mAP,PR,ROC,F1