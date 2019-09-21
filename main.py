#code:utf8
import torch
import os
from losses.main import get_loss
import numpy as np
from tqdm import tqdm
from torch import nn
import math
import val
import models.bn_inception as network
import models.embed as embed
import torchvision.models as model
from sample_data.sample_data import Preprocess
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics.pairwise
import sklearn.cluster
import sklearn.metrics.cluster
import threading

max_prec=0 
max_history=0

def run(cfg):
	print(cfg.NET)
	if cfg.NET=="bn_inception_v2":
		net = network.bn_inception(pretrained = True)
	if cfg.NET=="densenet201":
		net = model.densenet201(pretrained = True)
	embed.embed(net, sz_embedding=cfg.EMBEDDING_WIDTH,normalize_output = True, net_id = cfg.NET)

	if cfg.USE_CUDA==1: 
		net.cuda()
	metric_loss = get_loss(n_input=cfg.N, k=cfg.K, tau=0,n_pos=cfg.POS_SAMPLE_NUM, margin=cfg.MARGIN,input_dim=cfg.EMBEDDING_WIDTH,output_dim=cfg.TRAIN_CLASS,batch_size=cfg.BATCH_SIZE,method=cfg.METHOD).cuda()

	optimizer = torch.optim.Adam(
    [
        { # embedding parameters
            'params': net.embedding_layer.parameters(), 
            'lr' : cfg.EMBD_LR
        },
        { # architecture parameters, excluding embedding layer
            'params': list(
                set(
                    net.parameters()
                ).difference(
                    set(net.embedding_layer.parameters())
                )
            ), 
            'lr' : cfg.NET_LR
        },
        { # metric loss parameters
            'params': metric_loss.parameters(), 
            'lr': cfg.METRIC_LOSS_LR
        },
    ],
    eps = 1e-2,
    weight_decay = cfg.WEIGHT_DECAY
)
	#for i in metric_loss.named_parameters():
	#	print(i)
	model_name=cfg.MODEL_PATH+str(cfg.MARGIN)+str(cfg.EMBEDDING_WIDTH)+str(cfg.METRIC_LOSS_PARAM)+str(cfg.METHOD)+str(cfg.K)+str(cfg.POS_SAMPLE_NUM)+".pkl"
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SCHEDULER_STEP, gamma = cfg.GAMMA)    #[10,30,70]
	print("metric_rate:",cfg.METRIC_LOSS_PARAM)
	if cfg.TRAINING_OLD==1:
		print("Load model params")
		net.load_state_dict(torch.load(model_name))

	preprocess = Preprocess(root=cfg.DATA_ROOT,use_cuda=cfg.USE_CUDA,train_batch_size=cfg.BATCH_SIZE,test_batch_size=cfg.TEST_BATCH_SIZE,method=cfg.METHOD,dataset_name=cfg.DATASET,with_bounding_box=cfg.WITH_BOUNDING_BOX,download=cfg.DOWNLOAD,n_pos=cfg.POS_SAMPLE_NUM,N=cfg.N)
	metric = 'cosine'
	print("embd_size=",cfg.EMBEDDING_WIDTH,"dataset=",cfg.DATASET,"batch_size=",cfg.BATCH_SIZE,"GPU ID=",cfg.GPU_NUM)
	print("EMBD_LR=",cfg.EMBD_LR,"SOFTMAX_LOSS_LR=",cfg.SOFTMAX_LOSS_LR,"NET_LR=",cfg.NET_LR)
	if cfg.METHOD==0:
		print("K=",cfg.K,"N=",cfg.N,"N+=",cfg.POS_SAMPLE_NUM,cfg.BATCH_SIZE,'margin=',cfg.MARGIN)
	
	run_num=0
	sparsity=0
	err_pos=0
	old_err_pos=0
	total_sparsity=0
	total_err_pos=0
	totalEpochLoss=0
	#X1, T1, _=val.inference(net,preprocess.test_loader)
	#save_log(X1,T1,metric,run_num,totalEpochLoss,total_sparsity,net,model_name)
	for epoch in range(cfg.EPOCH):
	#train
		scheduler.step()
		train_loader=iter(preprocess.train_loader)
		
		for _ in tqdm(range(len(preprocess.train_loader))):
			if run_num%cfg.SHOW_PER_ITER==cfg.SHOW_PER_ITER-1:
				X1, T1, _=val.inference(net,preprocess.test_loader)
				
				if cfg.MULTI_THREAD == 1:
					t = threading.Thread(target=save_log,args=(X1,T1,metric,run_num,totalEpochLoss,total_err_pos,net,model_name))#create threading
					t.setDaemon(True)
					t.start()
				else:
					save_log(X1,T1,metric,run_num,totalEpochLoss,total_sparsity,net,model_name)
				
				totalEpochLoss=0

			data_ = train_loader.next()
			batch_img, real_y, img_name = data_

			optimizer.zero_grad()
			out=net(batch_img.cuda())
			loss_metric,err_pos,sparsity=metric_loss(out,real_y.cuda())
			loss=loss_metric

			total_err_pos=err_pos
			totalEpochLoss=totalEpochLoss+loss.data/cfg.SHOW_PER_ITER
			if math.isnan(loss.data)==False:
				loss.backward()
				optimizer.step()
			else:
				print(loss.data)
				
			run_num+=1
		print("\r\nEpoch:",epoch)
		

def save_log(X1,T1,metric,run_num,totalEpochLoss,total_err_pos,net,model_name):
	global max_prec
	global max_history
	print(max_history)
	print(model_name)
	if cfg.DATASET == 'snapshop':
		recall=val.test_snapshop(X1, T1,metric,cfg.TEST_K)
		print("iter:",run_num,"loss:",totalEpochLoss)
		recall_test="&"+str('%.2f'%recall[0])+"&"+str('%.2f'%recall[1])+"&"+str('%.2f'%recall[2])+"\\\\"
		print(recall)
		torch.save(net.state_dict(), model_name)
		if recall[0]>max_prec:
			max_prec=recall[0]
			max_history = recall_test
			torch.save(net.state_dict(), model_name)	

	else:
		
		preck_test,recallk_test=val.test(X1, T1, cfg.TEST_K,metric)
		nmi,mAP,PR,ROC,F1 = 0,0,0,0,0 
		print("iter:",run_num,"loss:",totalEpochLoss)
		prec_recall_test="&"+str('%.2f'%preck_test[0])+"&"+str('%.2f'%preck_test[1])+"&"+str('%.2f'%preck_test[2])+"&"+str('%.2f'%preck_test[3])+"&"+str('%.2f'%recallk_test[1])+"&"+str('%.2f'%recallk_test[2])+"&"+str('%.2f'%recallk_test[3])+"\\\\"
		print(prec_recall_test)
		torch.save(net.state_dict(),model_name+str('%.2f'%recallk_test[0]))	
		if recallk_test[0]>max_prec:
			max_prec=recallk_test[0]
			max_history = prec_recall_test
			torch.save(net.state_dict(),model_name+str('%.2f'%recallk_test[0]))	
			
			#nmi,mAP,PR,ROC,F1 = val.test_some_scores(X1, T1,metric,cfg.TEST_CLASS)
			#show_result = "&"+str('%.2f'%preck_test[0])+"&"+str('%.2f'%preck_test[1])+"&"+str('%.2f'%preck_test[2])+"&"+str('%.2f'%preck_test[3])+"&"+str('%.2f'%recallk_test[1])+"&"+str('%.2f'%recallk_test[2])+"&"+str('%.2f'%recallk_test[3])+"&"+str('%.2f'%nmi)+"&"+str('%.2f'%mAP)+"&"+str('%.2f'%F1)+"\\\\"
			#print(show_result)
			
		
		#ratio=total_sparsity/total_err_pos
		'''
		output = open("change_triplets"+str(cfg.DATASET)+'sampling.txt', 'a')

		output.write(str(run_num))
		output.write(' ')
		output.write(str(total_err_pos))
		output.write(' ')
		#output.write(str('%.2f'%preck_test[0]))
		output.write(' ')
		output.write('\r')
		output.close()
		'''
		
		

if __name__ == '__main__':
	import config.config_CUB200_2011 as cfg
	#import config.config_CARS196 as cfg
	#import config.config_ONLINE_PRODUCT as cfg
	os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_NUM
	run(cfg)
	
