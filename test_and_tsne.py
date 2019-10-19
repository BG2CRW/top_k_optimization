import torch
#import config.config_CUB200_2011 as cfg
#import config.config_CARS196 as cfg 
import config.config_ONLINE_PRODUCT as cfg 
import os
from losses.main import get_loss
import numpy as np
from tqdm import tqdm
from torch import nn
import math
import val
from sklearn.manifold import TSNE
from time import time
import matplotlib as mpl
from PIL import Image
mpl.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sample_data.sample_data import Preprocess
import models.bn_inception as network
import models.embed as embed
import torchvision.models as model

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_NUM

def plot_embedding(data, path):
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)
	random_arr = np.random.choice(60000,2990,replace=False)
	fig = plt.figure(figsize=(80,60))
	ax = plt.subplot(111)
	plt.xticks([])
	plt.yticks([])
	ax.axis('off') 
	#for i in tqdm(range(int(data.shape[0]/3))):
	for i in tqdm(range(random_arr.shape[0])):
		ax1 = fig.add_axes([data[random_arr[i], 0], data[random_arr[i], 1],0.02,0.02])
		img = Image.open(path[random_arr[i]])
		ax1.imshow(img)
		ax1.axis('off')   
	return fig

def draw_tsne(X ,path):
	print('Computing t-SNE embedding')
	
	data=X.cpu().numpy()
	if not os.path.exists('tsne.npz'):
		tsne = TSNE(n_components=2, init='pca', random_state=0)
		result = tsne.fit_transform(data)
		np.savez("tsne",result)
	else:
		npzfile=np.load('tsne.npz')
		result = npzfile['arr_0']
		
	fig = plot_embedding(result, path)
	fig.savefig("str(cfg.DATASET)+tsne.jpg")
	#plt.show(fig)

def run(flag):
	print(cfg.NET)

	if cfg.NET=="bn_inception_v2":
		net_id=1
	if cfg.NET=="densenet201":
		net_id=0
	if "CUB" in cfg.DATASET:
		dataset_id=0
	if "CAR" in cfg.DATASET:
		dataset_id=1
	if "Online" in cfg.DATASET:
		dataset_id=2	
	prefix="model_param/paper_result/"
	net_list=["densenet201_","bn_inception_v2_"]
	dataset_list=["cub200_2011_","cars196_","online_"]
	method_list=["angular","contrastive","hardmining","lifted","npair","preck","proxynca","samplingmatters","semihard","triplet"]
	preck_test=[0,0,0,0,0,0,0]
	recallk_test=[0,0,0,0,0,0,0]
	nmi,mAP,PR,ROC,F1=0,0,0,0,0

	name=prefix+net_list[net_id]+dataset_list[dataset_id]+method_list[flag]
	print(name)
	if cfg.METHOD==0 or cfg.METHOD==8:
		metric = 'cosine'
	else:
		metric = 'euclidean'

	if not os.path.exists(name+'.npz'):
		if cfg.NET=="bn_inception_v2":
			net = network.bn_inception(pretrained = True)
			net_id=1
		if cfg.NET=="densenet201":
			net = model.densenet201(pretrained = True)
			net_id=0
		embed.embed(net, sz_embedding=cfg.EMBEDDING_WIDTH,normalize_output = True, net_id = cfg.NET)


		if cfg.USE_CUDA==1: 
			net.cuda()
		print("Load model params")
		#print(cfg.MODEL_PATH+str(cfg.MARGIN)+str(cfg.EMBEDDING_WIDTH)+str(cfg.DATASET)+str(cfg.METRIC_LOSS_PARAM)+'x'+str(cfg.METHOD)+str(cfg.CE_LOSS_PARAM)+'x'+str(cfg.SOFTMAX_METHOD)+str(cfg.K)+str(cfg.POS_SAMPLE_NUM)+".pkl")
		#net.load_state_dict(torch.load(cfg.MODEL_PATH+str(cfg.MARGIN)+str(cfg.EMBEDDING_WIDTH)+str(cfg.DATASET)+str(cfg.METRIC_LOSS_PARAM)+'x'+str(cfg.METHOD)+str(cfg.CE_LOSS_PARAM)+'x'+str(cfg.SOFTMAX_METHOD)+str(cfg.K)+str(cfg.POS_SAMPLE_NUM)+".pkl"))

		net.load_state_dict(torch.load(name+".pkl"))

		print("Index all dataset")
		preprocess = Preprocess(root=cfg.DATA_ROOT,use_cuda=cfg.USE_CUDA,train_batch_size=cfg.BATCH_SIZE,test_batch_size=cfg.TEST_BATCH_SIZE,method=cfg.METHOD,dataset_name=cfg.DATASET,with_bounding_box=cfg.WITH_BOUNDING_BOX,download=cfg.DOWNLOAD,n_pos=cfg.POS_SAMPLE_NUM,N=cfg.N)
		print("Done!")

		print("embd_size=",cfg.EMBEDDING_WIDTH,"dataset=",cfg.DATASET)

		if cfg.METHOD==0 or cfg.METHOD==8:
			print("tau=",cfg.TAU,"K=",cfg.K,"N=",cfg.N,"N+=",cfg.POS_SAMPLE_NUM,"embd_width=",cfg.EMBEDDING_WIDTH,"batch_size=",cfg.BATCH_SIZE,'margin=',cfg.MARGIN)
		
		X1, T1, path1=val.inference(net,preprocess.test_loader)
		#X2, T2, path2=val.inference(net,preprocess.test_train_loader)
		print("inference finished!")
		X1=X1.numpy()
		T1=T1.numpy()
		np.savez(name,X1,T1,path1)
	else:
		npzfile=np.load(name+'.npz')
		X1 = npzfile['arr_0']
		T1 = npzfile['arr_1']
		path1 = npzfile['arr_2']
	
	
	X1=torch.from_numpy(X1)
	T1=torch.from_numpy(T1)
	
	#val.get_some_case(X1, T1, path1, 10, metric)
	preck_test,recallk_test=val.test(X1, T1,cfg.TEST_K,metric)
	#print("prec@K:",preck_test,"recall@K:",recallk_test)
	#preck_train,recallk_train=val.test(X2, T2,cfg.TEST_K,metric)
	#print("prec@K:",preck_train,"recall@K:",recallk_train)

	nmi,mAP,PR,ROC,F1 = val.test_some_scores(X1, T1, path1, metric,cfg.TEST_CLASS,name)

	show_result = name+"&"+str('%.2f'%preck_test[0])+"&"+str('%.2f'%preck_test[1])+"&"+str('%.2f'%preck_test[2])+"&"+str('%.2f'%preck_test[3])+"&"+str('%.2f'%recallk_test[1])+"&"+str('%.2f'%recallk_test[2])+"&"+str('%.2f'%recallk_test[3])+"&"+str('%.2f'%(nmi*100))+"&"+str('%.2f'%(mAP*100))+"&"+str('%.2f'%(F1*100))+"\\\\"
	print(show_result)
	#print(PR,ROC)
	
	#draw_tsne(X1,path1)
	
	
	

if __name__ == '__main__':
	run(7)
	#run(0)
	#for i in range(5):   #0,5   5,10
	#	run(i)

