import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import numpy as np
import random
import torch
import os
import sample_data.download_split_data as D
import threading
import queue
from torch.utils.data.sampler import Sampler
from collections import defaultdict

def default_loader(path):
	try:
		img = Image.open(path)
		return img.convert('RGB')
	except:
		print("Cannot read image: {}".format(path))
class ourData(Dataset):
	def __init__(self, img_path, txt_path, data_transforms=None, loader = default_loader):
		with open(txt_path) as input_file:
			lines = input_file.readlines()
			self.img_name = [os.path.join(img_path, line.split()[1]) for line in lines]
			self.img_label = [int(line.split(' ')[0])-1 for line in lines]
		self.data_transforms = data_transforms
		self.loader = loader
		self.sample_nums = len(self.img_label )

		classes = list(set(self.img_label))
		# Generate Index Dictionary for every class
		Index = defaultdict(list)
		for i, label in enumerate(self.img_label):
			Index[label].append(i)
		self.Index = Index

	def __len__(self):
		return len(self.img_name)

	def __getitem__(self, item):
		img_name = self.img_name[item]
		label = self.img_label[item]
		#print(img_name)
		img = self.loader(img_name)

		if self.data_transforms is not None:
			try:
				img = self.data_transforms(img)
			except:
				print("Cannot transform image: {}".format(img_name))
		return img, label, img_name

class Identity(): # used for skipping transforms
	def __call__(self, im):
		return im


class ScaleIntensities():
	def __init__(self, in_range, out_range):
		""" Scales intensities. For example [-1, 1] -> [0, 255]."""
		self.in_range = in_range
		self.out_range = out_range

	def __call__(self, tensor):
		tensor = (
			tensor - self.in_range[0]
		) / (
			self.in_range[1] - self.in_range[0]
		) * (
			self.out_range[1] - self.out_range[0]
		) + self.out_range[0]
		return tensor


def make_transform(sz_resize = 256, sz_crop = 227, mean = [128, 117, 104], #224/227
		std = [1, 1, 1], rgb_to_bgr = True, is_train = True, 
		intensity_scale = [[0, 1], [0, 255]]):
	return transforms.Compose([
		transforms.Compose([ # train: horizontal flip and random resized crop
			transforms.RandomResizedCrop(sz_crop),
			transforms.RandomHorizontalFlip(),
		]) if is_train else transforms.Compose([ # test: else center crop
			transforms.Resize(sz_resize),
			transforms.CenterCrop(sz_crop),
		]),
		transforms.ToTensor(),
		ScaleIntensities(
			*intensity_scale) if intensity_scale is not None else Identity(),
		transforms.Normalize(
			mean=mean,
			std=std,
		),
		transforms.Lambda(
			lambda x: x[[2, 1, 0], ...]
		) if rgb_to_bgr else Identity()
	])


class OurSampler(Sampler):
	def __init__(self, data_source, batch_size,method,n_pos,N):
		self.data_source = data_source
		self.index_dic = data_source.Index
		self.pids = list(self.index_dic.keys())
		self.num_samples = len(self.pids)
		self.batch_size = (batch_size//2) *2
		self.method = method
		self.classes = len(data_source.Index)   
		self.times = self.data_source.sample_nums//self.batch_size
		self.n_pos = n_pos
		self.N = N
	def __len__(self):
		return self.times*self.batch_size

	def __iter__(self):
		ret = []
		if self.method==2 or self.method==5 or self.method==7 or self.method==1:#clustering or lifted or contrastive or proxy_nca  有重复类
			for _ in range(self.times):
				y_=[0 for i in range(self.batch_size)]
				for i in range(self.batch_size//2):
					y_[i]=random.randint(0,self.classes-1)
					y_[i+self.batch_size//2]=y_[i]
				for i in range(2*self.batch_size//2):
					ret.append(self.index_dic[int(y_[i])][random.randint(0,len(self.index_dic[int(y_[i])])-1)])
		if self.method==3 or self.method==4 or self.method==6:#npair or angular or triplet   无重复类
			for _ in range(self.times):
				y_=np.array(random.sample(range(self.classes-1), self.batch_size//2))
				y_=np.concatenate([y_,y_]).tolist()
				real_index_bag=[]
				for i in range(self.batch_size):
					while True:
						temp_index=random.randint(0,len(self.index_dic[int(y_[i])])-1)
						real_index=self.index_dic[int(y_[i])][temp_index]
						if real_index not in real_index_bag:
							real_index_bag.append(real_index)
							ret.append(real_index)
							break
		if self.method==0:#prec@k loss
			for _ in range(self.times):
				flag=0  
				N=self.N
				batch_size = self.batch_size//(N+1)
				K_arr=[]	
				real_y=torch.zeros((N+1)*batch_size)
				for i in range(batch_size):
					classid=random.randint(0,self.classes-1) #random choose a class 1-11318 training class#11317
					temp=self.index_dic[classid][:len(self.index_dic[classid])]
					if self.n_pos+1>len(temp):
						flag=0
					if self.n_pos+1<len(temp):
						flag=1
					K=np.min([self.n_pos+1,len(temp)])-1
					K_arr.append(K)

					for j in range (K+1):
						real_y[j+i*(N+1)]=classid
					for k in range(N-K):
						while True:
							neg_id=random.randint(0,self.classes-1)
							if neg_id!=classid:
								break
						real_y[k+K+1+i*(self.N+1)]=neg_id
				pointer=0
				if flag==0:
					for i in range(batch_size):
						for j in range(K_arr[i]+1):
							temp_index=self.index_dic[int(real_y[pointer])][j]
							ret.append(temp_index)
							pointer+=1
						for j in range(N-K_arr[i]):
							temp_index=self.index_dic[int(real_y[pointer])][random.randint(0,len(self.index_dic[int(real_y[pointer])])-1)]
							ret.append(temp_index)
							pointer+=1
				else:
					for i in range((N+1)*batch_size):
						temp_index=self.index_dic[int(real_y[i])][random.randint(0,len(self.index_dic[int(real_y[i])])-1)]
						ret.append(temp_index)
		if self.method==8 :#another prec@k loss
			for _ in range(self.times):
				cnt=0  
				sample_classes=98
				slices = random.sample([i for i in range(self.classes)], sample_classes)
				break_flag=False
				for i in range(sample_classes):
					category_num=len(self.index_dic[int(slices[i])])
					candidate=self.index_dic[slices[i]][:category_num]
					if category_num>=self.n_pos+1:
						candidate=random.sample([candidate[i] for i in range(len(candidate))],self.n_pos+1)
					for j in range(len(candidate)):
						cnt+=1
						ret.append(candidate[j])
						if cnt==self.batch_size:
							break_flag=True
							break
					if break_flag==True:
						break
		if self.method==9 :#sampling matters loss
			for _ in range(self.times):
				cnt=0  
				sample_classes=98
				slices = random.sample([i for i in range(self.classes)], sample_classes)
				break_flag=False
				n_pos=2
				for i in range(sample_classes):
					category_num=len(self.index_dic[int(slices[i])])
					candidate=self.index_dic[slices[i]][:category_num]
					if category_num>=n_pos:
						candidate=random.sample([candidate[i] for i in range(len(candidate))],n_pos)
					for j in range(len(candidate)):
						cnt+=1
						ret.append(candidate[j])
						if cnt==self.batch_size:
							break_flag=True
							break
					if break_flag==True:
						break
		return iter(ret)


class Preprocess():
	def __init__(self,root,use_cuda,train_batch_size,test_batch_size,method,dataset_name,with_bounding_box,download,n_pos,N):
		self.root = os.path.join(root, dataset_name)
		self.use_cuda=use_cuda
		self.test_batch_size=test_batch_size
		self.train_batch_size=train_batch_size
		self.method=method
		self.size = 227
		self.n_pos=n_pos
		self.N=N

		if 'CARS196' == dataset_name:
			if download==True:
				D.download_split_data.download_and_split_CARS196(root)
			train_img_path = self.root
			test_img_path = self.root
			train_txt_path = os.path.join(self.root,'train.txt')
			test_txt_path = os.path.join(self.root,'test.txt')
			if with_bounding_box==True:
				train_txt_path = os.path.join(self.root,'bounding_train.txt')
				test_txt_path = os.path.join(self.root,'bounding_test.txt')
		if 'CUB200-2011' == dataset_name:
			if download==True:
				D.download_split_data.download_and_split_CUB200_2011(root)
			train_img_path = os.path.join(self.root,'CUB_200_2011')
			test_img_path = os.path.join(self.root,'CUB_200_2011')
			train_txt_path = os.path.join(self.root,'CUB_200_2011','new_train.txt')
			test_txt_path = os.path.join(self.root,'CUB_200_2011','new_test.txt')
			if with_bounding_box==True:
				train_txt_path = os.path.join(self.root,'CUB_200_2011','new_bounding_train.txt')
				test_txt_path = os.path.join(self.root,'CUB_200_2011','new_bounding_test.txt')
		if 'Stanford_Online_Products' == dataset_name:
			if download==True:
				D.download_split_data.download_and_split_Stanford_Online_Products(root)
			train_img_path = self.root
			test_img_path = self.root
			train_txt_path = os.path.join(self.root,'new_train.txt')
			test_txt_path = os.path.join(self.root,'new_test.txt')
		self.dataset_train = ourData(img_path=train_img_path,
										txt_path=train_txt_path,
										data_transforms=make_transform()) 
		self.dataset_test = ourData(img_path=test_img_path,
										txt_path=test_txt_path,
										data_transforms=make_transform(is_train=False)) 

		if self.method==0:
			self.train_batch_size = (self.N+1)*self.train_batch_size

		print("Dataset: "+dataset_name," with bounding_box: ",with_bounding_box)
		print("Total images in training sets: ",self.dataset_train.sample_nums)
		self.test_loader = data.DataLoader(dataset=self.dataset_test,batch_size=self.test_batch_size,shuffle=False,num_workers = 4,pin_memory = True)
		self.test_train_loader = data.DataLoader(dataset=self.dataset_train,batch_size=self.test_batch_size,shuffle=False,num_workers = 4,pin_memory = True)
		self.train_loader = data.DataLoader(dataset=self.dataset_train,batch_size=self.train_batch_size,
			sampler=OurSampler(self.dataset_train, batch_size=self.train_batch_size,method=self.method,
				n_pos=self.n_pos,N=self.N),num_workers = 10,drop_last=True,pin_memory = True)

	def test_loader(self):
		return self.test_loader
	def test_train_loader(self):
		return self.test_train_loader
	def train_loader(self):
		return self.train_loader

