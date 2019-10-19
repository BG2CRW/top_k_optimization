import os
import sys
import requests
from six.moves import urllib
import tarfile
import scipy.io
from PIL import Image
from tqdm import tqdm
import linecache
import numpy as np
import zipfile

def maybe_download(filename, data_dir, SOURCE_URL):
	"""Download the data from Yann's website, unless it's already here."""
	filepath = os.path.join(data_dir, filename)
	print(filepath)
	if not os.path.exists(filepath):
		filepath, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
def extract(data_dir,target_path):
	tar = tarfile.open(data_dir, "r:gz")
	file_names = tar.getnames()
	for file_name in file_names:
		tar.extract(file_name,target_path)
	tar.close()
def extract_zip(data_dir,target_path):
	print(data_dir,target_path)
	f = zipfile.ZipFile(data_dir,'r')
	for file in f.namelist():
		f.extract(file,target_path)
def download_and_split_all_datasets(DATA_ROOT):
	download_and_split_CUB200_2011(DATA_ROOT)
	download_and_split_CARS196(DATA_ROOT)
	download_and_split_Stanford_Online_Products(DATA_ROOT)

def download_and_split_CUB200_2011(DATA_ROOT):
	#CUB200-2011
	if not os.path.exists(os.path.join(DATA_ROOT,'CUB200-2011')):
		os.mkdir(os.path.join(DATA_ROOT,'CUB200-2011'))
	print('Download CUB_200_2011.tgz...')
	maybe_download('CUB_200_2011.tgz', os.path.join(DATA_ROOT,'CUB200-2011'),'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz')
	print('Download segmentations.tgz...')
	maybe_download('segmentations.tgz', os.path.join(DATA_ROOT,'CUB200-2011'),'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz')
	print('Extracting CUB_200_2011...')
	extract(os.path.join(DATA_ROOT,'CUB200-2011','CUB_200_2011.tgz'), os.path.join(DATA_ROOT,'CUB200-2011'))
	print('Extracting segmentations.tgz...')
	extract(os.path.join(DATA_ROOT,'CUB200-2011','segmentations.tgz'), os.path.join(DATA_ROOT,'CUB200-2011'))
	print('Resplit datasets...')
	root_cub=os.path.join(DATA_ROOT,'CUB200-2011','CUB_200_2011')
	if not os.path.exists(os.path.join(root_cub,'bounding_images')):
		os.mkdir(os.path.join(root_cub,'bounding_images'))
	f=open(os.path.join(root_cub,'images.txt'))
	f_train=open(os.path.join(root_cub,'new_train.txt'),'w')
	f_test=open(os.path.join(root_cub,'new_test.txt'),'w')
	f_bounding_train=open(os.path.join(root_cub,'new_bounding_train.txt'),'w')
	f_bounding_test=open(os.path.join(root_cub,'new_bounding_test.txt'),'w')
	for line in f:
		if int(line.split(' ')[1].split('.')[0])<=100:
			temp=line.strip().split(' ')[1].split('.')[0]+' '+'images/'+line.split(' ')[1]
			f_train.write(temp)
			temp=line.strip().split(' ')[1].split('.')[0]+' '+'bounding_images/'+line.split(' ')[1]
			f_bounding_train.write(temp)
		else:
			temp=line.strip().split(' ')[1].split('.')[0]+' '+'images/'+line.split(' ')[1]
			f_test.write(temp)
			temp=line.strip().split(' ')[1].split('.')[0]+' '+'bounding_images/'+line.split(' ')[1]
			f_bounding_test.write(temp)
	f.close()
	f_train.close()
	f_test.close()

	f = linecache.getlines(os.path.join(root_cub,'bounding_boxes.txt'))
	f_name=linecache.getlines(os.path.join(root_cub,'images.txt'))
	for i in tqdm(range(len(f))):
		axes=f[i].strip().split(' ')
		name=f_name[i].strip().split(' ')[1]
		path=os.path.join(root_cub,'images',name)
		new_path=os.path.join(root_cub,'bounding_images',name)
		if not os.path.exists(os.path.join(root_cub,'bounding_images',name.split('/')[0])):
			os.mkdir(os.path.join(root_cub,'bounding_images',name.split('/')[0]))
		img=Image.open(path)
		axes=([int(float(axes[1])),int(float(axes[2])),int(float(axes[3])),int(float(axes[4]))])
		img_cut=img.crop([axes[0],axes[1],axes[0]+axes[2],axes[1]+axes[3]])
		img_cut.save(new_path)

def download_and_split_CARS196(DATA_ROOT):
	#CARS196
	if not os.path.exists(os.path.join(DATA_ROOT,'CARS196')):
		os.mkdir(os.path.join(DATA_ROOT,'CARS196'))
	print('Downloading CARS196 cars_train.tgz...')
	maybe_download('cars_train.tgz', os.path.join(DATA_ROOT,'CARS196'),'http://imagenet.stanford.edu/internal/car196/cars_train.tgz')
	print('Downloading CARS196 cars_test.tgz...')
	maybe_download('cars_test.tgz', os.path.join(DATA_ROOT,'CARS196'),'http://imagenet.stanford.edu/internal/car196/cars_test.tgz')
	print('Downloading CARS196 car_devkit.tgz...')
	maybe_download('car_devkit.tgz', os.path.join(DATA_ROOT,'CARS196'),'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz')
	print('Extracting CARS196 cars_train.tgz...')
	extract(os.path.join(DATA_ROOT,'CARS196','cars_train.tgz'), os.path.join(DATA_ROOT,'CARS196'))
	print('Extracting CARS196 cars_test.tgz...')
	extract(os.path.join(DATA_ROOT,'CARS196','cars_test.tgz'), os.path.join(DATA_ROOT,'CARS196'))
	print('Extracting CARS196 car_devkit.tgz...')
	extract(os.path.join(DATA_ROOT,'CARS196','car_devkit.tgz'), os.path.join(DATA_ROOT,'CARS196'))
	maybe_download('cars_test_annos_withlabels.mat', os.path.join(DATA_ROOT,'CARS196','devkit'),'http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat')
	print('Resplit datasets...')
	data1=scipy.io.loadmat(os.path.join(DATA_ROOT,'CARS196','devkit','cars_train_annos.mat'))
	data2=scipy.io.loadmat(os.path.join(DATA_ROOT,'CARS196','devkit','cars_test_annos_withlabels.mat'))
	if not os.path.exists(os.path.join(DATA_ROOT,'CARS196','bounding_train')):
		os.mkdir(os.path.join(DATA_ROOT,'CARS196','bounding_train'))
	if not os.path.exists(os.path.join(DATA_ROOT,'CARS196','bounding_test')):
		os.mkdir(os.path.join(DATA_ROOT,'CARS196','bounding_test'))

	f_train=open(os.path.join(DATA_ROOT,'CARS196','bounding_train.txt'),'w')
	f_test=open(os.path.join(DATA_ROOT,'CARS196','bounding_test.txt'),'w')
	f_common_train=open(os.path.join(DATA_ROOT,'CARS196','train.txt'),'w')
	f_common_test=open(os.path.join(DATA_ROOT,'CARS196','test.txt'),'w')
	for i in tqdm(range(len(data1['annotations'][0]))):
		img=Image.open(os.path.join(DATA_ROOT,'CARS196','cars_train',data1['annotations'][0][i][5][0]))
		img_cut=img.crop((data1['annotations'][0][i][0][0][0],data1['annotations'][0][i][1][0][0],data1['annotations'][0][i][2][0][0],data1['annotations'][0][i][3][0][0]))
		img_cut.save(os.path.join(DATA_ROOT,'CARS196','bounding_train',data1['annotations'][0][i][5][0]))
		if data1['annotations'][0][i][4][0][0]<=98:
			f_train.write(str(data1['annotations'][0][i][4][0][0])+' '+os.path.join('bounding_train',data1['annotations'][0][i][5][0])+'\r\n')
			f_common_train.write(str(data1['annotations'][0][i][4][0][0])+' '+os.path.join('cars_train',data1['annotations'][0][i][5][0])+'\r\n')
		else:
			f_test.write(str(data1['annotations'][0][i][4][0][0])+' '+os.path.join('bounding_train',data1['annotations'][0][i][5][0])+'\r\n')
			f_common_test.write(str(data1['annotations'][0][i][4][0][0])+' '+os.path.join('cars_train',data1['annotations'][0][i][5][0])+'\r\n')
	for i in tqdm(range(len(data2['annotations'][0]))):
		img=Image.open(os.path.join(DATA_ROOT,'CARS196','cars_test',data2['annotations'][0][i][5][0]))
		img_cut=img.crop((data2['annotations'][0][i][0][0][0],data2['annotations'][0][i][1][0][0],data2['annotations'][0][i][2][0][0],data2['annotations'][0][i][3][0][0]))
		img_cut.save(os.path.join(DATA_ROOT,'CARS196','bounding_test',data2['annotations'][0][i][5][0]))
		if data2['annotations'][0][i][4][0][0]<=98:
			f_train.write(str(data2['annotations'][0][i][4][0][0])+' '+os.path.join('bounding_test',data2['annotations'][0][i][5][0])+'\r\n')
			f_common_train.write(str(data2['annotations'][0][i][4][0][0])+' '+os.path.join('cars_test',data2['annotations'][0][i][5][0])+'\r\n')
		else:
			f_test.write(str(data2['annotations'][0][i][4][0][0])+' '+os.path.join('bounding_test',data2['annotations'][0][i][5][0])+'\r\n')
			f_common_test.write(str(data2['annotations'][0][i][4][0][0])+' '+os.path.join('cars_test',data2['annotations'][0][i][5][0])+'\r\n')
	f_train.close()
	f_test.close()

def download_and_split_Stanford_Online_Products(DATA_ROOT):
	#Stanford_Online_Products
	if not os.path.exists(os.path.join(DATA_ROOT,'Stanford_Online_Products')):
		os.mkdir(os.path.join(DATA_ROOT,'Stanford_Online_Products'))
	print('Download Stanford_Online_Products.zip...')

	maybe_download('Stanford_Online_Products.zip', DATA_ROOT,'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip')
	print('Extracting Stanford_Online_Products.zip...')
	extract_zip(os.path.join(DATA_ROOT,'Stanford_Online_Products.zip'), DATA_ROOT)
	f_train=open(os.path.join(DATA_ROOT,'Stanford_Online_Products','new_train.txt'),'w')
	f_test=open(os.path.join(DATA_ROOT,'Stanford_Online_Products','new_test.txt'),'w')
	f1=open(os.path.join(DATA_ROOT,'Stanford_Online_Products','Ebay_train.txt'))
	f2=open(os.path.join(DATA_ROOT,'Stanford_Online_Products','Ebay_test.txt'))
	for line in f1:
		if line.split()[0][0]!='i':
			cls=line.split()[1]
			pth=line.split()[3]
			txt=cls+' '+pth+'\r'
			f_train.write(txt)
	for line in f2:
		if line.split()[0][0]!='i':
			cls=line.split()[1]
			pth=line.split()[3]
			txt=cls+' '+pth+'\r'
			f_test.write(txt)
	f_train.close()
	f_test.close()
	f1.close()
	f2.close()

if __name__ == "__main__":
	#download_and_split_all_datasets('/export/home/datasets')
	#split_snapshop('/export/home/datasets')
	download_and_split_CARS196('/export/home/datasets')
