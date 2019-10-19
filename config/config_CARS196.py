DATA_ROOT="/export/home/datasets"
DATASET = 'CARS196'
NET = "bn_inception_v2"    #"densenet201" 
EMBEDDING_WIDTH = 64
DATASET_NUM = 8054   #train
WITH_BOUNDING_BOX = True
DOWNLOAD = False
TEST_K = 10
MULTI_THREAD=1

#general parameter
USE_CUDA = 1
EPOCH = 100000
LAMDA=1
METRIC_LOSS_PARAM = LAMDA
CE_LOSS_PARAM = 1-LAMDA
TRAINING_OLD = 0
TRAIN_CLASS = 98    #cifar100:80 CUB200:100 CARS196:98
TEST_CLASS = TRAIN_CLASS
MODEL_PATH = "model_param/"+NET+"_cars196_bbox"+str(WITH_BOUNDING_BOX)+"_"
#our parameter
K = 5               #3 5  10
N = 55              #
POS_SAMPLE_NUM =10   #8 12  20
TAU = 0  #0.1           #10->2   
METRIC_LOSS_LR = 1e-3
GPU_NUM = '2'
METHOD = 9 #0:smooth prec@k loss 1:proxy-nca 2:clustering loss  3:npair loss  4:angular loss  
		   #5:lifted loss 6:triplet loss 7:contrastive loss 8:another prec@k loss 9:sampling matters
SCHEDULER_STEP = [10, 30, 70]
AUX_METHOD=23#22:transductive other:none
if METHOD==0: #smooth prec@k loss
	SOFTMAX_METHOD = 14   #1:ProxyNCA loss  10:arcface 1 1:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	MARGIN = 0.4      #up to 0.4 
	SOFTMAX_MARGIN = 0.5   #L-Softmax:2  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 500
	BATCH_SIZE = 1   #0:20     1:200     2:500     3:500
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4

	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=0.1

if METHOD == 1:#proxy-nca loss
	MARGIN = None
	SOFTMAX_METHOD = 14
	SOFTMAX_MARGIN = 3    #L-Softmax:3  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 64
	BATCH_SIZE = 64    #<98
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 8:      #0:another prec@k loss 
	SOFTMAX_METHOD = 14   #1:ProxyNCA loss  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	MARGIN = 0.05     #up to 0.4 
	SOFTMAX_MARGIN = 0.5   #L-Softmax:2  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 200
	BATCH_SIZE = 64   #0:20     1:200     2:500     3:500
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 3:#npair loss
	MARGIN = 0.3
	SOFTMAX_METHOD = 14   #1:ProxyNCA loss  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	SOFTMAX_MARGIN = 4    #L-Softmax:3  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 300
	BATCH_SIZE = 64    #<98
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 4:#angular loss
	MARGIN = 45   #45 degree
	SOFTMAX_METHOD = 14
	SOFTMAX_MARGIN = 3    #L-Softmax:3  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 300
	BATCH_SIZE = 64    #<98
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 2:  #clustering loss 
	MARGIN = 1  
	SOFTMAX_METHOD = 14   #1:ProxyNCA loss  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	SOFTMAX_MARGIN = 3    #L-Softmax:3  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 300
	BATCH_SIZE = 30    #<98
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 5:      #lifted loss
	MARGIN = 0.3
	SOFTMAX_METHOD = 14   #1:ProxyNCA loss  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	SOFTMAX_MARGIN = 2    #L-Softmax:3  A-Softmax:4  LMCL:2  arcface:0.5
	SHOW_PER_ITER = 300
	BATCH_SIZE = 64
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 6:      #triplet loss
	MARGIN = 0.2 
	SOFTMAX_METHOD = 14    #10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	SOFTMAX_MARGIN = 2   #L-Softmax:2  A-Softmax:4 LMCL:2
	SHOW_PER_ITER = 200
	BATCH_SIZE = 64
	EMBD_LR = 1e-5
	METRIC_LOSS_LR = 1e-4
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 7:     #contrastive loss
	SOFTMAX_METHOD = 14    #  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	MARGIN = 0.3           #L-Softmax:2  A-Softmax:4 LMCL:2
	SOFTMAX_MARGIN = 2
	SHOW_PER_ITER = 150
	BATCH_SIZE = 64
	EMBD_LR = 1e-5
	METRIC_LOSS_LR = 1e-4
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

if METHOD == 9:     #sampling matters
	SOFTMAX_METHOD = 10    #  10:arcface 11:LMCL loss  12:A-Softmax loss 13:L-Softmax loss  14:Common softmax loss
	MARGIN = 0.1           #L-Softmax:2  A-Softmax:4 LMCL:2
	SOFTMAX_MARGIN = 2
	SHOW_PER_ITER = 200
	BATCH_SIZE = 70
	EMBD_LR = 1e-5
	METRIC_LOSS_LR = 1e-4
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1

TEST_BATCH_SIZE = BATCH_SIZE
