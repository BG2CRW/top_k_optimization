DATA_ROOT="~/datasets"
DATASET = 'Stanford_Online_Products' #0:CIFAR100  1:CUB200   2:CARS196  3:online product
NET ="bn_inception_v2" # "densenet201" 
EMBEDDING_WIDTH = 512
DATASET_NUM = 59552   #train
WITH_BOUNDING_BOX = False
DOWNLOAD = True
TEST_K = 10
TEST_BATCH_SIZE = 128
#general parameter
USE_CUDA = 1
EPOCH = 1000
LAMDA = 1
METRIC_LOSS_PARAM = LAMDA
TRAINING_OLD = 1
TRAIN_CLASS = 11318   #cifar100:80 CUB200:100 CARS196:98
TEST_CLASS = 11316
MODEL_PATH = "model_param/"+NET+"_online_"
K = 5               #3 5  10
N = 20             #
POS_SAMPLE_NUM = 10  #8 10  20
MULTI_THREAD=1  

GPU_NUM = '1'
METHOD = 0
METRIC_LOSS_LR = 1e-3
SCHEDULER_STEP = [15, 100, 200]

if METHOD == 0:     #0:prec@k loss 
	MARGIN = 0.2      #up to 0.4 
	SHOW_PER_ITER = 500
	BATCH_SIZE = 68   #0:20     1:200     2:500     3:500
	EMBD_LR = 1e-5
	SOFTMAX_LOSS_LR = 1e-4
	NET_LR = 1e-3
	WEIGHT_DECAY = 5e-4
	GAMMA=1e-1
