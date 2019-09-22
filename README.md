# top_k_optimization in PyTorch 1.0
- **This repository has the source code for the paper "Sampling Wisely: Deep Image by Top-k Precision Optimization" (ICCV2019).**

## Citing this work
If you find this work useful in your research, please consider citing:

    @inproceedings{luICCV19,
        Author = {Jing Lu and Chaofan Xu and Wei Zhang and Lingyu Duan and Tao Mei},
        Title = {Sampling Wisely: Deep Image by Top-k Precision Optimization},
        Booktitle = {IEEE International Conference on Computer Vision (ICCV)},
        Year = {2019}
    }

## DATASETS
- **auto downloader and spliter** 
- **CUB200-2011(with or without bounding box)** 
- **CARS196(with or without bounding box)** 
- **Stanford Online Product** 

## VALIDATIONS
- **precsion@k** 
- **recall@k** 
- **NMI** 
- **F1** 
- **mAP** 

## TOOLS
- **TSNE**
- **bad case** 

## Installation
1. Install pytorch1.0, run`conda install pytorch torchvision -c pytorch`

2. Run `conda install future requests six pillow`

3. Run `pip install sklearn tqdm`

4. Run `cd top_k_optimization`

5. Choose right config file in main.py and set it whether you would download and split download and whether you need the dataset with bounding boxes in *_config.py and run `python main.py`

## Validation
Draw TSNE picture

Choose right config file in test_and_tsne.py and set it whether you would download and split download and whether you need the dataset with bounding boxes in *_config.py
Run `python test_and_tsne.py'
