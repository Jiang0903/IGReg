# IGReg: Image-Geometry-Assisted Point Cloud Registration via Selective Correlation Fusion

This is the official repository for  `IGReg: Image-Geometry-Assisted Point Cloud Registration via Selective Correlation Fusion`

Authors: [Zongyi Xu](https://scholar.google.com.hk/citations?user=PUseiVAAAAAJ), Xinqi Jiang, Xinyu Gao, Rui Gao, [Changjun Gu](https://scholar.google.com.hk/citations?user=TepTtrQAAAAJ), [Qianni Zhang](https://scholar.google.com.hk/citations?user=XR6C9BoAAAAJ), [Weisheng Li](https://scholar.google.com.hk/citations?user=M17E3HEAAAAJ) and [Xinbo Gao](https://scholar.google.com.hk/citations?user=VZVTOOIAAAAJ).

## Introduction

 Point cloud registration suffers from repeated patterns and low geometric structures in indoor scenes. The recent transformer utilises attention mechanism to capture the global correlations in feature space and improves the registration performance. However, for indoor scenarios, global correlation loses its advantages as it cannot distinguish real useful features and noise. To address this problem, we propose an imagegeometry-assisted point cloud registration method by integrating image information into point features and selectively fusing the geometric consistency with respect to reliable salient areas. Firstly, an Intra-Image-Geometry fusion module is proposed to integrate the texture and structure information into the point feature space by the cross-attention mechanism. Initial corresponding superpoints are acquired as salient anchors in the source and target. Then, a selective correlation fusion module is designed to embed the correlations between the salient anchors and points. During training, the saliency location and selective correlation fusion modules exchange information iteratively to identify the most reliable salient anchors and achieve effective feature fusion. The obtained distinctive point cloud features allow for accurate correspondence matching, leading to the success of indoor point cloud registration. Extensive experiments are conducted on 3DMatch and 3DLoMatch datasets to demonstrate the outstanding performance of the proposed approach compared to the state-of-the-art, particularly in those geometrically challenging cases such as repetitive patterns and low-geometry regions.

![](assets/pipeline.png)

## Environment Setup
Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n IGReg python==3.8
conda activate IGReg

# Install pytorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop
```

Code has been tested with Ubuntu 20.04, GCC 9.4.0, Python 3.8, PyTorch 1.13.1, CUDA 11.7.


## Pre-trained Weights

To extract the feature of color images, we need to download [the pretrained weight of 2D encoder](https://drive.google.com/file/d/1-HVsB60B9JkXmDAgufYEIY_12liPUWzm/view?usp=sharing) and put it in `weight` directory.

We also provide [the pretrained weight of IGReg](https://drive.google.com/file/d/1-DWb2v0mpDR9Skp6wjt_prjfrQHkzVac/view?usp=sharing) which was trained on 3DMatch datasets. Please download it and put into `weight` directory.

## 3DMatch

### Data preparation

We put the point clouds, RGB images and camera information of 3DMatch dataset together. The point cloud files were the same as [PREDATOR](https://github.com/prs-eth/OverlapPredator). Please download the preprocessed dataset from [here](https://drive.google.com/file/d/1-2kIOYYWeLHIn_mpRN0qP5iNT2uxofWF/view?usp=sharing) and put into the `data` directory. The data should be organized as follows:

```text
--data--3DMatch--metadata
              |--data--train--7-scenes-chess--camera-intrinsics.txt
                    |      |               |--cloud_bin_0_0_pose.txt
                    |      |               |--cloud_bin_0_0.png
                    |      |               |--cloud_bin_0_1.pose.txt
                    |      |               |--cloud_bin_0_1.png
                    |      |               |--cloud_bin_0.info.txt
                    |      |               |--cloud_bin_0.pth
                    |      |               |--...
                    |      |--...
                    |--test--7-scenes-redkitchen--camera-intrinsics.txt
                          |                    |--cloud_bin_0_0_pose.txt
                          |                    |--cloud_bin_0_0.png
                          |                    |--cloud_bin_0_1.pose.txt
                          |                    |--cloud_bin_0_1.png
                          |                    |--cloud_bin_0.info.txt
                          |                    |--cloud_bin_0.pth
                          |                    |--...
                          |--...
```

### Training

The code for 3DMatch is in `experiments/3dmatch_IGReg`. Use the following command for training.

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Testing

Use the following command for testing 3DMatch dataset. `EPOCH` is the epoch id.
```bash
cd experiments/3dmatch_IGReg
export CUDA_VISIBLE_DEVICES=0
python test.py  --test_epoch=EPOCH --benchmark=3DMatch
python eval.py  --test_epoch=EPOCH --benchmark=3DMatch --method=lgr
```
You can also use the bash `experiments/3dmatch_IGReg/eval_all.sh` for testing.


We also provide pretrained weights in `weight`, use the following command to test the pretrained weights.


```bash
cd experiments/3dmatch_IGReg
export CUDA_VISIBLE_DEVICES=0
python test.py --snapshot=../../weights/3dmatch.pth.tar --benchmark=3DMatch
python eval.py --benchmark=3DMatch --method=lgr
```

Replace `3DMatch` with `3DLoMatch` to evaluate on 3DLoMatch.

## TUM RGB-D SLAM Dataset

### Data preparation
Following the work of [Fast and Robust Iterative Closest Point](https://ieeexplore.ieee.org/abstract/document/9336308), we preprocess the TUM RGB-D SLAM dataset in the same way. Please download the preprocessed dataset from [here](https://drive.google.com/file/d/1-3DFJRKe3X2KDGuBcjp_8GT6ALzCqb9l/view?usp=sharing) and put into the `data` directory. The data should be organied like 3DMatch dataset.The data should be organized as follows:

```text
--data--tum--metadata
          |--data--test--rgbd_dataset_freiburg1_360--camera-intrinsics.txt
                      |                           |--cloud_bin_0.info.txt
                      |                           |--cloud_bin_0.png
                      |                           |--cloud_bin_0.pose.txt
                      |                           |--cloud_bin_0.pth
                      |                           |--...
                      |--...
```

### Testing
To demonstrate the generalisability of IGReg, we directly use the model trained on 3DMatch to test the TUM RGB-D SLAM dataset. Use the following command for testing.
```bash
cd experiments/tum_IGReg
export CUDA_VISIBLE_DEVICES=0
python test.py --snapshot=../../weights/3dmatch.pth.tar
python eval.py
```


## Citation

```bibtex
To be released.
```

## Acknowledgements

- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [IMFNet](https://github.com/XiaoshuiHuang/IMFNet)
- [PCR-CG](https://github.com/Gardlin/PCR-CG)
- [TUM RGB-D SLAM Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
