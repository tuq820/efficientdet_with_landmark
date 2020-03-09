# EfficientDet with  landmark

## Introduction

add landmark to efficientdet

## How to use our code

### 1.Data

prepare data and label , example:label.txt classes.txt, front 4 is box  x1,y1,x2,y2, then is landmark  x1, y1..... 

label like: # xxx.png

classname1 x1 y1 x2 y2 ptx1 pty1 ptx2 pty2 ......

classname2 x1 y1 x2 y2 ptx1 pty1 ptx2 pty2 ......

### 2.Train

python3 train.py




## Requirements

* **python 3.6**
* **pytorch 1.2**
* **opencv (cv2)**
* **tensorboard**
* **tensorboardX** (This library could be skipped if you do not use SummaryWriter)
* **pycocotools**
* **efficientnet_pytorch**

## References
- Mingxing Tan, Ruoming Pang, Quoc V. Le. "EfficientDet: Scalable and Efficient Object Detection." [EfficientDet](https://arxiv.org/abs/1911.09070).
- Our implementation borrows almost parts from [RetinaNet.Pytorch](https://github.com/yhenon/pytorch-retinanet) -
- Our implementation almost reference [EfficientDet](https://github.com/signatrix/efficientdet) [RetinaFace_Pytorch](https://github.com/supernotman/RetinaFace_Pytorch)

## 

