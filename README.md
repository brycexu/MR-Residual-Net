This repository contains my evaluations of Merge and Run in Binarized Residual Neural Work

Copyright: Xianda Xu xiandaxu@std.uestc.edu.cn

Baseline
--------
### Model: ResNet-18

Paper: (https://arxiv.org/abs/1512.03385)

Netscope: [Network](http://ethereon.github.io/netscope/#/gist/20db0c9bcdf859d2ffa0a5a55fe9b979)

Full-Precise Accuracy on Cifar-10: 93.28% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/Base.png"/></div>

Binarized Accuracy on Cifar-10: 90.50% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/Base (binarized).png"/></div>

Merge and Run
---------
### Model: MR-ResNet-20 (the number of layers is almost identical to the baseline ResNet-18 model)

Paper: (https://arxiv.org/abs/1611.07718)

Netscope: [Network](http://ethereon.github.io/netscope/#/gist/46029162791a6f9b6a9e54e7742c12d4)

Full-Precise Accuracy on Cifar-10: 92.15% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR.png"/></div>

Binarized Accuracy on Cifar-10: 87.77% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR-18 (binarized).png"/></div>

### Model: MR-ResNet-32 (the depth is identical to the baseline ResNet-18 model)

Paper: (https://arxiv.org/abs/1611.07718)

Netscope: Coming soon

Full-Precise Accuracy on Cifar-10: 93.39% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR-32.png"/></div>

Binarized Accuracy on Cifar-10: 91.26% with 80 epoches

<div align=center><img width="453" height="200" src="https://github.com/brycexu/MR-Residual-Net/blob/master/Images/MR-32 (binarized).png"/></div>

## Binarization Principle

* Keep full-precision on the first convolutional layer and the last linear layer.

* In binarized convolutional layers, all weights are binarized and scaled in propagation (https://arxiv.org/abs/1603.05279). But here, the scale factors are not learnt but all set to 1.

* BatchNorms in binarized blocks have no affine weights and bias parameters.

* Since activations are not binarized, ReLU is used instead of HardTanh (https://arxiv.org/pdf/1602.02830).

* Downsampling in Merge and Run models is done by firstly concatenating left-branch and right-branch and secondly using a convolusion (kernel-size:1, stride:2, padding:0)





















