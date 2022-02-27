# CSR-Net
 This is the code for [CSR-Net: Camera Spectral Response Network for Dimensionality Reduction and Classiﬁcation in Hyperspectral Imagery](https://www.mdpi.com/2072-4292/12/20/3294), Remote Sensing 2020, by Yunhao Zou, Ying Fu, Yinqiang Zheng and Wei Li.

## Introduction
In this work, we present a CNN architecture called CSR-Net for hyperspectral image classification. Our model can achieve the optimal camera spectral response (CSR) functions for HSI classiﬁcation. More importantly, the learned CSR can be directly used to reduce data dimensions when capturing images as well as guarantee the classiﬁcation accuracy.

## Code
### Prerequisite
- python 3.6
- pytorch 1.8.0
- numpy
- scikit-learn
- matplotlib
### Training and Testing
Please run ```sh train.sh``` for training and testing.

## Citation
If you find this work useful for your research, please cite: 
```
@article{zou2020csr,
  title={CSR-Net: Camera Spectral Response Network for Dimensionality Reduction and Classification in Hyperspectral Imagery},
  author={Zou, Yunhao and Fu, Ying and Zheng, Yinqiang and Li, Wei},
  journal={Remote Sensing},
  volume={12},
  number={20},
  pages={3294},
  year={2020},
}
```
## Acknowledgment
This codes are inspired by [pResNet-HSI](https://github.com/mhaut/pResNet-HSI) and [DANet](https://github.com/junfu1115/DANet/)