<p align="center"><img width="40%" src="logo/gluon_logo.png" /></p>

--------------------------------------------------------------------------------

This repository is a **gluon/mxnet** version of [yunjey's pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial). The origin repository is elegent and thanks for the [yunjey](https://github.com/yunjey)'s great work.

There is an office and perfect tutorial by mxnet' team: [Deep Learning - The Straight Dope](http://gluon.mxnet.io/)

## Table of Contents

#### 1. Basics

- [Gluon Basis](https://github.com/AceCoooool/gluon-example/tree/master/example/1-basics/1_basic.py)
- [Linear Regression](https://github.com/AceCoooool/gluon-example/tree/master/example/1-basics/2_linear.py)
- [Logistic Regression](https://github.com/AceCoooool/gluon-example/tree/master/example/1-basics/3_logistic.py)
- [Feedforward Neural Network](https://github.com/AceCoooool/gluon-example/tree/master/example/1-basics/4_network.py)

#### 2. Intermediate

- [Convolutional Neural Network](https://github.com/AceCoooool/gluon-example/tree/master/example/2-intermediate/1_cnn.py)
- [Deep Residual Network](https://github.com/AceCoooool/gluon-example/tree/master/example/2-intermediate/2_resnet.py)
- [Recurrent Neural Network](https://github.com/AceCoooool/gluon-example/tree/master/example/2-intermediate/3_rnn.py)
- [Bidirectional Recurrent Neural Network](https://github.com/AceCoooool/gluon-example/tree/master/example/2-intermediate/4_birnn.py)
- [Language Model (RNN-LM)](https://github.com/AceCoooool/gluon-example/tree/master/example/2-intermediate/5_language.py)
- [Generative Adversarial Network](https://github.com/AceCoooool/gluon-example/tree/master/example/2-intermediate/6_gan.py)

#### 3. Advanced

- [Neural Style Transfer](https://github.com/AceCoooool/gluon-example/tree/master/example/3-advanced/1_style_transfer.py)
- [Variational Auto-Encoder](https://github.com/AceCoooool/gluon-example/tree/master/example/3-advanced/2_vae.py)

## Note

1. You can run it through: `python xxxx.py`
2. The **Intermediate** and **Advanced** can switch from `cpu` or `gpu` by change the `gpu=True`，Some examples also provide `symbol=True` for switch the `imperative` or `symbolic`（but some examples I don not know how to do it）
3. It's my first time to use gluon/mxnet , so there must be some "bad behavior" in the achievement.

## Gluon vs PyTorch


|   Model   | Acc(%) | time(s) | Model  | Acc(%) | time(s) |
| :-------: | :----: | :-----: | :----: | :----: | :-----: |
|   NN(g)   |   97   | 8.1326  | CNN(g) |   98   | 12.1198 |
|   NN(p)   |   97   | 19.6208 | CNN(p) |   98   | 22.7114 |
| ResNet(g) |   81   | 442.88  | RNN(g) |   97   | 12.832  |
| ResNet(p) |   85   | 505.09  | RNN(p) |   97   | 14.257  |
| BIRNN(g)  |   97   | 17.908  | LM(g)  |   ??   | 350.584 |
| BIRNN(p)  |   97   | 18.481  | LM(p)  |   ??   | 305.738 |

Note:

- The models in the tutorial, not represent other situations.
- GPU: 1060 (6g) 
- (g) --- gluon, (p) --- pytorch
- pytorch: [0.2.0](http://pytorch.org/),  mxnet: [0.12.0](https://github.com/apache/incubator-mxnet/releases), python: 3.5