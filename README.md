<p align="center"><img width="40%" src="logo/gluon_logo.png" /></p>

--------------------------------------------------------------------------------

This repository is a **gluon/mxnet** version of [yunjey's pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial). 

There is an office and perfect tutorial by mxnet' team: [Deep Learning - The Straight Dope](http://gluon.mxnet.io/)

## Table of Contents

#### 1. Basics

- [Gluon Basis](./example/1-basics/1_basic.py)
- [Linear Regression](./example/1-basics/2_linear.py)
- [Logistic Regression](./example/1-basics/3_logistic.py)
- [Feedforward Neural Network](./example/1-basics/4_network.py)

#### 2. Intermediate

- [Convolutional Neural Network](./example/2-intermediate/1_cnn.py)
- [Deep Residual Network](./example/2-intermediate/2_resnet.py)
- [Recurrent Neural Network](./example/2-intermediate/3_rnn.py)
- [Bidirectional Recurrent Neural Network](./example/2-intermediate/4_birnn.py)
- [Language Model (RNN-LM)](./example/2-intermediate/5_language.py)
- [Generative Adversarial Network](./example/2-intermediate/6_gan.py)

#### 3. Advanced

- [Neural Style Transfer](./example/3-advanced/1_style_transfer.py)
- [Variational Auto-Encoder](./example/3-advanced/2_vae.py)

## Note

1. You can run it through: `python xxxx.py`
2. The **Intermediate** and **Advanced** can switch from `cpu` or `gpu` by change the `gpu=True`，Some examples also provide `symbol=True` for switch the `imperative` or `symbolic`（but some examples I don not know how to do it）
3. It's my first time to use gluon/mxnet , so there must be some "bad behavior" in the achievement.

