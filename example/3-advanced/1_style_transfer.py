from mxnet.gluon import nn, model_zoo
from mxnet import autograd, gluon, ndarray as nd
import mxnet as mx
import numpy as np
import sys
sys.path.append('..')
from cvtransform import load_image
from save_img import save_images

# Hyper Parameters
context_file = './data/context.png'
style_file = './data/style.png'
max_size = 400
lr = 0.003
style_weight = 100
total_step = 1000
log_step = 10
sample_step = 100

# choose cpu or gpu
gpu = True
ctx = mx.gpu() if gpu else mx.cpu()


# Pretrained VGGNet
class VGGNet(nn.HybridSequential):
    def __init__(self):
        super(VGGNet, self).__init__()
        # Select conv1_1 ~ conv5_1 activation maps
        self.select = [0, 5, 10, 19, 28]
        self.vgg = model_zoo.vision.vgg19(pretrained=True, ctx=ctx).features

    def hybrid_forward(self, F, x):
        # Extract 5 conv activation maps from an input image
        features = []
        for name, layer in enumerate(self.vgg):
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


vgg = VGGNet()

# Load content and style images
context = nd.array(load_image(context_file, max_size), ctx=ctx)
style = nd.array(load_image(style_file, shape=context.shape[2:]), ctx=ctx)

# Initialization and optimizer
target = context.copy()
target.attach_grad()
optimizer = mx.optimizer.Adam(learning_rate=lr, beta1=0.5, beta2=0.999)

for step in range(total_step):
    with autograd.record():
        target_features = vgg(target)
        context_features = vgg(context)
        style_features = vgg(style)
        style_loss = nd.zeros((1,), ctx=ctx)
        context_loss = nd.zeros((1,), ctx=ctx)
        for f1, f2, f3 in zip(target_features, context_features, style_features):
            # Compute content loss (target and content image)
            context_loss = context_loss + nd.mean((f1 - f2) ** 2)
            # Reshape conv features
            _, c, h, w = f1.shape
            f1 = f1.reshape((c, h * w))
            f3 = f3.reshape((c, h * w))
            # Compute gram matrix
            f1 = nd.linalg_gemm2(f1, f1, transpose_b=1)
            f3 = nd.linalg_gemm2(f3, f3, transpose_b=1)
            # Compute style loss (target and style image)
            style_loss = style_loss + nd.mean((f1 - f3) ** 2) / (c * h * w)
        # Compute total loss
        loss = context_loss + style_weight * style_loss
    # backprop and optimize
    loss.backward()
    optimizer.update(step, target, target.grad, optimizer.create_state(step, target))

    if (step + 1) % log_step == 0:
        print('Step [%d/%d], Content Loss: %.4f, Style Loss: %.4f'
              % (step + 1, total_step, context_loss.sum().asscalar(), style_loss.sum().asscalar()))

    if (step + 1) % sample_step == 0:
        # Save the generated image
        img = target.copy().asnumpy()
        img = np.clip((img.transpose((0, 2, 3, 1)) - (-2.12, -2.04, -1.8)) / (4.37, 4.46, 4.44), 0, 1)
        save_images(img, [1, 1], './data/output-%d.png' % (step + 1))
