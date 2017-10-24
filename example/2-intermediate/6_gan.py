from mxnet.gluon import data, nn
from mxnet import gluon, autograd, ndarray as nd
import mxnet as mx
import numpy as np
import sys
sys.path.append('..')
from cvtransform import Normalize
from save_img import save_images

# choose cpu or gpu --- default cpu, choose imperative or symbolic
gpu, symbol = True, True
ctx = mx.gpu() if gpu else mx.cpu()

# Hyper Parameters
batch_size = 100
image_frame_dim = int(np.floor(np.sqrt(batch_size)))
lr_d = 0.0003
lr_g = 0.0003
num_epochs = 80

# Image processing
train_augs = [Normalize(0.5, 0.5)]


def transform(img, label):
    img = img.asnumpy().astype('float32') / 255
    img = (img - 0.5) / 0.5
    return img, label


# MNIST dataset
mnist = data.vision.MNIST(train=True, transform=transform)

# Data loader
data_loader = data.DataLoader(mnist, batch_size, shuffle=False)

# Discriminator
D = nn.HybridSequential()
with D.name_scope():
    D.add(
        nn.Dense(256),
        nn.LeakyReLU(0.2),
        nn.Dense(256),
        nn.LeakyReLU(0.2),
        nn.Dense(1),
        nn.Activation('sigmoid')
    )

D.initialize(ctx=ctx)

# Generator
G = nn.HybridSequential()
with G.name_scope():
    G.add(
        nn.Dense(256),
        nn.LeakyReLU(0.2),
        nn.Dense(256),
        nn.LeakyReLU(0.2),
        nn.Dense(784),
        nn.Activation('tanh')
    )

G.initialize(ctx=ctx)

if symbol:
    D.hybridize()
    G.hybridize()

# Binary cross entropy loss and optimizer
criterion = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
d_optimizer = gluon.Trainer(D.collect_params(), 'adam', {'learning_rate': lr_d})
g_optimizer = gluon.Trainer(G.collect_params(), 'adam', {'learning_rate': lr_g})

# Start training
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Build mini-batch dataset
        images = nd.array(images.reshape((batch_size, -1)), ctx=ctx)
        # Create the labels which are later used as input for the BCE loss
        real_labels = nd.ones((batch_size,), ctx=ctx)
        fake_labels = nd.zeros((batch_size,), ctx=ctx)
        # ============= Train the discriminator ============= #
        # Compute BCE_Loss using real and fake images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # First term of the loss is always zero since fake_labels == 0
        # Second term of the loss is always zero since real_labels == 1
        z = nd.random_normal(shape=(batch_size, 64), ctx=ctx)
        with autograd.record():
            outputs1 = D(images)
            d_loss_real = criterion(outputs1, real_labels)
            fake_images = G(z)
            outputs2 = D(fake_images)
            d_loss_fake = criterion(outputs2, fake_labels)
            d_loss = d_loss_real.mean() + d_loss_fake.mean()
        # Backprop + Optimize
        d_loss.backward()
        d_optimizer.step(batch_size)

        # =============== Train the generator =============== #
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        z = nd.random_normal(shape=(batch_size, 64), ctx=ctx)
        with autograd.record():
            fake_images = G(z)
            outputs = D(fake_images)
            g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step(batch_size)

        if (i + 1) % 300 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                  % (epoch, 200, i + 1, 600, d_loss.sum().asscalar(), g_loss.sum().asscalar(),
                     outputs1.mean().asscalar(), outputs2.mean().asscalar()))

    # Save real images
    if (epoch + 1) == 1:
        images = images.asnumpy().reshape((batch_size, 28, 28, 1))
        save_images(images, [image_frame_dim, image_frame_dim], './data/real_images.png')

    # Save sampled images
    fake_images = fake_images.asnumpy().reshape((batch_size, 28, 28, 1))
    save_images(fake_images, [image_frame_dim, image_frame_dim], './data/fake_images-%d.png' % (epoch + 1))
