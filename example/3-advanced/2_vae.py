from mxnet import gluon, autograd, ndarray as nd
from mxnet.gluon import data, nn
import mxnet as mx
import numpy as np
import sys
sys.path.append('..')
from save_img import save_images

# choose cpu or gpu
gpu = True
ctx = mx.gpu() if gpu else mx.cpu()

# Hyper Parameters
lr = 0.001
num_epochs = 50
batch_size = 100
frame_dim = int(np.floor(np.sqrt(batch_size)))

# MNIST dataset
dataset = data.vision.MNIST(train=True)

# Data loader
data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


# VAE model
class VAE(nn.HybridBlock):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        with self.name_scope():
            encoder = nn.HybridSequential()
            encoder.add(
                nn.Dense(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dense(z_dim * 2)  # 2 for mean and variance.
            )

            decoder = nn.HybridSequential()
            decoder.add(
                nn.Dense(h_dim, activation='relu'),
                nn.Dense(image_size, activation='sigmoid')
            )
        self.encoder = encoder
        self.decoder = decoder

    def reparametrize(self, mu, log_var):
        # z = mean + eps * sigma where eps is sampled from N(0, 1).
        eps = nd.random.normal(shape=(mu.shape[0], mu.shape[1]))
        z = mu + eps * nd.exp(log_var / 2)
        return z

    def hybrid_forward(self, F, x):
        h = self.encoder(x)
        mu, log_var = nd.split(h, 2, axis=1)
        z = self.reparametrize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var

    def sample(self, z):
        return self.decoder(z)


vae = VAE()
vae.initialize(ctx=ctx)

# optimizer and loss
optimizer = gluon.Trainer(vae.collect_params(), 'adam', {'learning_rate': lr})
criterion = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

# fixed inputs for debugging
fixed_z = nd.random.normal(shape=(batch_size, 20), ctx=ctx)
fixed_x, _ = next(data_iter)
save_images(fixed_x.asnumpy().reshape((batch_size, 28, 28, 1)), [frame_dim, frame_dim], './data/real_images.png')
fixed_x = fixed_x.reshape((batch_size, -1)).astype('float32') / 255

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape((batch_size, -1)).astype('float32') / 255
        images = images.as_in_context(ctx)
        with autograd.record():
            out, mu, log_var = vae(images)
            # Compute reconstruction loss and kl divergence
            reconst_loss = criterion(out, images).sum() * images.size / batch_size
            kl_divergence = nd.sum(0.5 * (mu ** 2 + nd.exp(log_var) - log_var - 1))
            total_loss = reconst_loss + kl_divergence
        # Backprop + Optimize
        total_loss.backward()
        optimizer.step(batch_size)

        if i % 100 == 0:
            print("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.7f"
                  % (epoch + 1, num_epochs, i + 1, iter_per_epoch, total_loss.sum().asscalar(),
                     reconst_loss.sum().asscalar(), kl_divergence.sum().asscalar()))
    # Save the reconstructed images
    reconst_images, _, _ = vae(fixed_x.as_in_context(ctx))
    save_images(reconst_images.asnumpy().reshape((batch_size, 28, 28, 1)), [frame_dim, frame_dim],
                './data/reconst_images_%d.png' % (epoch + 1))
