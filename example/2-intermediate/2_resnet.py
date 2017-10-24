from mxnet import gluon, image, autograd
from mxnet.gluon import nn, data
import mxnet.ndarray as nd
import mxnet as mx
import sys
sys.path.append('..')
from cvtransform import Scale, RandomHorizontalFlip, RandomCrop

# choose cpu or gpu --- default cpu, choose imperative or symbolic
gpu, symbol = True, True
ctx = mx.gpu() if gpu else mx.cpu()

# Hyper Parameters
lr = 0.001
num_classes = 10
num_epochs = 80
batch_size = 100

# Image Preprocessing
# train_augs = [image.ResizeAug(40), image.HorizontalFlipAug(.5), image.RandomCropAug([32, 32])]
train_augs = [Scale(40), RandomHorizontalFlip(), RandomCrop(32)]


def get_transform(augs):
    def transform(img, label):
        if augs is not None:
            img = img.asnumpy()
            for f in augs:
                img = f(img)
        return img, label

    return transform


# CIFAR-10 Dataset
train_dataset = data.vision.CIFAR10(train=True, transform=get_transform(train_augs))
test_dataset = data.vision.CIFAR10(train=False, transform=get_transform(None))

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Residual Block
class ResidualBlock(nn.HybridBlock):
    def __init__(self, channels, same_shape=True):
        super(ResidualBlock, self).__init__()
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)


# ResNet Module
resnet = nn.HybridSequential()
with resnet.name_scope():
    resnet.add(
        nn.Conv2D(16, kernel_size=3, padding=1),
        nn.BatchNorm(),
        nn.Activation(activation='relu'),
        ResidualBlock(16),
        ResidualBlock(16),
        ResidualBlock(32, same_shape=False),
        ResidualBlock(32),
        ResidualBlock(64, same_shape=False),
        ResidualBlock(64),
        nn.AvgPool2D(pool_size=8),
        nn.Flatten(),
        nn.Dense(num_classes)
    )

resnet.initialize(ctx=ctx)
if symbol:
    resnet.hybridize()

# Loss and Optimizer
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(resnet.collect_params(), 'adam', {'learning_rate': lr})

# Training
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = nd.array(imgs).astype('float32').transpose((0, 3, 1, 2)) / 255
        imgs, labels = imgs.as_in_context(ctx), labels.as_in_context(ctx)
        with autograd.record():
            outputs = resnet(imgs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(batch_size)

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.sum().asscalar()))

    # Decaying Learning Rate
    if (epoch + 1) % 20 == 0:
        optimizer.set_learning_rate(optimizer.learning_rate / 3)

# Test
correct = 0
total = 0
for imgs, labels in test_loader:
    imgs = nd.array(imgs).astype('float32').transpose((0, 3, 1, 2)) / 255
    imgs, labels = imgs.as_in_context(ctx), labels.as_in_context(ctx)
    outputs = resnet(imgs)
    predict = outputs.argmax(1).astype('int32')
    total += labels.shape[0]
    correct += (predict == labels).sum().asscalar()

print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
