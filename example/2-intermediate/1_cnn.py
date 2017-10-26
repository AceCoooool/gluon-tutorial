from mxnet.gluon import data, nn
from mxnet import gluon, autograd
import mxnet as mx
import time

# choose cpu or gpu --- default cpu
gpu = True
ctx = mx.gpu() if gpu else mx.cpu()

start = time.time()
# Hyper Parameters
num_epochs = 5
input_size = 784
batch_size = 100
learning_rate = 0.001

# MNIST Dataset
train_dataset = data.vision.MNIST(train=True)
test_dataset = data.vision.MNIST(train=False)

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size, shuffle=False)

# CNN Model (2 conv layer)
cnn = nn.Sequential()
with cnn.name_scope():
    cnn.add(
        nn.Conv2D(16, 5, padding=2),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(2),
        nn.Conv2D(32, 5, padding=2),
        nn.BatchNorm(axis=1),
        nn.Activation(activation='relu'),
        nn.MaxPool2D(2),
        nn.Flatten(),
        nn.Dense(10)
    )

cnn.initialize(ctx=ctx)

# Loss and Optimizer
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(cnn.collect_params(), 'adam', {'learning_rate': learning_rate})

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.astype('float32').reshape((-1, 1, 28, 28)) / 255
        images, labels = images.as_in_context(ctx), labels.as_in_context(ctx)
        # Forward + Backward + Optimize
        with autograd.record():
            outputs = cnn(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(batch_size)

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.sum().asscalar()))

# Test the Model
total, correct = 0, 0
for images, labels in test_loader:
    images = images.astype('float32').reshape((-1, 1, 28, 28)) / 255
    images, labels = images.as_in_context(ctx), labels.as_in_context(ctx)
    outputs = cnn(images)
    predict = outputs.argmax(1).astype('int32')
    total += labels.shape[0]
    correct += (predict == labels).sum().asscalar()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print('total time:', time.time() - start)
