import mxnet as mx
from mxnet.gluon import data, nn
from mxnet import gluon, autograd, init
import time

gpu = False
ctx = mx.gpu() if gpu else mx.cpu()

start = time.time()

# Hyper Parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST Dataset
train_dataset = data.vision.MNIST(train=True)
test_dataset = data.vision.MNIST(train=False)

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size, shuffle=False)


# Neural Network Model (1 hidden layer)
class Net(nn.Block):
    def __init__(self, hidden_size, num_classes):
        super(Net, self).__init__()
        with self.name_scope():
            self.fc1 = nn.Dense(hidden_size, activation='relu')
            self.fc2 = nn.Dense(num_classes)

    def forward(self, x):
        return self.fc2(self.fc1(x))


net = Net(hidden_size, num_classes)
net.initialize(init=init.Xavier(), ctx=ctx)

# Loss and Optimizer
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.astype('float32').reshape((-1, input_size)) / 255
        images, labels = images.as_in_context(ctx), labels.as_in_context(ctx)
        # Forward + Backward + Optimize
        with autograd.record():
            outputs = net(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(batch_size)

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.sum().asscalar()))

# Test the Model
total, correct = 0, 0
for images, labels in test_loader:
    images = images.astype('float32').reshape((-1, input_size)) / 255
    images, labels = images.as_in_context(ctx), labels.as_in_context(ctx)
    outputs = net(images)
    predict = outputs.argmax(1).astype('int32')
    total += labels.shape[0]
    correct += (predict == labels).sum().asscalar()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print('total time:', time.time() - start)
