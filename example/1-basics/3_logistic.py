from mxnet import gluon, autograd
from mxnet.gluon import data, nn

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST Dataset (Images and Labels)
train_dataset = data.vision.MNIST(train=True)
test_dataset = data.vision.MNIST(train=False)

# Dataset Loader (Input Pipline)
train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size, shuffle=False)


# Model
class LogisticRegression(nn.Block):
    def __init__(self, num_classes):
        super(LogisticRegression, self).__init__()
        with self.name_scope():
            self.linear = nn.Dense(num_classes)

    def forward(self, x):
        return self.linear(x)


model = LogisticRegression(num_classes)
model.initialize()

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': learning_rate})

# Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.astype('float32').reshape((-1, input_size)) / 255
        # Forward + Backward + Optimize
        with autograd.record():
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(batch_size)
        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.sum().asscalar()))

# Test the Model
correct, total = 0, 0
for images, labels in test_loader:
    images = images.astype('float32').reshape((-1, input_size)) / 255
    outputs = model(images)
    predict = outputs.argmax(1).astype('int32')
    total += labels.shape[0]
    correct += (predict == labels).sum().asscalar()

print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
