from mxnet import gluon, autograd, ndarray as nd
from mxnet.gluon import data, nn, rnn
import mxnet as mx

# choose cpu or gpu --- default cpu
gpu = True
ctx = mx.gpu() if gpu else mx.cpu()

# Hyper Parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

# MNIST Dataset
train_dataset = data.vision.MNIST(train=True)
test_dataset = data.vision.MNIST(train=False)

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size, shuffle=False)


# BiRNN Model (Many-to-One)
class BiRNN(gluon.Block):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        with self.name_scope():
            self.lstm = rnn.LSTM(hidden_size, num_layers, layout='NTC', bidirectional=True)
            self.fc = nn.Dense(num_classes)

    def forward(self, x):
        # Set initial states
        h0 = nd.zeros((self.num_layers * 2, x.shape[0], self.hidden_size), ctx=ctx)
        c0 = nd.zeros((self.num_layers * 2, x.shape[0], self.hidden_size), ctx=ctx)
        # Forward propagate RNN
        out, _ = self.lstm(x, [h0, c0])
        return self.fc(out[:, out.shape[1] - 1, :])


rnn = BiRNN(hidden_size, num_layers, num_classes)
rnn.initialize(ctx=ctx)

# Loss and Optimizer
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(rnn.collect_params(), 'adam', {'learning_rate': learning_rate})

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.astype('float32').reshape((-1, sequence_length, input_size)) / 255
        images, labels = images.as_in_context(ctx), labels.as_in_context(ctx)
        # Forward + Backward + Optimize
        with autograd.record():
            outputs = rnn(images)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(batch_size)

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.sum().asscalar()))

# Test the Model
total, correct = 0, 0
for images, labels in test_loader:
    images = images.astype('float32').reshape((-1, sequence_length, input_size)) / 255
    images, labels = images.as_in_context(ctx), labels.as_in_context(ctx)
    outputs = rnn(images)
    predict = outputs.argmax(1).astype('int32')
    total += labels.shape[0]
    correct += (predict == labels).sum().asscalar()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
