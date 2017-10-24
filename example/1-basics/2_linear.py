from mxnet import gluon, autograd, ndarray as nd
from mxnet.gluon import nn
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy Dataset
x_train = nd.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]])

y_train = nd.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]])


# Linear Regression Model
class LinearRegression(nn.Block):
    def __init__(self, output_size):
        super(LinearRegression, self).__init__()
        with self.name_scope():
            self.linear = nn.Dense(output_size)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression(output_size)
model.initialize()

# Loss and Optimizer
criterion = gluon.loss.L2Loss()
optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': learning_rate})

# Train the Model
for epoch in range(num_epochs):
    with autograd.record():
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step(x_train.shape[0])

    if (epoch + 1) % 5 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, nd.sum(loss).asscalar()))

# Plot the graph
predict = model(x_train).asnumpy()
x_train, y_train = x_train.asnumpy(), y_train.asnumpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predict, label='Fitted line')
plt.legend()
plt.show()
