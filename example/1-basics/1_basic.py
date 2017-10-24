import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from mxnet import autograd, gluon, ndarray as nd
from mxnet.gluon.model_zoo import vision

# ======================= Basic autograd example 1 =======================#
# Create ndarray
x = nd.array([1])
w = nd.array([2])
b = nd.array([3])

for k in [x, w, b]:
    k.attach_grad()

# Build a computational graph.
with autograd.record():
    y = w * x + b

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)  # x.grad = 2
print(w.grad)  # w.grad = 1
print(b.grad)  # b.grad = 1

# ======================== Basic autograd example 2 =======================#
x = nd.random_normal(0, 1, shape=(5, 3))
y = nd.random_normal(0, 1, shape=(5, 2))

# Build a linear layer.
linear = gluon.nn.Dense(2)
linear.initialize()
print(linear.weight)
print(linear.bias)

# Build Loss and Optimizer.
criterion = gluon.loss.L2Loss()
optimizer = gluon.Trainer(linear.collect_params(), 'sgd', {'learning_rate': 0.01})

# Forward propagation.
with autograd.record():
    pred = linear(x)
    loss = criterion(pred, y)

print('loss: ', nd.sum(loss).asscalar())

# Backpropagation
loss.backward()
print('dL/dw: ', linear.weight.grad())
print('dL/db: ', linear.bias.grad())

print('weight(before): ', linear.weight.data())
print('bias(before): ', linear.bias.data())
optimizer.step(batch_size=5)  # batch_size: every grad divide it
# after_data = before_data - 0.01*grad/batch_size
print('weight(after): ', linear.weight.data())
print('bias(after): ', linear.bias.data())

# Print out the loss after optimization.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', nd.sum(loss).asscalar())

# ======================== Loading data from numpy ========================#
a = np.array([[1, 2], [3, 4]])
b = nd.array(a)  # convert numpy array to ndarray
c = b.asnumpy()  # convert ndarray to numpy array
print(a)
print(b)
print(c)

# ===================== Implementing the input pipline =====================#
# demo data
num_example = 20
num_input = 2
batch_size = 4
X = nd.random_normal(shape=(num_example, num_input))
y = 2 * X[:, 0] + 1.1 * X[:, 1] + 0.6

# construct dataset.
dataset = gluon.data.ArrayDataset(X, y)
data, label = dataset[0]
print('data_0: ', data)
print('label_0: ', label)

# Data Loader
data_loader = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

# Mini-batch images and labels.
data_iter = iter(data_loader)
data, label = next(data_iter)
print('data_batch: ', data)
print('label_batch: ', label)

# Actual usage of data loader is as below.
for data, label in data_loader:
    print(data, label)
    # Your training code will be written here
    break


# ===================== Input pipline for custom dataset =====================#
# You should build custom dataset as below.
class CustomDataset(gluon.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names
        pass

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. transform).
        # 3. Return a data pair (e.g. image and label).
        pass

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0


# ========================== Using pretrained model ==========================#
# Download and load pretrained resnet.
resnet = vision.resnet18_v1(pretrained=True)

# If you want to finetune only top layer of the model.
for param in resnet.collect_params().values():
    param.grad_req = 'null'

# Replace classifier layer for finetuning.
newclass = nn.HybridSequential(prefix=resnet.prefix)  # it's better to use same prefix
with newclass.name_scope():
    newclass.add(nn.MaxPool2D(pool_size=1, strides=1, padding=0, ceil_mode=True))
    newclass.add(nn.Dense(100))

newclass.initialize()
resnet.classifier = newclass
print(resnet)

# For test.
image = nd.random_normal(shape=(10, 3, 256, 256))
output = resnet(image)
print(output.shape)

# ============================ Save and load the model ============================#
# Save and load only the model parameters(recommended).
resnet.save_params('model.params')  # note: if not same prefix, there is an error
resnet.load_params('model.params', mx.cpu())
