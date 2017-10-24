from mxnet import gluon, init, autograd, ndarray as nd
from mxnet.gluon import rnn, nn
import mxnet as mx
from data.data_utils import Corpus

# TODO: this achievement is not efficient
# choose cpu or gpu --- default cpu
gpu = True
ctx = mx.gpu() if gpu else mx.cpu()

# Hyper Parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000  # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

# Load Penn Treebank Dataset
train_path = './data/train.txt'
sample_path = './data/sample.txt'
corpus = Corpus()
ids = corpus.get_data(train_path, batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.shape[1] // seq_length


# RNN Based Language Model
class RNNLM(gluon.Block):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        with self.name_scope():
            self.embed = nn.Embedding(vocab_size, embed_size, weight_initializer=init.Uniform(0.1))
            self.lstm = rnn.LSTM(hidden_size, num_layers, layout='NTC')
            self.linear = nn.Dense(vocab_size, weight_initializer=init.Uniform(0.1))

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        # Forward propagate RNN
        out, h = self.lstm(x, h)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape((out.shape[0] * out.shape[1], out.shape[2]))
        # Decode hidden states of all time step
        out = self.linear(out)
        return out, h


model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)
model.initialize(ctx=ctx)

# Loss and Optimizer
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': learning_rate})

# Training
for epoch in range(num_epochs):
    # Initial hidden and memory states
    states = [nd.zeros((num_layers, batch_size, hidden_size), ctx=ctx),
              nd.zeros((num_layers, batch_size, hidden_size), ctx=ctx)]
    for i in range(0, ids.shape[1] - seq_length, seq_length):
        # Get batch inputs and targets
        inputs = ids[:, i:i + seq_length].as_in_context(ctx)
        targets = ids[:, (i + 1):i + 1 + seq_length].as_in_context(ctx)
        states = [state.detach() for state in states]
        with autograd.record():
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape((targets.size,)))
        loss.backward()
        grads = [i.grad(ctx) for i in model.collect_params().values()]
        gluon.utils.clip_global_norm(grads, 0.5 * batch_size)
        optimizer.step(batch_size)
        step = (i + 1) // seq_length
        if step % 100 == 0:
            print('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                  (epoch + 1, num_epochs, step, num_batches, (loss / 1000).sum().asscalar(),
                   (loss / 1000).sum().exp().asscalar()))

# Sampling
with open(sample_path, 'w') as f:
    # Set intial hidden ane memory states
    states = [nd.zeros((num_layers, 1, hidden_size), ctx=ctx),
              nd.zeros((num_layers, 1, hidden_size), ctx=ctx)]
    # Select one word id randomly
    prob = nd.ones((vocab_size,)) / vocab_size
    input1 = nd.sample_multinomial(prob).reshape((1, 1)).as_in_context(ctx)
    for i in range(num_samples):
        output, state = model(input1, states)
        # Sample a word id
        prob = nd.exp(output).reshape((-1,))
        prob = prob / prob.sum()
        word_id = nd.sample_multinomial(prob, 1)
        input1 = word_id.reshape((1, 1))
        word = corpus.dictionary.idx2word[word_id.asscalar()]
        word = '\n' if word == '<eos>' else word + ' '
        f.write(word)

        if (i + 1) % 100 == 0:
            print('Sampled [%d/%d] words and save to %s' % (i + 1, num_samples, sample_path))
