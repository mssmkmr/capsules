from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block
mx.random.seed(1)

###########################
#  Speficy the context we'll be using
###########################
ctx = mx.cpu()

###########################
#  Load up our dataset
###########################
batch_size = 64
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                    batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

class SimpleCapsule(Block):
    def __init__(self, units, in_units=0, **kwargs):
        super(SimpleCapsule, self).__init__(**kwargs)
        with self.name_scope():
            self.w = self.params.get('w', shape=(in_units, units))
    def forward(self, x):
        with x.context:
            u = nd.dot(x, self.w.data())
            return u

class Capsule(Block):
    def __init__(self, units, in_units=0, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        with self.name_scope():
            self.w = self.params.get('w', shape=(in_units, units))
            self.b = self.params.get('b', shape=(1, units),  differentiable=False)
    def forward(self, x):
        with x.context:
            c = nd.softmax(self.b.data(), axis=1)
            u = nd.dot(x, self.w.data())
            s = nd.multiply(c, u)
            s_nrm = nd.sum(s*s)
            fact = s_nrm / ( 1. + s_nrm)
            v = fact * s / nd.sqrt(s_nrm)
            self.u_v = nd.sum(nd.multiply(u, v))
            return u
    def update_b(self):
        self.b.set_data(self.b.data() + self.u_v)

class CapNet(Block):
    def __init__(self, **kwargs):
        super(CapNet, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(256, 9, activation='relu')
            self.w = []
            for i in range(32):
                setattr(self, "caps{}".format(i), SimpleCapsule(8, 20))
                self.w += [self.params.get('w{}'.format(i), shape=(8, 16))]
            self.digitcap = SimpleCapsule(10, 16)
    def forward(self, x):
        with x.context:
            x = self.conv1(x)
            x = nd.split(x, num_outputs=32, axis=1)
            cups = []
            for i in range(32):
                xi = getattr(self, "caps{}".format(i))(x[i])
                xi = nd.dot(xi, self.w[i].data())
                cups += [xi]
            x = nd.concat(*cups)
            x = self.digitcap(x)
            x = nd.sum(x, axis=[1, 2])
        return x
    def update_b(self):
        for i in range(32):
            getattr(self,  "caps{}".format(i)).update_b()

net = gluon.nn.Sequential()
capnet = CapNet()
with net.name_scope():
    net.add(capnet)
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})
metric = mx.metric.Accuracy()

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.

    for i, (data, label) in enumerate(data_iterator):
        with autograd.record():
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, 10)
            output = net(data)

        metric.update([label], [output])
    return metric.get()[1]

epochs = 1000  # Low number for testing, set higher when you run!
moving_loss = 0.

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
            cross_entropy.backward()
        # capnet.update_b() # if use normal Capsule
        trainer.step(data.shape[0])

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Train_acc %s, Test_acc %s" % (e, train_accuracy, test_accuracy))
filename = "model.net"
net.save_params(filename)
