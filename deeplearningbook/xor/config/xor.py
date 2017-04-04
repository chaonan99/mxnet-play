import mxnet as mx
import numpy as np
import logging
logging.baseConfig(level=logging.INFO)


def get_network():
    net = mx.sym.Variable('data')
    net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
    net = mx.sym.Activation(net, name='relu1', act_type="relu")
    net = mx.sym.FullyConnected(net, name='fc2', num_hidden=10)
    net = mx.sym.SoftmaxOutput(net, name='softmax')
    return net


def get_data():


def main():
   np.

if __name__ == '__main__':
    main()
