"""MXNet XOR Example
^^^^^
Mix all components in one file

[Author] chaonan99
[E-mail] chenhaonan1995@gmail.com
[Create] 2017/04
[Last Modified] 2017/04/04
"""

import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)


class Config:
    minibatch_size = 640
    num_classes = 2
    num_batches = {'train': 20000, 'validation': 100}
    num_epoches = 5

    @property
    def input_shape(self):
        return (self.minibatch_size, 2)

config = Config()


class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad

class SimpleIter:
    def __init__(self, num_batches):
        self.batch_size = config.minibatch_size
        self.num_batches = num_batches
        self.data_shape = config.input_shape
        self.label_shape = (config.minibatch_size, )
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [('data', self.data_shape)]

    @property
    def provide_label(self):
        return [('softmax_label', self.label_shape)]

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = np.random.rand(np.array(self.data_shape).prod())
            data = data.reshape(self.batch_size, -1) * 2 - 1
            label = (data.prod(axis=1) > 0).astype(int)
            return SimpleBatch(data=[mx.nd.array(data)], label=[mx.nd.array(label)], pad=0)
        else:
            raise StopIteration


def get_iter(dataset_name):
    return SimpleIter(config.num_batches[dataset_name])


def get_network():
    net = mx.sym.Variable('data')
    net = mx.sym.FullyConnected(net, name='fc1', num_hidden=4)
    net = mx.sym.Activation(net, name='relu1', act_type="relu")
    net = mx.sym.FullyConnected(net, name='fc2', num_hidden=config.num_classes)
    net = mx.sym.SoftmaxOutput(net, name='softmax')
    mod = mx.mod.Module(symbol=net, context=mx.cpu(),
        data_names=['data'], label_names=['softmax_label'])
    return mod


def predict(mod):
    it = mx.io.NDArrayIter(data={'data': np.array([[-0.5,-0.5], [-0.5,0.5], [0.5,-0.5], [0.5,0.5]])},
        label={'softmax_label': np.array([0, 1, 1, 0])})
    mod_predict = mx.mod.Module(symbol=mod.symbol)
    mod_predict.bind(data_shapes=it.provide_data, label_shapes=it.provide_label)
    mod2.set_params(*mod.get_params())
    return mod2.predict(it)


# def train():
#     mod = get_network()
#     train_iter = get_iter('train')
#     val_iter = get_iter('validation')
#     mod.bind(data_shapes=train_iter.provide_data,
#         label_shapes=train_iter.provide_label)
#     mod.init_params(initializer=mx.init.Xavier())
#     mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
#     metric = mx.metric.create('acc')
#     # train one epoch, i.e. going over the data iter one pass
#     for i in range(config.num_epoches):
#         for batch in train_iter:
#             mod.forward(batch, is_train=True)       # compute predictions
#             mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
#             mod.backward()                          # compute gradients
#             mod.update()                            # update parameters using SGD


def main():
    mod = get_network()
    log_training = mx.callback.log_train_metric(10)
    mod.fit(train_data=get_iter('train'),
        eval_data=get_iter('validation'),
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=config.num_epoches,
        batch_end_callback=log_training)
    y = predict(mod)


if __name__ == '__main__':
    main()
