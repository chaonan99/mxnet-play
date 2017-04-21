import time
import logging
import mxnet as mx

from common import config
import dataset
import model

def main():
    mod = model.get_network()
    train_iter = dataset.get_iter('train')
    val_iter = dataset.get_iter('validation')
    mod.bind(data_shapes=train_iter.provide_data,
        label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier())
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
    metric = mx.metric.create('acc')
    # train one epoch, i.e. going over the data iter one pass
    for i in range(config.num_epoches):
        for batch in train_iter:
            mod.forward(batch, is_train=True)       # compute predictions
            mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
            mod.backward()                          # compute gradients
            mod.update()                            # update parameters using SGD

if __name__ == '__main__':
    main()