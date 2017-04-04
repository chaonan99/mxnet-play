import mxnet as mx

def get_network():
    net = mx.sym.Variable('data')
    net = mx.sym.FullyConnected(net, name='fc1', num_hidden=4)
    net = mx.sym.Activation(net, name='relu1', act_type="relu")
    net = mx.sym.FullyConnected(net, name='fc2', num_hidden=2)
    net = mx.sym.SoftmaxOutput(net, name='softmax')
    mod = mx.mod.Module(symbol=net, context=mx.gpu(),
        data_names=['data'], label_names=['softmax_label'])
    return mod