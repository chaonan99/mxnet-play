from common import config

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