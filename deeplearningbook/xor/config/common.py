class Config:
    minibatch_size = 640
    num_classes = 2
    num_batches = {'train': 20000, 'validation': 100}
    num_epoches = 5

    @property
    def input_shape(self):
        return (self.minibatch_size, 2)

config = Config()