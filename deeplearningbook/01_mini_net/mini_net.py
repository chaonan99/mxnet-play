"""Minimal NN with some basic experiments
^^^^^
This doesn't require MxNet at all!
Reference: http://cs.stanford.edu/people/karpathy/cs231nfiles/minimal_net.html

[Author] chaonan99
[E-mail] chenhaonan1995@gmail.com
[Create] 2017/04
[Last Modified] 2017/04/21
"""


import numpy as np
import matplotlib.pyplot as plt


class Config:
    N = 100     # number of points per class
    D = 2       # dimensionality
    K = 3       # number of classes
    h = 100     # size of hidden layer
    reg = 1e-3  # regulization strength
    lr = 1e-0   # learning rate
    epoch = 1000
    @property
    def num_examples(self):
        return self.K * self.N
config = Config()


def get_data():
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y


def visulize_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


def train_network(b_fill=0):
    X, y = get_data()
    # initialize parameters randomly
    W = 0.01 * np.random.randn(config.D,config.h)
    b = np.zeros((1,config.h))
    b.fill(b_fill)
    W2 = 0.01 * np.random.randn(config.h,config.K)
    b2 = np.zeros((1,config.K))
    # if not b_zero:
    #     b2.fill(0.1)

    losses = []

    for i in range(config.epoch):

        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2

        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(config.num_examples),y])
        data_loss = np.sum(corect_logprobs)/config.num_examples
        reg_loss = 0.5*config.reg*np.sum(W*W) + 0.5*config.reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        losses.append(loss)
        # if i % 100 == 0:
        #     print("iteration %d: loss %f" % (i, loss))

        # compute the gradient on scores
        dscores = probs
        dscores[range(config.num_examples),y] -= 1
        dscores /= config.num_examples

        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        # add regularization gradient contribution
        dW2 += config.reg * W2
        dW += config.reg * W

        # perform a parameter update
        W += -config.lr * dW
        b += -config.lr * db
        W2 += -config.lr * dW2
        b2 += -config.lr * db2

    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

    return losses


"""Experiment 1
Initialize bias with small positive number may increase
converging speed when using ReLU nonlinearity.
"""
def experiment1():
    loss_b_zero = []
    loss_b_positive = []
    for i in range(20):
        loss_b_zero.append(train_network(b_fill=0.0))
        loss_b_positive.append(train_network(b_fill=0.01))
    loss_b_zero = np.array(loss_b_zero).mean(axis=0)
    loss_b_positive = np.array(loss_b_positive).mean(axis=0)
    plt.plot(range(config.epoch), loss_b_positive, 'r', range(config.epoch), loss_b_zero, 'b')
    plt.show()


def main():
    experiment1()


if __name__ == '__main__':
    main()
