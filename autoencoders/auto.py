import theano
import theano.tensor as T
import numpy as np
import scipy
import cPickle as pickle


def load_data(file):
    return pickle.load(open(file))


def relu(x):
    return T.maximum(T.cast(0., theano.config.floatX), x)


def train_model(data, train_fn):
    train, valid, test = data
    print "Epoch\tLoss"
    for i in range(300):
        print "%s\t%s" % (i, train_fn(train[0]))


def build_model():
    rng = np.random.RandomState(1234)
    n_hidden = 30
    n_out = 10
    learn_rate = 10.0
    inp_size = 784
    X = T.matrix('input')

    # First Hidden Layer's weights
    W1 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(inp_size, n_hidden)),
                                dtype=theano.config.floatX))
    # First Hidden Layer's bias
    b1 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(n_hidden,)),
                                   dtype=theano.config.floatX))
    # First Hidden Layer's output
    h1 = relu(T.dot(X, W1) + b1)

    # Second Hidden Layer's weights
    W2 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(n_hidden, n_hidden)),
                                dtype=theano.config.floatX))
    # Second Hidden Layer's bias
    b2 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(n_hidden,)),
                                   dtype=theano.config.floatX))
    # Second Hidden Layer's output
    h2 = relu(T.dot(h1, W2) + b2)

    # Output Layer's weights
    W3 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(n_hidden, inp_size)),
                                   dtype=theano.config.floatX))
    # Output Layer's bias
    b3 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(inp_size)),
                                   dtype=theano.config.floatX))

    # Output Layer's output
    out = T.nnet.sigmoid(T.dot(h2, W3) + b3)
    # The loss function we're optimising
    loss = -T.mean(X * T.log(out) + (1 - X) * T.log(1 - out))

    # Collect all parameters
    par = [W1, b1, W2, b2, W3, b3]
    # Get the gradients
    grad = [T.grad(loss, i) for i in par]
    # Collect all parameters
    updates = [(p, p - learn_rate * g) for p, g in zip(par, grad)]
    print "building function..."
    train_fn = theano.function([X], loss, updates=updates,
                               allow_input_downcast=True)

    # Compile a function to measure accuracy
    prediction = out
    eval_accuracy = theano.function([X], prediction,
                                    allow_input_downcast=True)

    return train_fn, eval_accuracy


def getimages(x):
    im = np.zeros((280, 280))
    for i in range(10):
        for j in range(10):
            im[i*28: (i+1)*28, j*28:(j+1)*28] = x[(j*10 + i), :].reshape((28, 28))
    return im


def main():
    data = load_data('../data/mnist.pkl')
    im = getimages(data[0][0])
    scipy.misc.imsave('0.jpg', im)
    train_fn, predict = build_model()
    train_model(data, train_fn)
    pred = getimages(predict(data[0][0]))
    scipy.misc.imsave('1.jpg', pred)


if __name__ == '__main__':
    main()