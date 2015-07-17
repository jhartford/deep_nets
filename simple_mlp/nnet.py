import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle


def load_data(file):
    return pickle.load(open(file))


def relu(x):
    return T.maximum(T.cast(0., theano.config.floatX), x)


def main():
    train, valid, test = load_data('../data/mnist.pkl')
    rng = np.random.RandomState(1234)
    n_hidden = 50
    n_out = 10
    learn_rate = 0.1

    X = T.matrix('input')
    y = T.ivector('output')

    W1 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(784, n_hidden)),
                                dtype=theano.config.floatX))

    b1 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(n_hidden,)),
                                   dtype=theano.config.floatX))

    h1 = relu(T.dot(X, W1) + b1)

    W2 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(n_hidden, 10)),
                                   dtype=theano.config.floatX))
    b2 = theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(10)),
                                   dtype=theano.config.floatX))

    out = T.nnet.softmax(T.dot(h1, W2) + b2)

    loss = -T.mean(T.log(out)[T.arange(y.shape[0]), y])
    par = [W1, b1, W2, b2]
    grad = [T.grad(loss, i) for i in par]
    updates = [(p, p - learn_rate * g) for p, g in zip(par, grad)]
    prediction = T.argmax(out, axis=1)
    accuracy = T.mean(T.eq(prediction, y))
    print "building function..."
    train_fn = theano.function([X, y], [loss], updates=updates,
                               allow_input_downcast=True)

    eval_accuracy = theano.function([X, y], accuracy,
                                    allow_input_downcast=True)

    for i in range(500):
        #loss, accuracy = 
        print i, train_fn(train[0], train[1]), eval_accuracy(test[0], test[1])


if __name__ == '__main__':
    main()