import theano
import theano.tensor as T
import numpy as np
import scipy
import cPickle as pickle


def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)


def theano_shared(value, **kwargs):
    var = theano.shared(floatX(value), **kwargs)
    return var


def adadelta(params, gradients, rho=0.95, eps=1e-6, offset_lr=1.0):
    #  Theano variables for storing intermediate values
    grad_2 = [theano_shared(np.zeros(p.get_value().shape)) for p in params]
    delta_2 = [theano_shared(np.zeros(p.get_value().shape)) for p in params]
    #  del_x = [theano_shared(np.zeros(p.get_value().shape)) for p in params]

    accumulate_grad = [(g2, rho * g2 + (1-rho) * (g)**2) for
                       g2, g in zip(grad_2, gradients)]

    deltas = [-(T.sqrt(d2 + eps)) / (T.sqrt(g2 + eps)) * (g)
              for g2, d2, g in zip(grad_2, delta_2, gradients)]

    accumulate_update = [(dx, rho * dx + (1-rho) * x**2) for
                         dx, x in zip(delta_2, deltas)]
    apply_update = [(p, p + offset_lr * x) for p, x in zip(params, deltas)]
    updates = accumulate_grad + accumulate_update + apply_update
    adaptive_lr = False
    return updates


def load_data(file):
    return pickle.load(open(file))


def relu(x):
    return T.maximum(T.cast(0., theano.config.floatX), x)


def train_model(data, train_fn):
    train, valid, test = data
    print "Epoch\tLoss"
    for i in range(50):
        print "%s\t%s" % (i, train_fn(train[0]))


def build_model():
    rng = np.random.RandomState(1234)
    s_rng = T.shared_randomstreams.RandomStreams(123)
    n_hidden = 50
    learn_rate = 10.0
    inp_size = 784
    n_latent = 20
    X = T.matrix('input')
    dim = [(n_hidden, n_latent, '1'),  # W1 ... b_i = W_i[0]
           (inp_size, n_hidden, '2'),  # W2
           (n_hidden, inp_size, '3'),  # W3
           (n_latent, n_hidden, '4'),  # W4
           (n_latent, n_hidden, '5'),  # W5
           ]

    weights = []
    biases = []
    for d in dim:
        weights.append(theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                       size=(d[0], d[1])),
                       dtype=theano.config.floatX), name='W%s' % d[2]))
        biases.append(theano.shared(np.array(rng.uniform(low=-0.1, high=0.1,
                      size=(d[0])),
                      dtype=theano.config.floatX), name='b%s' % d[2]))

    # Encoder
    h = T.tanh(T.dot(X, weights[2].T) + biases[2])
    mu = T.dot(h, weights[3].T) + biases[3]
    sig = T.exp(T.dot(h, weights[4].T) + biases[4])

    eps = s_rng.normal(size=mu.shape)
    z = mu + sig * eps

    y = T.nnet.sigmoid(T.dot(T.tanh(T.dot(z, weights[0].T) + biases[0]),
                       weights[1].T) + biases[1])
    # The loss function we're optimising
    loss = -T.mean(X * T.log(y) + (1 - X) * T.log(1 - y))

    # Collect all parameters
    par = weights + biases
    # Get the gradients
    for p in par:
        print str(p), p.get_value().shape
    grad = [T.grad(loss, i) for i in par]
    # Collect all parameters
    # updates = [(p, p - learn_rate * g) for p, g in zip(par, grad)]
    updates = adadelta(par, grad)
    print "building function..."
    train_fn = theano.function([X], loss, updates=updates,
                               allow_input_downcast=True)

    get_par = theano.function([X], [mu, sig])

    # Compile a function to measure accuracy
    n = T.iscalar()
    mu_fitted = T.vector()
    sig_fitted = T.vector()
    eps = s_rng.normal(size=mu_fitted.shape)

    z_decode = mu_fitted + sig_fitted * eps
    y = T.nnet.sigmoid(T.dot(T.tanh(T.dot(z_decode, weights[0].T) + biases[0]),
                       weights[1].T) + biases[1])

    tile = T.repeat(y, n).reshape((y.shape[0], n)).T
    samp = s_rng.binomial(n=1, p=tile)

    sample = theano.function([n, mu_fitted, sig_fitted], samp,
                             allow_input_downcast=True)

    return train_fn, sample, par, get_par


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
    train_fn, sample, par, get_par = build_model()
    train_model(data, train_fn)
    mu, sig = get_par(data[0][0])
    print mu.shape, sig.shape
    #for p in par:
    #    print str(p), p.get_value()
    s = sample(100, mu[0, :], sig[0, :])
    print s.shape
    pred = getimages(s)
    scipy.misc.imsave('1.jpg', pred)


if __name__ == '__main__':
    main()