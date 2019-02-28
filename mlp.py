import numpy as np

def cross_ent_loss(y, g):
    return -np.mean(y * np.log(g) + (1 - y) * np.log(1 - g))

def onehot(y, k):
    onehot_encoded = np.zeros((y.size, k))
    onehot_encoded[np.arange(y.size), y.ravel()] = 1.0

    return onehot_encoded

def ReLU(a):
    return np.maximum(0, a)

def ReLU_grad(a):
    return np.multiply(np.ones(a.shape), (a > 0))

def softmax(a):
    exps = np.exp(a)

    return np.divide(exps, np.expand_dims(np.sum(exps, axis=1), 1))

class Layer:
    def __init__(self, input_size, output_size, batch_size):
        self.x = np.empty((batch_size, input_size + 1))
        self.W = np.empty((output_size, input_size + 1))
        self.dW = np.empty(self.W.shape)
        self.delta = None
        self.init_weight(input_size, output_size)

    def init_weight(self, fan_in, fan_out):
        np.multiply(np.random.randn(fan_out, fan_in), fan_in ** -0.5, out=self.W[:, :-1])
        self.W[:, -1] = 1.0

    def __call__(self, x):
        np.copyto(self.x[:x.shape[0],:x.shape[1]], x)
        self.x[:, x.shape[1]] = 1.0
        self.x[x.shape[0]:, x.shape[1] + 1:] = 0.0
        self.a = np.matmul(self.x, self.W.T)

        return self.a


    def backward(self, delta):
        self.delta = delta
        np.matmul(delta.T, self.x, out=self.dW)
        dx = np.matmul(delta, self.W[:, :-1])

        return dx

class Activation:
    def __init__(self, activation=ReLU):
        self.a = None
        self.z = None
        self.activation = activation
        if activation == ReLU:
            self.grad = ReLU_grad
        elif activation == tanh:
            self.grad = tanh_grad

    def __call__(self, a):
        self.a = a
        self.z = self.activation(a)

        return self.z

    def backward(self, dz):
        return np.multiply(self.grad(self.a), dz)

class MLP:
    def __init__(self, input_size, layer_sizes, output_size, batch_size):
        self.layers = []
        for s_1, s_2 in zip([input_size] + layer_sizes, layer_sizes):
            self.layers.append(Layer(s_1, s_2, batch_size))
            self.layers.append(Activation())
        self.layers.append(Layer(layer_sizes[-1], output_size, batch_size))
        self.g = None
        self.n_layers = len(self.layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.g = softmax(x)

        return self.g

    def backward(self, y):
        delta = self.g - y

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
