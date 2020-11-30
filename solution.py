import pickle
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot",
                 normalization=False
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
            if normalization:
                self.normalize()
        else:
            self.train, self.valid, self.test = None, None, None

    def uniform_dist(self, n_in, n_out):
        if self.seed is not None:
            np.random.seed(self.seed)
        return

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            n_in = all_dims[layer_n - 1]
            n_out = all_dims[layer_n]
            uniform_dist = np.random.uniform(low=-np.sqrt(6 / (n_in + n_out)), high=np.sqrt(6 / (n_in + n_out)),
                                             size=(n_in, n_out))
            self.weights[f"W{layer_n}"] = uniform_dist
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            return np.where(x > 0, 1, 0)
        else:
            return np.maximum(0, x)

    def sigmoid(self, x, grad=False):
        if grad:
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        else:
            return 1 / (1 + np.exp(-x))

    def tanh(self, x, grad=False):
        if grad:
            return 1 - np.power((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), 2)
        else:
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def leakyrelu(self, x, grad=False):
        alpha = 0.01
        if grad:
            return np.where(x > 0, 1, alpha)
        else:
            return np.where(x >= 0, x, alpha * x)

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)

        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        elif self.activation_str == "leakyrelu":
            return self.leakyrelu(x, grad)
        else:
            raise Exception("invalid")

    def softmax(self, x):
        z = x - np.max(x)
        if x.ndim == 1:
            return np.exp(z) / np.sum(np.exp(z))
        else:
            return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def forward(self, x):
        cache = {"Z0": x}
        for layer_n in range(1, self.n_hidden + 2):
            cache[f"A{layer_n}"] = np.dot(cache[f"Z{layer_n - 1}"],self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
            if layer_n < self.n_hidden + 1:
                cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])
            else:
                cache[f"Z{layer_n}"] = self.softmax(cache[f"A{layer_n}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        pass
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - self.lr * grads[f"db{layer}"]


    def one_hot(self, y):
        return np.eye(self.n_classes)[y].astype(int)

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        y = labels
        a = self.one_hot(prediction)
        cost = (- 1/self.batch_size) * np.sum(y*np.log(a)+(13-y)*np.log(1-a))
        return cost

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                pass

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        # WRITE CODE HERE
        pass
        return 0

    def normalize(self):
        # WRITE CODE HERE
        # compute mean and std along the first axis
        pass
