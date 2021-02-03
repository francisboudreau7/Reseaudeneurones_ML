import pickle
import numpy as np
import matplotlib.pyplot as plt

class NN(object):
    def __init__(self,
                 hidden_dims=(512, 120, 120,120,120,120),
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
        for layer in range(self.n_hidden+1,0,-1):
            if layer == self.n_hidden + 1:
                grads[f"dA{layer}"] = output - labels
            else:
                grads[f"dA{layer}"] = grads[f"dZ{layer}"] * self.activation(cache[f"A{layer}"], grad = True)
            grads[f"dW{layer}"] = np.dot(cache[f"Z{layer-1}"].T,grads[f"dA{layer}"])/self.batch_size
            grads[f"db{layer}"] = np.sum(grads[f"dA{layer}"],axis = 0,keepdims=True)/self.batch_size
            if layer > 1:
                grads[f"dZ{layer-1}"] = np.dot(grads[f"dA{layer}"], self.weights[f"W{layer}"].T)
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
        cost = -np.sum(labels*np.log(prediction))/len(prediction)
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
                cache = self.forward(minibatchX)
                grads = self.backward(cache,minibatchY)
                self.update(grads)
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
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy

    def normalize(self):
        # WRITE CODE HERE
        # compute mean and std along the first axis
        pass

def main():
    neural_net = NN(seed=0,lr=0.03,batch_size=100,datapath='svhn.pkl')
    epoch = 31
    neural_net.train_loop(30)
    print(neural_net.train_logs['train_accuracy'])
    print(neural_net.evaluate())

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(np.arange(1,epoch),neural_net.train_logs['validation_accuracy'],'-r',label = "validation")
    ax1.plot(np.arange(1,epoch),neural_net.train_logs['train_accuracy'],'-b',label = "train")
    ax2.plot(np.arange(1,epoch), neural_net.train_logs['validation_loss'], '-r',label="validation")
    ax2.plot(np.arange(1,epoch), neural_net.train_logs['train_loss'], '-b',label="train")
    ax1.set_ylabel("Precision")
    ax2.set_ylabel("Loss")
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc="upper right")
    plt.show()
if __name__ == "__main__":
    main()