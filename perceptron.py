import numpy as np
import matplotlib.pyplot as plt


class Layer:

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        self.weights = weights if weights is not None else np.random.uniform(-0.5, 0.5, size=(n_input, n_neurons))
        self.activation = activation
        self.bias = bias if bias is not None else 1
        self.activated_output = None
        self.output = None
        self.error = None

    def activate(self, x):
        output = np.dot(x, self.weights) + self.bias
        self.output = output
        self.activated_output = self._apply_activation(output)
        return self.activated_output

    def _apply_activation(self, x):
        if self.activation is None:
            return x

        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))

        return x

    def apply_activation_derivative(self, x):

        if self.activation is None:
            return x

        if self.activation == 'sigmoid':
            return x * (1 - x)

        return x


class NeuralNetwork:

    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def predict(self, X):
        ff = self.feed_forward(X)
        return ff

    def back_propagation(self, X, y, learning_rate):

        output = self.feed_forward(X)

        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = (y - output) * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = next_layer.error.dot(next_layer.weights.T) * layer.apply_activation_derivative(
                    layer.activated_output)

        for i in range(len(self._layers)):
            layer = self._layers[i]
            last_output = np.atleast_2d(X if i == 0 else self._layers[i - 1].activated_output)
            layer.weights += last_output.T.dot(layer.error) * learning_rate
            layer.bias += np.sum(layer.error, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, learning_rate, max_epochs):
        rmses = []

        for epoch in range(max_epochs):
            self.back_propagation(X, y, learning_rate)
            mse = np.square(np.subtract(y, nn.feed_forward(X)))
            rmse = np.mean(np.sqrt(np.sum(mse, axis=1)))
            rmses.append(rmse)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, RMSE: {rmse}')

        return rmses


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add_layer(Layer(4, 2, 'sigmoid'))
    nn.add_layer(Layer(2, 4, 'sigmoid'))

    X = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    y = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    errors = nn.train(X, y, 0.2, 5000)
    plt.plot(errors, label='RMSE')
    plt.title('Changes in RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.ylim(0)
    plt.legend()
    plt.show()

    print("Data input: \n" + str(X))
    print("Predicted output: \n" + str(np.round(nn.predict(X), 3)))
