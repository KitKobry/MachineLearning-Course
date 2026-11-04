
import random
from value import Value


class Neuron:
    def __init__(self, n_inputs):
        self.weights = [
            Value(random.uniform(-1, 1))
            for _ in range(n_inputs)
        ]
        self.bias = Value(0.0)

    def __call__(self, x, activation=True):
        result = self.bias

        for weight, value in zip(self.weights, x):
            result = result + weight * value

        if activation:
            return result.relu()

        return result


class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.hidden = [
            Neuron(n_inputs)
            for _ in range(n_hidden)
        ]

        self.output = [
            Neuron(n_hidden)
            for _ in range(n_outputs)
        ]

    def __call__(self, x):
        hidden_values = [
            neuron(x) for neuron in self.hidden
        ]

        outputs = [
            neuron(hidden_values, activation=False)
            for neuron in self.output
        ]

        return outputs[0].sigmoid()

    def parameters(self):
        params = []

        for neuron in self.hidden + self.output:
            params.extend(neuron.weights)
            params.append(neuron.bias)

        return params


def binary_cross_entropy(prediction, target):
    eps = 1e-7

    return -(
        target * (prediction + eps).log()
        + (1 - target) * (1 - prediction + eps).log()
    )
