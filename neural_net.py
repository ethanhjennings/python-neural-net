'''
Implementation of a general purpose deep neural network.
'''

from itertools import islice
import concurrent.futures
import json
import math
import pickle
import random
import time

import numpy as np

np.seterr(all='raise')

# Constants that can be tuned:

BATCH_SIZE = 32

# Weight/bias initalization functions

def random_init(rows, cols=None):
    if cols is not None:
        spread = math.sqrt(1/cols)
        return np.random.normal(loc=0, scale=spread, size=(rows, cols))
    else:
        return np.zeros(rows)

# Activation functions:

def sigmoid(data):
    return 1.0 / (1.0 + np.exp(-1.0 * data))

def d_sigmoid(data):
    s = sigmoid(data)
    return s*(1.0-s)

def relu(data):
    # Rectifier Linear Unit (reLU)
    data[data <= 0] = 0
    return data

def d_relu(data):
    data[data <= 0] = 0
    data[data > 0] = 1
    return data

def elu(data):
    alpha = 1
    return np.piecewise(data, [data <= 0, data > 0], [lambda x: alpha*(np.exp(np.clip(x, -10, 10)) - 1), lambda x: x])

def d_elu(data):
    alpha = 1
    return np.piecewise(data, [data <= 0, data > 0], [lambda x: alpha*np.exp(np.clip(x, -10, 10)), lambda x: 1])

# Cost functions:
def mean_squared_error(actual, expected):
    # Divided by two so the derviative is cleaner
    return ((expected - actual) ** 2)/2

def d_mean_squared_error(actual, expected):
    # Divided by two for clenliness
    return actual - expected

def l1_cost(actual, expected):
    return ((np.abs(actual - expected)) ** 1.5).mean()

def d_l1_cost(actual, expected):
    # Divided by two for cleanliness
    delta = actual - expected
    return np.sign(delta) * 1.5 * (np.abs(delta)) ** 0.5

class NeuralNetwork:
    def __init__(self, 
            layer_sizes,
            output_file,
            learning_rate=0.05,
            using_dropout=True,
            dropout_prob=0.2,
            init_func=random_init,
            act_func=relu, 
            act_func_deriv=d_relu,
            error_func=mean_squared_error,
            error_func_deriv=d_mean_squared_error):

        self.layer_sizes = layer_sizes
        self.output_file = output_file
        self.learning_rate = learning_rate
        self.using_dropout = using_dropout
        self.dropout_prob = dropout_prob
        self.init_func = init_func
        self.act_func = act_func
        self.act_func_deriv = act_func_deriv
        self.error_func=error_func
        self.error_func_deriv=error_func_deriv
        self.weights = self._gen_random_weights(layer_sizes)
        self.biases = self._gen_random_biases(layer_sizes)

    def _gen_random_weights(self, layer_sizes):
        weights = []
        for i in range(1, len(layer_sizes)):
            prev_layer_size = layer_sizes[i-1]
            layer_size = layer_sizes[i]
            weights.append(self.init_func(layer_size, prev_layer_size))
        return weights

    def _gen_random_biases(self, layer_sizes):
        return [self.init_func(size, None) for size in layer_sizes[1:]]

    def _feedforward(self, inputs, dropout_mask=None):
        activations = [inputs]

        weighted_inputs = []
        for i, (weight_matrix, bias_vec) in enumerate(zip(self.weights, self.biases)):            
            weighted_input = weight_matrix.dot(activations[-1]) + bias_vec
            activation = self.act_func(weighted_input)
            
            # Apply dropout
            if dropout_mask is not None:
                activation *= dropout_mask[i+1]

            weighted_inputs.append(weighted_input)
            activations.append(activation)

        return activations, weighted_inputs

    def _backpropogation(self, input_activations, expected_outputs):
        '''Calculate network's gradient for a single training example.'''

        weight_gradient = []
        bias_gradient = []

        # Create randomized dropout mask with `dropout_prob` that any neuron
        # is dropped and scale up surviving neurons accordingly
        if self.using_dropout:
            keep_prob = 1 - self.dropout_prob
            dropout_mask = [(np.random.random(s) < keep_prob) / keep_prob 
                    for s in self.layer_sizes]
        else:
            dropout_mask = None

        # Evaluate network normally in the forward direction
        activations, weighted_inputs = self._feedforward(input_activations, dropout_mask)

        last_idx = len(self.layer_sizes)-2

        # Compute weight and bias gradients for each layer in reverse
        for i in range(last_idx, -1, -1):
            if i == last_idx: 
                # Output layer
                layer_errors = (self.error_func_deriv(activations[-1], expected_outputs) * 
                    self.act_func_deriv(weighted_inputs[-1]))
            else: 
                # Input and hidden layers
                layer_errors = (self.weights[i+1].transpose().dot(next_layer_errors) *
                    self.act_func_deriv(weighted_inputs[i]))

                # Apply dropout
                if self.using_dropout:
                    layer_errors *= dropout_mask[i + 1]
                
            weight_gradient.insert(0, np.outer(layer_errors,activations[i]))
            bias_gradient.insert(0, layer_errors)
            next_layer_errors = layer_errors

        return weight_gradient, bias_gradient

    def run(self, inputs):
        ''' Evaluate this neural network for a single set of inputs
            Takes in a np array of input activations with each element in range [0, 1]
            Returns activations of the output neurons as np array in range [0, 1]
        '''
        activations, weighted_input = self._feedforward(inputs)
        return activations[-1]

    def train(self, training_examples, test_examples):
        ''' Train this neural network with training data/labels and evaluate performance.
            training_examples and test_examples must be a list of tuples of input and expected 
            output activations (i.e. labels):
                 [(input activations, expected outputs), ...]
            You must call this method before run() or the network will give undefined results.
        '''

        global BATCH_SIZE

        percent_corrects = []

        print("Training params:")
        print("Activation function: " + self.act_func.__name__)
        print("Layer sizes: " + str(self.layer_sizes))

        for i in range(0, 50000):
            print(str(min([g.min() for g in self.weights])) + "," + str(max([g.max() for g in self.weights])))
            print(str(min([g.min() for g in self.biases])) + "," + str(max([g.max() for g in self.biases])))
            print("Saving progress to " + self.output_file + "." + str(i))
            pickle.dump(self, open(self.output_file + "." + str(i), 'wb'))

            print("Learning trial #" + str(i))
            self.dropout_prob -= 0.001
            self.dropout_prob = max(self.dropout_prob, 0)
            if i > 500:
                self.learning_rate = 0.01
            if i > 1000:
                self.learning_rate = 0.001
            if i > 2000:
                self.learning_rate = 0.0001
            print("Learning rate: " + str(self.learning_rate) + " batch size: " + str(BATCH_SIZE))

            guesses_and_answers = [(np.argmax(self.run(test_input)), np.argmax(test_label)) for test_input, test_label in test_examples]
            num_correct = len([1 for guess, answer in guesses_and_answers if guess == answer])
            total = len(test_examples)
            print("Test: " + str(num_correct) + "/" + str(total) + " correct: " + str(num_correct/total))

            start = time.time()

            # Break up training data into random batches of size BATCH_SIZE to save time when learning
            batches = [islice(training_examples, BATCH_SIZE) for i in range(0, int(60000/BATCH_SIZE))]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                for j, batch in enumerate(batches):
                    # Calculate gradient for this batch
                    results = list(executor.map(self._backpropogation, *zip(*batch)))

                    mean_weight_gradient      = np.mean([r[0] for r in results], axis=0) * self.learning_rate
                    mean_bias_gradient        = np.mean([r[1] for r in results], axis=0) * self.learning_rate

                    # Update weights and biases
                    self.weights = [weight_matrix - grad for weight_matrix, grad in zip(self.weights, mean_weight_gradient)]
                    self.biases =  [bias_vec      - grad for bias_vec,      grad in zip(self.biases,  mean_bias_gradient)]

            end = time.time()
            print("Epoch took " + str(end-start) + "s")
        print("Finished learning!")

    def _numerical_gradient(self, inputs, expected_outputs):
        ''' This estimates a gradient using finite difference. It should be really
            close to the backpropogtion functions's gradient, but is orders of magnitude
            slower so it should only be used for testing the accuracy of backprop. '''

        EPSILON = 0.00000001

        gradient = []

        for i, weight_mat in enumerate(self.weights):
            grad_weight_mat = np.zeros(shape=weight_mat.shape)
            for r, row in enumerate(weight_mat):
                for c, col in enumerate(row):
                    self.weights[i][r, c] += EPSILON
                    cost_a = self.error_func(self._feedforward(inputs)[0][-1], expected_outputs)

                    self.weights[i][r, c] -= 2*EPSILON
                    cost_b = self.error_func(self._feedforward(inputs)[0][-1], expected_outputs)

                    self.weights[i][r, c] += EPSILON

                    grad_weight_mat[r, c] = (cost_a - cost_b)/(EPSILON)
            gradient.append(grad_weight_mat)

        bias_gradient = []
        for i, bias_vec in enumerate(self.biases):
            bias_vec = np.zeros(shape=bias_vec.shape)
            for b, bias in enumerate(bias_vec):
                self.biases[i][b] += EPSILON
                cost_a = self.error_func(self._feedforward(inputs)[0][-1], expected_outputs)

                self.biases[i][b] -= 2*EPSILON
                cost_b = self.error_func(self._feedforward(inputs)[0][-1], expected_outputs)

                self.biases[i][b] += EPSILON

                bias_vec[b] = (cost_a - cost_b)/(EPSILON)
            bias_gradient.append(bias_vec)

        activations, weighted_inputs = self._feedforward(inputs)
        error = self.error_func(activations[-1], expected_outputs)

        return gradient, bias_gradient, error

    def to_json(self):
        ''' Convert network to json.'''
        weights_list = [w.tolist() for w in self.weights]
        biases_list = [b.tolist() for b in self.biases]
        return json.dumps({"weights": weights_list, "biases": biases_list})
