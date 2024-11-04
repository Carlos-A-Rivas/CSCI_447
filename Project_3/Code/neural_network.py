import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from dataset import dataset

class neural_net:
    def __init__(self, data: dataset, prediction_type_flag: str, hidden_layer_count=0, network_shape=[], hidden_node_count=1, epochs=100, momentum=.9, learning_rate=.01, batch_size=10, suppress_plots=True):
        self.suppress_plots = suppress_plots
        self.epochs = epochs
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_layer_count = hidden_layer_count
        if hidden_layer_count == 0:
            hidden_node_count = 0
        else:
            hidden_node_count = [hidden_node_count] * hidden_layer_count
        self.tune_set = data.tune_set
        self.validate_set = data.validate_set
        self.prediction_type = prediction_type_flag

        if self.prediction_type == "classification":
            self.class_count = len(np.unique(self.tune_set[:,-1]))
        else:
            self.class_count = 0

        input_size = self.tune_set.shape[1] - 1
        self.input_size = input_size
        if network_shape == []:
            self.network_shape = [input_size] + (hidden_node_count if hidden_node_count else []) + ([self.class_count] if (self.prediction_type == "classification") else [1])
        else:
            self.network_shape = network_shape
        self.biases = []
        self.weights = []
        self.bias_velocity = []
        self.weight_velocity = []

    def init_weights_biases_momentum(self):
        '''
        Initializes weights based on the network shape list
        '''
        self.biases = [np.random.randn(next_size, 1) for next_size in self.network_shape[1:]]
        self.weights = [np.random.randn(next_size, cur_size) for cur_size, next_size in zip(self.network_shape[:-1], self.network_shape[1:])]
        self.bias_velocity = [np.zeros(bias.shape) for bias in self.biases]
        self.weight_velocity = [np.zeros(weight.shape) for weight in self.weights]
        #print(self.biases)


    def for_prop(self, input: np):
        '''
        Feeds forward a single example through the network
        '''
        output = input
        for bias, weight in zip(self.biases[:-1], self.weights[:-1]):
            output = self.sigmoid(np.dot(weight, output) + bias)
        # Not sure right now if output will be correct for regression, but life goes on.
        # The following lines choose the output activation function based on prediction type
        bias, weight = self.biases[-1], self.weights[-1]
        # MIGHT NEED TO RESHAPE THE DOT PRODUCT ON THE LINE BELOW
        output = (np.dot(weight, output) + bias)
        if self.prediction_type == "classification":
            output = self.softmax(output)
        return output
    
    def get_training_data(self, i: int):
        '''
        method needs to take the set of fold i-(i-1) and and compile those into its own array.
        Then format the data as follows: each example = (attributes, label)
        i is used to indicate which training set you want returned
        '''
        desired_data = np.concatenate([self.validate_set[j] for j in range(10) if j != i])
        training_data = [(example[:-1], example[-1]) for example in desired_data]
        return training_data
    
    def get_testing_data(self, i: int):
        '''
        method needs to take the set of fold i-(i-1) and and compile those into its own array.
        Then format the data as follows: each example = (attributes, label)
        i is used to indicate which training set you want returned
        '''
        desired_data = self.validate_set[i]
        testing_data = [(example[:-1], example[-1]) for example in desired_data]
        return testing_data
    
    def get_tuning_data(self):
        '''
        method needs to take the set of fold i-(i-1) and and compile those into its own array.
        Then format the data as follows: each example = (attributes, label)
        i is used to indicate which training set you want returned
        '''
        desired_data = self.tune_set
        tuning_data = [(example[:-1], example[-1]) for example in desired_data]
        return tuning_data

    def grad_desc(self, training_data, epochs, momentum, learning_rate, batch_size):
        '''
        Takes in a traing set from get_training_data. The format is a list of tuples, where each tuple
        represents an example. Within each tuple the first value is the feature vector and the second
        value is the label.
        '''
        example_count = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0, example_count, batch_size)]
            for mini_batch in mini_batches:
                self.update_weights(mini_batch, momentum, learning_rate)

    def update_weights(self, mini_batch, momentum, learning_rate):
        '''
        NEEDS COMMENTING/PARSING
        '''
        bias_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradient = [np.zeros(weight.shape) for weight in self.weights]

        # Compute gradients for the mini-batch
        for feature, label in mini_batch:
            #print(type(label))
            if not np.isnan(label):
                delta_bias_gradient, delta_weight_gradient = self.epoch(feature, label)
                bias_gradient = [gradient + delta for gradient, delta in zip(bias_gradient, delta_bias_gradient)]
                weight_gradient = [gradient + delta for gradient, delta in zip(weight_gradient, delta_weight_gradient)]
        
        # Update velocities and apply updates with momentum
        self.bias_velocity = [momentum * velocity - (learning_rate / len(mini_batch)) * gradient for velocity, gradient in zip(self.bias_velocity, bias_gradient)]
        self.weight_velocity = [momentum * velocity - (learning_rate / len(mini_batch)) * gradient for velocity, gradient in zip(self.weight_velocity, weight_gradient)]

        # Update weights and biases
        self.biases = [bias + velocity for bias, velocity in zip(self.biases, self.bias_velocity)]
        self.weights = [bias + velocity for bias, velocity in zip(self.weights, self.weight_velocity)]

    def epoch(self, feature, label):
        '''
        NEEDS COMMENTING/PARSING
        '''
        #print(f"\n\nLABEL: {label}\n\n")
        bias_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradient = [np.zeros(weight.shape) for weight in self.weights]
        # feedforward
        activation = feature
        activations = [feature] # list to store all the activations, layer by layer
        weighted_inputs = [] # list to store all the z vectors, layer by layer
        #print(f"Biases: {self.biases[-1]}")
        #print(f"Weights: {self.weights[-1]}")
        for bias, weight in zip(self.biases[:-1], self.weights[:-1]):
            #print(f"Activation:\n{activation.shape}\n\nWeight:\n{weight.shape}\n\nBias:\n{bias.shape}\n\n\n")
            weighted_input = np.dot(weight, activation.reshape(-1,1)) + bias
            #print(f"Weighted Input:\n{weighted_input.shape}")
            activation = self.sigmoid(weighted_input)
            weighted_inputs.append(weighted_input)
            activations.append(activation)
        # The output layer uses different activation functions
        bias, weight = self.biases[-1], self.weights[-1]
        #print(f"Activation:\n{activation}\n\nWeight:\n{weight.shape}\n\nBias:\n{bias.shape}\n\n\n")
        #print(f"Dot Product Shape:\n{np.dot(weight, activation).reshape(-1,1)}\n\n")
        weighted_input = np.dot(weight, activation.reshape(-1,1)) + bias
        #weighted_input = np.dot(weight, activation)
        #print(f"Weighted Input (Should be two scalars):\n{(weighted_input.shape)}")
        activation = weighted_input
        if self.prediction_type == "classification":
            activation = self.softmax(weighted_input)
        weighted_inputs.append(weighted_input)
        activations.append(activation)
        #print(f"Activations:\n{activations}\n\nWeighted Inputs:\n{weighted_inputs}")
    


        # backward pass
        # NEED TO ONE-HOT ENCODE THE LABELS FOR CLASSIFICATION SETS TO MAKE THE DELTA LINE WORK
        # PROLLY WON'T NEED LOSS PRIME METHOD
        if self.prediction_type == "classification":
            one_hot_label = [0] * self.class_count
            one_hot_label[int(label)] = 1
            one_hot_label = np.array(one_hot_label).reshape(-1, 1)
        else:
            one_hot_label = label
        #print(f"Activations[-1]:\n{activations[-1]}\nOne-Hot Label:\n{one_hot_label}")
        delta = (activations[-1] - one_hot_label)# * self.softmax(weighted_inputs[-1])
        bias_gradient[-1] = delta
        #print(f"Delta:\n{delta}\n\nActivations:\n{activations[-2].reshape(1,-1)}\n\n")
        weight_gradient[-1] = np.dot(delta, activations[-2].reshape(1,-1))# # CHECK THIS LINE FOR COMPREHENSION

        for layer_idx in range(2, len(self.network_shape)):
            weighted_input = weighted_inputs[-layer_idx]
            activation_prime = self.sigmoid_prime(weighted_input)
            delta = np.dot(self.weights[-layer_idx+1].transpose(), delta) * activation_prime
            # add logic to convert scalar if delta is 1x1
            if delta.shape == (1,1):
                bias_gradient[-layer_idx] = delta.reshape(-1)
                weight_gradient[-layer_idx] = (delta.reshape(-1) * activations[-layer_idx-1].transpose())
            else:
                bias_gradient[-layer_idx] = delta
                weight_gradient[-layer_idx] = np.dot(delta, activations[-layer_idx-1].reshape(1,-1))
        #print("GOT TO THE END")
        return (bias_gradient, weight_gradient)

    def tune(self):
        # CONSIDER REMOVING THE TQDM ON EPOCHS
        hidden_node_vals = [1, 3, 5, 7, 9]
        epoch_vals = [10, 50, 100, 200, 500]
        momentum_vals = [0.5, 0.7, 0.9, 0.95, 0.99]
        learning_rate_vals = [0.0001, 0.001, 0.01, 0.1, 1.0]
        batch_size_vals = [16, 32, 64, 128, 256]

        hidden_node_scores = []
        epoch_scores = []
        momentum_scores = []
        learning_rate_scores = []
        batch_size_scores = []

        # Hidden node Count Tuning
        if (self.hidden_layer_count > 0):
            hidden_node_combinations = list(itertools.product(hidden_node_vals, repeat=self.hidden_layer_count))
            for combination in tqdm(hidden_node_combinations, desc="Tuning Hidden Node Count", leave=False):
                self.network_shape = [self.input_size] + (list(combination)) + ([self.class_count] if (self.prediction_type == "classification") else [1])
                hidden_node_score = self.train_test(tuning_flag=True, epochs=self.epochs, momentum=self.momentum, learning_rate=self.learning_rate, batch_size=self.batch_size)
                hidden_node_scores.append(np.mean(hidden_node_score))
            hidden_node_scores = np.array(hidden_node_scores)
            if self.prediction_type == "classification":
                self.network_shape = [self.input_size] + (list(hidden_node_combinations[np.argmax(hidden_node_scores)])) + ([self.class_count] if (self.prediction_type == "classification") else [1])
            else:
                self.network_shape = [self.input_size] + (list(hidden_node_combinations[np.argmin(hidden_node_scores)])) + ([self.class_count] if (self.prediction_type == "classification") else [1])          
            print(f"Tuned Network Shape: {self.network_shape}")

        # Epoch tuning
        for epochs in tqdm(epoch_vals, desc="Tuning Epochs", leave=False):
            epoch_score = self.train_test(tuning_flag=True, epochs=epochs, momentum=self.momentum, learning_rate=self.learning_rate, batch_size=self.batch_size)
            epoch_scores.append(np.mean(epoch_score))
        epoch_scores = np.array(epoch_scores)
        if self.prediction_type == "classification":
            self.epochs = epoch_vals[np.argmax(epoch_scores)]
        else:
            self.epochs = epoch_vals[np.argmin(epoch_scores)]
        print(f"Tuned Epoch Value: {self.epochs}")

        # Momentum Tuning
        for momentum in tqdm(momentum_vals, desc="Tuning Momentum", leave=False):
            momentum_score = self.train_test(tuning_flag=True, epochs=self.epochs, momentum=momentum, learning_rate=self.learning_rate, batch_size=self.batch_size)
            momentum_scores.append(np.mean(momentum_score))
        momentum_scores = np.array(momentum_scores)
        if self.prediction_type == "classification":
            self.momentum = momentum_vals[np.argmax(momentum_scores)]
        else:
            self.momentum = momentum_vals[np.argmin(momentum_scores)]
        print(f"Tuned Momentum Value: {self.momentum}")

        # Learning rate tuning
        for learning_rate in tqdm(learning_rate_vals, desc="Tuning Learning Rate", leave=False):
            learning_rate_score = self.train_test(tuning_flag=True, epochs=self.epochs, momentum=self.momentum, learning_rate=learning_rate, batch_size=self.batch_size)
            learning_rate_scores.append(np.mean(learning_rate_score))
        learning_rate_scores = np.array(learning_rate_scores)
        if self.prediction_type == "classification":
            self.learning_rate = learning_rate_vals[np.argmax(learning_rate_scores)]
        else:
            self.learning_rate = learning_rate_vals[np.argmin(learning_rate_scores)]
        print(f"Tuned Learning Rate: {self.learning_rate}")

        # Batch size tuning
        for batch_size in tqdm(batch_size_vals, desc="Tuning Batch Size", leave=False):
            batch_size_score = self.train_test(tuning_flag=True, epochs=self.epochs, momentum=self.momentum, learning_rate=self.learning_rate, batch_size=batch_size)
            batch_size_scores.append(np.mean(batch_size_score))
        batch_size_scores = np.array(batch_size_scores)
        if self.prediction_type == "classification":
            self.batch_size = batch_size_vals[np.argmax(batch_size_scores)]
        else:
            self.batch_size = batch_size_vals[np.argmin(batch_size_scores)]
        print(f"Tuned Batch Size: {self.batch_size}")

        return [self.network_shape, self.epochs, self.momentum, self.learning_rate, self.batch_size]
    
    def train_test(self, tuning_flag: bool, epochs=100, momentum=.9, learning_rate=.01, batch_size=10):
        scores = []
        if tuning_flag:
            for i in range(10):
                self.init_weights_biases_momentum()
                self.grad_desc(self.get_training_data(i), epochs, momentum, learning_rate, batch_size)
                score = self.loss(self.get_tuning_data())
                scores.append(score)
        else:
            for i in tqdm(range(10), desc="Evaluating Test Data", leave=False):
                self.init_weights_biases_momentum()
                self.grad_desc(self.get_training_data(i), self.epochs, self.momentum, self.learning_rate, self.batch_size)
                score = self.loss(self.get_testing_data(i))
                scores.append(score)
        return np.array(scores)
    
    def loss(self, test_data):
        if self.prediction_type == "classification":
            results = [(np.argmax(self.for_prop(example)), label) for (example, label) in test_data if not np.isnan(label)]
            correct_results = sum(int(example == label) for (example, label) in results)
            total_examples = len(results)
            return correct_results / total_examples
        else:
            results = [(self.for_prop(x), y) for (x, y) in test_data if not np.isnan(y)]
            # Ensure predictions and labels are both 1D arrays of the same length
            predictions = np.array([prediction.flatten()[0] if prediction.size == 1 else np.argmax(prediction) for (prediction, label) in results], dtype=float)
            labels = np.array([label for (prediction, label) in results], dtype=float)

            # Calculate MSE
            mse = np.mean((predictions - labels) ** 2)
            return mse
            

            '''
            results = [(self.for_prop(x), y) for (x, y) in test_data if not np.isnan(y)]
            predictions = np.array([prediction.flatten() for (prediction, label) in results], dtype=float).reshape(-1)
            labels = np.array([label for (prediction, label) in results], dtype=float).reshape(-1)
            mse = np.mean((predictions - labels) ** 2)
            return mse
            '''
        
    '''
    def evaluate(self, test_data):
        """Return the accuracy of the network on the test data, excluding any NaN labels."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data if not np.isnan(y) and len(x) == 2]
        correct_predictions = sum(int(x == y) for (x, y) in test_results)
        total_examples = len(test_results)
        return correct_predictions / total_examples if total_examples > 0 else 0
    '''

    '''
    def loss_prime(self):
        return
    '''
    def sigmoid(self, input: np):
        return 1.0/(1.0+np.exp(-input))
    def sigmoid_prime(self, input: np):
        return self.sigmoid(input)*(1-self.sigmoid(input))
    def softmax(self, input):
        exp = np.exp(input - np.max(input))
        return exp / np.sum(exp)
    '''
    # Since loss output is in a slightly different format for neural nets, we might need an final loss method to output final performance
    def final_loss(self):
        return
    '''
    # THIS PROLLY NEEDS EDITING
    def plot_loss(self, metrics: list, parameter: str, increment):
        '''
        This function plots the loss performance for each epoch. This allows us to visualize at how many epochs
        performance drops off.
        '''
        # Extract # of epochs and loss metrics
        metrics = np.array(metrics)
        epochs = np.arange(1, metrics.shape[0] + 1) * increment
        loss1 = metrics[:, 0]
        loss2 = metrics[:, 1]

        # Create loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss1, label='Loss Metric 1', marker='o')
        plt.plot(epochs, loss2, label='Loss Metric 2', marker='o')
        plt.xlabel(f'{parameter} Value')
        plt.ylabel('Loss')
        plt.title(f'Loss Metrics vs. {parameter} value')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
    