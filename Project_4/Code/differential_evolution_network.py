from backprop_network import backprop_nn
from dataset import dataset
import numpy as np
import random
from joblib import Parallel, delayed
import tqdm as tqdm


class DE_nn(backprop_nn):
    def __init__(self, data: dataset, prediction_type_flag: str, network_shape: list, population_size=50, epochs=100, scaling_factor=0.7, crossover_rate=0.7):
        self.epochs = epochs
        self.pop_size = population_size
        self.f = scaling_factor
        self.cr = crossover_rate
        self.tune_set = data.tune_set
        self.validate_set = data.validate_set
        self.prediction_type = prediction_type_flag

        if self.prediction_type == "classification":
            self.class_count = len(np.unique(self.tune_set[:,-1]))
        else:
            self.class_count = 0
            
        input_size = self.tune_set.shape[1] - 1
        self.input_size = input_size
        self.network_shape = network_shape
        self.population = []
        self.model = ()
    def init_population(self, pop_size):
        '''
        Initializes a population of weights and biases
        '''
        self.population = []
        for i in range(pop_size):
            biases = [np.random.randn(next_size, 1) for next_size in self.network_shape[1:]]
            weights = [np.random.randn(next_size, cur_size) for cur_size, next_size in zip(self.network_shape[:-1], self.network_shape[1:])]
            self.population.append((weights,biases))
    def donor_vector(self, f):
        # Step 1: Select 3 candidates
        candidates = random.sample(self.population,3)
        # Step 2: For each candidate, mutate the weights and biases
        weights_1, biases_1 = candidates[0]
        weights_2, biases_2 = candidates[1]
        weights_3, biases_3 = candidates[2]
        new_weights = []
        new_biases = []

        for w1, w2, w3 in zip(weights_1, weights_2, weights_3):
            new_weight = w1 + f * (w2 - w3)
            new_weights.append(new_weight)

        # Apply the equation for each bias vector
        for b1, b2, b3 in zip(biases_1, biases_2, biases_3):
            new_bias = b1 + f * (b2 - b3)
            new_biases.append(new_bias)

        # Return the donor vector (weights, biases)
        return (new_weights, new_biases)
    def trial_vector(self, target_candidate, f, cr):
        # Unpack target candidate and donor vector
        target_weights, target_biases = target_candidate
        donor_vector = self.donor_vector(f)
        donor_weights, donor_biases = donor_vector

        # Initialize lists to store trial weights and biases
        trial_weights = []
        trial_biases = []

        # Apply crossover to weights
        for target_weight, donor_weight in zip(target_weights, donor_weights):
            # Perform element-wise crossover
            mask = np.random.rand(*target_weight.shape) < cr
            trial_weight = np.where(mask, donor_weight, target_weight)
            trial_weights.append(trial_weight)

        # Apply crossover to biases
        for target_bias, donor_bias in zip(target_biases, donor_biases):
            # Perform element-wise crossover
            mask = np.random.rand(*target_bias.shape) < cr
            trial_bias = np.where(mask, donor_bias, target_bias)
            trial_biases.append(trial_bias)

        # Return the trial vector (weights, biases)
        return (trial_weights, trial_biases)
    def evaluate_fitness(self, test_data, target, trial):
        '''
        Returns the best performing model between the target and trial
        '''
        target_fitness = self.loss(test_data, target)
        trial_fitness = self.loss(test_data, trial)
        if self.prediction_type == "classification":
            if max(target_fitness, trial_fitness) == target_fitness:
                return target
            else:
                return trial
        else:
            if min(target_fitness, trial_fitness) == target_fitness:
                return target
            else:
                return trial
    def for_prop(self, input: np, weights, biases):
        '''
        Feeds forward a single example through the network
        '''
        output = input
        for bias, weight in zip(biases[:-1], weights[:-1]):
            output = self.sigmoid(np.dot(weight, output) + bias)        #for each weight calculate the output of the activation function
                
        bias, weight = biases[-1], weights[-1]
        #For regression, use a linear combination for output activation
        #For classification, use a softmax output activation
        output = (np.dot(weight, output) + bias)    
        if self.prediction_type == "classification":
            output = self.softmax(output)
        return output
    def loss(self, test_data, model):
        '''
        This method calculates the loss based on our evaluation metrics
        For classification: 0/1 loss
        For regression: Mean squared error
        '''
        weights, biases = model
        if self.prediction_type == "classification":
            results = [(np.argmax(self.for_prop(example, weights, biases)), label) for (example, label) in test_data if not np.isnan(label)]
            correct_results = sum(int(example == label) for (example, label) in results)
            total_examples = len(results)
            return correct_results / total_examples
        else:
            results = [(self.for_prop(x, weights, biases), y) for (x, y) in test_data if not np.isnan(y)]
            # Ensure predictions and labels are both 1D arrays of the same length
            predictions = np.array([prediction.flatten()[0] if prediction.size == 1 else np.argmax(prediction) for (prediction, label) in results], dtype=float)
            labels = np.array([label for (prediction, label) in results], dtype=float)

            # Calculate MSE
            mse = np.mean((predictions - labels) ** 2)
            return mse
    def best_candidate(self, test_data):
        '''
        Returns the best model among a population
        '''
        scores = []
        for candidate in self.population:
            scores.append(self.loss(test_data, candidate))
        scores = np.array(scores)
        if self.prediction_type == "classification":
            return self.population[np.argmax(scores)]
        else:
            return self.population[np.argmin(scores)]
    def evolve(self, test_data, epochs, f, cr):
        '''
        This method evolves the models in the population
        '''
        for i in range(epochs):
            new_population = []
            for candidate in self.population:
                trial_vector = self.trial_vector(candidate, f, cr)
                new_candidate = self.evaluate_fitness(test_data, candidate, trial_vector)
                new_population.append(new_candidate)
            self.population = new_population
        self.model = self.best_candidate(test_data)
    def train_test(self, tuning_flag: bool, pop_size=50, epochs=100, f=0.7, cr=0.5):
        '''
        This method should take in the hyperparameters determined during tuning. It should use those hyperparameter
        values to train and test the model and return the calculated loss scores.
        '''

        # Define a function that encapsulates the work for each iteration
        def train_single_model(i):
            # Use self to access methods and attributes from the class
            if tuning_flag:
                self.init_population(pop_size)
                self.evolve(self.get_training_data(i), epochs, f, cr)
            else:
                self.init_population(self.pop_size)
                self.evolve(self.get_training_data(i), self.epochs, self.f, self.cr)
            return self.loss(self.get_tuning_data() if tuning_flag else self.get_testing_data(i), self.model)

        # Parallel execution using joblib
        scores = Parallel(n_jobs=12)(
            delayed(train_single_model)(i)
            for i in tqdm(range(10), desc="Evaluating Models", leave=False)
        )
        return np.array(scores)
    def tune(self, tuning_pop_size=False, tuning_epochs=True, tuning_f=True, tuning_cr=True):
        '''
        Four parameters need to be tuned: Population size, epochs, scaling factor, crossover rate
        '''
        pop_size_vals = [10, 50, 100, 200]
        epoch_vals = [10, 50, 100, 200, 500]
        f_vals = [0.4, 0.5, 0.7, 0.9, 1.0]
        cr_vals = [0.1, 0.3, 0.5, 0.7, 0.9]

        pop_size_scores = []
        epoch_scores = []
        f_scores = []
        cr_scores = []

        if tuning_pop_size:
            # Population Size Tuning
            #Try all four pop size values and return the value that performs the best
            for pop_size in tqdm(pop_size_vals, desc="Tuning Population Size", leave=False):
                pop_size_score = self.train_test(tuning_flag=True, pop_size=pop_size, epochs=self.epochs, f=self.f, cr=self.cr)
                pop_size_scores.append(np.mean(pop_size_score))
            pop_size_scores = np.array(pop_size_scores)
            if self.prediction_type == "classification":
                self.pop_size = pop_size_vals[np.argmax(pop_size_scores)]
            else:
                self.pop_size = pop_size_vals[np.argmin(pop_size_scores)]
            print(f"Tuned Population Size: {self.pop_size}")
        
        if tuning_epochs:
            # Epoch Tuning
            #Try all five epoch values and return the value that performs the best
            for epoch in tqdm(epoch_vals, desc="Tuning Epochs", leave=False):
                epoch_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=epoch, f=self.f, cr=self.cr)
                epoch_scores.append(np.mean(epoch_score))
            epoch_scores = np.array(epoch_scores)
            if self.prediction_type == "classification":
                self.epochs = epoch_vals[np.argmax(epoch_scores)]
            else:
                self.epochs = epoch_vals[np.argmin(epoch_scores)]
            print(f"Tuned Epoch Value: {self.epochs}")

        if tuning_f:
            # Scaling Factor Tuning
            #Try all five epoch values and return the value that performs the best
            for f in tqdm(f_vals, desc="Tuning Scaling Factor", leave=False):
                f_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=self.epochs, f=f, cr=self.cr)
                f_scores.append(np.mean(f_score))
            f_scores = np.array(f_scores)
            if self.prediction_type == "classification":
                self.f = f_vals[np.argmax(f_scores)]
            else:
                self.f = f_vals[np.argmin(f_scores)]
            print(f"Tuned Scaling Factor: {self.f}")

        if tuning_cr:
            # Crossover Rate Tuning
            #Try all five epoch values and return the value that performs the best
            for cr in tqdm(cr_vals, desc="Tuning Crossover Rate", leave=False):
                cr_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=self.epochs, f=self.f, cr=cr)
                cr_scores.append(np.mean(cr_score))
            cr_scores = np.array(cr_scores)
            if self.prediction_type == "classification":
                self.cr = cr_vals[np.argmax(cr_scores)]
            else:
                self.cr = cr_vals[np.argmin(cr_scores)]
            print(f"Tuned Crossover Rate: {self.cr}")


        return [self.pop_size, self.epochs, self.f, self.cr]