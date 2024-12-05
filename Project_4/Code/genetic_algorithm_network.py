from differential_evolution_network import DE_nn
from dataset import dataset
import numpy as np
from joblib import Parallel, delayed
import tqdm as tqdm

# All files were developed collaboratively
class GA_nn(DE_nn):
    def __init__(self, data: dataset, prediction_type_flag: str, network_shape: list, population_size=50, epochs=100, selection_size=0.4, mutation_rate=0.05, crossover_rate=0.5):
        '''
        Initializes the hyperparameters, network architecture, etc. If tuning is not called, default hyperparameters are used
        '''
        self.pop_size = population_size
        self.epochs = epochs
        self.select_size = selection_size
        self.mr = mutation_rate
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
    def select_candidates(self, test_data, select_size):
        '''
        Returns the best and worst 'select_size' models among the population
        '''
        # Evaluate all candidates
        scores = []
        for candidate in self.population:
            scores.append(self.loss(test_data, candidate))
        scores = np.array(scores)
        
        # Sort candidates based on prediction type (classification or regression)
        if self.prediction_type == "classification":
            # For classification, higher scores are better
            sorted_indices = scores.argsort()[::-1]  # Sort in descending order
        else:
            # For regression, lower scores are better
            sorted_indices = scores.argsort()  # Sort in ascending order

        # Select the top 'select_size' candidates
        top_indices = sorted_indices[:select_size]
        top_candidates = [self.population[i] for i in top_indices]

        # Select the bottom 'select_size' candidates
        bottom_indices = sorted_indices[-select_size:]
        bottom_candidates = [self.population[i] for i in bottom_indices]

        return top_candidates, bottom_candidates
    def crossover(self, candidates, cr):

        new_candidates = []

        for i in range(0, len(candidates), 2):
            # Initialize lists to store trial weights and biases
            weights = []
            biases = []

            # Access the current pair of items
            parent_1 = candidates[i]
            if i + 1 < len(candidates):
                parent_2 = candidates[i + 1]
            parent_1_weights, parent_1_biases = parent_1
            parent_2_weights, parent_2_biases = parent_2

            # Apply crossover to weights
            for parent_1_weight, parent_2_weight in zip(parent_1_weights, parent_2_weights):
                # Perform element-wise crossover
                mask = np.random.rand(*parent_1_weight.shape) < cr
                weight = np.where(mask, parent_2_weight, parent_1_weight)
                weights.append(weight)

            # Apply crossover to biases
            for parent_1_bias, parent_2_bias in zip(parent_1_biases, parent_2_biases):
                # Perform element-wise crossover
                mask = np.random.rand(*parent_1_bias.shape) < cr
                bias = np.where(mask, parent_2_bias, parent_1_bias)
                biases.append(bias)
            
            new_candidates.append((weights,biases))

        # Return the crossed-over candidates
        return new_candidates   
    def mutate(self, candidates, mr):
        '''
        Mutates the weights and biases of the candidates based on a given mutation probability.
        Args:
        - candidates: List of candidates, where each candidate is a tuple (weights, biases).
        - mutation_prob: Probability of mutation for each weight and bias (float between 0 and 1).

        Returns:
        - mutated_candidates: List of mutated candidates.
        '''
        mutated_candidates = []

        # Iterate over all candidates
        for weights, biases in candidates:
            # Create new lists to store mutated weights and biases
            new_weights = []
            new_biases = []

            # Iterate over each weight array in the candidate
            for weight in weights:
                if np.random.rand() < mr:
                    # Mutate weight by adding a small random noise
                    mutated_weight = weight + np.random.normal(0, 0.1, size=weight.shape)
                else:
                    # No mutation, keep original weight
                    mutated_weight = weight
                new_weights.append(mutated_weight)

            # Iterate over each bias array in the candidate
            for bias in biases:
                if np.random.rand() < mr:
                    # Mutate bias by adding a small random noise
                    mutated_bias = bias + np.random.normal(0, 0.1, size=bias.shape)
                else:
                    # No mutation, keep original bias
                    mutated_bias = bias
                new_biases.append(mutated_bias)

            # Append the mutated candidate to the new list
            mutated_candidates.append((new_weights, new_biases))

        return mutated_candidates
    def selection(self, test_data, select_size, mr, cr):
        '''
        Returns a new population of equal size to the original, with replaced candidates
        that have been through tournament, crossover, and mutation.
        '''
        # Function to check if two tuples containing lists of numpy arrays are equal
        def are_tuples_equal(tuple1, tuple2):
            list1_weights, list1_biases = tuple1
            list2_weights, list2_biases = tuple2
            # Check if all weight arrays are equal
            if len(list1_weights) != len(list2_weights) or len(list1_biases) != len(list2_biases):
                return False
            for w1, w2 in zip(list1_weights, list2_weights):
                if not np.array_equal(w1, w2):
                    return False
            # Check if all bias arrays are equal
            for b1, b2 in zip(list1_biases, list2_biases):
                if not np.array_equal(b1, b2):
                    return False
            return True

        # Step 1: Select X best and X worst candidates
        top_candidates, bottom_candidates = self.select_candidates(test_data, select_size)
        # Step 2: Crossover the best candidates
        new_candidates = self.crossover(top_candidates, cr)
        # Step 3: Mutate the best candidates
        new_candidates = self.mutate(new_candidates, mr)
        # Step 4: Replace the X worst candidates with the new candidates
        new_population = []
        # Iterate over list_c and replace matching items with the corresponding item from list_b
        for candidate in self.population:
            found_match = False
            # Check if item matches anything in list_a
            for i, bottom_candidate in enumerate(bottom_candidates):
                if are_tuples_equal(candidate, bottom_candidate):
                    # Replace with the corresponding item in list_b
                    new_population.append(top_candidates[i])
                    found_match = True
                    break
            if not found_match:
                # If no match, keep the original item
                new_population.append(candidate)
        return new_population
    def evolve(self, test_data, epochs, select_size, mr, cr):
        for i in range(epochs):
            new_population = self.selection(test_data, select_size, mr, cr)
            self.population = new_population
        self.model = self.best_candidate(test_data)
    def train_test(self, tuning_flag: bool, pop_size=50, epochs=100, select_size=0.4, mr=0.05, cr=0.5):
        '''
        This method should take in the hyperparameters determined during tuning. It should use those hyperparameter
        values to train and test the model and return the calculated loss scores.
        '''

        # Define a function that encapsulates the work for each iteration
        def train_single_model(i):
            # Use self to access methods and attributes from the class
            if tuning_flag:
                self.init_population(pop_size)
                self.evolve(self.get_training_data(i), epochs, select_size, mr, cr)
            else:
                self.init_population(self.pop_size)
                self.evolve(self.get_training_data(i), self.epochs, int(self.select_size*self.pop_size), self.mr, self.cr)
            return self.loss(self.get_tuning_data() if tuning_flag else self.get_testing_data(i), self.model)

        # Parallel execution using joblib
        scores = Parallel(n_jobs=12)(
            delayed(train_single_model)(i)
            for i in tqdm(range(10), desc="Evaluating Models", leave=False)
        )
        return np.array(scores)
    def tune(self, tuning_pop_size=False, tuning_epochs=True, tuning_select_size=True, tuning_mr=True, tuning_cr=True):
        '''
        Five parameters need to be tuned: Population size, epochs, selection size, mutation rate, crossover rate
        '''
        pop_size_vals = [10, 50, 100, 200]
        epoch_vals = [10, 50, 100, 200, 500]
        select_size_vals = [0.2, 0.4, 0.6, 0.8]
        mr_vals = [0.01, 0.03, 0.05, 0.07, 0.1]
        cr_vals = [0.1, 0.3, 0.5, 0.7, 0.9]

        pop_size_scores = []
        epoch_scores = []
        select_size_scores = []
        mr_scores = []
        cr_scores = []

        if tuning_pop_size:
            # Population Size Tuning
            #Try all four pop size values and return the value that performs the best
            for pop_size in tqdm(pop_size_vals, desc="Tuning Population Size", leave=False):
                pop_size_score = self.train_test(tuning_flag=True, pop_size=pop_size, epochs=self.epochs, select_size=int(self.select_size*pop_size), mr=self.mr, cr=self.cr)
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
                epoch_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=epoch, select_size=int(self.select_size*self.pop_size), mr=self.mr, cr=self.cr)
                epoch_scores.append(np.mean(epoch_score))
            epoch_scores = np.array(epoch_scores)
            if self.prediction_type == "classification":
                self.epochs = epoch_vals[np.argmax(epoch_scores)]
            else:
                self.epochs = epoch_vals[np.argmin(epoch_scores)]
            print(f"Tuned Epoch Value: {self.epochs}")

        if tuning_select_size:
            # Scaling Factor Tuning
            #Try all five epoch values and return the value that performs the best
            for select_size in tqdm(select_size_vals, desc="Tuning Selection Size", leave=False):
                select_size_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=self.epochs, select_size=int(select_size*self.pop_size), mr=self.mr, cr=self.cr)
                select_size_scores.append(np.mean(select_size_score))
            select_size_scores = np.array(select_size_scores)
            if self.prediction_type == "classification":
                self.select_size = select_size_vals[np.argmax(select_size_scores)]
            else:
                self.select_size = select_size_vals[np.argmin(select_size_scores)]
            print(f"Tuned Selection Size: {self.select_size}")

        if tuning_mr:
            # Crossover Rate Tuning
            #Try all five epoch values and return the value that performs the best
            for mr in tqdm(mr_vals, desc="Tuning Mutation Rate", leave=False):
                mr_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=self.epochs, select_size=int(self.select_size*self.pop_size), mr=mr, cr=self.cr)
                mr_scores.append(np.mean(mr_score))
            mr_scores = np.array(mr_scores)
            if self.prediction_type == "classification":
                self.mr = mr_vals[np.argmax(mr_scores)]
            else:
                self.mr = mr_vals[np.argmin(mr_scores)]
            print(f"Tuned Mutation Rate: {self.mr}")

        if tuning_cr:
            # Crossover Rate Tuning
            #Try all five epoch values and return the value that performs the best
            for cr in tqdm(cr_vals, desc="Tuning Crossover Rate", leave=False):
                cr_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=self.epochs, select_size=int(self.select_size*self.pop_size), mr=self.mr, cr=cr)
                cr_scores.append(np.mean(cr_score))
            cr_scores = np.array(cr_scores)
            if self.prediction_type == "classification":
                self.cr = cr_vals[np.argmax(cr_scores)]
            else:
                self.cr = cr_vals[np.argmin(cr_scores)]
            print(f"Tuned Crossover Rate: {self.cr}")


        return [self.pop_size, self.epochs, self.select_size, self.mr, self.cr]   