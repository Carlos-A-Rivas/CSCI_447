from dataset import dataset
import numpy as np
from tqdm import tqdm

class PSO_nn:
    def __init__(self, data: dataset, prediction_type_flag: str, network_shape: list, population_size=50, epochs=50, social_update_rate=0.5, cognitive_update_rate=0.5, inertia=1):
        '''
        Initializes the hyperparameters, network architecture, etc. If tuning is not called, default hyperparameters are used
        '''
        self.epochs = epochs
        self.pop_size = population_size
        self.c1 = cognitive_update_rate  # Cognitive update rate
        self.c2 = social_update_rate     # Social update rate
        self.w = inertia                 # Inertia weight
        self.tune_set = data.tune_set
        self.validate_set = data.validate_set
        self.prediction_type = prediction_type_flag

        if self.prediction_type == "classification":
            self.class_count = len(np.unique(self.tune_set[:, -1]))
        else:
            self.class_count = 0

        input_size = self.tune_set.shape[1] - 1
        self.input_size = input_size
        self.network_shape = network_shape
        self.population = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_scores = []
        self.global_best_position = None
        self.global_best_score = None
        self.model = ()

    def init_population(self):
        '''
        Initializes a population of particles with random weights and velocities.
        '''
        self.population = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_scores = []

        for i in range(self.pop_size):
            biases = [np.random.randn(next_size, 1) for next_size in self.network_shape[1:]]
            weights = [np.random.randn(next_size, cur_size) for cur_size, next_size in zip(self.network_shape[:-1], self.network_shape[1:])]
            particle = (weights, biases)
            self.population.append(particle)

            # Initialize velocities 
            vel_biases = [np.zeros_like(bias) for bias in biases]
            vel_weights = [np.zeros_like(weight) for weight in weights]
            velocity = (vel_weights, vel_biases)
            self.velocities.append(velocity)

            # Initialize personal bests
            self.personal_best_positions.append(particle)
            score = self.loss(self.get_training_data(i % 10), particle)
            self.personal_best_scores.append(score)

        # Initialize global best
        best_idx = np.argmin(self.personal_best_scores) if self.prediction_type != "classification" else np.argmax(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx]
        self.global_best_score = self.personal_best_scores[best_idx]

    def update_velocity(self, idx):
        '''
        Updates the velocity of a particle based on inertia, cognitive update rate, and social update rate.
        '''
        particle = self.population[idx]
        velocity = self.velocities[idx]
        personal_best = self.personal_best_positions[idx]
        global_best = self.global_best_position

        new_vel_weights = []
        new_vel_biases = []

        for v_w, x_w, pbest_w, gbest_w in zip(velocity[0], particle[0], personal_best[0], global_best[0]):
            r1 = np.random.rand(*x_w.shape)
            r2 = np.random.rand(*x_w.shape)
            cognitive_component = self.c1 * r1 * (pbest_w - x_w)
            social_component = self.c2 * r2 * (gbest_w - x_w)
            new_velocity_w = self.w * v_w + cognitive_component + social_component
            new_vel_weights.append(new_velocity_w)

        for v_b, x_b, pbest_b, gbest_b in zip(velocity[1], particle[1], personal_best[1], global_best[1]):
            r1 = np.random.rand(*x_b.shape)
            r2 = np.random.rand(*x_b.shape)
            cognitive_component = self.c1 * r1 * (pbest_b - x_b)
            social_component = self.c2 * r2 * (gbest_b - x_b)
            new_velocity_b = self.w * v_b + cognitive_component + social_component
            new_vel_biases.append(new_velocity_b)

        velocity_limit = 3  #Limit max velocity
        new_vel_weights = [np.clip(vw, -velocity_limit, velocity_limit) for vw in new_vel_weights]
        new_vel_biases = [np.clip(vb, -velocity_limit, velocity_limit) for vb in new_vel_biases]

        self.velocities[idx] = (new_vel_weights, new_vel_biases)

    def update_position(self, idx):
        '''
        Updates the position of a particle
        '''
        particle = self.population[idx]
        velocity = self.velocities[idx]

        new_weights = []
        new_biases = []

        for x_w, v_w in zip(particle[0], velocity[0]):
            new_w = x_w + v_w
            new_weights.append(new_w)

        for x_b, v_b in zip(particle[1], velocity[1]):
            new_b = x_b + v_b
            new_biases.append(new_b)

        self.population[idx] = (new_weights, new_biases)

    def evaluate_fitness(self, idx, test_data):
        '''
        Evaluates the fitness of a particle and updates personal and global bests if necessary
        '''
        particle = self.population[idx]
        score = self.loss(test_data, particle)

        # Update personal best
        if self.prediction_type == "classification":
            if score > self.personal_best_scores[idx]:
                self.personal_best_positions[idx] = particle
                self.personal_best_scores[idx] = score
        else:
            if score < self.personal_best_scores[idx]:
                self.personal_best_positions[idx] = particle
                self.personal_best_scores[idx] = score

        # Update global best
        if self.prediction_type == "classification":
            if score > self.global_best_score:
                self.global_best_position = particle
                self.global_best_score = score
        else:
            if score < self.global_best_score:
                self.global_best_position = particle
                self.global_best_score = score

    def for_prop(self, input_vector: np.ndarray, weights, biases):
        '''
        Feeds forward a single example through the network.
        '''
        output = input_vector
        for bias, weight in zip(biases[:-1], weights[:-1]):
            output = self.sigmoid(np.dot(weight, output) + bias)
        bias, weight = biases[-1], weights[-1]
        output = np.dot(weight, output) + bias
        if self.prediction_type == "classification":
            output = self.softmax(output)
        return output

    def loss(self, test_data, model):
        '''
        Calculates the loss based on evaluation metrics.
        For classification: 0/1 loss
        For regression: Mean squared error
        '''
        weights, biases = model
        if self.prediction_type == "classification":
            results = [(np.argmax(self.for_prop(example.reshape(-1,1), weights, biases)), int(label)) for (example, label) in test_data if not np.isnan(label)]
            correct_results = sum(int(pred == label) for (pred, label) in results)
            total_examples = len(results)
            return correct_results / total_examples
        else:
            results = [(self.for_prop(x.reshape(-1,1), weights, biases), y) for (x, y) in test_data if not np.isnan(y)]
            predictions = np.array([prediction.flatten()[0] for (prediction, label) in results], dtype=float)
            labels = np.array([label for (prediction, label) in results], dtype=float)
            mse = np.mean((predictions - labels) ** 2)
            return mse

    def best_candidate(self, test_data):
        '''
        Returns the best performing partiicle among the population.
        '''
        scores = []
        for candidate in self.population:
            scores.append(self.loss(test_data, candidate))
        scores = np.array(scores)
        if self.prediction_type == "classification":
            return self.population[np.argmax(scores)]
        else:
            return self.population[np.argmin(scores)]

    def evolve(self, test_data, epochs):
        '''
        Evolves the population over a number of epochs.
        '''
        for epoch in tqdm(range(epochs), desc="Evolving Particles", leave=False):
            for idx in range(self.pop_size):
                self.update_velocity(idx)
                self.update_position(idx)
                self.evaluate_fitness(idx, test_data)
        self.model = self.global_best_position

    def train_test(self, tuning_flag: bool, pop_size=50, epochs=50, c1=0.5, c2=0.5, w=1):
        '''
        Trains and tests the model, returns the calculated loss scores.
        '''
        scores = []
        if tuning_flag:
            for i in range(10):
                self.pop_size = pop_size
                self.epochs = epochs
                self.c1 = c1
                self.c2 = c2
                self.w = w
                self.init_population()
                self.evolve(self.get_training_data(i), epochs)
                score = self.loss(self.get_tuning_data(), self.model)
                scores.append(score)
        else:
            for i in tqdm(range(10), desc="Evaluating Test Data", leave=False):
                self.init_population()
                self.evolve(self.get_training_data(i), self.epochs)
                score = self.loss(self.get_testing_data(i), self.model)
                scores.append(score)
        return np.array(scores)

    def tune(self):
        '''
        Tunes the hyperparameters: population size, epochs, cognitive update rate, social update rate, inertia weight.
        '''
        pop_size_vals = [10, 50, 100]
        epoch_vals = [10, 50, 75, 100, 200]
        c1_vals = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]        #Cognitive update rate
        c2_vals = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]        #Social Update Rate
        w_vals = [0.4, 0.7, 1.0]                        #Inertia

        # Tune Population Size
        pop_size_scores = []
        for pop_size in tqdm(pop_size_vals, desc="Tuning Population Size", leave=False):
            pop_size_score = self.train_test(tuning_flag=True, pop_size=pop_size, epochs=self.epochs, c1=self.c1, c2=self.c2, w=self.w)
            pop_size_scores.append(np.mean(pop_size_score))
        pop_size_scores = np.array(pop_size_scores)
        if self.prediction_type == "classification":
            self.pop_size = pop_size_vals[np.argmax(pop_size_scores)]
        else:
            self.pop_size = pop_size_vals[np.argmin(pop_size_scores)]
        print(f"Tuned Population Size: {self.pop_size}")

        # Tune Epochs
        epoch_scores = []
        for epoch in tqdm(epoch_vals, desc="Tuning Epochs", leave=False):
            epoch_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=epoch, c1=self.c1, c2=self.c2, w=self.w)
            epoch_scores.append(np.mean(epoch_score))
        epoch_scores = np.array(epoch_scores)
        if self.prediction_type == "classification":
            self.epochs = epoch_vals[np.argmax(epoch_scores)]
        else:
            self.epochs = epoch_vals[np.argmin(epoch_scores)]
        print(f"Tuned Epochs: {self.epochs}")

        # Tune Cognitive Update Rate(c1)
        c1_scores = []
        for c1 in tqdm(c1_vals, desc="Tuning Cognitive Update Rate", leave=False):
            c1_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=self.epochs, c1=c1, c2=self.c2, w=self.w)
            c1_scores.append(np.mean(c1_score))
        c1_scores = np.array(c1_scores)
        if self.prediction_type == "classification":
            self.c1 = c1_vals[np.argmax(c1_scores)]
        else:
            self.c1 = c1_vals[np.argmin(c1_scores)]
        print(f"Tuned Cognitive Coefficient: {self.c1}")

        # Tune Social Update Rate (c2)
        c2_scores = []
        for c2 in tqdm(c2_vals, desc="Tuning Social Update Rate", leave=False):
            c2_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=self.epochs, c1=self.c1, c2=c2, w=self.w)
            c2_scores.append(np.mean(c2_score))
        c2_scores = np.array(c2_scores)
        if self.prediction_type == "classification":
            self.c2 = c2_vals[np.argmax(c2_scores)]
        else:
            self.c2 = c2_vals[np.argmin(c2_scores)]
        print(f"Tuned Social Coefficient: {self.c2}")

        # Tune Inertia Weight (w)
        w_scores = []
        for w in tqdm(w_vals, desc="Tuning Inertia Weight", leave=False):
            w_score = self.train_test(tuning_flag=True, pop_size=self.pop_size, epochs=self.epochs, c1=self.c1, c2=self.c2, w=w)
            w_scores.append(np.mean(w_score))
        w_scores = np.array(w_scores)
        if self.prediction_type == "classification":
            self.w = w_vals[np.argmax(w_scores)]
        else:
            self.w = w_vals[np.argmin(w_scores)]
        print(f"Tuned Inertia Weight: {self.w}")

        return [self.pop_size, self.epochs, self.c1, self.c2, self.w]

    def sigmoid(self, input_vector):
        '''
        Sigmoid activation function.
        '''
        return 1 / (1 + np.exp(-input_vector))

    def softmax(self, input_vector):
        '''
        Softmax activation function for classification output layer.
        '''
        e_x = np.exp(input_vector - np.max(input_vector))
        return e_x / e_x.sum(axis=0)

    def get_training_data(self, i: int):
        '''
        method needs to take in training data and compile 9 of the 10 folds (not fold I) into an array
        we then want to format the data as follows: each example = (attributes, label)
        I is used to indicate which fold is the hold out fold
        '''
        desired_data = np.concatenate([self.validate_set[j] for j in range(10) if j != i])  #Get all folds other than fold I and compile into its own array
        training_data = [(example[:-1], example[-1]) for example in desired_data]   #Format properly
        return training_data
    def get_testing_data(self, i: int):
        '''
        method needs to take in training data and compile 1 of the 10 folds (fold I) into an array
        Then format the data as follows: each example = (attributes, label)
        i is used to indicate which training set you want returned
        '''
        desired_data = self.validate_set[i]         #Get the test set
        testing_data = [(example[:-1], example[-1]) for example in desired_data] #Format properly
        return testing_data
    def get_tuning_data(self):
        '''
        method needs to take in the tuning set and properly format it
        Then format the data as follows: each example = (attributes, label)
        i is used to indicate which training set you want returned
        '''
        desired_data = self.tune_set  #Get the tuning set
        tuning_data = [(example[:-1], example[-1]) for example in desired_data] #Format properly
        return tuning_data