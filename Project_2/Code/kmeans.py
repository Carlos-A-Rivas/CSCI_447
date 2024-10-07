import os
import numpy as np
import random
import csv
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from dataset import dataset

class kmeans:
    def __init__(self, data: dataset, prediction_type_flag: str, k_c = 1, k_n = 1, sigma = 1.0):
        '''
        - Set a variable equal to the tune and validation sets
        - instantiate self variables
        '''
        self.k_n = k_n
        self.sigma = sigma
        self.k_c = k_c
        self.tune_set = data.tune_set
        self.validate_set = data.validate_set
        self.prediction_type = prediction_type_flag
        self.predictions = []
        self.answers = []
        self.centroids = []


        return
    def plot_loss(self, metrics: list, parameter: str, increment):
        # Extract the number of epochs and loss metrics
        metrics = np.array(metrics)
        epochs = np.arange(1, metrics.shape[0] + 1) * increment  # Assuming epochs start from 1
        loss1 = metrics[:, 0]  # First loss metric
        loss2 = metrics[:, 1]  # Second loss metric

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss1, label='Loss Metric 1', marker='o')
        plt.plot(epochs, loss2, label='Loss Metric 2', marker='o')

        # Adding labels and title
        plt.xlabel(f'{parameter} Value')
        plt.ylabel('Loss')
        plt.title(f'Loss Metrics vs. {parameter} value')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()
        plt.close()
    def tune(self, epochs=15, k_c_increment=1, k_n_increment=1, sigma_increment=1):
        '''
        Tune number of clusters (k_c), number of neighbors (k_n), and sigma (for regression).
        Performance is averaged across all 10 folds. This process repeats for a specified number
        of epochs with the hyperparameters incrementing on each epoch.
        '''
        self.cluster()
        # Initialize the tuning lists to store performance metrics
        k_c_scores = []
        k_n_scores = []
        sigma_scores = []
        
        # Initialize hyperparameters
        self.k_c = k_c_increment
        self.k_n = k_n_increment
        self.sigma = sigma_increment

        # Tune k_c (number of clusters)
        for i in tqdm(range(epochs), desc="Tuning K_c..."):
            self.k_c += k_c_increment
            self.cluster()  # Re-run clustering with updated k_c
            if self.prediction_type == 'regression':
                k_c_scores.append(self.regress(True))
            else:
                k_c_scores.append(self.classify(True))

        self.plot_loss(k_c_scores, 'K_c', k_c_increment)
        
        # Tune k_n (number of neighbors)
        for i in tqdm(range(epochs), desc="Tuning K_n..."):
            self.k_n += k_n_increment
            if self.prediction_type == 'regression':
                k_n_scores.append(self.regress(True))
            else:
                k_n_scores.append(self.classify(True))

        self.plot_loss(k_n_scores, 'K_n', k_n_increment)
        
        # Tune sigma (only for regression)
        if self.prediction_type == 'regression':
            for i in tqdm(range(epochs), desc="Tuning Sigma..."):
                self.sigma += sigma_increment
                sigma_scores.append(self.regress(True))
            self.plot_loss(sigma_scores, 'Sigma', sigma_increment)

        # Select best parameters based on performance
        k_c_scores = np.array(k_c_scores)
        if(self.prediction_type == 'classification'):
            best_k_c_epochs = np.argmax(k_c_scores, axis=0)
        else:
            best_k_c_epochs = np.argmin(k_c_scores,axis=0)
        self.k_c = (round(np.mean(best_k_c_epochs)) + 1) * k_c_increment
        print(f"Tuned k_c: {self.k_c}")

        k_n_scores = np.array(k_n_scores)
        if(self.prediction_type == 'classification'):
            best_k_n_epochs = np.argmax(k_n_scores, axis=0)
        else:
            best_k_n_epochs = np.argmin(k_n_scores,axis=0)
        self.k_n = (round(np.mean(best_k_n_epochs)) + 1) * k_n_increment
        print(f"Tuned k_n: {self.k_n}")

        if self.prediction_type == 'regression':
            sigma_scores = np.array(sigma_scores)
            best_sigma_epochs = np.argmin(sigma_scores, axis=0)
            self.sigma = (round(np.mean(best_sigma_epochs[0] + 1))) * sigma_increment
            print(f"Tuned sigma: {self.sigma}")

        return

   
    def cluster(self):
        centroids_list = []
        
        # Get into correct fold
        for fold_idx in tqdm(range(10), leave=False): 
            model = np.concatenate([self.validate_set[i] for i in range(10) if i != fold_idx]) 
            model[model == 'null'] = np.nan
            model = model.astype(float)

            # Remove rows with NaN values from the model
            model = model[~np.isnan(model).any(axis=1)]
            
            # Initialize centroids
            centroids = model[np.random.choice(model.shape[0], self.k_c, replace=False)]
            
            
            prev_centroids = np.copy(centroids)
            convergence_threshold = 0.05

            max_iterations = 50
            iteration = 0

            while iteration < max_iterations:
                distances = np.linalg.norm(model[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)

                prev_centroids = centroids.copy()

                for i in range(self.k_c):
                    if np.any(labels == i):
                        centroids[i] = np.nanmean(model[labels == i], axis=0)
                    else:
                        centroids[i] = model[np.random.choice(model.shape[0])]

                relative_change = np.abs(centroids - prev_centroids) / (np.abs(prev_centroids) + 1e-10)
                
                if np.all(relative_change < convergence_threshold):
                    break

                iteration += 1
                

            if iteration == max_iterations:
                print("Warning: Maximum iterations reached without convergence.")


  

            # Output final centroids
            centroids_list.append(centroids)

        self.centroids = np.array(centroids_list)
        #print(f"Final Centroids Shape: {self.centroids.shape}")

    
    def classify(self, tuning_flag):
        '''
        classify holdout set repeat for each fold
        '''
        self.cluster()
        predictions = []
        answers = []
        hold_out_fold = self.tune_set
        for fold_idx in tqdm(range(10), leave=False):
            if (tuning_flag == False):
                hold_out_fold = self.validate_set[fold_idx]
            model = self.centroids[fold_idx]
            #print(model.shape)
            #print(hold_out_fold.shape)

            for test_point in hold_out_fold:
                if (test_point[0] != 'null'):
                    true_label = test_point[-1]
                    neighbor_indices = self.get_neighbors(model, test_point, self.k_n)
                    #print(f"Neighbor Indices:\n{neighbor_indices}")
                    neighbor_labels = model[neighbor_indices, -1]
                    #print(f"Neighbor Labels: {neighbor_labels}")
                    label_counts = Counter(neighbor_labels)
                    predicted_label = label_counts.most_common(1)[0][0]

                    predictions.append(predicted_label)
                    answers.append(true_label)

        self.predictions = np.array(predictions)
        self.predictions = np.rint(self.predictions).astype(int).astype(str)
        self.predictions = np.array(self.predictions)
        self.answers = np.array(answers)
        #print(f"Predictions: {self.predictions}")
        #print(f"Answers: {self.answers}")
        Loss_values = self.calculate_loss()
        #print(f"Loss Values: {Loss_values}")
        return Loss_values   
        
    def regress(self, tuning_flag):
        self.cluster()
        predictions = []
        answers = []
        hold_out_fold = self.tune_set
        for fold_idx in tqdm(range(10), leave=False):
            if (tuning_flag == False):
                hold_out_fold = self.validate_set[fold_idx]
            model = self.centroids[fold_idx]
            #print(model.shape)
            #print(hold_out_fold.shape)

            for test_point in hold_out_fold:
                if (test_point[0] != 'null'):
                    true_label = test_point[-1]
                    neighbor_indices = self.get_neighbors(model, test_point, self.k_n)
                    #print(f"Neighbor Indices:\n{neighbor_indices}")
                    nearest_neighbors = model[neighbor_indices]
                    #print(f"Nearest Neighbors: {nearest_neighbors}")
                    neighbor_values = nearest_neighbors[:, -1]

                    distances = np.array([np.linalg.norm(test_point[:-1].astype(float) - neighbor[:-1].astype(float)) for neighbor in nearest_neighbors])
                
                    rbf_weights = np.exp(- (distances ** 2) / (2 * self.sigma ** 2))
                    #print(f"Should be equal to last indice of the nearest neighbors: {nearest_neighbors[:, -1]}")
                    weighted_sum = np.sum(rbf_weights * nearest_neighbors[:, -1].astype(float))
                    weight_total = np.sum(rbf_weights)

                    predicted_value = weighted_sum / weight_total if weight_total != 0 else np.mean(neighbor_values.astype(float))

                    predictions.append(predicted_value)
                    answers.append(true_label)

        self.predictions = np.array(predictions)
        self.predictions = np.rint(self.predictions).astype(int).astype(str)
        self.answers = np.array(answers)
        #print(f"Predictions:{self.predictions}")
        #print(f"Answers: {self.answers}")
        Loss_values = self.calculate_loss()
        #print(f"Loss Values: {Loss_values}")
        return Loss_values
    def euclidean_distance(self, point1: np, point2: np):
        # np.linalg.norm calculates the euclidean distances between two points
        #print(f"Point 1 type: {point1.shape}")
        #print(f"Point 2 type: {point2.shape}")
        return np.linalg.norm(point1 - point2)
    def calculate_loss(self):
            '''
            Classifiction: 0/1 loss, F1 score
            Regression: Mean squared error, Mean absolute

            '''
            loss = []
            if(self.prediction_type == "classification"):
                #print(self.predictions)
                #print(f"Answers: {self.answers}")
                accuracy = np.mean(self.predictions == self.answers)
                loss.append(float(accuracy))

                unique_classes = np.unique(self.answers)
                f1_scores = []
                for cls in unique_classes:
                    true_positives = sum((self.predictions == cls) & (self.answers == cls))
                    predicted_positives = sum(self.predictions == cls)
                    actual_positives = sum(self.answers == cls)

                    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
                    recall = true_positives / actual_positives if actual_positives > 0 else 0

                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0
                    f1_scores.append(f1)

                loss.append(float(np.mean(f1_scores)))

            else:
                mse = np.mean(self.answers.astype(float) - self.predictions.astype(float)) ** 2
                loss.append(float(mse))

                mae = np.mean(np.abs(self.answers.astype(float) - self.predictions.astype(float)))
                loss.append(float(mae))
            return loss
    def get_neighbors(self, model: np, test_point: np, k_n: int):
        '''
        - Feed this function a NxN numpy array where the first dimension is num of examples and the second dimension is num of freatures
        - The second argument is the reference point
        - the third argument is the point that is being referenced for distances
        - The method returns the class/regression value of the k_n nearest neighbors
        '''
        #print(f"Model shape: {model.shape}")
        

        distances = np.zeros(model.shape[0], dtype=float)
        #print(f"Distances Shape: {distances.shape}")
        for i, model_point in enumerate(model):
            # calculate euclidean distance
            # COULD ALWAYS SWAP THIS FUNCTION CALL FOR THE ONE LINER
            if (model_point[-1] != "null"):
                #print(f"test point: {test_point}")
                #print(f"model point: {model_point}")
                distances[i] = self.euclidean_distance(test_point[:-1].astype(float), model_point[:-1].astype(float))
            else:
                distances[i] = float('inf')
        # np.partitions moves the K_n smallest values in an np array to the front of the array. We then slice the array to get the k_n smallest values
        #smallest_distances = np.partition(distances, k_n)[:k_n]
        #print(f"Smallest distances: {smallest_distances}")
        neighbor_indices = np.argsort(distances)[:k_n]
        #print(f"Neighbor Indices:\n{neighbor_indices}")
        #print(type(nearest_neighbors))
        # CURRENTLY RETURNS THE INDICES OF THE NEAREST NEIGHBORS
        return neighbor_indices