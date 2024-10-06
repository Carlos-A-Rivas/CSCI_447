import os
import numpy as np
import random
import csv
from tqdm import tqdm
from collections import Counter

class knn:
    def __init__(self, data: dataset, prediction_type_flag: str):
        '''
        - Set a variable equal to the tune and validation sets
        - instantiate self variables
        '''
        self.k_n = 6
        self.sigma = 1.0
        self.tune_set = data.tune_set
        self.validate_set = data.validate_set
        self.prediction_type = prediction_type_flag
        self.predictions = []
        self.answers = []
        return
    def tune(self, prediction_type_flag: str, epochs=50, k_n=1, sigma=1):
        '''
        Use default parameters to predict the tune set using each set of 9 partitions as the model.
        Performance should be calculated and averaged across the ENTIRE set of models with the given
        hyperparameter. A hyperparameter is incremented, and predictions is re-run. This process
        repeats until the desired number of epochs are reached.
        '''
        return
    def classify(self):
        '''
        classify holdout set repeat for each fold
        '''
        for fold_idx in tqdm(range(10)):
            hold_out_fold = self.validate_set[fold_idx]
            model = np.concatenate([self.validate_set[i] for i in range(10) if i != fold_idx])
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

                    self.predictions.append(predicted_label)
                    self.answers.append(true_label)

        self.predictions = np.array(self.predictions)
        self.answers = np.array(self.answers)
        Loss_values = self.calculate_loss()
        print(f"Loss Values: {Loss_values}")
        return Loss_values
    
    
    def regress(self):
        '''
        regress each hold out set repeat for each fold
        '''
        for fold_idx in tqdm(range(10)):
            hold_out_fold = self.validate_set[fold_idx]
            model = np.concatenate([self.validate_set[i] for i in range(10) if i != fold_idx])
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

                    self.predictions.append(predicted_value)
                    self.answers.append(true_label)

        self.predictions = np.array(self.predictions)
        self.answers = np.array(self.answers)
        Loss_values = self.calculate_loss()
        print(f"Loss Values: {Loss_values}")
    

    # NEEDS HEAVY EDITING
    def calculate_loss(self):
            '''
            Classifiction: 0/1 loss, F1 score
            Regression: Mean squared error, Mean absolute

            '''
            loss = {}
            if(self.prediction_type == "classification"):
                accuracy = np.mean(self.predictions == self.answers)
                loss["0/1 Loss"] = accuracy

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

                loss["F1 Score"] = np.mean(f1_scores)

            else:
                mse = np.mean(self.answers.astype(float) - self.predictions.astype(float)) ** 2
                loss["MSE"] = mse

                mae = np.mean(np.abs(self.answers.astype(float) - self.predictions.astype(float)))
                loss["MAE"] = mae
            return loss
    



    def euclidean_distance(self, point1: np, point2: np):
        # np.linalg.norm calculates the euclidean distances between two points
        #print(f"Point 1 type: {point1.shape}")
        #print(f"Point 2 type: {point2.shape}")
        return np.linalg.norm(point1 - point2)
    def get_neighbors(self, model: np, test_point: np, k_n: int):
        '''
        - Feed this function a NxN numpy array where the first dimension is num of examples and the second dimension is num of freatures
        - The second argument is the reference point
        - the third argument is the point that is being referenced for distances
        - The method returns the class/regression value of the k_n nearest neighbors
        '''
        #print(f"Model shape: {model.shape}")
        distances = np.zeros((model.shape[0]), dtype=float)
        #print(f"Distances Shape: {distances.shape}")
        for i, model_point in enumerate(model):
            # calculate euclidean distance
            # COULD ALWAYS SWAP THIS FUNCTION CALL FOR THE ONE LINER
            if (model_point[0] != "null"):
                #print(f"test point: {test_point}")
                #print(f"model point: {model_point}")
                distances[i] = self.euclidean_distance(test_point[:-1].astype(float), model_point[:-1].astype(float))
            else:
                distances[i] = 10000000
        # np.partitions moves the K_n smallest values in an np array to the front of the array. We then slice the array to get the k_n smallest values
        smallest_distances = np.partition(distances, k_n)[:k_n]
        #print(f"Smallest distances: {smallest_distances}")
        neighbor_indices = np.where(np.isin(distances, smallest_distances))[0]
        #print(f"Neighbor Indices:\n{neighbor_indices}")
        nearest_neighbors = model[neighbor_indices]
        #print(type(nearest_neighbors))
        # CURRENTLY RETURNS THE INDICES OF THE NEAREST NEIGHBORS
        return neighbor_indices