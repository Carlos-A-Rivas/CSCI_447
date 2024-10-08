from dataset import dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter

class enn:
    def __init__(self, data: dataset, prediction_type_flag: str, k_n=1, sigma=1.0, epsilon=1, suppress_plots=True):
        '''
        - Set a variable equal to the tune and validation sets
        - instantiate self variables
        '''
        self.suppress_plots = suppress_plots
        self.k_n = k_n
        self.sigma = sigma
        self.epslion = epsilon
        self.tune_set = data.tune_set
        self.validate_set = data.validate_set
        self.prediction_type = prediction_type_flag
        self.predictions = []
        self.answers = []
        self.reduced_models = []
        
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
    def tune(self, epochs=15, k_n_increment=1, sigma_increment=1):
        '''
        for fold_idx, fold in enumerate(self.validate_set):
            print(f"Number of examples in OG data fold {fold_idx+1}: {len(fold)}")
        for fold_idx, fold in enumerate(reduced_models):
            print(f"Number of examples in reduced data fold {fold_idx+1}: {len(fold)}")
        '''
        self.reduced_models = self.reduce_dataset(self.validate_set)
        k_n_scores = []
        sigma_scores = []
        self.k_n = k_n_increment
        self.sigma = sigma_increment
        for i in tqdm(range(epochs), desc="Tuning K_n..."):
            self.k_n += k_n_increment
            if (self.prediction_type == 'regression'):
                k_n_scores.append(self.regress(True))
            else:
                k_n_scores.append(self.classify(True))
        if (self.suppress_plots == False):
            self.plot_loss(k_n_scores, 'K_n', k_n_increment)
            

        if (self.prediction_type == 'regression'):    
            for i in tqdm(range(epochs), desc="Tuning sigma..."):
                self.sigma += sigma_increment
                sigma_scores.append(self.regress(True))
            if (self.suppress_plots == False):
                self.plot_loss(sigma_scores, 'Sigma', sigma_increment)

        k_n_scores = np.array(k_n_scores)
        if (self.prediction_type == 'regression'):
            best_k_n_epochs = np.argmin(k_n_scores, axis=0)
        else:
            best_k_n_epochs = np.argmax(k_n_scores, axis=0)
        self.k_n = (round(np.mean(best_k_n_epochs)) + 1) * k_n_increment
        print(f"Tuned k_n: {self.k_n}")
        if (self.prediction_type == 'regression'):
            # CURRENTLY IS ONLY USING MSE TO TUNE SIGMA
            sigma_scores = np.array(sigma_scores)
            best_sigma_epochs = np.argmin(sigma_scores, axis=0)
            self.sigma = (round(np.mean(best_sigma_epochs[0] + 1))) * sigma_increment
            print(f"Tuned sigma: {self.sigma}")
        return  
    def reduce_dataset(self, initial_set: np, epsilon = 0.05):
        reduced_models = []
        padded_folds = []

        for fold_idx in tqdm(range(10), desc="reducing dataset...", leave=False):
            removal_indices = []
            model = np.concatenate([initial_set[i] for i in range(10) if i != fold_idx])
            #print(f"Fold {fold_idx} Model Shape: {model.shape}")
            #print(hold_out_fold.shape)
            for test_point_idx, test_point in enumerate(model):
                if (test_point[0] != 'null'):
                    # Create a new array excluding the test point
                    self_classify_model = np.delete(model, test_point_idx, axis=0)
                    true_label = test_point[-1]
                    neighbor_indices = self.get_neighbors(self_classify_model, test_point, self.k_n)
                    #print(f"Neighbor Indices:\n{neighbor_indices}")
                    if (self.prediction_type == "classification"):
                        neighbor_labels = self_classify_model[neighbor_indices, -1]
                        #print(f"Neighbor Labels: {neighbor_labels}")
                        label_counts = Counter(neighbor_labels)
                        predicted_label = label_counts.most_common(1)[0][0]
                        if (predicted_label != true_label):
                            removal_indices.append(test_point_idx)
                    else:
                        nearest_neighbors = self_classify_model[neighbor_indices]
                        #print(f"Nearest Neighbors: {nearest_neighbors}")
                        neighbor_values = nearest_neighbors[:, -1]
                        distances = np.array([np.linalg.norm(test_point[:-1].astype(float) - neighbor[:-1].astype(float)) for neighbor in nearest_neighbors])
                        rbf_weights = np.exp(- (distances ** 2) / (2 * self.sigma ** 2))
                        #print(f"Should be equal to last indice of the nearest neighbors: {nearest_neighbors[:, -1]}")
                        weighted_sum = np.sum(rbf_weights * nearest_neighbors[:, -1].astype(float))
                        weight_total = np.sum(rbf_weights)

                        predicted_value = weighted_sum / weight_total if weight_total != 0 else np.mean(neighbor_values.astype(float))
                        if ((abs(float(predicted_value) - float(true_label)) <= epsilon * float(predicted_value)) == False):
                            removal_indices.append(test_point_idx)
            #print(f"Fold {fold_idx+1} Shape: {np.delete(model, removal_indices, axis=0).shape}")
            reduced_models.append(np.delete(model, removal_indices, axis=0))
        
        #print(reduced_models[0])
        max_rows = max(fold.shape[0] for fold in reduced_models)
        # Pad each array to have the same number of rows (max_rows)
        for fold in reduced_models:
            pad_width = max_rows - fold.shape[0]
            padded_fold = np.pad(fold, ((0, pad_width), (0, 0)), mode='constant', constant_values='null')
            padded_folds.append(padded_fold)
        # Stack the padded arrays into a 3D array
        padded_reduced_models = np.stack(padded_folds)
        print(padded_reduced_models.shape)
        return padded_reduced_models
        #else: # regression


        #return
    def classify(self, tuning_flag=False):
        '''
        classify holdout set repeat for each fold
        '''
        Loss_values = np.zeros((10, 2))
        predictions = []
        answers = []
        hold_out_fold = self.tune_set
        for fold_idx in tqdm(range(10), leave=False):
            if (tuning_flag == False):
                hold_out_fold = self.validate_set[fold_idx]
            model = self.reduced_models[fold_idx]
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
            self.answers = np.array(answers).astype(float)
            self.answers = np.rint(self.answers).astype(int).astype(str)
            #print(f"Predictions: {self.predictions}")
            #print(f"Answers: {self.answers}")
            Loss_values[fold_idx] = self.calculate_loss()
            predictions = []
            answers = []

        if tuning_flag:
            average_loss = np.mean(Loss_values, axis=0)
            return average_loss  
        else:
            print(f"Loss: {Loss_values}")
            return Loss_values  
    def regress(self, tuning_flag=False):
        '''
        regress each hold out set repeat for each fold
        '''
        Loss_values = np.zeros((10, 2))  
        predictions = []
        answers = []
        hold_out_fold = self.tune_set
        for fold_idx in tqdm(range(10), leave=False):
            if (tuning_flag == False):
                hold_out_fold = self.validate_set[fold_idx]
            model = self.reduced_models[fold_idx]
            #print(model.shape)
            #print(hold_out_fold.shape)

            for test_point in hold_out_fold:
                if (test_point[0] != 'null'):
                    true_label = test_point[-1]
                    neighbor_indices = self.get_neighbors(model, test_point, self.k_n)
                    #print(f"Neighbor Indices:\n{neighbor_indices}")
                    nearest_neighbors = model[neighbor_indices]
                    nearest_neighbors = nearest_neighbors[~np.any(nearest_neighbors == 'null', axis=1)]
                    #print(f"Nearest Neighbors: {nearest_neighbors}")
                    neighbor_values = nearest_neighbors[:, -1]

                    distances = np.array([np.linalg.norm(test_point[:-1].astype(float) - neighbor[:-1].astype(float)) for neighbor in nearest_neighbors if neighbor[0] != 'null'])
                
                    rbf_weights = np.exp(- (distances ** 2) / (2 * self.sigma ** 2))
                    #print(f"Should be equal to last indice of the nearest neighbors: {nearest_neighbors[:, -1]}")
                    weighted_sum = np.sum(rbf_weights * nearest_neighbors[:, -1].astype(float))
                    weight_total = np.sum(rbf_weights)

                    predicted_value = weighted_sum / weight_total if weight_total != 0 else np.mean(neighbor_values.astype(float))

                    predictions.append(predicted_value)
                    answers.append(true_label)
                    
            self.predictions = np.array(predictions)
            self.predictions = np.rint(self.predictions).astype(int).astype(str)
            self.answers = np.array(answers).astype(float)
            self.answers = np.rint(self.answers).astype(int).astype(str)
            #print(f"Predictions: {self.predictions}")
            #print(f"Answers: {self.answers}")
            Loss_values[fold_idx] = self.calculate_loss()
            predictions = []
            answers = []

        if tuning_flag:
            average_loss = np.mean(Loss_values, axis=0)
            return average_loss  
        else:
            return Loss_values
    def calculate_loss(self):
            '''
            Classifiction: 0/1 loss, F1 score
            Regression: Mean squared error, Mean absolute

            '''
            loss = []
            if(self.prediction_type == "classification"):
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
        nearest_neighbors = model[neighbor_indices]
        #print(type(nearest_neighbors))
        # CURRENTLY RETURNS THE INDICES OF THE NEAREST NEIGHBORS
        return neighbor_indices