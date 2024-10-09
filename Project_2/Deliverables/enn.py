from dataset import dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter

class enn:
    '''
    The all-inclusive class for running the ENN algorithm
    '''
    def __init__(self, data: dataset, prediction_type_flag: str, k_n=1, sigma=1.0, epsilon=1, suppress_plots=True):
        '''
        The constructor initializes self variables.
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
    def plot_loss(self, metrics: list, parameter: str, increment):
        '''
        This function plots the loss performance for each epoch. This allows us to visualize at how many epochs
        performance drops off.
        '''
        # Extract # of epochs and loss metrics
        metrics = np.array(metrics)
        epochs = np.arange(1, metrics.shape[0] + 1) * increment  # Assuming epochs start from 1
        loss1 = metrics[:, 0]  # First loss metric
        loss2 = metrics[:, 1]  # Second loss metric

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
    def tune(self, epochs=15, k_n_increment=1, sigma_increment=1):
        '''
        First step for ENN is to reduce the dataset based on tuned KNN parameters.
        Use default parameters to predict the tune set using each set of 9 partitions as the model.
        Performance should be calculated and averaged across the ENTIRE set of models with the given
        hyperparameter. A hyperparameter is incremented, and predictions are re-run. This process
        repeats until the desired number of epochs are reached.
        '''
        # Reduce dataset
        self.reduced_models = self.reduce_dataset(self.validate_set)

        k_n_scores = []
        sigma_scores = []
        self.k_n = k_n_increment
        self.sigma = sigma_increment

        # Tuning K_n
        for i in tqdm(range(epochs), desc="Tuning K_n..."):
            self.k_n += k_n_increment
            if (self.prediction_type == 'regression'):
                k_n_scores.append(self.regress(True))
            else:
                k_n_scores.append(self.classify(True))
        if (self.suppress_plots == False):
            self.plot_loss(k_n_scores, 'K_n', k_n_increment)
            
        # Tuning Sigma
        if (self.prediction_type == 'regression'):    
            for i in tqdm(range(epochs), desc="Tuning sigma..."):
                self.sigma += sigma_increment
                sigma_scores.append(self.regress(True))
            if (self.suppress_plots == False):
                self.plot_loss(sigma_scores, 'Sigma', sigma_increment)

        # Choosing which epoch had the best performance
        k_n_scores = np.array(k_n_scores)
        if (self.prediction_type == 'regression'):
            best_k_n_epochs = np.argmin(k_n_scores, axis=0)
        else:
            best_k_n_epochs = np.argmax(k_n_scores, axis=0)
        self.k_n = (round(np.mean(best_k_n_epochs+1))) * k_n_increment
        print(f"Tuned k_n: {self.k_n}")
        if (self.prediction_type == 'regression'):
            sigma_scores = np.array(sigma_scores)
            best_sigma_epochs = np.argmin(sigma_scores, axis=0)
            self.sigma = (round(np.mean(best_sigma_epochs[0] + 1))) * sigma_increment
            print(f"Tuned sigma: {self.sigma}")
    def reduce_dataset(self, initial_set: np, epsilon = 0.05):
        '''
        This function reduces each 9 fold dataset by predicting each point in itself, and
        if the prediction is incorrect the point gets removed.
        '''
        reduced_models = []
        padded_folds = []

        # Creates a model using 9 folds
        for fold_idx in tqdm(range(10), desc="reducing dataset...", leave=False):
            removal_indices = []
            model = np.concatenate([initial_set[i] for i in range(10) if i != fold_idx])
            for test_point_idx, test_point in enumerate(model):
                if (test_point[0] != 'null'):
                    # Create a new array excluding the test point
                    self_classify_model = np.delete(model, test_point_idx, axis=0)
                    true_label = test_point[-1]
                    neighbor_indices = self.get_neighbors(self_classify_model, test_point, self.k_n)
                    # classify a point in the model
                    if (self.prediction_type == "classification"):
                        neighbor_labels = self_classify_model[neighbor_indices, -1]
                        label_counts = Counter(neighbor_labels)
                        predicted_label = label_counts.most_common(1)[0][0]
                        if (predicted_label != true_label):
                            removal_indices.append(test_point_idx)
                    # regress a point in the model
                    else:
                        nearest_neighbors = self_classify_model[neighbor_indices]
                        # RBF Calculation
                        neighbor_values = nearest_neighbors[:, -1]
                        distances = np.array([np.linalg.norm(test_point[:-1].astype(float) - neighbor[:-1].astype(float)) for neighbor in nearest_neighbors])
                        rbf_weights = np.exp(- (distances ** 2) / (2 * self.sigma ** 2))
                        weighted_sum = np.sum(rbf_weights * nearest_neighbors[:, -1].astype(float))
                        weight_total = np.sum(rbf_weights)
                        predicted_value = weighted_sum / weight_total if weight_total != 0 else np.mean(neighbor_values.astype(float))
                        if ((abs(float(predicted_value) - float(true_label)) <= epsilon * float(predicted_value)) == False):
                            removal_indices.append(test_point_idx)
            reduced_models.append(np.delete(model, removal_indices, axis=0))

        # Pad each array to have the same number of rows (max_rows)
        max_rows = max(fold.shape[0] for fold in reduced_models)
        for fold in reduced_models:
            pad_width = max_rows - fold.shape[0]
            padded_fold = np.pad(fold, ((0, pad_width), (0, 0)), mode='constant', constant_values='null')
            padded_folds.append(padded_fold)
        padded_reduced_models = np.stack(padded_folds)
        print(padded_reduced_models.shape)
        return padded_reduced_models
    def classify(self, tuning_flag=False):
        '''
        There are two modes for this method. The first is regular classify, where the method classifies the holdout set
        and repeats for each fold. The other mode is tune, where instead of classifying the holdout set the method
        classifies the tune set.
        '''
        Loss_values = np.zeros((10, 2))
        predictions = []
        answers = []
        hold_out_fold = self.tune_set

        # Creates a model using 9 folds
        for fold_idx in tqdm(range(10), leave=False):
            if (tuning_flag == False):
                hold_out_fold = self.validate_set[fold_idx]
            model = self.reduced_models[fold_idx]
            # for each data point, get the nearest neighbors, determine the predicted class
            for test_point in hold_out_fold:
                if (test_point[0] != 'null'):
                    true_label = test_point[-1]
                    neighbor_indices = self.get_neighbors(model, test_point, self.k_n)
                    neighbor_labels = model[neighbor_indices, -1]
                    label_counts = Counter(neighbor_labels)
                    predicted_label = label_counts.most_common(1)[0][0]
                    # Compile predictions and true values
                    predictions.append(float(predicted_label))
                    answers.append(true_label)

            # Format the predictions and true values
            self.predictions = np.array(predictions)
            self.predictions = np.rint(self.predictions).astype(int).astype(str)
            self.answers = np.array(answers).astype(float)
            self.answers = np.rint(self.answers).astype(int).astype(str)
            # Calculate loss for the current fold and store it
            Loss_values[fold_idx] = self.calculate_loss()
            predictions = []
            answers = []

        # If we are tuning we return average loss across the folds, otherwise we return the loss for each fold
        if tuning_flag:
            average_loss = np.mean(Loss_values, axis=0)
            return average_loss  
        else:
            print(f"Loss: {Loss_values}")
            return Loss_values  
    def regress(self, tuning_flag=False):
        '''
        There are two modes for this method. The first is regular regression, where the method regresses the holdout set
        and repeats for each fold. The other mode is tune, where instead of regressing the holdout set the method
        regresses the tune set.
        '''
        Loss_values = np.zeros((10, 2))  
        predictions = []
        answers = []
        hold_out_fold = self.tune_set

        # iterate through each fold
        for fold_idx in tqdm(range(10), leave=False):
            if (tuning_flag == False):
                hold_out_fold = self.validate_set[fold_idx]
            model = self.reduced_models[fold_idx]
            # for each data point, get the nearest neighbors, determine the predicted value
            for test_point in hold_out_fold:
                if (test_point[0] != 'null'):
                    true_label = test_point[-1]
                    neighbor_indices = self.get_neighbors(model, test_point, self.k_n)
                    nearest_neighbors = model[neighbor_indices]
                    nearest_neighbors = nearest_neighbors[~np.any(nearest_neighbors == 'null', axis=1)]
                    # RBF Calculation
                    neighbor_values = nearest_neighbors[:, -1]
                    distances = np.array([np.linalg.norm(test_point[:-1].astype(float) - neighbor[:-1].astype(float)) for neighbor in nearest_neighbors if neighbor[0] != 'null'])
                    rbf_weights = np.exp(- (distances ** 2) / (2 * self.sigma ** 2))
                    weighted_sum = np.sum(rbf_weights * nearest_neighbors[:, -1].astype(float))
                    weight_total = np.sum(rbf_weights)
                    predicted_value = weighted_sum / weight_total if weight_total != 0 else np.mean(neighbor_values.astype(float))
                    # Compile predictions and true values
                    predictions.append(predicted_value)
                    answers.append(true_label)
            # Format the predictions and true values
            self.predictions = np.array(predictions)
            self.answers = np.array(answers)
            # Calculate loss for the current fold and store it
            Loss_values[fold_idx] = self.calculate_loss()

        # If we are tuning we return average loss across the folds, otherwise we return the loss for each fold
        if tuning_flag:
            average_loss = np.mean(Loss_values, axis=0)
            return average_loss  
        else:
            return Loss_values
    def calculate_loss(self):
            '''
            Classifiction: 0/1 loss, F1 score
            Regression: Mean squared error, Mean absolute
            Calculates the loss for the repsective prediction type.
            '''
            loss = []
            # Classification loss calculation
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
            # Regression loss calculation
            else:
                mse = np.mean(self.answers.astype(float) - self.predictions.astype(float)) ** 2
                loss.append(float(mse))

                mae = np.mean(np.abs(self.answers.astype(float) - self.predictions.astype(float)))
                loss.append(float(mae))
            return loss
    def euclidean_distance(self, point1: np, point2: np):
        '''
        np.linalg.norm calculates the euclidean distances between two points
        '''
        return np.linalg.norm(point1 - point2)
    def get_neighbors(self, model: np, test_point: np, k_n: int):
        '''
        - Feed this function a NxN numpy array where the first dimension is num of examples and the second dimension is num of freatures
        - The second argument is the reference point
        - the third argument is the point that is being referenced for how many neighbors to consider
        - The method returns the class/regression value of the k_n nearest neighbors
        '''
        distances = np.zeros((model.shape[0]), dtype=float)
        for i, model_point in enumerate(model):
            # calculate euclidean distance
            if (model_point[-1] != "null"):
                distances[i] = self.euclidean_distance(test_point[:-1].astype(float), model_point[:-1].astype(float))
            else:
                distances[i] = float('inf')
        # np.partitions moves the K_n smallest values in an np array to the front of the array. We then slice the array to get the k_n smallest values
        neighbor_indices = np.argsort(distances)[:k_n]
        nearest_neighbors = model[neighbor_indices]
        return neighbor_indices