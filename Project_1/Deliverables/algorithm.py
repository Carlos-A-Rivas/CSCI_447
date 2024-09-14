from dataset import dataset
from collections import defaultdict
import numpy as np
import csv

class algorithm:
    def __init__(self, data_array:dataset, which_data:str):
        # The algorithm takes in a pre-processed dataset object, there shouldn't be much to do with the constructor except maybe a setter for the dataset
        self.data_array = data_array.partitions
        self.feature_probs = {}
        self.class_priors = {}
        self.predictions = []
        self.answers = []
        self.all_f1_scores_per_class = []

        # Determines label type
        cancer_labels = ['2','4']
        glass_labels = ['1','2','3','4','5','6','7']
        iris_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        soybean_labels = ['D1','D2','D3','D4']
        votes_labels = ['republican','democrat']
        if (which_data == 'cancer'):
            self.labels = cancer_labels
        elif (which_data == 'glass'):
            self.labels = glass_labels
        elif (which_data == 'iris'):
            self.labels = iris_labels
        elif (which_data == 'soybean'):
            self.labels = soybean_labels
        elif (which_data == 'votes'):
            self.labels = votes_labels
        
        # Stores the class labels in an array where the first indice is the partition and the second indice stores class labels for each example
        self.labels_2d = []
        for fold_i in range(len(self.data_array)):
            self.labels_2d.append([])
            for example_i in range(len(self.data_array[fold_i])):
                self.labels_2d[fold_i].append(self.data_array[fold_i][example_i][-1])

    def train_predict(self):
        '''
        Trains an individual model, then tries to classify the remaining fold. Repeats 10 times to produce 10 separate models.
        '''
        num_folds = len(self.data_array)

        # Loop through each fold
        for fold_idx in range(num_folds):
            # call the get_train_test method to store training data, training labels, test data, and test labels
            train_data, train_labels, test_data, test_labels = self._get_train_test(fold_idx)
        
            # Store each class label in the training data within a set
            unique_classes = set(train_labels)
            
            # Instantiate default dictionaries to store class frequency, feature frequency, and the total number of features
            class_counts = defaultdict(int)
            feature_counts = defaultdict(lambda: defaultdict(int))   #creates a nested dictionary to store feature frequencies by class
            feature_totals = defaultdict(int)
            
            num_examples = len(train_labels)  #store the total number of examples         
            for example, label in zip(train_data, train_labels):
                class_counts[label] += 1  # Increment the count for the class
                for feature_idx, feature_value in enumerate(example):
                    feature_counts[label][feature_idx, feature_value] += 1  # Increment the number of instances of a feature within a given class
                    feature_totals[label] += 1 #Increment the count for the feature

            # Calculate the class priors
            self.class_priors[fold_idx] = {}
            # Iterate through each class
            for cls in unique_classes:
                # Calculate the prior probability of each class
                self.class_priors[fold_idx][cls] = (class_counts[cls]) / (num_examples)

            # Calculate feature probabilities
            self.feature_probs[fold_idx] = defaultdict(lambda: defaultdict(float))  #create a nested dictionary to storefeaturee probbilities for each fold 
            for label in unique_classes: #iterate throughg each class
                for feature, count in feature_counts[label].items():  #iterate through the feature counts for each clas
                    self.feature_probs[fold_idx][label][feature] = (count + 1) / (feature_totals[label] + len(feature_counts[label])) #calculate the feature probabilities and store them in the nested dictionary
                    #Self.feature_probs stores the probabilities where the outer dictionary is the fold, the middle dictionary is the class, and the inner dictionary is the feature probabilities


            # Use the calculated prior probabilities and feature probabilities to classify the test data.
            fold_predictions = [] 
            for example in test_data:  #iteraate through each example in the test data
                predicted_label = self.predict(example, fold_idx) #call the predict method to classify an example for a specified fold
                fold_predictions.append(predicted_label) #store the predicted class label for the fold
            self.predictions.append(fold_predictions) # store the predicted class for each fold
        self.class_counts = class_counts
        self.train_data = train_data
        self.feature_counts = feature_counts
        

    def predict(self, example, fold_idx):
        """
        Predict the label of a single example using the trained model for a specific fold.
        """
        class_probs = {}
        train_data, train_labels, test_data, test_labels = self._get_train_test(fold_idx)  # call _get_train_test method to reinstantiate train_labels
        unique_classes = set(train_labels) #create a set of the class labels in the training data
        # Compute the probbility of an example belonging to each class
        for label in unique_classes:
            #iterate through each fold
            if label in self.class_priors[fold_idx]:
                prob = self.class_priors[fold_idx][label] #get the probability of a class in the fold
            else:
                # If the class didn't show up in training, apply a very small probability for it
                prob = 1 / (len(self.class_priors[fold_idx]) + 1)
            # iterate through each feature in an example
            for feature_idx, feature_value in enumerate(example):
                # If a feature has an associated probability for the class, mutiply the probablity by the prior probability
                if (feature_idx, feature_value) in self.feature_probs[fold_idx][label]:
                    prob *= self.feature_probs[fold_idx][label][(feature_idx, feature_value)]
                else:
                    # If the feature didn't show up in training, apply a very small probabilty for it
                    prob *= 1 / (len(self.feature_probs[fold_idx][label]) + 1)
            class_probs[label] = prob
            #store the probability that the example belongs to each class
        # Return the most probable class
        if class_probs:
            return max(class_probs, key=class_probs.get)
        else:
            #If class_probs calculation fails, return None
            return None  # or a default class, if you prefer

    def _get_train_test(self, fold_idx):
        """
        Helper method to split data into training and testing sets based on fold_idx.
        """
        test_data = self.data_array[fold_idx] #pull outthe test fold and store it
        test_labels = self.labels_2d[fold_idx] #pull out the test labels and store them

        # Combine all other folds for training
        train_data = []
        train_labels = []
        for i in range(len(self.data_array)): #iterate through each fold
            if i != fold_idx:         #ignore ethe test fold
                train_data.extend(self.data_array[i]) #store training data as an array
                train_labels.extend(self.labels_2d[i]) #store training labels in an array

        return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)  #Convert to numpy arrays and return all four arrays

    def calculate_loss(self):
        """ 
        This method will analyze the classification, and determine, TP, TN, FP, FN, F1 loss, 0/1 loss, etc.
        These results need to be saved
        We will also want to create a box plot where the x-axis is the dataset we worked with (10 datasets), and y axis is a performance metric (whatever we want). For each dataset there will be 10 datapoints in the box, each point represents the performance of one of the folds.
        """
        self.zero_one_losses = []  
        self.f1_scores = []
        for fold_index in range(10): #iterate through all the folds
            test_fold = self.data_array[fold_index] #pull out the test fold
            #Create dictionaries to store TP, FP, and FN
            true_positives = {label: 0 for label in self.labels}      
            false_positives = {label: 0 for label in self.labels}
            false_negatives = {label: 0 for label in self.labels}
            correct_predictions = 0
            total_predictions = len(test_fold)  # Save the number of examples in the test set
            #Iterate through each prediction
            for i in range(total_predictions):
                true_label = self.data_array[fold_index][i][-1] #store the correct label for the example
                predicted_label = self.predictions[fold_index][i] #Store the predicted label for the example
                #Calculate TP, FP, and FN
                if predicted_label == true_label:
                    #If the predicted label matches the true label, increment correct predictions and true positives
                    correct_predictions += 1
                    true_positives[true_label] += 1
                else:
                    #If the predicted label doesn't match the true label, increment false negatives and false positives
                    if predicted_label in self.labels: #check if the predicted label is valid before incrementing false positives
                        false_positives[predicted_label] += 1 
                    false_negatives[true_label] += 1



            #Calculate 0/1 loss
            loss = 1 - (correct_predictions / total_predictions)
            self.zero_one_losses.append(loss)


            #Calculate F1 scores
            f1_scores_per_class = []
            for label in self.labels:  # Iterate through each class
                # For each class calculate precision, p = TP / (TP +FP)
                precision = (true_positives[label] / (true_positives[label] + false_positives[label])
                        if (true_positives[label] + false_positives[label]) > 0 else 0)
                # For each class calculate recall, r = TP / (TP +FN)
                recall = (true_positives[label] / (true_positives[label] + false_negatives[label])
                        if (true_positives[label] + false_negatives[label]) > 0 else 0)
                # for each class calculate F1 score, f1 = 2 x (p x r)/(p + r)
                f1 = ((2 * (precision * recall)) / (precision + recall)) if (precision + recall) > 0 else 0
                f1_scores_per_class.append(f1)
            self.all_f1_scores_per_class.append(f1_scores_per_class)
            #Average the class f1 scores to get a full model f1 score
            average_f1 = sum(f1_scores_per_class) / len(self.labels)
            self.f1_scores.append(average_f1)
    

    def save_fold_data_to_csv(self, fold_indice, class_prob_file, attribute_prob_file):
        '''
        Saves the weights from a trained model into two separate CSV files. One file contains the class probabilities, the other contains the attribute probabilities
        '''
        with open(class_prob_file, mode='w', newline='') as class_file:
            class_writer = csv.writer(class_file)
            class_writer.writerow(["Fold Index", "Class", "Probability"])
            if fold_indice in self.class_priors:
                for class_name, probability in self.class_priors[fold_indice].items():
                    class_writer.writerow([fold_indice, class_name, probability])
        with open(attribute_prob_file, mode='w', newline='') as attribute_file:
            attribute_writer = csv.writer(attribute_file)
            attribute_writer.writerow(["Fold Index", "Class", "(Attribute value, Attribute)", "Probability"])
            if fold_indice in self.feature_probs:
                for class_name, attributes in self.feature_probs[fold_indice].items():
                    for attr_tuple, probability in attributes.items():
                        attribute_writer.writerow([fold_indice, class_name, attr_tuple, probability])


