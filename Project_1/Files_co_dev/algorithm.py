import numpy as np
from dataset import dataset
import matplotlib as plt


def make_plots(f1_scores, loss_scores, names_list):
    f1_data = [f1_scores[i] for i in range(len(f1_scores))]
    loss_data = [loss_scores[i] for i in range(len(loss_scores))]

    positions = np.arange(len(f1_scores))
    width = 0.4

    plt.figure(figsize=(8, 6))
    plt.boxplot(f1_data, positions=positions, widths=width, patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='blue'),
                whiskerprops=dict(color='blue'), capprops=dict(color='blue'),
                flierprops=dict(markerfacecolor='blue', marker='o'))
    plt.xticks(positions, names_list)
    plt.xlabel('Datasets')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores Across Datasets')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.boxplot(loss_data, positions=positions, widths=width, patch_artist=True,
                boxprops=dict(facecolor='lightcoral'), medianprops=dict(color='red'),
                whiskerprops=dict(color='red'), capprops=dict(color='red'),
                flierprops=dict(markerfacecolor='red', marker='o'))
    plt.xticks(positions, names_list)
    plt.xlabel('Datasets')
    plt.ylabel('0/1 Loss Score')
    plt.title('0/1 Loss Scores Across Datasets')
    plt.tight_layout()
    plt.show()


class algorithm:
    def __init__(self, data_array:dataset, which_data:str):
        # The algorithm takes in a pre-processed dataset object, there shouldn't be much to do with the constructor except maybe a setter for the dataset
        self.data_array = data_array

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

    def train(self):
        # Runs Q(), Runs F().

        # Q() calculates the class prior probabilities, this create be an array 10 * the # of classes in the datset (10 for each validation set).
        self.all_class_counts = []
        self.class_counts = [0] * len(self.labels)
        self.probs = [0] * len(self.labels)
        self.example_count = 0
        self.class_probs = []
        for i in range(10):
            self.rotated_data = self.data_array.partitions[i:] + self.data_array.partitions[:i]
            self.rotated_data = self.rotated_data[:-1]
            self.class_counts = [0] * len(self.labels)
            self.example_count = 0
            for partition in range(9):
                for example in range(len(self.rotated_data[partition])):
                    self.example_count += 1
                    for label in range(len(self.labels)):
                        if (self.rotated_data[partition][example][-1] == self.labels[label]):
                            self.class_counts[label] += 1
            for i, count in enumerate(self.class_counts):
                self.probs[i] = count/(self.example_count)
            self.class_probs.append(self.probs.copy())
            self.all_class_counts.append(self.class_counts.copy())




        # F() calculates the individual attribute probabilities for each class. This creates an array of 10 * # of classes * # of attributes * # of attribute possibilites.
        # NsubCi is the number of examples in each class. This is contained in self.all_class_counts where the dimensions are Model * # of classes. Each location contains the # of examples for a given class
        # self.attribute_value_counts_per_fold dimensions: # of models * list # of attributes * dictionary for attribute values
        d = self.data_array.attribute_count - 1
        # This will store the attribute value counts for each fold (10 sets of dictionaries)
        self.attribute_value_counts_per_fold = []
        # Perform 10-fold cross-validation
        for i in range(10):
            # Initialize a dictionary for each attribute (column) to count occurrences for the current fold
            attribute_value_counts = [{} for _ in range(d)]
            self.rotated_data = self.data_array.partitions[i:] + self.data_array.partitions[:i]
            self.rotated_data = self.rotated_data[:-1]
            # Count attribute value occurrences
            for partition in self.rotated_data:
                for example in partition:
                    for a in range(d): # d already excludes the label indice
                        # If the attribute value hasn't been seen before at this index, initialize its count
                        if example[a] not in attribute_value_counts[a]:
                            attribute_value_counts[a][example[a]] = 0
                        attribute_value_counts[a][example[a]] += 1
            
            # Store the attribute value counts for this fold
            self.attribute_value_counts_per_fold.append(attribute_value_counts)

        '''
        # Print the attribute value counts for each fold to verify
        for fold, counts in enumerate(self.attribute_value_counts_per_fold):
            print(f"Fold {fold+1}:")
            for attribute_index, count_dict in enumerate(counts):
                print(f"  Attribute {attribute_index}: {count_dict}")
        '''


        self.attribute_probs = []
        # Now that data is collected, we need to create the loop that does the math with the values.
        for i in range(10):
            self.attribute_probs.append([])
            for classs in range(len(self.all_class_counts[i])):
                self.attribute_probs[i].append([])
                for attribute in range(d):
                    self.attribute_probs[i][classs].append({})
                    keys = list(self.attribute_value_counts_per_fold[i][attribute].keys())
                    for key in keys:
                        # calculate the probability, and put it into self.attribute_probs[i][classs][attribute].append(probabbility_value)
                        numerator = self.attribute_value_counts_per_fold[i][attribute].get(key)
                        N = self.all_class_counts[i][classs]
                        probability_value = (numerator + 1)/(N + d)
                        self.attribute_probs[i][classs][attribute][key] = probability_value

        print(f"Number of attribute possibilities: {len(self.attribute_value_counts_per_fold[0][0])} Actual Dimension: {len(self.attribute_probs[0][0][0])}")
        print(f"Number of attributes: {len(self.attribute_value_counts_per_fold[0])} Actual Dimension: {len(self.attribute_probs[0][0])}")
        print(f"Number of classes: {len(self.all_class_counts[0])} Actual Dimension: {len(self.attribute_probs[0])}")
        print(f"Number of folds: 10 Actual Dimension: {len(self.attribute_probs)}")

        # Once everything is run we should have trained 10 models for this method call. We will probably want to store each of these models in an array containing the other 2 arrays produced from Q and F, meaning there will be a ~7D array :/
        # We will also want to save each of the models for both validation purposes, and for the event processing fails downstream we won't have to re-train
        
        
        
        return
    
    '''
        d = self.data_array.attribute_count-1
        # This list will end up having dimensions # of models * # of attributes: each indice contains the # of examples that have the attribute
        self.attribute_values = [[] for _ in range(d)]
        self.example_matches = [[0 for _ in range (d)] for _ in range(10)]
        #print(f'SELF.EXAMPLE MATCHES: {self.example_matches}')
        
        # Produces a list of all of the possible attribute values in the dataset
        for partition in range(len(self.data_array.partitions)):
            for example in range(len(self.data_array.partitions[partition])):
                for a, attribute in enumerate(self.data_array.partitions[partition][example][:-1]):
                    if attribute not in self.attribute_values[a]:
                        self.attribute_values[a].append(attribute)
        #print(f"Self.attribute_values: {self.attribute_values}")

        for i in range(10):
            self.rotated_data = self.data_array.partitions[i:] + self.data_array.partitions[:i]
            self.rotated_data = self.rotated_data[:-1]
            for partition in range(9):
                for example in range(len(self.rotated_data[partition])):
                    for attribute in range(d):
                        if (self.rotated_data[partition][example][attribute] in self.attribute_values[attribute]):
                            self.example_matches[i][attribute] += 1

                        
                        for value in self.attribute_values[attribute]:
                            # Every time we enter this if statement we want to count if the attribute value has been repeated for the example
                            if (self.rotated_data[partition][example][attribute] == value):
                                self.example_matches[i][attribute] += 1
                        
            
            for i, count in enumerate(self.class_counts):
                self.probs[i] = count/(self.example_count)
            self.class_probs.append(self.probs.copy())
            
        '''

        
    def validate(self):
        # This method runs the algorithm on the validation folds. We want to make sure we are running the proper model on the proper fold, so if you trained on folds 1-9, you would start the for loop at 10 and decrement (for example).
        # This will produce 10 classified folds, where we will want to save the results from the classification.

        fold_order = [9] + list(range(0, 9))  #sets the order for which partition is our validation set
        self.answers = []
        self.predictions = []
        for i in range(10):  #goes through each model
            test_fold_index = fold_order[i]
            class_prob_index = fold_order[(i + 1) % 10]

        
            test_fold = self.data_array.partitions[test_fold_index]
            attribute_probs = self.attribute_probs[class_prob_index]
            
            self.correct_preditions = 0
            total_predictions = len(test_fold)

            for example in test_fold:
                features = example[:-1]
                true_label = example[-1]
                self.answers.append(true_label)
                max_class_prob = -float('inf')
                predicted_class = None

                for classs in range(len(self.all_class_counts[i])):
                    class_prob = self.class_probs[class_prob_index][classs]

                    for feature_index, feature_value in enumerate(features):
                        '''
                        print(f"i: {i}")
                        print(f"classs: {classs}")
                        print(f"featire index: {feature_index}")
                        print(f"Feature Value: {feature_value}")
                        print(f"Feature Value: {type(feature_value)}")
                        print(f"Attribute probability keys: {type(attribute_probs[i][classs])}")
                        '''
                        feature_prob = attribute_probs[classs][feature_index].get(feature_value)
                        class_prob *= feature_prob


                    if class_prob > max_class_prob:
                        max_class_prob = class_prob
                        predicted_class = self.labels[classs]

                predicted_label = predicted_class
                self.predictions.append(predicted_label)
    def calculate_loss(self):
        # This method will analyze the classification, and determine, TP, TN, FP, FN, F1 loss, 0/1 loss, etc.
        # These results need to be saved
        # We will also want to create a box plot where the x-axis is the dataset we worked with (10 datasets), and y axis is a performance metric (whatever we want). For each dataset there will be 10 datapoints in the box, each point represents the performance of one of the folds.
        self.zero_one_losses = []
        self.f1_scores = []
        fold_order = [9] + list(range(0, 9))
        for fold_index in fold_order:
            test_fold = self.data_array.partitions[fold_index]

            true_positives = {label: 0 for label in self.labels}
            false_positives = {label: 0 for label in self.labels}
            false_negatives = {label: 0 for label in self.labels}
            correct_predictions = 0
            total_predictions = len(test_fold)


            for i in range(total_predictions):
                true_label = self.answers[i]
                predicted_label = self.predictions[i]


                if predicted_label == true_label:
                    correct_predictions += 1

                if predicted_label == true_label:
                    true_positives[true_label] += 1
                else:
                    false_positives[predicted_label] += 1
                    false_negatives[true_label] += 1
                

            loss = 1 - (correct_predictions / total_predictions)
            self.zero_one_losses.append(loss)



            f1_scores_per_class = []
            for label in self.labels:
                precision = true_positives[label] / (true_positives[label] + false_positives[label]) if (true_positives[label] + false_positives[label]) > 0 else 0
                recall = true_positives[label] / (true_positives[label] + false_negatives[label]) if (true_positives[label] + false_negatives[label]) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores_per_class.append(f1)
            
            average_f1 = sum(f1_scores_per_class) / len(self.labels)
            self.f1_scores.append(average_f1)
