##########################################################    
## MY VALIDATE
##########################################################
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


                for classs in range(len(self.all_class_counts[class_prob_index])):
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
                        #print(feature_value, type(feature_prob))
                        #print(type(feature_prob))
                        if(feature_prob is None):
                            class_prob *= 1
                            #print(type(feature_prob))
                        else:
                            class_prob *= feature_prob


                    if class_prob > max_class_prob:
                        max_class_prob = class_prob
                        predicted_class = self.labels[classs]

                predicted_label = predicted_class
                self.predictions.append(predicted_label)



##########################################################    
## GPT CREATED VALIDATE VALIDATE
##########################################################


##########################################################    
## MY TRAINER
##########################################################
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
                        #print(f"Example Array: {example}")
                        #print(f"Example Array Length: {len(example)}")
                        #print(f"attribute_values_counts Array: {attribute_value_counts}")
                        #print(f"attribute_values_counts Lenth: {len(attribute_value_counts)}")
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







#################################################################
### GPT EDITED TRAINER
#################################################################
    def train(self):
        # Runs Q(), Runs F().

        # Q() calculates the class prior probabilities.
        self.all_class_counts = []
        self.class_counts = [0] * len(self.labels)
        self.probs = [0] * len(self.labels)
        self.example_count = 0
        self.class_probs = []

        # Perform 10-fold cross-validation
        for i in range(10):
            # Rotate the data for the current fold
            self.rotated_data = self.data_array.partitions[i:] + self.data_array.partitions[:i]
            self.rotated_data = self.rotated_data[:-1]
            self.class_counts = [0] * len(self.labels)  # Reset class counts for the new fold
            self.example_count = 0  # Reset example count for the new fold

            # Count examples for each class
            for partition in range(9):  # Use 9 partitions for training
                for example in range(len(self.rotated_data[partition])):
                    self.example_count += 1  # Total examples count
                    for label in range(len(self.labels)):
                        if self.rotated_data[partition][example][-1] == self.labels[label]:
                            self.class_counts[label] += 1  # Count examples per class

            # Calculate class prior probabilities
            for i, count in enumerate(self.class_counts):
                self.probs[i] = count / self.example_count  # P(Class)
            self.class_probs.append(self.probs.copy())  # Store the probabilities for this fold
            self.all_class_counts.append(self.class_counts.copy())  # Store the class counts for this fold

        # F() calculates the individual attribute probabilities for each class.
        d = self.data_array.attribute_count - 1  # Number of attributes (excluding label)

        # Store attribute value counts for each fold
        self.attribute_value_counts_per_fold = []

        # Perform 10-fold cross-validation
        for i in range(10):
            # Initialize a dictionary for each attribute to count occurrences for the current fold
            attribute_value_counts = [{} for _ in range(d)]
            self.rotated_data = self.data_array.partitions[i:] + self.data_array.partitions[:i]
            self.rotated_data = self.rotated_data[:-1]

            # Count attribute value occurrences per class
            for partition in self.rotated_data:
                for example in partition:
                    for a in range(d):  # Loop over attributes (exclude label index)
                        attr_value = example[a]
                        if attr_value not in attribute_value_counts[a]:
                            attribute_value_counts[a][attr_value] = 0
                        attribute_value_counts[a][attr_value] += 1  # Increment count for this attribute value
            
            # Store attribute value counts for this fold
            self.attribute_value_counts_per_fold.append(attribute_value_counts)

        # Now calculate the attribute probabilities
        self.attribute_probs = []
        for i in range(10):
            self.attribute_probs.append([])  # Create a list for each fold
            for classs in range(len(self.all_class_counts[i])):  # Loop over each class
                self.attribute_probs[i].append([])  # Create a list for each class
                for attribute in range(d):  # Loop over each attribute
                    self.attribute_probs[i][classs].append({})
                    keys = list(self.attribute_value_counts_per_fold[i][attribute].keys())
                    for key in keys:
                        # Ensure that the numerator comes from correct class counts
                        numerator = self.attribute_value_counts_per_fold[i][attribute].get(key, 0)
                        N = self.all_class_counts[i][classs]  # Number of examples in the class
                        probability_value = (numerator + 1) / (N + len(keys))  # Laplace Smoothing applied
                        self.attribute_probs[i][classs][attribute][key] = probability_value  # Store the probability

        # Optional: Debugging print statements to verify attribute counts and probabilities
        '''
        for fold, counts in enumerate(self.attribute_value_counts_per_fold):
            print(f"Fold {fold+1}:")
            for attribute_index, count_dict in enumerate(counts):
                print(f"  Attribute {attribute_index}: {count_dict}")
        for fold, probs in enumerate(self.attribute_probs):
            print(f"Fold {fold+1}:")
            for classs, attributes in enumerate(probs):
                print(f"  Class {classs}:")
                for attribute, values in enumerate(attributes):
                    print(f"    Attribute {attribute}: {values}")
        '''


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
    







###################################################################
### GPT CREATED TRAINER
###################################################################
import numpy as np
from collections import defaultdict

class algorithm:
    def __init__(self, data, labels):
        self.data = data  # data: 3D list with folds, examples, and features
        self.labels = labels  # list of labels corresponding to the data
        self.feature_probs = {}
        self.class_priors = {}
    
    def train(self):
        num_folds = len(self.data)
        
        # Loop through each fold
        for fold_idx in range(num_folds):
            # Prepare training and validation data
            train_data, train_labels, test_data, test_labels = self._get_train_test(fold_idx)
            
            # Get the list of unique classes
            unique_classes = set(train_labels)
            
            # Initialize dictionaries for storing class priors and feature likelihoods
            class_counts = defaultdict(int)
            feature_counts = defaultdict(lambda: defaultdict(int))
            feature_totals = defaultdict(int)
            
            num_examples = len(train_labels)
            
            # Step 1: Calculate priors and feature likelihoods
            for example, label in zip(train_data, train_labels):
                class_counts[label] += 1
                for feature_idx, feature_value in enumerate(example):
                    feature_counts[label][feature_idx, feature_value] += 1
                    feature_totals[label] += 1

            # Step 2: Calculate prior probabilities for each class
            self.class_priors[fold_idx] = {cls: class_counts[cls] / num_examples for cls in unique_classes}

            # Step 3: Calculate conditional probabilities P(feature | class)
            self.feature_probs[fold_idx] = defaultdict(lambda: defaultdict(float))
            for label in unique_classes:
                for feature, count in feature_counts[label].items():
                    self.feature_probs[fold_idx][label][feature] = (count + 1) / (feature_totals[label] + len(feature_counts[label]))
            
            # Optionally: Perform evaluation on the test fold if desired here

    def _get_train_test(self, fold_idx):
        """
        Helper method to split data into training and testing sets based on fold_idx.
        """
        test_data = self.data[fold_idx]
        test_labels = self.labels[fold_idx]

        # Combine all other folds for training
        train_data = []
        train_labels = []
        for i in range(len(self.data)):
            if i != fold_idx:
                train_data.extend(self.data[i])
                train_labels.extend(self.labels[i])
        
        return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)





############################
## OG VALIDATE
###########################


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

        
            test_fold = self.data_array[test_fold_index]
            attribute_probs = self.attribute_probs[class_prob_index]
            
            self.correct_preditions = 0
            total_predictions = len(test_fold)

            for example in test_fold:
                features = example[:-1]
                true_label = example[-1]
                self.answers.append(true_label)
                max_class_prob = -float('inf')
                predicted_class = None


                for classs in range(len(self.all_class_counts[class_prob_index])):
                    class_prob = self.class_probs[class_prob_index][classs]

                    for feature_index, feature_value in enumerate(features):
                        '''
    
                        '''
                        print(f"i: {i}")
                        print(f"classs: {classs}")
                        print(f"featire index: {feature_index}")
                        print(f"Feature Value: {feature_value}")
                        print(f"Feature Value: {type(feature_value)}")
                        print(f"Attribute probability keys: {type(attribute_probs[i][classs])}")
                        '''
                        '''
                        feature_prob = attribute_probs[classs][feature_index].get(feature_value)
                        #print(feature_value, type(feature_prob))
                        #print(type(feature_prob))
                        if(feature_prob is None):
                            class_prob *= 1
                            #print(type(feature_prob))
                        else:
                            class_prob *= feature_prob


                    if class_prob > max_class_prob:
                        max_class_prob = class_prob
                        predicted_class = self.labels[classs]

                predicted_label = predicted_class
                self.predictions.append(predicted_label)
    '''