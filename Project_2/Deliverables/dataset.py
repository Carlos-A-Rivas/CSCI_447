import os
import numpy as np
import random
import csv

# All files were developed collaboratively

class dataset:
    '''
    The dataset class handles initial data loading along with all pre-processing tasks
    '''
    def __init__(self, data_path: str, processed_flag: str):
        '''
        The constructor initializes all of the self variables, and loads the data from the original .data file.
        '''
         # Instantiate self variables
        self.intake_data = []
        self.tune_set = []
        self.validate_set = []
        self.ninety_data = []

        # Data is being read in from original .DATA file
        if (processed_flag == False):
            # Separating the .data file into lines, and shuffling the lines
            with open(data_path, 'r') as file:
                lines = file.readlines()
            # Deliminate strings into lists
            for i in range(len(lines)):
                lines[i] = lines[i].strip()
                lines[i] = lines[i].split(',')  
            # Make the list into a numpy array
            self.intake_data = np.array(lines)
    def normalize(self):
        '''
        performs mim-max normalization on the last column of the intake data (example value). This will only be used for regression data.
        '''
        values = self.intake_data[:,-1].astype(float)
        normalized_values = (values - values.min()) / (values.max() - values.min())
        self.intake_data[:, -1] = normalized_values
    def continuize(self):
        '''
        This method goes through each item in the data array, and if the item is not a number, it is replaced with a number (continuization).
        If there are no non-numbers in the dataset, all the numbers are converted to floats.
        '''
        string_to_int = {}
        next_int = 0
        # This function continuizes a single element so it can be vectorized
        def convert_to_num(value):
            nonlocal next_int
            try:
                # Try to convert to float
                return float(value)
            except ValueError:
                # If conversion fails, map the string a number
                if value not in string_to_int:
                    string_to_int[value] = next_int
                    next_int += 1
                return string_to_int[value]

        # Apply convert_to_num to each element in the array
        vectorization = np.vectorize(convert_to_num, otypes=[float])
        self.intake_data = vectorization(self.intake_data)
    def impute(self):
        '''
        Replaces question marks in a dataset with a random value between 1 and 10.
        '''
        for ex_idx in range(len(self.intake_data)):
            for att_idx in range(len(self.intake_data[ex_idx])):
                # if this statement is entered that means there is a missing piece of attribute data, so imputation needs to occur at this location
                if (self.intake_data[ex_idx][att_idx] == '?'):
                    # This will be the imputation method using range 1-10
                        self.intake_data[ex_idx][att_idx] = str(random.randint(1,10))
    def shuffle(self):
        '''
        This method will shuffle the self.intake_data by examples.
        '''
        np.random.shuffle(self.intake_data)
    def sort(self, prediction_type_flag):
        '''
        Sorts the data by its class/target value. We can assume all labels are the last indice of an example.
        The prediction_type_flag essentially tells us if the last indice can be converted to a float or not. Regression datasets are sorted by value
        '''
        if prediction_type_flag == "regression":
            #print('REGRESSION')
            sorted_data = self.intake_data[self.intake_data[:, -1].astype(np.float32).argsort()]
        else:
            #print("CLASSIFICATION")
            sorted_data = self.intake_data[self.intake_data[:, -1].argsort()]
        self.intake_data = sorted_data
    def split(self):
        '''
        Puts the first 10% of the data into its own array (self.tune_set), then the remaining data (self.validate_set) into its own array.
        We should end up with two arrays, both are sorted and stratified. The validation will still need to be separated into partitions.
        '''
        tune_data = []
        for i, example in enumerate(self.intake_data):
            if(i % 10) == 0:
                tune_data.append(example)
            else:
                self.ninety_data.append(example)
        self.tune_set = np.array(tune_data)
        self.ninety_data = np.array(self.ninety_data)
    def fold(self):
        '''
        This method folds self.validate_set into stratified partitions
        '''
        # shape should be (10, # of examples, # of attributes)
        shape = (10, (len(self.ninety_data) // 10) + 1, len(self.ninety_data[0]))
        self.validate_set = np.full(shape, 'null')
        fold_counts = np.zeros(10)

        # splits data into folds
        for i, example in enumerate(self.ninety_data):
            fold_index = i % 10
            example_position = fold_counts[fold_index]  #This finds the next null example
            self.validate_set[fold_index, int(example_position)] = example
            fold_counts[fold_index] += 1
    def shuffle_splits(self):
        '''
        Shuffles the tune set and validate set after they are complete and stratified
        '''
        np.random.shuffle(self.tune_set)
        for partition_idx, partition in enumerate(self.validate_set):
            np.random.shuffle(partition)
    def remove_attribute(self, indice=0):
        '''
        Takes in an attribute indice, and removes that entire indice from the dataset. This can be used to remove ID numbers
        '''
        self.intake_data = np.delete(self.intake_data, indice, 1)    
    def save(self, filename: str):
        """
        saves the tune set and validation set to a csv file for inspection purposes.
        """
        #get/create the path to the folder that the file should be saved to
        folder_path = os.path.expanduser(f"~/CSCI_447/Project_2/Datasets/processed_data")  
        os.makedirs(folder_path, exist_ok=True)
        tune_file_path = os.path.join(folder_path, (filename+'_tune_set.csv'))
        validate_file_path = os.path.join(folder_path, (filename+'_validate_set.csv'))

        # save the tune set
        shape_info = None
        with open(tune_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            if shape_info:
                writer.writerow(["shape"] + list(shape_info))
            writer.writerows(self.tune_set)

        # save the validation set
        reshaped_array = np.array([[';'.join(row) for row in batch] for batch in self.validate_set])
        shape_info = self.validate_set.shape
        with open(validate_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            if shape_info:
                writer.writerow(["shape"] + list(shape_info))
            writer.writerows(reshaped_array)
    def extract(self, file_path: str):
        """
        Loads data from a CSV file and converts it back to a numpy array in the original format.
        """
        tune_file_path = file_path+'_tune_set.csv'
        validate_file_path = file_path+'_validate_set.csv'

        # extract the tune set
        with open(tune_file_path, mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)
        self.tune_set = np.array(rows, dtype=str)

        # extract the validate set
        with open(validate_file_path, mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)
        shape_info = tuple(map(int, rows[0][1:]))
        data = rows[1:]
        reconstructed_data = [[cell.split(';') for cell in row] for row in data]
        self.validate_set = np.array(reconstructed_data, dtype=str).reshape(shape_info)