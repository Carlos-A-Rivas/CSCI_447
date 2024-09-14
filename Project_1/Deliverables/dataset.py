# DATASET.PARTITIONS FORMAT: dataset.partitions[fold][example][attribute]


import random
import os

class dataset:
    '''
    The dataset class is used to pre-process raw .data files. The methods contained in this class have to be used based on the condition of the incoming data.
    '''
    def __init__(self, data_path, label_location, processed_flag):
        if (processed_flag == False):
            # Separating the .data file into lines, and shuffling the lines
            with open(data_path, 'r') as file:
                lines = file.readlines()
            random.shuffle(lines)
            self.data_lines = lines

            # Deliminate strings into lists
            for i in range(len(self.data_lines)):
                self.data_lines[i] = self.data_lines[i].strip()
                self.data_lines[i] = self.data_lines[i].split(',')

            # If the labels are in the first location, move them to the last location in the line
            if (label_location == 'first'):
                for i in range(len(self.data_lines)):
                    first_item = self.data_lines[i].pop(0)
                    self.data_lines[i].append(first_item)

            # Folding the data...
            self.line_count = len(self.data_lines)
            partition_size = self.line_count//10
            partition_remainder = self.line_count%10
            self.partitions = [self.data_lines[i * partition_size:(i + 1) * partition_size] for i in range(10)] # Separating the lines list into 10 lists of lines
            if (partition_remainder != 0):
                for i in range(partition_remainder):
                    self.partitions[:][i].append(self.data_lines[-(i+1)])
        else:
            #extract processed data
            self.partitions = []
            with open(data_path, 'r') as file:  #open csv file of processed data
                for line in file:
                    line = line.strip() #strip each line of any white space or newlines
                    examples = line.split(";") #Split each line into examples by each semicolon
                    attributes = [list(map(str, item.split(","))) for item in examples] #seperate each attribute and remove commas
                    self.partitions.append(attributes) # put into 3d array

        # General dataset information setters
        self.partition_count = 10
        self.attribute_count = len(self.partitions[0][0])

    def save(self, save_file_name, save_folder):
        # Saves the data based on our convention: Each line is a partition, semicolons separate examples, commas separate attributes/labels
        folder_path = os.path.expanduser(f"{save_folder}/processed_data_new")  
        os.makedirs(folder_path, exist_ok=True)
        #get/create the path to the folder that the file should be saved to
        file_path = os.path.join(folder_path, save_file_name)
        #create the file path
        with open(f"{file_path}.csv", "w") as file:
            #open a csv file in the desired location
            for line in self.partitions:
                partition_lines = ";".join([",".join(map(str, sub_array)) for sub_array in line])
                #for each partition, join each example by a semi colon and each attribute by a comma
                file.write(partition_lines + "\n")
                #write each partition into the file with each 
        return

    def imputate(self, voter_bool):
        # Replaces question marks in a dataset with a random value between the min/max of an attribute value
        # Breast cancer has a range of 1-10 for the attribute that is missing values
        # voter_bool indicates whether you are working with the voter dataset, where attributes are strings of 'y' or 'n' instead of numbers
        voter_options = ['y', 'n']
        
        for partition in range(len(self.partitions)):
            for example in range(len(self.partitions[partition])):
                for attribute in range(len(self.partitions[partition][example])):
                    # if this statement is entered that means there is a missing piece of attribute data, so imputation needs to occur at this location
                    if (self.partitions[partition][example][attribute] == '?'):
                        # This will be the imputation method using 'y' or 'n'
                        if (voter_bool == True):
                            self.partitions[partition][example][attribute] = random.choice(voter_options)
                        # This will be the imputation method using range 1-10
                        else:
                            self.partitions[partition][example][attribute] = str(random.randint(1,10))



    def discretize(self, divisions, glass_iris):
        # divisions: how much you want the bins to be divided by
        # glass_iris: boolean that indicates whether you are working with the glass or iris dataset. True = glass, False = Iris
        glass_extrema = [[1.5112, 1.5339],[10.73,17.38],[0,4.49],[0.29,3.5],[69.81,75.41],[0,6.21],[5.43,16.19],[0,3.15],[0,0.51]]
        iris_extrema = [[4.3,7.9],[2.0,4.4],[1.0,6.9],[0.1,2.5]]
        glass_ranges = []
        iris_ranges = []
        # Creates the increment level for the bin ranges
        for extrema in glass_extrema:
            glass_ranges.append(round((extrema[1]-extrema[0])/divisions,6))
        for extrema in iris_extrema:
            iris_ranges.append(round((extrema[1]-extrema[0])/divisions,2))

        
        self.bin_range = []
        # Takes a continous attribute and "bins" the attribute into discrete groups. May need to create a dictionary
        for partition in range(len(self.partitions)):
            for example in range(len(self.partitions[partition])):
                for attribute in range((len(self.partitions[partition][example])-1)):
                    # Run glass discretization
                    if (glass_iris == True):
                        entered = False
                        for i in range(divisions):
                            min = (glass_extrema[attribute][0] + (glass_ranges[attribute] * i))
                            max = (glass_extrema[attribute][0] + (glass_ranges[attribute] * (i+1)))
                            if partition == 0 and example == 0 and attribute == 0:
                                self.bin_range.append([])
                                self.bin_range[i].append(min)
                                self.bin_range[i].append(max)
                            if ((float(self.partitions[partition][example][attribute]) >= min) and (float(self.partitions[partition][example][attribute]) <= max)):
                                self.partitions[partition][example][attribute] = str(i)
                                entered = True
                        if entered == False:
                            self.partitions[partition][example][attribute] = str(int(float(self.partitions[partition][example][attribute])))
                    # Run Iris discretization
                    else:
                        entered = False
                        for i in range(divisions):
                            if ((float(self.partitions[partition][example][attribute]) >= (iris_extrema[attribute][0] + (iris_ranges[attribute] * i))) and (float(self.partitions[partition][example][attribute]) <= (iris_extrema[attribute][0] + (iris_ranges[attribute] * (i+1))))):
                                self.partitions[partition][example][attribute] = str(i)
                                entered = True
                            '''
                            if entered == False:
                                self.partitions[partition][example][attribute] = str(int(float(self.partitions[partition][example][attribute])))
                            ''' 

    def add_noise(self):
        #selects 10% of the features at random and shuffles the values within each feature, thus introducing noise into the data
        num_to_shuffle = max(1, int(round(0.1 * (self.attribute_count- 1)))) #get the numbr of attributes to shuffle based on 10% of features bing shuffled
        attribute_noise = random.sample(range(self.attribute_count - 1), num_to_shuffle)# randomly choose an attribute to shuffle
        for attribute_indice in attribute_noise: #loop through each attribute to shuffle in each example and partition
            attributes_values = []
            for partition in self.partitions:
                for example in partition:
                    attributes_values.append(example[attribute_indice]) #store attribute values in a list
                    
            
            random.shuffle(attributes_values) #shuffle selected attribute values
            
            i = 0
            for partition in self.partitions:  #loop through each partition and example
                    for example in partition:
                        example[attribute_indice] = attributes_values[i] #put the attribute value back into the data after being shuffled
                        i += 1
    
    def remove_attribute(self, indice=0):
        # Takes in an attribute indice, and removes that entire indice from the dataset. This can be used to remove ID numbers
        for partition in range(len(self.partitions)):
            for example in range(len(self.partitions[partition])):
                    self.partitions[partition][example].pop(0)
    
    def fix_data(self):
        for partition in range(len(self.partitions)):
            for i, example in enumerate(self.partitions[partition]):
                    if len(example) <= 2:
                        self.partitions[partition].pop(i)   