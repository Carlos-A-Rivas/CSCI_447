class dataset:
    def __init__(self, data_path: str, processed_flag: str):
        '''
        - INSTANTIATE ALL self VARIABLES IN THE INIT
        - take in the .data file, process it where we get a numpy array of strings where dimensions are as follows: self.intake_data[example][features]
        - MAKE SURE TO ADD EXTRACT FUNCTIONALITY FOR BOTH THE TUNING SET AND VALIDATION SET
        '''
        # FINN ADDS UP HERE
        self.intake_data = []
        self.tune_set = []
        self.validate_set = []
        self.ninety_data = []
        # CARLOS ADDS DOWN HERE

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

        '''
        # Data is being extracted from a saved CSV File
        else:
            #extract_data()
        '''

    def continuize(self):
        '''
        This method takes in the indices that need to be continuized. This will look like replacing values that are strings with numbers.
        We want to make sure we call this method BEFORE we shuffle so that we do not have to keep track of which number corresponds to which
        original value. We can figure this out later
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
        vectorization = np.vectorize(convert_to_num)
        self.intake_data = vectorization(self.intake_data)
        return
    def impute(self):
        # Replaces question marks in a dataset with a random value between the min/max of an attribute value
        # Breast cancer has a range of 1-10 for the attribute that is missing values
        for ex_idx in range(len(self.intake_data)):
            for att_idx in range(len(self.intake_data[ex_idx])):
                # if this statement is entered that means there is a missing piece of attribute data, so imputation needs to occur at this location
                if (self.intake_data[ex_idx][att_idx] == '?'):
                    # This will be the imputation method using range 1-10
                        self.intake_data[ex_idx][att_idx] = str(random.randint(1,10))
        return
    def shuffle(self):
        '''
        ONLY CALLED AFTER CONTINUIZING AND IMPUTING
        - This method will shuffle the self.intake_data by examples
        - Consider adding a flag where this can shuffle higher dimensional array (not explicitly necessary)
        '''
        np.random.shuffle(self.intake_data)
        return
    def sort(self, prediction_type_flag):
        '''
        - Sorts the data by its class/target value. We can assume all labels are the last indice of an example.
        - The prediction_type_flag essentially tells us if the last indice can be converted to a float or not. Regression datasets are sorted by value
        '''
        if prediction_type_flag == "regression":
            print('REGRESSION')
            sorted_data = self.intake_data[self.intake_data[:, -1].astype(float).argsort()]
        else:
            print("CLASSIFICATION")
            sorted_data = self.intake_data[self.intake_data[:, -1].argsort()]

        self.intake_data = sorted_data
        return
    def split(self):
        '''
        Puts the first 10% of the data into its own array (self.tune_set), then the remaining data (self.validate_set) into its own array.
        We should end up with two arrays, both are sorted and stratified. The validation still will need to be separated into partitions.
        '''
        tune_data = []

        for i, example in enumerate(self.intake_data):
            if(i % 10) == 0:
                tune_data.append(example)
            else:
                self.ninety_data.append(example)

        self.tune_set = np.array(tune_data)
        self.ninety_data = np.array(self.ninety_data)
        
        return
    def fold(self):
        '''
        This method folds self.validate_set into stratified partitions
        '''
        shape = (10, (len(self.ninety_data) // 10) + 1, len(self.ninety_data[0]))
        null_string = "null"
        self.validate_set = np.full(shape, null_string)
        fold_counts = np.zeros(10)

        for i, example in enumerate(self.ninety_data):
            fold_index = i % 10
            
            example_position = fold_counts[fold_index]  #This finds the next null example
            self.validate_set[fold_index, int(example_position)] = example

        
            fold_counts[fold_index] += 1
        return
    def shuffle_splits(self):
        '''
        Shuffles the tune set and validate set after they are complete and stratified
        '''
        np.random.shuffle(self.tune_set)
        for partition_idx, partition in enumerate(self.validate_set):
            np.random.shuffle(partition)
        return
    

    ## Don't worry about saving for now
    def save_validate_set(self, save_file_name, save_folder):
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
        #print(f"CSV file saved to {file_path}")
        return