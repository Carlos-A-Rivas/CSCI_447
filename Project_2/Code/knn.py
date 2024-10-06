from dataset import dataset
import numpy as np

class knn:
    def __init__(self, data: dataset, prediction_type_flag: str):
        '''
        - Set a variable equal to the tune and validation sets
        - instantiate self variables
        '''
        self.tune_set = data.tune_set
        self.validate_set = data.validate_set
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
        return
    def regress(self):
        return
    def calculate_loss(self):
        return
    def get_neighbors(self, model: np, test_point: np, k_n: int):
        '''
        - Feed this function a NxN numpy array where the first dimension is num of examples and the second dimension is num of freatures
        - The second argument is the reference point
        - the third argument is the point that is being referenced for distances
        - The method returns the class/regression value of the k_n nearest neighbors
        '''
        model
        test_point

        def euclidean_distance(point1: np, point2: np):
            # np.linalg.norm calculates the euclidean distances between two points
            #EDITED!!
            return np.linalg.norm(point1 - float(point2))
                
        
        distances = np.zeros((model.shape[0]), dtype=float)
        for i, model_point in enumerate(model):
            # calculate euclidean distance
            # COULD ALWAYS SWAP THIS FUNCTION CALL FOR THE ONE LINER
            if (model_point[0] != "null"):
                distances[i] = euclidean_distance(test_point, model_point[i][:-1])
        # np.partitions moves the K_n smallest values in an np array to the front of the array. We then slice the array to get the k_n smallest values
        smallest_distances = np.partition(distances, k_n)[:k_n]
        neighbor_indices = np.where((distances == smallest_distances))
        print(f"Neighbor Indices:\n{neighbor_indices}")
        # MIGHT NEED TO CONVERT NEIGHBOR_INDICES INTO A TUPLE
        nearest_neighbors = model[neighbor_indices][-1]
        # Technically this returns the class/regression value of the k_n nearest neighbors
        return nearest_neighbors