import csv
import numpy as np
import matplotlib.pyplot as plt
from dataset import dataset

def process_all(user: str, shuffle_split: bool):
    abalone_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/abalone.data', False)
    cancer_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/breast-cancer-wisconsin.data', False)
    fire_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/forestfires.data', False)
    glass_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/glass.data', False)
    machine_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/machine.data', False)
    soybean_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/soybean-small.data', False)

    abalone_data.continuize()
    abalone_data.normalize()
    abalone_data.shuffle()
    abalone_data.sort('regression')
    abalone_data.split()
    abalone_data.fold()

    cancer_data.remove_attribute()
    cancer_data.impute()
    cancer_data.shuffle()
    cancer_data.sort('classification')
    cancer_data.split()
    cancer_data.fold()

    fire_data.continuize()
    fire_data.normalize()
    fire_data.shuffle()
    fire_data.sort('regression')
    fire_data.split()
    fire_data.fold()

    glass_data.continuize()
    glass_data.remove_attribute()
    glass_data.shuffle()
    glass_data.sort('classification')
    glass_data.split()
    glass_data.fold()

    machine_data.continuize()
    machine_data.normalize()
    machine_data.shuffle()
    machine_data.sort('regression')
    machine_data.split()
    machine_data.fold()

    soybean_data.continuize()
    soybean_data.shuffle()
    soybean_data.sort('classification')
    soybean_data.split()
    soybean_data.fold()

    if (shuffle_split == True) :
        abalone_data.shuffle_splits()
        cancer_data.shuffle_splits()
        fire_data.shuffle_splits()
        glass_data.shuffle_splits()
        machine_data.shuffle_splits()
        soybean_data.shuffle_splits()

    abalone_data.save('abalone')
    cancer_data.save('cancer')
    fire_data.save('fire')
    glass_data.save('glass')
    machine_data.save('machine')
    soybean_data.save('soybean')

    return abalone_data, cancer_data, fire_data, glass_data, machine_data, soybean_data

def save_arrays_to_csv(arrays, filename):
    """
    Save 18 numpy arrays (each 10x2) to a single CSV file.
    
    Parameters:
    arrays (list of np.ndarray): List of 18 numpy arrays to be saved.
    filename (str): Name of the CSV file to save the arrays.
    """
    if len(arrays) != 18:
        raise ValueError("The input must contain 18 numpy arrays.")
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write metadata to identify where each array starts
        for idx, array in enumerate(arrays):
            writer.writerow([f"Array {idx+1}"])
            writer.writerows(array)
            writer.writerow([])  # Empty line for separation between arrays

def load_arrays_from_csv(filename):
    """
    Load 18 numpy arrays (each 10x2) from a CSV file.
    
    Parameters:
    filename (str): Name of the CSV file to load the arrays from.
    
    Returns:
    list of np.ndarray: List of 18 numpy arrays extracted from the CSV file.
    """
    arrays = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        current_array = []
        
        for row in reader:
            # Check for metadata line or empty line
            if len(row) == 0:
                continue
            elif "Array" in row[0]:
                if current_array:
                    arrays.append(np.array(current_array, dtype=float))
                    current_array = []
            else:
                current_array.append([float(val) for val in row])
        
        # Add the last array if present
        if current_array:
            arrays.append(np.array(current_array, dtype=float))
    
    if len(arrays) != 18:
        raise ValueError("Expected to extract 18 arrays, but found a different number.")
    
    return arrays

def make_plots_2(arrays, classification_names, regression_names, figure_size, rotation_val):
    '''
    This function creates boxplots for the metrics for classification and regression
    models across three datasets.

    Parameters:
    arrays (list of np.ndarray): List of 18 numpy arrays (each 10x2). First 9 are classification (0/1 Loss, F1),
                                 second 9 are regression (MSE, MAE).
    classification_names (list of str): List of dataset names for classification (length should be 3).
    regression_names (list of str): List of dataset names for regression (length should be 3).
    figure_size (tuple): Size of the figure.
    rotation_val (int): Rotation angle for x-axis labels.
    '''
    if len(arrays) != 18:
        raise ValueError("Expected a list of 18 numpy arrays.")

    # Split the arrays into classification and regression
    classification_arrays = arrays[:9]
    regression_arrays = arrays[9:]

    # Extract metrics for classification
    loss_data = [classification_arrays[i][:, 0] for i in range(len(classification_arrays))]
    f1_data = [classification_arrays[i][:, 1] for i in range(len(classification_arrays))]

    # Extract metrics for regression
    mse_data = [regression_arrays[i][:, 0] for i in range(len(regression_arrays))]
    mae_data = [regression_arrays[i][:, 1] for i in range(len(regression_arrays))]

    num_models = 3  # KNN, ENN, K-Means

    # Positions for boxplots - 3 models per dataset
    def calculate_positions(num_datasets, num_models, spacing, width):
        positions = []
        for i in range(num_datasets):
            base_position = i * spacing * num_models
            positions.extend([base_position, base_position + width, base_position + 2 * width])
        return positions

    width = 0.25
    spacing = .5

    positions_classification = calculate_positions(len(classification_names), num_models, spacing, width)
    positions_regression = calculate_positions(len(regression_names), num_models, spacing, width)

    # Define colors for the models
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    model_names = ['KNN', 'ENN', 'K-Means']

    # Plotting 0/1 Loss Scores (Classification)
    plt.figure(figsize=figure_size)
    for i in range(num_models):
        plt.boxplot([loss_data[j] for j in range(i, len(loss_data), num_models)],
                    positions=[positions_classification[j] for j in range(i, len(positions_classification), num_models)],
                    widths=width, patch_artist=True, boxprops=dict(facecolor=colors[i]),
                    medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                    capprops=dict(color='black'), flierprops=dict(markerfacecolor=colors[i], marker='o'),
                    label=model_names[i])

    plt.xticks([i * spacing * num_models + width for i in range(len(classification_names))], classification_names, rotation=rotation_val)
    plt.xlabel('Classification Datasets')
    plt.ylabel('0/1 Loss Score')
    plt.title('0/1 Loss Scores Across Classification Datasets (Higher is Better)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plotting F1 Scores (Classification)
    plt.figure(figsize=figure_size)
    for i in range(num_models):
        plt.boxplot([f1_data[j] for j in range(i, len(f1_data), num_models)],
                    positions=[positions_classification[j] for j in range(i, len(positions_classification), num_models)],
                    widths=width, patch_artist=True, boxprops=dict(facecolor=colors[i]),
                    medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                    capprops=dict(color='black'), flierprops=dict(markerfacecolor=colors[i], marker='o'),
                    label=model_names[i])

    plt.xticks([i * spacing * num_models + width for i in range(len(classification_names))], classification_names, rotation=rotation_val)
    plt.xlabel('Classification Datasets')
    plt.ylabel('Average F1 Scores Across Classes')
    plt.title('Average F1 Scores Across Classification Datasets (Higher is Better)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plotting Mean Squared Error (Regression)
    plt.figure(figsize=figure_size)
    for i in range(num_models):
        plt.boxplot([mse_data[j] for j in range(i, len(mse_data), num_models)],
                    positions=[positions_regression[j] for j in range(i, len(positions_regression), num_models)],
                    widths=width, patch_artist=True, boxprops=dict(facecolor=colors[i]),
                    medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                    capprops=dict(color='black'), flierprops=dict(markerfacecolor=colors[i], marker='o'),
                    label=model_names[i])

    plt.xticks([i * spacing * num_models + width for i in range(len(regression_names))], regression_names, rotation=rotation_val)
    plt.xlabel('Regression Datasets')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error Across Regression Datasets (Lower is Better)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plotting Mean Absolute Error (Regression)
    plt.figure(figsize=figure_size)
    for i in range(num_models):
        plt.boxplot([mae_data[j] for j in range(i, len(mae_data), num_models)],
                    positions=[positions_regression[j] for j in range(i, len(positions_regression), num_models)],
                    widths=width, patch_artist=True, boxprops=dict(facecolor=colors[i]),
                    medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                    capprops=dict(color='black'), flierprops=dict(markerfacecolor=colors[i], marker='o'),
                    label=model_names[i])

    plt.xticks([i * spacing * num_models + width for i in range(len(regression_names))], regression_names, rotation=rotation_val)
    plt.xlabel('Regression Datasets')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error Across Regression Datasets (Lower is Better)')
    plt.legend()
    plt.tight_layout()
    plt.show()