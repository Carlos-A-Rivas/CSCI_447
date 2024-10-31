import csv
import numpy as np
import matplotlib.pyplot as plt
from dataset import dataset

# All files were developed collaboratively

def process_all(user: str, shuffle_split: bool):
    # instantiates the datasets
    abalone_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/abalone.data', False)
    cancer_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/breast-cancer-wisconsin.data', False)
    fire_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/forestfires.data', False)
    glass_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/glass.data', False)
    machine_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/machine.data', False)
    soybean_data = dataset('/home/'+user+'/CSCI_447/Project_2/Datasets/soybean-small.data', False)

    # abalone processing
    abalone_data.oh_encode()
    abalone_data.normalize("regression")
    abalone_data.shuffle()
    abalone_data.sort('regression')
    abalone_data.split()
    abalone_data.fold()

    # cancer processing
    cancer_data.oh_encode()
    cancer_data.normalize("classification")
    cancer_data.remove_attribute()
    cancer_data.impute()
    cancer_data.shuffle()
    cancer_data.sort('classification')
    cancer_data.split()
    cancer_data.fold()

    # fire processing
    fire_data.oh_encode()
    fire_data.normalize("regression")
    fire_data.shuffle()
    fire_data.sort('regression')
    fire_data.split()
    fire_data.fold()

    # glass processing
    glass_data.oh_encode()
    glass_data.remove_attribute()
    glass_data.normalize("classification")
    glass_data.shuffle()
    glass_data.sort('classification')
    glass_data.split()
    glass_data.fold()

    # machine processing
    machine_data.oh_encode()
    machine_data.normalize("regression")
    machine_data.shuffle()
    machine_data.sort('regression')
    machine_data.split()
    machine_data.fold()

    # soybean processing
    soybean_data.oh_encode()
    soybean_data.normalize("classification")
    soybean_data.shuffle()
    soybean_data.sort('classification')
    soybean_data.split()
    soybean_data.fold()

    # Final shuffle of the data
    if (shuffle_split == True) :
        abalone_data.shuffle_splits()
        cancer_data.shuffle_splits()
        fire_data.shuffle_splits()
        glass_data.shuffle_splits()
        machine_data.shuffle_splits()
        soybean_data.shuffle_splits()

    # save the datasets
    abalone_data.save('abalone')
    cancer_data.save('cancer')
    fire_data.save('fire')
    glass_data.save('glass')
    machine_data.save('machine')
    soybean_data.save('soybean')

    return abalone_data, cancer_data, fire_data, glass_data, machine_data, soybean_data

def save_arrays_to_csv(arrays: list, filename: str):
    """
    Save 18 numpy arrays to a single CSV file. This is used to save final performance data after
    running all algorithms, so that the data can be visualized later without having to re-run
    the entire program.
    """
    if len(arrays) != 18:
        raise ValueError("The input must contain 18 numpy arrays.")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for idx, array in enumerate(arrays):
            writer.writerow([f"Array {idx+1}"])
            writer.writerows(array)
            writer.writerow([])

def load_arrays_from_csv(filename: str):
    """
    Load 18 numpy arrays from a CSV file. Used for loading performance metrics.
    """
    arrays = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        current_array = []
        for row in reader:
            if len(row) == 0:
                continue
            elif "Array" in row[0]:
                if current_array:
                    arrays.append(np.array(current_array, dtype=float))
                    current_array = []
            else:
                current_array.append([float(val) for val in row])
        if current_array:
            arrays.append(np.array(current_array, dtype=float))
    if len(arrays) != 18:
        raise ValueError("Expected to extract 18 arrays, but found a different number.")
    return arrays

def make_plots_2(arrays: list, classification_names: list, regression_names: list, figure_size: tuple, rotation_val: int, save_path: str):
    '''
    This function creates boxplots for the 2 performance metrics for both classification and regression.
    This function will output 4 plots, 0/1 loss for classification, F1 score for classification,
    MSE for regression, and MAE for regression.
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
    # Positions for boxplots - 3 models per dataset
    num_models = 3
    def calculate_positions(num_datasets, num_models, spacing, width):
        positions = []
        for i in range(num_datasets):
            base_position = i * spacing * num_models
            positions.extend([base_position, base_position + width, base_position + 2 * width])
        return positions
    # Consider making these input parameters
    width = 0.25
    spacing = .5
    positions_classification = calculate_positions(len(classification_names), num_models, spacing, width)
    positions_regression = calculate_positions(len(regression_names), num_models, spacing, width)
    # Choose colors for each model
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    model_names = ['KNN', 'ENN', 'K-Means']

    # Plotting 0/1 Loss
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
    plt.savefig(f"{save_path}01_loss.png")
    plt.show()

    # Plotting F1 Score
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
    plt.savefig(f"{save_path}f1_score.png")
    plt.show()

    # Plotting MSE
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
    plt.savefig(f"{save_path}mse.png")
    plt.show()

    # Plotting MAE
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
    plt.savefig(f"{save_path}mae.png")
    plt.show()