import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_plots(f1_scores, loss_scores, names_list, figure_size, rotation_val):
    '''
    This function creates the boxplots shown in our submitted paper.
    '''

    f1_data = [f1_scores[i] for i in range(len(f1_scores))]
    loss_data = [loss_scores[i] for i in range(len(loss_scores))]

    positions = np.arange(len(f1_scores))
    width = 0.4

    plt.figure(figsize=figure_size)
    plt.boxplot(f1_data, positions=positions, widths=width, patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='blue'),
                whiskerprops=dict(color='blue'), capprops=dict(color='blue'),
                flierprops=dict(markerfacecolor='blue', marker='o'))
    plt.xticks(positions, names_list, rotation=rotation_val)
    plt.xlabel('Datasets')
    plt.ylabel('Average F1 Scores Across Classes')
    plt.title('Average F1 Scores Across Datasets (Higher is Better)')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=figure_size)
    plt.boxplot(loss_data, positions=positions, widths=width, patch_artist=True,
                boxprops=dict(facecolor='lightcoral'), medianprops=dict(color='red'),
                whiskerprops=dict(color='red'), capprops=dict(color='red'),
                flierprops=dict(markerfacecolor='red', marker='o'))
    plt.xticks(positions, names_list, rotation=rotation_val)
    plt.xlabel('Datasets')
    plt.ylabel('0/1 Loss Score')
    plt.title('0/1 Loss Scores Across Datasets (Lower is Better)')
    plt.tight_layout()
    plt.show()


def create_comparison_tables(unaltered_loss, altered_loss, unaltered_f1, altered_f1, class_labels):
    '''
    This function prints out statistics that we entered into our submitted paper.
    '''
    unaltered_loss = np.array(unaltered_loss)
    altered_loss = np.array(altered_loss)
    unaltered_f1 = np.array(unaltered_f1)
    altered_f1 = np.array(altered_f1)

    def calculate_stats(data):
        '''
        calculates basic statistics
        '''
        return {
            'mean': np.mean(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'std': np.std(data, axis=0)
        }
    unaltered_loss_stats = calculate_stats(unaltered_loss)
    altered_loss_stats = calculate_stats(altered_loss)
    unaltered_f1_stats = calculate_stats(unaltered_f1)
    altered_f1_stats = calculate_stats(altered_f1)

    def create_stats_df(loss_stats, f1_stats, group_name, class_labels):
        '''
        Puts stats into a pandas dataframe
        '''
        rows = []
        # Add the 0/1 loss row
        rows.append([group_name, '0/1 Loss', '—', loss_stats['mean'], loss_stats['min'], loss_stats['max'], loss_stats['std']])
        # Add rows for each class's F1 score
        for i, cls in enumerate(class_labels):
            rows.append([group_name, 'F1', cls, f1_stats['mean'][i], f1_stats['min'][i], f1_stats['max'][i], f1_stats['std'][i]])
        return pd.DataFrame(rows, columns=['Training Data', 'Measure', 'Positive Class', 'Mean', 'Min.', 'Max.', 'Std.'])
    unaltered_df = create_stats_df(unaltered_loss_stats, unaltered_f1_stats, 'Unaltered', class_labels)
    altered_df = create_stats_df(altered_loss_stats, altered_f1_stats, 'Altered', class_labels)