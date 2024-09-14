from dataset import dataset
from algorithm import algorithm
from additional_functions import make_plots
from additional_functions import create_comparison_tables

def main():
    # Instantiate Datasets
    cancer_nopre = dataset('/home/carlos/Machine_Learning_Practice/datasets/breast-cancer-wisconsin.data', 'last', False)
    glass_nopre = dataset('/home/carlos/Machine_Learning_Practice/datasets/glass.data', 'last', False)
    votes_nopre = dataset('/home/carlos/Machine_Learning_Practice/datasets/house-votes-84.data', 'first', False)
    iris_nopre = dataset('/home/carlos/Machine_Learning_Practice/datasets/iris.data', 'last', False)
    soybean_nopre = dataset('/home/carlos/Machine_Learning_Practice/datasets/soybean-small.data', 'last', False)

    cancer_pre = dataset('/home/carlos/Machine_Learning_Practice/datasets/breast-cancer-wisconsin.data', 'last', False)
    glass_pre = dataset('/home/carlos/Machine_Learning_Practice/datasets/glass.data', 'last', False)
    votes_pre = dataset('/home/carlos/Machine_Learning_Practice/datasets/house-votes-84.data', 'first', False)
    iris_pre = dataset('/home/carlos/Machine_Learning_Practice/datasets/iris.data', 'last', False)
    soybean_pre = dataset('/home/carlos/Machine_Learning_Practice/datasets/soybean-small.data', 'last', False)

    # Perform bare processing on nopre datasets
    cancer_nopre.remove_attribute()
    cancer_nopre.imputate(False)
    votes_nopre.imputate(True)
    glass_nopre.remove_attribute()
    glass_nopre.discretize(10, True)
    iris_nopre.discretize(10, False)
    iris_nopre.fix_data()

    # Perform more involved pre-processing on pre datasets
    # Cancer pre-processing
    cancer_pre.remove_attribute()
    cancer_pre.imputate(False)
    cancer_pre.add_noise()

    # Glass pre-processing
    glass_pre.remove_attribute()
    glass_pre.discretize(10, True)
    glass_pre.add_noise()

    # Votes pre-processing
    votes_pre.imputate(True)
    votes_pre.add_noise()

    # Iris pre-processing
    iris_pre.fix_data()
    iris_pre.discretize(10, False)
    iris_pre.add_noise()

    # Soybean pre-processing
    soybean_pre.add_noise()


    # Save all of the datasets
    # save(self, save_file_name, save_folder):
    cancer_nopre.save('cancer_nopre', '/home/carlos/Machine_Learning_Practice')
    glass_nopre.save('glass_nopre', '/home/carlos/Machine_Learning_Practice')
    votes_nopre.save('votes_nopre', '/home/carlos/Machine_Learning_Practice')
    iris_nopre.save('iris_nopre', '/home/carlos/Machine_Learning_Practice')
    soybean_nopre.save('soybean_nopre', '/home/carlos/Machine_Learning_Practice')

    cancer_pre.save('cancer_pre', '/home/carlos/Machine_Learning_Practice')
    glass_pre.save('glass_pre', '/home/carlos/Machine_Learning_Practice')
    votes_pre.save('votes_pre', '/home/carlos/Machine_Learning_Practice')
    iris_pre.save('iris_pre', '/home/carlos/Machine_Learning_Practice')
    soybean_pre.save('soybean_pre', '/home/carlos/Machine_Learning_Practice')


    # Instantiate the algorithm class
    cancer_nopre_processing = algorithm(cancer_nopre, 'cancer')
    glass_nopre_processing = algorithm(glass_nopre, 'glass')
    votes_nopre_processing = algorithm(votes_nopre, 'votes')
    iris_nopre_processing = algorithm(iris_nopre, 'iris')
    soybean_nopre_processing = algorithm(soybean_nopre, 'soybean')

    cancer_pre_processing = algorithm(cancer_pre, 'cancer')
    glass_pre_processing = algorithm(glass_pre, 'glass')
    votes_pre_processing = algorithm(votes_pre, 'votes')
    iris_pre_processing = algorithm(iris_pre, 'iris')
    soybean_pre_processing = algorithm(soybean_pre, 'soybean')


    # Train-Predict sequence
    cancer_nopre_processing.train_predict()
    glass_nopre_processing.train_predict()
    votes_nopre_processing.train_predict()
    iris_nopre_processing.train_predict()
    soybean_nopre_processing.train_predict()

    cancer_pre_processing.train_predict()
    glass_pre_processing.train_predict()
    votes_pre_processing.train_predict()
    iris_pre_processing.train_predict()
    soybean_pre_processing.train_predict()


    # Calculates the loss performance metrics
    cancer_nopre_processing.calculate_loss()
    glass_nopre_processing.calculate_loss()
    votes_nopre_processing.calculate_loss()
    iris_nopre_processing.calculate_loss()
    soybean_nopre_processing.calculate_loss()

    cancer_pre_processing.calculate_loss()
    glass_pre_processing.calculate_loss()
    votes_pre_processing.calculate_loss()
    iris_pre_processing.calculate_loss()
    soybean_pre_processing.calculate_loss()

    # Visualize the data and statistics
    zero_one_list = [cancer_nopre_processing.zero_one_losses, cancer_pre_processing.zero_one_losses, glass_nopre_processing.zero_one_losses, glass_pre_processing.zero_one_losses, votes_nopre_processing.zero_one_losses, votes_pre_processing.zero_one_losses, iris_nopre_processing.zero_one_losses, iris_pre_processing.zero_one_losses, soybean_nopre_processing.zero_one_losses, soybean_pre_processing.zero_one_losses]
    f1_scores_list = [cancer_nopre_processing.f1_scores, cancer_pre_processing.f1_scores, glass_nopre_processing.f1_scores, glass_pre_processing.f1_scores, votes_nopre_processing.f1_scores, votes_pre_processing.f1_scores, iris_nopre_processing.f1_scores, iris_pre_processing.f1_scores, soybean_nopre_processing.f1_scores, soybean_pre_processing.f1_scores]
    names_list = ['Cancer Data w/o Noise','Cancer Data w/ Noise','Glass Data w/o Noise','Glass Data w/ Noise','Voter Data w/o Noise','Voter Data w/ Noise','Iris Data w/o Noise','Iris Data w/ Noise','Soybean Data w/o Noise','Soybean Data w/ Noise']
    figure_size = (12,6)
    make_plots(f1_scores_list, zero_one_list, names_list, figure_size, 20)

    # Cancer Statistics
    cancer_main_table, cancer_p_value_table = create_comparison_tables(cancer_nopre_processing.zero_one_losses, cancer_pre_processing.zero_one_losses, cancer_nopre_processing.all_f1_scores_per_class, cancer_pre_processing.all_f1_scores_per_class, cancer_nopre_processing.labels)
    print("Cancer Main Table:")
    print(cancer_main_table)
    print("\n Cancer P-value Table:")
    print(cancer_p_value_table)

    # Glass Statistics
    print(glass_nopre_processing.all_f1_scores_per_class)
    glass_main_table, glass_p_value_table = create_comparison_tables(glass_nopre_processing.zero_one_losses, glass_pre_processing.zero_one_losses, glass_nopre_processing.all_f1_scores_per_class, glass_pre_processing.all_f1_scores_per_class, glass_nopre_processing.labels)
    print("\nglass Main Table:")
    print(glass_main_table)
    print("\n glass P-value Table:")
    print(glass_p_value_table)

    # Votes Statistics
    votes_main_table, votes_p_value_table = create_comparison_tables(votes_nopre_processing.zero_one_losses, votes_pre_processing.zero_one_losses, votes_nopre_processing.all_f1_scores_per_class, votes_pre_processing.all_f1_scores_per_class, votes_nopre_processing.labels)
    print("\nvotes Main Table:")
    print(votes_main_table)
    print("\n votes P-value Table:")
    print(votes_p_value_table)

    # Iris Statistics
    print(iris_pre_processing.all_f1_scores_per_class)
    iris_main_table, iris_p_value_table = create_comparison_tables(iris_nopre_processing.zero_one_losses, iris_pre_processing.zero_one_losses, iris_nopre_processing.all_f1_scores_per_class, iris_pre_processing.all_f1_scores_per_class, iris_nopre_processing.labels)
    print("\niris Main Table:")
    print(iris_main_table)
    print("\n iris P-value Table:")
    print(iris_p_value_table)

    # Soybean Statistics
    soybean_main_table, soybean_p_value_table = create_comparison_tables(soybean_nopre_processing.zero_one_losses, soybean_pre_processing.zero_one_losses, soybean_nopre_processing.all_f1_scores_per_class, soybean_pre_processing.all_f1_scores_per_class, soybean_nopre_processing.labels)
    print("\nsoybean Main Table:")
    print(soybean_main_table)
    print("\n soybean P-value Table:")
    print(soybean_p_value_table)

main()