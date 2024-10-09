from knn import knn
from enn import enn
from kmeans import kmeans
from additional_functions import process_all
from additional_functions import save_arrays_to_csv
from additional_functions import load_arrays_from_csv
from additional_functions import make_plots_2

'''
THE CODE FOR THE VIDEO DEMO WAS MODIFIED TO ADD ADDITIONAL PRINT STATEMENTS. THAT CODE CAN BE FOUND IN THE VIDEO DEMO JUPYTER NOTEBOOK FILE
'''

def main():
    abalone_data, cancer_data, fire_data, glass_data, machine_data, soybean_data = process_all('carlthedog3', True)

    # instantiating kmeans algorithms
    cancer_kmeans = kmeans(cancer_data,'classification')
    glass_kmeans = kmeans(glass_data,'classification')
    soybean_kmeans = kmeans(soybean_data,'classification')
    abalone_kmeans = kmeans(abalone_data, 'regression')
    fire_kmeans = kmeans(fire_data, 'regression')
    machine_kmeans = kmeans(machine_data, 'regression')

    # tuning each kmeans algorithm
    cancer_kmeans.tune()
    glass_kmeans.tune(k_c_increment=3)
    soybean_kmeans.tune()
    abalone_kmeans.tune()
    fire_kmeans.tune()
    machine_kmeans.tune()

    # getting final results for each kmeans algorithm
    cancer_kmeans_results = cancer_kmeans.classify()
    glass_kmeans_results = glass_kmeans.classify()
    soybean_kmeans_results = soybean_kmeans.classify()
    abalone_kmeans_results = abalone_kmeans.regress()
    fire_kmeans_results = fire_kmeans.regress()
    machine_kmeans_results = machine_kmeans.regress()

    # instantiating knn algorithms
    cancer_knn = knn(cancer_data, 'classification')
    glass_knn = knn(glass_data, "classification")
    soybean_knn = knn(soybean_data, "classification")
    abalone_knn = knn(abalone_data, 'regression')
    fire_knn = knn(fire_data, 'regression')
    machine_knn = knn(machine_data, 'regression')

    # tuning each knn algorithm
    cancer_knn.tune(15)
    glass_knn.tune(15)
    soybean_knn.tune(10)
    abalone_knn.tune(5)
    fire_knn.tune(15)
    machine_knn.tune(10)

    # getting final results for each knn algorithm
    cancer_knn_results = cancer_knn.classify()
    glass_knn_results = glass_knn.classify()
    soybean_knn_results = soybean_knn.classify()
    abalone_knn_results = abalone_knn.regress()
    fire_knn_results = fire_knn.regress()
    machine_knn_results = machine_knn.regress()

    # instantiating enn algorithms (initialize hyperparameters for dataset reduction as those found from knn)
    cancer_enn = enn(cancer_data, 'classification', k_n=cancer_knn.k_n, sigma=cancer_knn.sigma)
    glass_enn = enn(glass_data, "classification", k_n=glass_knn.k_n, sigma=glass_knn.sigma)
    soybean_enn = enn(soybean_data, "classification", k_n=soybean_knn.k_n, sigma=soybean_knn.sigma)
    abalone_enn = enn(abalone_data, 'regression', k_n=abalone_knn.k_n, sigma=abalone_knn.sigma)
    fire_enn = enn(fire_data, 'regression', k_n=fire_knn.k_n, sigma=fire_knn.sigma)
    machine_enn = enn(machine_data, 'regression', k_n=machine_knn.k_n, sigma=machine_knn.sigma)

    # tuning each enn algorithm
    cancer_enn.tune(15)
    glass_enn.tune(10)
    soybean_enn.tune(10)
    abalone_enn.tune(5)
    fire_enn.tune(10)
    machine_enn.tune(25)


    # getting final results for each enn algorithm
    cancer_enn_results = cancer_enn.classify()
    glass_enn_results = glass_enn.classify()
    soybean_enn_results = soybean_enn.classify()
    abalone_enn_results = abalone_enn.regress()
    fire_enn_results = fire_enn.regress()
    machine_enn_results = machine_enn.regress()

    # print results for each algorithm
    print(f"Cancer KNN Loss:\n{cancer_knn_results}")
    print(f"Glass KNN Loss:\n{glass_knn_results}")
    print(f"Soybean KNN Loss:\n{soybean_knn_results}")
    print(f"Abalone KNN Loss:\n{abalone_knn_results}")
    print(f"Fire KNN Loss:\n{fire_knn_results}")
    print(f"Machine KNN Loss:\n{machine_knn_results}")

    print(f"Cancer ENN Loss:\n{cancer_enn_results}")
    print(f"Glass ENN Loss:\n{glass_enn_results}")
    print(f"Soybean ENN Loss:\n{soybean_enn_results}")
    print(f"Abalone ENN Loss:\n{abalone_enn_results}")
    print(f"Fire ENN Loss:\n{fire_enn_results}")
    print(f"Machine ENN Loss:\n{machine_enn_results}")

    print(f"Cancer K-Means Loss:\n{cancer_kmeans_results}")
    print(f"Glass K-Means Loss:\n{glass_kmeans_results}")
    print(f"Soybean K-Means Loss:\n{soybean_kmeans_results}")
    print(f"Abalone K-Means Loss:\n{abalone_kmeans_results}")
    print(f"Fire K-Means Loss:\n{fire_kmeans_results}")
    print(f"Machine K-Means Loss:\n{machine_kmeans_results}")

    # saving the performance data
    classification_arrays = [cancer_knn_results,cancer_enn_results,cancer_kmeans_results,
                         glass_knn_results,glass_enn_results,glass_kmeans_results,
                         soybean_knn_results,soybean_enn_results,soybean_kmeans_results]
    regression_arrays = [abalone_knn_results,abalone_enn_results,abalone_kmeans_results,
                        fire_knn_results,fire_enn_results,fire_kmeans_results,
                        machine_knn_results,machine_enn_results,machine_kmeans_results]
    arrays_to_save = []
    arrays_to_save.extend(classification_arrays)
    arrays_to_save.extend(regression_arrays)
    save_arrays_to_csv(arrays_to_save, 'normalized_data.csv')
    
    # loading the performance data from a saved file
    loaded_arrays = load_arrays_from_csv('normalized_data.csv')

    # plot the performance data
    classification_dataset_names = ['Cancer', 'Glass', 'Soybean']
    regression_dataset_names = ['Abalone', 'Fire', 'Machine']
    make_plots_2(loaded_arrays, classification_dataset_names, regression_dataset_names, (8, 5), 0)

main()