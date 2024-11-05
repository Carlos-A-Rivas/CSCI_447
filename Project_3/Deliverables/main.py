from dataset import dataset
from neural_network import neural_net
from additional_functions import process_all
from additional_functions import save_arrays_to_csv
from additional_functions import load_arrays_from_csv
from additional_functions import make_plots_2

def main():

    abalone_data, cancer_data, fire_data, glass_data, machine_data, soybean_data = process_all('m28r778', True)

    # Classification Sets
    cancer_net_0 = neural_net(cancer_data, "classification", hidden_layer_count=0, epochs=10,momentum=.5,learning_rate=.01,batch_size=16)
    cancer_net_1 = neural_net(cancer_data, "classification", hidden_layer_count=1, network_shape=[9,1,2],epochs=50,momentum=.7,learning_rate=.1,batch_size=16)
    cancer_net_2 = neural_net(cancer_data, "classification", hidden_layer_count=2, network_shape=[9,1,5,2],epochs=50,momentum=.9,learning_rate=.01,batch_size=16)

    glass_net_0 = neural_net(glass_data, "classification", hidden_layer_count=0, epochs=100,momentum=.5,learning_rate=.001,batch_size=16)
    glass_net_1 = neural_net(glass_data, "classification", hidden_layer_count=1, network_shape=[9,1,6],epochs=500,momentum=.95,learning_rate=.01,batch_size=32)
    glass_net_2 = neural_net(glass_data, "classification", hidden_layer_count=2, network_shape=[9,1,9,6],epochs=500,momentum=.95,learning_rate=.01,batch_size=32)

    soybean_net_0 = neural_net(soybean_data, "classification", hidden_layer_count=0, epochs=10,momentum=.99,learning_rate=.01,batch_size=16)
    soybean_net_1 = neural_net(soybean_data, "classification", hidden_layer_count=1, network_shape=[35,1,4],epochs=500,momentum=.99,learning_rate=.01,batch_size=16)
    soybean_net_2 = neural_net(soybean_data, "classification", hidden_layer_count=2, network_shape=[35,1,9,4],epochs=500,momentum=.95,learning_rate=.1,batch_size=256)


    # Regression Sets
    abalone_net_0 = neural_net(abalone_data, "regression", hidden_layer_count=0, epochs=200,momentum=.9,learning_rate=.01,batch_size=32)
    abalone_net_1 = neural_net(abalone_data, "regression", hidden_layer_count=1, network_shape=[8,1,1],epochs=500,momentum=.95,learning_rate=.01,batch_size=16)
    abalone_net_2 = neural_net(abalone_data, "regression", hidden_layer_count=2, network_shape=[8,1,9,1],epochs=500,momentum=.99,learning_rate=.001,batch_size=16)

    fire_net_0 = neural_net(fire_data, "regression", hidden_layer_count=0, epochs=200,momentum=.9,learning_rate=.01,batch_size=16)
    fire_net_1 = neural_net(fire_data, "regression", hidden_layer_count=1, network_shape=[12,1,1],epochs=200,momentum=.95,learning_rate=.01,batch_size=16)
    fire_net_2 = neural_net(fire_data, "regression", hidden_layer_count=2, network_shape=[12,1,1,1],epochs=500,momentum=.95,learning_rate=.001,batch_size=16)

    machine_net_0 = neural_net(machine_data, "regression", hidden_layer_count=0, epochs=500,momentum=.95,learning_rate=.01,batch_size=16)
    machine_net_1 = neural_net(machine_data, "regression", hidden_layer_count=1, network_shape=[9,1,1],epochs=500,momentum=.99,learning_rate=.01,batch_size=16)
    machine_net_2 = neural_net(machine_data, "regression", hidden_layer_count=2, network_shape=[9,1,7,1],epochs=500,momentum=.99,learning_rate=.01,batch_size=16)

    #Abalone 
    abalone_0_score = abalone_net_0.train_test(tuning_flag=False)
    abalone_1_score = abalone_net_1.train_test(tuning_flag=False)
    abalone_2_score = abalone_net_2.train_test(tuning_flag=False)

    #Fire
    fire_0_score = fire_net_0.train_test(tuning_flag=False)
    fire_1_score = fire_net_1.train_test(tuning_flag=False)
    fire_2_score = fire_net_2.train_test(tuning_flag=False)

    #Machine
    machine_0_score = machine_net_0.train_test(tuning_flag=False)
    machine_1_score = machine_net_1.train_test(tuning_flag=False)
    machine_2_score = machine_net_2.train_test(tuning_flag=False)

    #Cancer
    cancer_0_score = cancer_net_0.train_test(tuning_flag=False)
    cancer_1_score = cancer_net_1.train_test(tuning_flag=False)
    cancer_2_score = cancer_net_2.train_test(tuning_flag=False)

    #Glass
    glass_0_score = glass_net_0.train_test(tuning_flag=False)
    glass_1_score = glass_net_1.train_test(tuning_flag=False)
    glass_2_score = glass_net_2.train_test(tuning_flag=False)

    #Soybean
    soybean_0_score = soybean_net_0.train_test(tuning_flag=False)
    soybean_1_score = soybean_net_1.train_test(tuning_flag=False)
    soybean_2_score = soybean_net_2.train_test(tuning_flag=False)

    #Plots
    classification_arrays = [cancer_0_score,cancer_1_score,cancer_2_score,
                         glass_0_score,glass_1_score,glass_2_score,
                         soybean_0_score,soybean_1_score,soybean_2_score]
    regression_arrays = [abalone_0_score,abalone_1_score,abalone_2_score,
                        fire_0_score,fire_1_score,fire_2_score,
                        machine_0_score,machine_1_score,machine_2_score]
    arrays_to_save = []
    arrays_to_save.extend(classification_arrays)
    arrays_to_save.extend(regression_arrays)
    print(arrays_to_save)
    save_arrays_to_csv(arrays_to_save, 'results.csv')
    loaded_arrays = load_arrays_from_csv('results.csv')
    #loaded_arrays[1] = cancer_1_score
    classification_arrays = loaded_arrays[:9]
    regression_arrays = loaded_arrays[9:]
    #loaded_arrays = classification_arrays
    #loaded_arrays.extend(regression_arrays)
    classification_dataset_names = ['Cancer', 'Glass', 'Soybean']
    regression_dataset_names = ['Abalone', 'Fire', 'Machine']

    make_plots_2(loaded_arrays, classification_dataset_names, regression_dataset_names, (8, 5), 0, '/home/m28r778/CSCI_447/Project_3/Code/Figures/')
