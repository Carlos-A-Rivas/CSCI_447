import numpy as np
from backprop_network import backprop_nn
from genetic_algorithm_network import GA_nn
from differential_evolution_network import DE_nn
from particle_swarm_optimization_network import PSO_nn
from additional_functions import process_all
from additional_functions import save_arrays_to_csv


def main():
    abalone_data, cancer_data, fire_data, glass_data, machine_data, soybean_data = process_all('carlthedog3', True)

    # Backpropogation NN Instantiation
    # Classification Sets
    cancer_back_0 = backprop_nn(cancer_data, "classification", hidden_layer_count=0, epochs=10,momentum=.5,learning_rate=.01,batch_size=16)
    cancer_back_1 = backprop_nn(cancer_data, "classification", hidden_layer_count=1, network_shape=[9,1,2],epochs=50,momentum=.7,learning_rate=.1,batch_size=16)
    cancer_back_2 = backprop_nn(cancer_data, "classification", hidden_layer_count=2, network_shape=[9,1,5,2],epochs=50,momentum=.9,learning_rate=.01,batch_size=16)

    glass_back_0 = backprop_nn(glass_data, "classification", hidden_layer_count=0, epochs=100,momentum=.5,learning_rate=.001,batch_size=16)
    glass_back_1 = backprop_nn(glass_data, "classification", hidden_layer_count=1, network_shape=[9,1,6],epochs=500,momentum=.95,learning_rate=.01,batch_size=32)
    glass_back_2 = backprop_nn(glass_data, "classification", hidden_layer_count=2, network_shape=[9,1,9,6],epochs=500,momentum=.95,learning_rate=.01,batch_size=32)

    soybean_back_0 = backprop_nn(soybean_data, "classification", hidden_layer_count=0, epochs=10,momentum=.99,learning_rate=.01,batch_size=16)
    soybean_back_1 = backprop_nn(soybean_data, "classification", hidden_layer_count=1, network_shape=[35,1,4],epochs=500,momentum=.99,learning_rate=.01,batch_size=16)
    soybean_back_2 = backprop_nn(soybean_data, "classification", hidden_layer_count=2, network_shape=[35,1,9,4],epochs=500,momentum=.95,learning_rate=.1,batch_size=256)


    # Regression Sets
    abalone_back_0 = backprop_nn(abalone_data, "regression", hidden_layer_count=0, epochs=200,momentum=.9,learning_rate=.01,batch_size=32)
    abalone_back_1 = backprop_nn(abalone_data, "regression", hidden_layer_count=1, network_shape=[8,1,1],epochs=500,momentum=.95,learning_rate=.01,batch_size=16)
    abalone_back_2 = backprop_nn(abalone_data, "regression", hidden_layer_count=2, network_shape=[8,1,9,1],epochs=500,momentum=.99,learning_rate=.001,batch_size=16)

    fire_back_0 = backprop_nn(fire_data, "regression", hidden_layer_count=0, epochs=200,momentum=.9,learning_rate=.01,batch_size=16)
    fire_back_1 = backprop_nn(fire_data, "regression", hidden_layer_count=1, network_shape=[12,1,1],epochs=200,momentum=.95,learning_rate=.01,batch_size=16)
    fire_back_2 = backprop_nn(fire_data, "regression", hidden_layer_count=2, network_shape=[12,1,1,1],epochs=500,momentum=.95,learning_rate=.001,batch_size=16)

    machine_back_0 = backprop_nn(machine_data, "regression", hidden_layer_count=0, epochs=500,momentum=.95,learning_rate=.01,batch_size=16)
    machine_back_1 = backprop_nn(machine_data, "regression", hidden_layer_count=1, network_shape=[9,1,1],epochs=500,momentum=.99,learning_rate=.01,batch_size=16)
    machine_back_2 = backprop_nn(machine_data, "regression", hidden_layer_count=2, network_shape=[9,1,7,1],epochs=500,momentum=.99,learning_rate=.01,batch_size=16)



    # Differential Evolution NN Instantiation
    # Classification Sets
    cancer_DE_0 = DE_nn(cancer_data, "classification", network_shape=[9,2],epochs=200,scaling_factor=0.4,crossover_rate=0.9)
    cancer_DE_1 = DE_nn(cancer_data, "classification", network_shape=[9,1,2],epochs=10,scaling_factor=0.5,crossover_rate=0.7)
    cancer_DE_2 = DE_nn(cancer_data, "classification", network_shape=[9,1,5,2],epochs=200,scaling_factor=0.7,crossover_rate=0.9)

    glass_DE_0 = DE_nn(glass_data, "classification", network_shape=[9,6],epochs=100,scaling_factor=0.4,crossover_rate=0.5)
    glass_DE_1 = DE_nn(glass_data, "classification", network_shape=[9,1,6],epochs=500,scaling_factor=0.4,crossover_rate=0.1)
    glass_DE_2 = DE_nn(glass_data, "classification", network_shape=[9,1,9,6],epochs=500,scaling_factor=0.4,crossover_rate=0.9)

    soybean_DE_0 = DE_nn(soybean_data, "classification", network_shape=[35,4],epochs=500,scaling_factor=0.5,crossover_rate=0.3)
    soybean_DE_1 = DE_nn(soybean_data, "classification", network_shape=[35,1,4],epochs=200,scaling_factor=0.5,crossover_rate=0.7)
    soybean_DE_2 = DE_nn(soybean_data, "classification", network_shape=[35,1,9,4],epochs=500,scaling_factor=0.5,crossover_rate=0.5)


    # Regression Sets
    abalone_DE_0 = DE_nn(abalone_data, "regression", network_shape=[8,1],epochs=500,scaling_factor=0.4,crossover_rate=0.7)
    abalone_DE_1 = DE_nn(abalone_data, "regression", network_shape=[8,1,1],epochs=500,scaling_factor=0.5,crossover_rate=0.9)
    abalone_DE_2 = DE_nn(abalone_data, "regression", network_shape=[8,1,9,1],epochs=500,scaling_factor=0.4,crossover_rate=0.1)

    fire_DE_0 = DE_nn(fire_data, "regression", network_shape=[12,1],epochs=500,scaling_factor=0.4,crossover_rate=0.7)
    fire_DE_1 = DE_nn(fire_data, "regression", network_shape=[12,1,1],epochs=500,scaling_factor=0.7,crossover_rate=0.1)
    fire_DE_2 = DE_nn(fire_data, "regression", network_shape=[12,1,1,1],epochs=50,scaling_factor=1.0,crossover_rate=0.3)

    machine_DE_0 = DE_nn(machine_data, "regression", network_shape=[9,1],epochs=500,scaling_factor=0.7,crossover_rate=0.5)
    machine_DE_1 = DE_nn(machine_data, "regression", network_shape=[9,1,1],epochs=100,scaling_factor=0.4,crossover_rate=0.9)
    machine_DE_2 = DE_nn(machine_data, "regression", network_shape=[9,1,7,1],epochs=200,scaling_factor=0.4,crossover_rate=0.1)


    # Genetic Algorithm NN Instantiation
    # Classification Sets
    cancer_GA_0 = GA_nn(cancer_data, "classification", network_shape=[9,2])
    cancer_GA_1 = GA_nn(cancer_data, "classification", network_shape=[9,1,2])
    cancer_GA_2 = GA_nn(cancer_data, "classification", network_shape=[9,1,5,2])

    glass_GA_0 = GA_nn(glass_data, "classification", network_shape=[9,6])
    glass_GA_1 = GA_nn(glass_data, "classification", network_shape=[9,1,6])
    glass_GA_2 = GA_nn(glass_data, "classification", network_shape=[9,1,9,6])

    soybean_GA_0 = GA_nn(soybean_data, "classification", network_shape=[35,4])
    soybean_GA_1 = GA_nn(soybean_data, "classification", network_shape=[35,1,4])
    soybean_GA_2 = GA_nn(soybean_data, "classification", network_shape=[35,1,9,4])


    # Regression Sets
    abalone_GA_0 = GA_nn(abalone_data, "regression", network_shape=[8,1])
    abalone_GA_1 = GA_nn(abalone_data, "regression", network_shape=[8,1,1])
    abalone_GA_2 = GA_nn(abalone_data, "regression", network_shape=[8,1,9,1])

    fire_GA_0 = GA_nn(fire_data, "regression", network_shape=[12,1])
    fire_GA_1 = GA_nn(fire_data, "regression", network_shape=[12,1,1])
    fire_GA_2 = GA_nn(fire_data, "regression", network_shape=[12,1,1,1])

    machine_GA_0 = GA_nn(machine_data, "regression", network_shape=[9,1])
    machine_GA_1 = GA_nn(machine_data, "regression", network_shape=[9,1,1])
    machine_GA_2 = GA_nn(machine_data, "regression", network_shape=[9,1,7,1])


    # PSO NN Instantiation
    #Classification Sets
    soybean_PSO_0 = PSO_nn(soybean_data, "classification", network_shape=[35,4])
    soybean_PSO_1 = PSO_nn(soybean_data, "classification", network_shape=[35,1,4])
    soybean_PSO_2 = PSO_nn(soybean_data, "classification", network_shape=[35,1,9,4])

    cancer_PSO_0 = PSO_nn(cancer_data, "classification", network_shape=[9,2])
    cancer_PSO_1 = PSO_nn(cancer_data, "classification", network_shape=[9,1,2])
    cancer_PSO_2 = PSO_nn(cancer_data, "classification", network_shape=[9,1,5,2])

    glass_PSO_0 = PSO_nn(glass_data, "classification", network_shape=[9,6])
    glass_PSO_1 = PSO_nn(glass_data, "classification", network_shape=[9,1,6])
    glass_PSO_2 = PSO_nn(glass_data, "classification", network_shape=[9,1,9,6])

    #Regression sets
    abalone_PSO_0 = PSO_nn(abalone_data, "regression", network_shape=[8,1])
    abalone_PSO_1 = PSO_nn(abalone_data, "regression", network_shape=[8,1,1])
    abalone_PSO_2 = PSO_nn(abalone_data, "regression", network_shape=[8,1,9,1])

    fire_PSO_0 = PSO_nn(fire_data, "regression", network_shape=[12,1])
    fire_PSO_1 = PSO_nn(fire_data, "regression", network_shape=[12,1,1])
    fire_PSO_2 = PSO_nn(fire_data, "regression", network_shape=[12,1,1,1])

    machine_PSO_0 = PSO_nn(machine_data, "regression", network_shape=[9,1])
    machine_PSO_1 = PSO_nn(machine_data, "regression", network_shape=[9,1,1])
    machine_PSO_2 = PSO_nn(machine_data, "regression", network_shape=[9,1,7,1])





    # Backpropogation Testing
    #abalone_net_0_parameters = abalone_net_0.tune()
    abalone_back_0_score = abalone_back_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_back_0_score)}")
    #abalone_net_1_parameters = abalone_net_1.tune()
    abalone_back_1_score = abalone_back_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_back_1_score)}")
    #abalone_net_2_parameters = abalone_net_2.tune()
    abalone_back_2_score = abalone_back_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_back_2_score)}")
    #fire_net_0_parameters = fire_net_0.tune()
    fire_back_0_score = fire_back_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_back_0_score)}")
    #fire_net_1_parameters = fire_net_1.tune()
    fire_back_1_score = fire_back_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_back_1_score)}")
    #fire_net_2_parameters = fire_net_2.tune()
    fire_back_2_score = fire_back_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_back_2_score)}")
    #machine_net_0_parameters = machine_net_0.tune()
    machine_back_0_score = machine_back_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_back_0_score)}")
    #machine_net_1_parameters = machine_net_1.tune()
    machine_back_1_score = machine_back_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_back_1_score)}")
    #machine_net_2_parameters = machine_net_2.tune()
    machine_back_2_score = machine_back_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_back_2_score)}")
    #cancer_net_0_parameters = cancer_net_0.tune()
    cancer_back_0_score = cancer_back_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_back_0_score)}")
    #cancer_net_1_parameters = cancer_net_1.tune()
    cancer_back_1_score = cancer_back_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_back_1_score)}")
    #cancer_net_2_parameters = cancer_net_2.tune()
    cancer_back_2_score = cancer_back_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_back_2_score)}")
    #glass_net_0_parameters = glass_net_0.tune()
    glass_back_0_score = glass_back_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_back_0_score)}")
    #glass_net_1_parameters = glass_net_1.tune()
    glass_back_1_score = glass_back_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_back_1_score)}")
    #glass_net_2_parameters = glass_net_2.tune()
    glass_back_2_score = glass_back_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_back_2_score)}")
    #soybean_net_0_parameters = soybean_net_0.tune()
    soybean_back_0_score = soybean_back_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_back_0_score)}")
    #soybean_net_1_parameters = soybean_net_1.tune()
    soybean_back_1_score = soybean_back_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_back_1_score)}")
    #soybean_net_2_parameters = soybean_net_2.tune()
    soybean_back_2_score = soybean_back_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_back_2_score)}")



    # Genetic Algorithm Testing
    cancer_GA_0_parameters = cancer_GA_0.tune()
    cancer_GA_0_score = cancer_GA_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_GA_0_score)}")
    cancer_GA_1_parameters = cancer_GA_1.tune()
    cancer_GA_1_score = cancer_GA_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_GA_1_score)}")
    cancer_GA_2_parameters = cancer_GA_2.tune()
    cancer_GA_2_score = cancer_GA_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_GA_2_score)}")
    glass_GA_0_parameters = glass_GA_0.tune()
    glass_GA_0_score = glass_GA_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_GA_0_score)}")
    glass_GA_1_parameters = glass_GA_1.tune()
    glass_GA_1_score = glass_GA_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_GA_1_score)}")
    glass_GA_2_parameters = glass_GA_2.tune()
    glass_GA_2_score = glass_GA_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_GA_2_score)}")
    soybean_GA_0_parameters = soybean_GA_0.tune()
    soybean_GA_0_score = soybean_GA_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_GA_0_score)}")
    soybean_GA_1_parameters = soybean_GA_1.tune()
    soybean_GA_1_score = soybean_GA_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_GA_1_score)}")
    soybean_GA_2_parameters = soybean_GA_2.tune()
    soybean_GA_2_score = soybean_GA_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_GA_2_score)}")
    abalone_GA_0_parameters = abalone_GA_0.tune()
    abalone_GA_0_score = abalone_GA_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_GA_0_score)}")
    abalone_GA_1_parameters = abalone_GA_1.tune(tuning_epochs=False)
    abalone_GA_1_score = abalone_GA_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_GA_1_score)}")
    abalone_GA_2_parameters = abalone_GA_2.tune()
    abalone_GA_2_score = abalone_GA_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_GA_2_score)}")
    fire_GA_0_parameters = fire_GA_0.tune()
    fire_GA_0_score = fire_GA_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_GA_0_score)}")
    fire_GA_1_parameters = fire_GA_1.tune()
    fire_GA_1_score = fire_GA_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_GA_1_score)}")
    fire_GA_2_parameters = fire_GA_2.tune()
    fire_GA_2_score = fire_GA_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_GA_2_score)}")
    machine_GA_0_parameters = machine_GA_0.tune()
    machine_GA_0_score = machine_GA_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_GA_0_score)}")
    machine_GA_1_parameters = machine_GA_1.tune()
    machine_GA_1_score = machine_GA_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_GA_1_score)}")
    machine_GA_2_parameters = machine_GA_2.tune()
    machine_GA_2_score = machine_GA_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_GA_2_score)}")


    # Differential Evolution Testing
    #cancer_DE_0_parameters = cancer_DE_0.tune()
    cancer_DE_0_score = cancer_DE_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_DE_0_score)}")
    #cancer_DE_1_parameters = cancer_DE_1.tune()
    cancer_DE_1_score = cancer_DE_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_DE_1_score)}")
    #cancer_DE_2_parameters = cancer_DE_2.tune()
    cancer_DE_2_score = cancer_DE_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_DE_2_score)}")
    #glass_DE_0_parameters = glass_DE_0.tune()
    glass_DE_0_score = glass_DE_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_DE_0_score)}")
    #glass_DE_1_parameters = glass_DE_1.tune()
    glass_DE_1_score = glass_DE_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_DE_1_score)}")
    #glass_DE_2_parameters = glass_DE_2.tune()
    glass_DE_2_score = glass_DE_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_DE_2_score)}")
    #soybean_DE_0_parameters = soybean_DE_0.tune()
    soybean_DE_0_score = soybean_DE_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_DE_0_score)}")
    #soybean_DE_1_parameters = soybean_DE_1.tune()
    soybean_DE_1_score = soybean_DE_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_DE_1_score)}")
    #soybean_DE_2_parameters = soybean_DE_2.tune()
    soybean_DE_2_score = soybean_DE_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_DE_2_score)}")
    #abalone_DE_0_parameters = abalone_DE_0.tune()
    abalone_DE_0_score = abalone_DE_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_DE_0_score)}")
    #abalone_DE_1_parameters = abalone_DE_1.tune(tuning_epochs=False)
    abalone_DE_1_score = abalone_DE_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_DE_1_score)}")
    #abalone_DE_2_parameters = abalone_DE_2.tune()
    abalone_DE_2_score = abalone_DE_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_DE_2_score)}")
    #fire_DE_0_parameters = fire_DE_0.tune()
    fire_DE_0_score = fire_DE_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_DE_0_score)}")
    #fire_DE_1_parameters = fire_DE_1.tune()
    fire_DE_1_score = fire_DE_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_DE_1_score)}")
    #fire_DE_2_parameters = fire_DE_2.tune()
    fire_DE_2_score = fire_DE_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_DE_2_score)}")
    #machine_DE_0_parameters = machine_DE_0.tune()
    machine_DE_0_score = machine_DE_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_DE_0_score)}")
    #machine_DE_1_parameters = machine_DE_1.tune()
    machine_DE_1_score = machine_DE_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_DE_1_score)}")
    #machine_DE_2_parameters = machine_DE_2.tune()
    machine_DE_2_score = machine_DE_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_DE_2_score)}")


    # Particle Swarm Optimization Testing
    soybean_PSO_0_parameters = soybean_PSO_0.tune()
    soybean_PSO_0_score = soybean_PSO_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_PSO_0_score)}")
    soybean_PSO_1_parameters = soybean_PSO_1.tune()
    soybean_PSO_1_score = soybean_PSO_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_PSO_1_score)}")
    soybean_PSO_2_parameters = soybean_PSO_2.tune()
    soybean_PSO_2_score = soybean_PSO_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(soybean_PSO_2_score)}")
    cancer_PSO_0_parameters = cancer_PSO_0.tune()
    cancer_PSO_0_score = cancer_PSO_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_PSO_0_score)}")
    cancer_PSO_1_parameters = cancer_PSO_1.tune()
    cancer_PSO_1_score = cancer_PSO_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_PSO_1_score)}")
    cancer_PSO_2_parameters = cancer_PSO_2.tune()
    cancer_PSO_2_score = cancer_PSO_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(cancer_PSO_2_score)}")
    glass_PSO_0_parameters = glass_PSO_0.tune()
    glass_PSO_0_score = glass_PSO_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_PSO_0_score)}")
    glass_PSO_1_parameters = glass_PSO_1.tune()
    glass_PSO_1_score = glass_PSO_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_PSO_1_score)}")
    glass_PSO_2_parameters = glass_PSO_2.tune()
    glass_PSO_2_score = glass_PSO_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(glass_PSO_2_score)}")
    abalone_PSO_0_parameters = abalone_PSO_0.tune()
    abalone_PSO_0_score = abalone_PSO_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_PSO_0_score)}")
    abalone_PSO_1_parameters = abalone_PSO_1.tune()
    abalone_PSO_1_score = abalone_PSO_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_PSO_1_score)}")
    abalone_PSO_2_parameters = abalone_PSO_2.tune()
    abalone_PSO_2_score = abalone_PSO_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(abalone_PSO_2_score)}")
    fire_PSO_0_parameters = fire_PSO_0.tune()
    fire_PSO_0_score = fire_PSO_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_PSO_0_score)}")
    fire_PSO_1_parameters = fire_PSO_1.tune()
    fire_PSO_1_score = fire_PSO_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_PSO_1_score)}")
    fire_PSO_2_parameters = fire_PSO_2.tune()
    fire_PSO_2_score = fire_PSO_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(fire_PSO_2_score)}")
    machine_PSO_0_parameters = machine_PSO_0.tune()
    machine_PSO_0_score = machine_PSO_0.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_PSO_0_score)}")
    machine_PSO_1_parameters = machine_PSO_1.tune()
    machine_PSO_1_score = machine_PSO_1.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_PSO_1_score)}")
    machine_PSO_2_parameters = machine_PSO_2.tune()
    machine_PSO_2_score = machine_PSO_2.train_test(tuning_flag=False)
    print(f"Average Performance: {np.mean(machine_PSO_2_score)}")


    # Saving Performance Data
    backprop_scores = [cancer_back_0_score,cancer_back_1_score,cancer_back_2_score,
                glass_back_0_score,glass_back_1_score,glass_back_2_score,
                soybean_back_0_score,soybean_back_1_score,soybean_back_2_score,
                abalone_back_0_score,abalone_back_1_score,abalone_back_2_score,
                fire_back_0_score,fire_back_1_score,fire_back_2_score,
                machine_back_0_score,machine_back_1_score,machine_back_2_score]

    GA_scores = [cancer_GA_0_score,cancer_GA_1_score,cancer_GA_2_score,
                glass_GA_0_score,glass_GA_1_score,glass_GA_2_score,
                soybean_GA_0_score,soybean_GA_1_score,soybean_GA_2_score,
                abalone_GA_0_score,abalone_GA_1_score,abalone_GA_2_score,
                fire_GA_0_score,fire_GA_1_score,fire_GA_2_score,
                machine_GA_0_score,machine_GA_1_score,machine_GA_2_score]

    DE_scores = [cancer_DE_0_score,cancer_DE_1_score,cancer_DE_2_score,
                glass_DE_0_score,glass_DE_1_score,glass_DE_2_score,
                soybean_DE_0_score,soybean_DE_1_score,soybean_DE_2_score,
                abalone_DE_0_score,abalone_DE_1_score,abalone_DE_2_score,
                fire_DE_0_score,fire_DE_1_score,fire_DE_2_score,
                machine_DE_0_score,machine_DE_1_score,machine_DE_2_score]
    
    PSO_scores = [cancer_PSO_0_score,cancer_PSO_1_score,cancer_PSO_2_score,
                glass_PSO_0_score,glass_PSO_1_score,glass_PSO_2_score,
                soybean_PSO_0_score,soybean_PSO_1_score,soybean_PSO_2_score,
                abalone_PSO_0_score,abalone_PSO_1_score,abalone_PSO_2_score,
                fire_PSO_0_score,fire_PSO_1_score,fire_PSO_2_score,
                machine_PSO_0_score,machine_PSO_1_score,machine_PSO_2_score]
    
    save_arrays_to_csv(backprop_scores, 'Backprop_Data.csv')
    save_arrays_to_csv(GA_scores, 'GA_Data.csv')
    save_arrays_to_csv(DE_scores, 'DE_Data.csv')
    save_arrays_to_csv(PSO_scores, 'PSO_Data.csv')

main()
