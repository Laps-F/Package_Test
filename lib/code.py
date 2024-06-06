import numpy
import random
import pickle
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from datetime import date
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, concatenate, GlobalAveragePooling1D, BatchNormalization
from deap import base, tools, creator
from tcn import TCN

#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`algorithms` module is intended to contain some specific algorithms
in order to execute very common evolutionary algorithms. The method used here
are more for convenience than reference as the implementation of every
evolutionary algorithm may vary infinitely. Most of the algorithms in this
module use operators registered in the toolbox. Generally, the keyword used are
:meth:`mate` for crossover, :meth:`mutate` for mutation, :meth:`~deap.select`
for selection and :meth:`evaluate` for evaluation.
You are encouraged to write your own algorithms in order to make them do what
you really want them to do.
"""

import random

filter_range_max = 64
kernel_range_max = 5
reduction_range_max = 32
max_dense_nodes = 512
norm_range_max = 2
max_dense_layers =  4
len_activation = 3
len_dilation = 7

def map_range(value, leftMin, leftMax, rightMin, rightMax):
    '''
    mapear entre um range para outro
    '''
    
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def pre_processing_ind(pop):
    for i in range (len(pop)):
        #attention
        pop[i][0] = round(pop[i][0])
        pop[i][1] = round(pop[i][1])
        #TCNN
        pop[i][2] = round(pop[i][2])
        pop[i][3] = int(round(map_range(pop[i][3], 0, 1, 0, norm_range_max)))
        pop[i][4] = int(round(pop[i][4]))
        pop[i][5] = round(map_range(pop[i][5], 0, 1, 3, filter_range_max))
        pop[i][6] = round(map_range(pop[i][6], 0, 1, 3, kernel_range_max))
        pop[i][7] = float(map_range(pop[i][7], 0, 1, 0, 0.7))
        pop[i][8] = round(map_range(pop[i][8],0,1,0,len_activation))
        pop[i][9] = round(pop[i][9])
        pop[i][10] = round(pop[i][10])
        pop[i][11] = round(map_range(pop[i][11],0,1,0,len_dilation))
        pop[i][12] = int(round(pop[i][12]))
        #LSTM
        pop[i][13] = round(pop[i][13])
        pop[i][14] = round(map_range(pop[i][14], 0, 1, 3, filter_range_max))
        pop[i][15] = round(pop[i][15])
        pop[i][16] = float(map_range(pop[i][16], 0, 1, 0, 0.7))
        #DNN
        j=17
        for d in range (max_dense_layers):
            pop[i][j] = round(pop[i][j])
            pop[i][j+1] = round(map_range(pop[i][j+1],0,1,4,max_dense_nodes))
            pop[i][j+2] = round(pop[i][j+2])
            pop[i][j+3] = round(map_range(pop[i][j+3],0,1,0,len_activation))
            pop[i][j+4] = float(map_range(pop[i][j+4], 0, 1, 0, 0.7))
            j += 5
        pop[i][j] = round(map_range(pop[i][j],0,1,0,4))
        
    for i in range (len(pop)):
        #TCNN
        if (pop[i][2]==0):
            for t in range (3, 13):
                pop[i][t] = 0

        #LSTM
        if (pop[i][13]==0):
            for l in range (14, 17):
                pop[i][l] = 0
        #DNN
        j= 17
        for d in range (4):
            if (pop[i][j]==0):
                for l in range (j+1, j+5):
                    pop[i][l] = 0         
            j += 5
            
    return pop
    
def varAnd(population, toolbox, cxpb, mutpb):
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.
    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.
    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimpleBest(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, pop_size=None):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring
    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    .. note::
        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    population_to_hof = [toolbox.clone(ind) for ind in population]
    population_to_hof = pre_processing_ind(population_to_hof)
    if halloffame is not None:
        halloffame.update(population_to_hof)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        

        # Replace the current population by the offspring
        population[:] = tools.selBest(individuals = population + offspring, k = pop_size)
        
        offspring_to_hof = [toolbox.clone(ind) for ind in offspring]
        offspring_to_hof = pre_processing_ind(offspring_to_hof)
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring_to_hof)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def prepare_toolbox(problem_instance, number_of_variables, bounds_low, bounds_up, pop_size):
    '''
    Prepara o toolbox DEAP
    
    Args:
        problem_instance (function): evaluate_individual().
        number_of_variables (int): número de variáveis da arquitetura.
        bounds_low, bounds_up (int): limite de variação das váriaveis.
        pop_size (int): tamanho da população
    Returns:
        toolbox DEAP.
    
    '''
    
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    
    toolbox = base.Toolbox()
    
    toolbox.register('evaluate', problem_instance)    
    toolbox.register("attr_float", uniform, bounds_low, bounds_up, number_of_variables)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                     low=bounds_low, up=bounds_up, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, 
                     low=bounds_low, up=bounds_up, eta=20.0, 
                     indpb=1.0/number_of_variables)


    toolbox.max_gen = 10    # max number of iteration
    toolbox.mut_prob = 1/number_of_variables
    toolbox.cross_prob = 0.3
    
    return toolbox

def evaluate_individual(genome):
    '''
    Treina o modelo
    
    Args:
        genome (array): arquitetura a ser decodificada em modelo.
    Returns:
        F-Score do modelo.
    '''
    
    n_epochs = 50
    
    model, lstm  = decode(genome, False) #decodifica o genome em modelo
    
    f1_metric = None
    sample_weight = class_weight.compute_sample_weight( class_weight='balanced', y=y_train)
    
    if lstm == 1:
        fit_params = {
          'x': [x_train, x_train],
          'y': y_train,
          'batch_size' : 64,
          'epochs': n_epochs,
          'verbose': 0,
          'sample_weight': sample_weight,
        }
        model.fit(**fit_params)
        predictions = model.predict([x_test,x_test])
        f1_metric = f1_score(y_test, predictions.round().astype(int)[:,0].squeeze(), average=None)[1]
        num_parameters = model.count_params()

    else:
        fit_params = {
          'x': x_train,
          'y': y_train,
          'batch_size' : 64,
          'epochs': n_epochs,
          'verbose': 0,
          'sample_weight': sample_weight,
        }
        model.fit(**fit_params)
        predictions = model.predict(x_test)
        
        f1_metric = f1_score(y_test, predictions.round().astype(int)[:,0].squeeze(), average=None)[1]
        num_parameters = model.count_params()

    return f1_metric,

def map_range(value, leftMin, leftMax, rightMin, rightMax):
    '''
    mapear entre um range para outro
    '''
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def channel_attention(features: int, reduction: int = 16, name: str = "") -> KM.Model:
    '''channel attention model
    Args:
        features (int): number of features for incoming tensor
        reduction (int, optional): Reduction ratio for the MLP to squeeze information across channels. Defaults to 16.
        name (str, optional): Defaults to "".
    Returns:
        KM.Model: channelwise attention appllier model
    '''
    input_tensor = KL.Input(shape=(None, features))
    # Average pool over a feature map across channels
    avg = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    # Max pool over a feature map across channels
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2], keepdims=True)
    # Number of features for middle layer of shared MLP
    reduced_features = int(features // reduction)
    dense1 = KL.Dense(reduced_features)
    avg_reduced = dense1(avg)
    max_reduced = dense1(max_pool)
    dense2 = KL.Dense(features)
    avg_attention = dense2(KL.Activation("relu")(avg_reduced))
    max_attention = dense2(KL.Activation("relu")(max_reduced))
    # Channel-wise attention
    overall_attention = KL.Activation("sigmoid")(avg_attention + max_attention)
    return KM.Model(
        inputs=input_tensor, outputs=input_tensor * overall_attention, name=name
    )

def decode(genome, verbose=False):
    #print(genome)
    '''
    Args:
        genome (array): arquitetura a ser decodificada em modelo.
        verbose (bool): print da arquitetura.
    Returns:
        model: tensorflow model.
        lstm_active (bool): lstm foi aplicada ou não.
    '''
    
    #definindo número de cadamas de CNN e DNN e tamanho máximo de filtros
    filter_range_max = 64
    kernel_range_max = 5
    reduction_range_max = 32
    max_dense_nodes = 512
    norm_range_max = 2
    max_dense_layers =  4
    

    optimizer = [
      'adam',
      'rmsprop',
      'adagrad',
      'adadelta'
    ]
    
    activation =  [
      'relu',
      'gelu',
      'elu',
    ]
    
    dilations = [
        [1,2,4,8],
        [1,1,1,1],
        [1,2,4,8,16],
        [1,1,1,1,1],
        [1,2,4,8,16,32],
        [1,1,1,1,1,1],
        [1,2,4,8,16,32,64],
        [1,1,1,1,1,1,1]
        
    ]
    attention_layer = [
        "active",
    ]
    
    padding = [
        'causal',
        'same'
    ]
    
    tcn_layer_shape = [
        "Active"
        "normalization",
        "num filters",
        "kernel_size",
        "activation",
        "skip_connection",
        "return_sequences",
        "dropout",
        "dilations",
        "kernel_initializer"
        "padding",
    ]
  
    
    kernel_initializer = [
        "glorot_uniform",
        "GlorotNormal"
    ]
    
    lstm_layer_shape = [
        "active",
        "units",
        "return_sequences",
        "dropout",
        
    ]

    dense_layer_shape = [
      "active",
      "num nodes",
      "batch normalization",
      "activation",
      "dropout",
    ]
    
    #definindo o tamanho de cada camada
    lstm_layer_size = len(lstm_layer_shape)
    tcn_layer_size = len(tcn_layer_shape)
    dense_layers = max_dense_layers # excluindo a softmax layer
    dense_layer_size = len(dense_layer_shape)

    input_shape = x_train.shape[1:]
    n_classes = 1

    # inicia construcao do model
    model1 = Sequential()
    model2 = Sequential()
    offset = 0

    #camada de de atenção
    if verbose==True:
        print('\n ==== Individuo =====')
    if round(genome[offset])== 1:
        if verbose==True:
            print("\nAttention 1:")
        model1.add(channel_attention(features=62, reduction=1))
    offset += 1
    if round(genome[offset])== 1:
        if verbose==True:
            print("\nAttention 2:")
        model2.add(channel_attention(features=62, reduction=1))
    offset += 1
    
    #camada LSTM
    tcn_active = 0
    
    if round(genome[offset])== 1:
        batch_norm = False
        weight_norm = False
        layer_norm = False
        tcn_active = 1
        if verbose==True:
            print('\n TCN:')

        if int(round(map_range(genome[offset+1], 0, 1, 0, norm_range_max))) ==  0:
            batch_norm = True
            if verbose==True:
                print('batch norm')
        elif int(round(map_range(genome[offset+1], 0, 1, 0, norm_range_max))) ==  1:
            weight_norm = True
            if verbose==True:
                print('weight norm')
        else:
            layer_norm = True
            if verbose==True:
                print('layer norm')
        
        tcn = TCN(input_shape=(26, 62),
                    kernel_initializer= kernel_initializer[int(round(genome[offset + 2]))],
                   nb_filters =  round(map_range(genome[offset + 3], 0, 1, 3, filter_range_max)),
                   kernel_size= round(map_range(genome[offset + 4], 0, 1, 3, kernel_range_max)),
                   nb_stacks = 1,
                   dropout_rate = float(map_range(genome[offset + 5], 0, 1, 0, 0.7)),
                   activation = activation[round(map_range(genome[offset + 6],0,1,0,2))],
                    use_skip_connections= True if round(genome[offset+7])== 1 else False,
                    return_sequences = True if round(genome[offset+8])== 1 else False,
                    dilations = dilations[round(map_range(genome[offset + 9],0,1,0,len(dilations)-1))],
                    use_batch_norm = batch_norm,
                    use_weight_norm = weight_norm,
                    use_layer_norm = layer_norm,
                    padding = padding[int(round(genome[offset + 10]))])
        model1.add(tcn)

        if verbose==True:
            print('kernel initializer=%s'% kernel_initializer[int(round(genome[offset + 2]))])
            print('Num_filters=%d' % round(map_range(genome[offset + 3], 0, 1, 3, filter_range_max)))
            print('Keel size=%d' % round(map_range(genome[offset + 4], 0, 1, 3, kernel_range_max)))
            print('Dropout=%f' % float(map_range(genome[offset + 5], 0, 1, 0, 0.7)))
            print('Activation=%s' % activation[round(map_range(genome[offset + 6],0,1,0,2))])
            if round(genome[offset + 7]) == 1:
                print('Skip Connections')
            if round(genome[offset+8])== 1:
                print("Return sequences")
            print('Dilations=%s'% dilations[round(map_range(genome[offset + 9],0,1,0,len(dilations)-1))])
            print('Padding=%s' % padding[int(round(genome[offset + 10]))])
            

        # adiciona a GAP layer
        if round(genome[offset+8])== 1:
            model1.add(GlobalAveragePooling1D())
            # adiciona operacao flatten 
            model1.add(Flatten())
            if verbose==True:
                print('GlobalAvarePooling\nFlaten')       
    
    offset += tcn_layer_size
    
    #Camada LSTM
    lstm_active = False
    if round(genome[offset])==1:
        if verbose==True:
            print('\nLSTM:')
        lstm = tf.keras.layers.LSTM(units = round(map_range(genome[offset + 1], 0, 1, 3, filter_range_max)), 
                                    input_shape = (26, 62), 
                                    return_sequences=True if round(genome[offset+2])== 1 else False,
                                    dropout = float(map_range(genome[offset + 3], 0, 1, 0, 0.7)),
                                   )
        if verbose == True:
            print('Units=%d' % round(map_range(genome[offset + 1], 0, 1, 3, filter_range_max)))
            print('Dropout=%f' % float(map_range(genome[offset + 3], 0, 1, 0, 0.7)))
            if round(genome[offset+2])== 1:
                print("Return sequences")
        model2.add(lstm)
        
        if round(genome[offset+2])== 1:
            # adiciona a GAP layer
            model2.add(GlobalAveragePooling1D())
            # adiciona operacao flatten 
            model2.add(Flatten())
            if verbose==True:
                print('GlobalAvarePooling\nFlaten')
        # se existir model1 concatena o model1 ao model2
        if tcn_active:
            model_concat = concatenate([model1.output, model2.output], axis=-1)
            lstm_active = True
        else:
            model1 = model2
            
    offset += lstm_layer_size
    
    
    #camadas Dense
    for i in range(dense_layers):
        if round(genome[offset])==1:
            # Se LSTM == 1 adiciona camadas ao modelo concatenado
            if lstm_active:
                dense = None
                dense = Dense(round(map_range(genome[offset + 1],0,1,4,max_dense_nodes)))
                model_concat = dense(model_concat)
                if round(genome[offset + 2]) == 1:
                    model_concat = BatchNormalization()(model_concat)
                model_concat = Activation(activation[round(map_range(genome[offset + 3],0,1,0,2))])(model_concat)
                model_concat = Dropout(float(map_range(genome[offset + 4], 0, 1, 0, 0.7)))(model_concat)
                
                if verbose==True:
                    print('\n Dense%d' % i)
                    print('Max Nodes=%d' % round(map_range(genome[offset + 1],0,1,4,max_dense_nodes)))
                    print('Activation=%s' % activation[round(map_range(genome[offset + 3],0,1,0,2))])
                    print('Dropout=%f' % float(map_range(genome[offset + 5], 0, 1, 0, 0.7)))
                offset += dense_layer_size
                
            else:
                dense = None
                dense = Dense(round(map_range(genome[offset + 1],0,1,4,max_dense_nodes)))
                model1.add(dense)
                if round(genome[offset + 2]) == 1:
                    model1.add(BatchNormalization())
                model1.add(Activation(activation[round(map_range(genome[offset + 3],0,1,0,2))]))
                model1.add(Dropout(float(map_range(genome[offset + 4], 0, 1, 0, 0.7))))
       

                if verbose==True:
                    print('\n Dense%d' % i)
                    print('Max Nodes=%d' % round(map_range(genome[offset + 1],0,1,4,max_dense_nodes)))
                    if round(genome[offset + 2]):
                        print('Batch Norm')
                    print('Activation=%s' % activation[round(map_range(genome[offset + 3],0,1,0,2))])
                    print('Dropout=%f' % float(map_range(genome[offset + 5], 0, 1, 0, 0.7)))
                offset += dense_layer_size

    #Se LSTM == 1 adiciona camada final ao modelo concatenado
    if lstm_active == 1:
        model_concat = Dense(n_classes, activation='sigmoid')(model_concat)
        model = Model(inputs=[model1.input, model2.input], outputs=model_concat)
        model.compile(loss='binary_crossentropy',
          optimizer=optimizer[round(map_range(genome[offset],0,1,0,len(activation)-1))],
          metrics=['accuracy'])
    else:
        model1.add(Dense(n_classes, activation='sigmoid'))
        model1.compile(loss='binary_crossentropy',
          optimizer=optimizer[round(map_range(genome[offset],0,1,0,len(optimizer)-1))],
          metrics=['accuracy'])
        model = model1
    if verbose==True:
        print('\n Optimizer=%s \n' % optimizer[round(map_range(genome[offset],0,1,0,len(optimizer)-1))])

    return model, lstm_active

def ga(toolbox, verbose=False, stats= None, checkpoint = None, ind=[False]):
    '''
    Algoritmo Genético
    
    args:
        toolbox (toolbox): toolbox DEAP.
        Verbose (bool): print da arquitetura.
        Stats (tools.Statistics): instância `Statistics` para armazenar as populações em cada iteração.
        checkpoint (string): caminho do resultado da busca anterior armazenada.
        ind (list): individuo (arquitetura) a ser adicionado a população.
    returns:
        population (list of arrays): população final.
        logbook (tools.logbook): Registros de evolução como uma lista cronológica de dicionários.
        hof (tools.HallOfFame): hall da fama que quardas as 10 arquiteturas com maior score
        
    '''
    
    if checkpoint:
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        hof = cp["hof"]
    
    else:
        population = toolbox.population(n=30)
        
        #adiciona uma arquitetura e mais 3 variações
        if any(ind):
            population[0] = ind
            for i in range (1,4):
                population[i][:11] = population[0][:11]
        hof = tools.HallOfFame(10,similar=numpy.array_equal)
    
    best_score = 0
    
    for i in range(4):
        
        population, logbook = eaSimpleBest(population=population, toolbox=toolbox, cxpb=toolbox.cross_prob, mutpb=toolbox.mut_prob, ngen=10, halloffame=hof, stats=stats, verbose=verbose, pop_size =30)
        
        '''
        Calculando o maior score e salvando o resultado.
        '''
        score = tools.Statistics(key=lambda ind: ind.fitness.values)
        score.register("max", numpy.max, axis=0)
        record = score.compile(population)
        
        checkpoint = "NAS_SaveRuns/save_run_dr(40)_{}_{}.pkl".format(i+1,date.today().strftime("%b-%d-%Y"))
        cp = dict(population = population, logbook=logbook, hof = hof)
        
        with open(checkpoint, "wb") as cp_file:
            pickle.dump(cp, cp_file)
        
        print('f-score: ', numpy.round(record['max']*100, 3))
        
        #se o modelo parar de evoluir encerra a busca.
        if numpy.round(record['max']*100, 3) > best_score:
            best_score = numpy.round(record['max']*100, 3)
        else:  
            break
    return population, logbook, hof

def main(X_train, X_test, Y_train, Y_test, ck):
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = X_train, X_test, Y_train, Y_test
    '''
    Colocando o objetivo para nosso problema: maximizar F1
    '''
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", numpy.ndarray, typecode='d', 
                fitness=creator.FitnessMax)

    '''
    Definindo númedo de camadas CNN e Dense.
    '''
    num_dense_layers = 4

    '''
    Calculando o número de variáveis.
    '''
    number_of_variables = 2 + 11 + 4 + num_dense_layers*5 + 1 #Atention, TCN, convlayers, lstm, denselayers, optimizer

    bounds_low, bounds_up = 0, 1 # valores sao remapeados em decode

    '''
    Definindo o tamanho da população e a quantidade de gerações (de 10 em 10).
    '''
    _pop_size = 30
    _max_gen_per_10 = 4


    toolbox = prepare_toolbox(evaluate_individual
                            , number_of_variables,
                            bounds_low, bounds_up,_pop_size)

    '''
    criando uma instância `Statistics` para armazenar as populações em cada iteração.
    '''
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("max", numpy.max, axis=0)

    res, logbook, hof = ga(toolbox, verbose=False, stats= stats, checkpoint = ck, ind=[])
