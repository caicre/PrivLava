import os

# Copyright 2021 Kuntai Cai
# caikt@comp.nus.edu.sg

# thread number for numpy (when it runs on CPU)
thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num


from .preprocess import read_preprocessed_data, postprocess
from .attribute_graph import AttributeGraph
from .markov_random_field import MarkovRandomField
import json
import numpy as np
import pickle
import time
from .preprocess import preprocess
from .attribute_hierarchy import get_one_level_hierarchy
import sys

def main(data, domain, budget, exp_name='exp', task='TVD', init_cliques=[], p_config=None):
    defalut_config = {

        'beta5':        0.00,   # construct inner Bayesian network
        
        'theta':       6,

        'score':        'pairwsie_TVD', # pairwsie_TVD is emperically better
        # 'score':        'pairwsie_MI',
        # 'score':        'pairwise_entropy',
 
        'score_R':                      False,
        'init_measure':                 0, # 0 inner Bayesian Network 
                                        # 1 all n way measure
                                        # 2 clique measure
                                        # 3 empty measure
        'supplement_2way':              False,
        'attr_measure':                 False,
        'enable_attribute_hierarchy':   False,
        # 'enable_attribute_hierarchy':   True,
        'last_estimation':              False,
        'init_model':                   True,
        'max_level_gap':                1,
        'init_marginal':                None,

        'estimation_iter_num':          2000,
        'print_interval':               50,
        'print':                        True,

        'structure_learning':           True,
        'max_clique_size':              2e6,
        'max_parameter_size':           1e7,
        'size_penalty':                 1e-8,
        'marginal_noise':               None,
        'marginal_hist':                None,
        'sensitivity':                  1,

        'estimation_method':            'mirror_descent',

        'max_measure_attr_num':         6,

        'convergence_ratio':            1.3,
        'final_convergence_ratio':      0.8,

        'use_exp_mech':                 -1,      # do not use exponential mechanism to select measures
        # 'use_exp_mech':                 0.05,
        'structure_entropy':            False,   # marginal_noise will be set 0 to calculate the entropy of structures

        'noise_type':                   'normal', # only support normal
        'save_model':                   True,
    }
    # There might be no enough resource to run PrivMRF on GPU
    # acs should be runned on cpu, nltcs is too small and doesn't have to be runned on GPU

    gpu = True

    cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    for path in ['./temp', './result', './out']:
        if not os.path.exists(path):
            os.mkdir(path)

    if defalut_config['use_exp_mech'] > 0:
        defalut_config['beta1'] = 0.12 # dependency graph, Markov network
        defalut_config['beta2'] = 0.55 # marginal distributions of initial marginals
        defalut_config['beta4'] = 0.33 # marginal distributions of newly selected marginals
    else:
        if defalut_config['init_measure'] == 3:
            defalut_config['beta1'] = 0.10 
            defalut_config['beta2'] = 0.0
            defalut_config['beta3'] = 0.10
            defalut_config['beta4'] = 0.80
        else:
            defalut_config['beta1'] = 0.10 # dependency graph, Markov network
            defalut_config['beta2'] = 0.50 # marginal distributions of initial marginals
            defalut_config['beta3'] = 0.10 # query L_1 norms
            defalut_config['beta4'] = 0.30 # marginal distributions of newly selected marginals

            defalut_config['t'] = 0.8
            # beta2, beta4 is no longer uesful, we use t to allocate budget for marginal distribution
            # we ensure that beta2 + beta4 = 1 - (beta1 + beta3)

    config = defalut_config.copy()
    if p_config is not None:
        for item in p_config:
            config[item] = p_config[item]

    if not config['print']:
        temp_stream = sys.stdout
        sys.stdout = open('./temp/log.txt', 'w')

    config['theta1'] = config['theta']
    config['theta2'] = config['theta']
    config['epsilon'] = 'not_applicable'
    config['budget'] = budget
    print('PrivMRF')

    data_name = '?'
    config['data'] = data_name
    
    # data, domain, attr_list = read_preprocessed_data(data_name, task)
    attr_list = get_one_level_hierarchy(domain)
    start_time = time.time()
    
    print('theta:', config['theta'])
    if config['structure_learning']:
        if config['init_model']:
            init_cliques = init_cliques.copy()
            if config['marginal_noise'] is not None:
                for marginal in config['marginal_noise']:
                    init_cliques.append(marginal)
            init_model = AttributeGraph(data, domain, attr_list, config, data_name)
            graph, measure_list, attr_list, attr_to_level, entropy = init_model.construct_model(init_cliques)
            # AttributeGraph.save_model(init_model, './temp/' + config['data'] + '_model.mdl')

        # return entropy

        # init_model = AttributeGraph.load_model('./temp/' + config['data'] + '_model.mdl')
        graph = init_model.graph
        measure_list = init_model.measure_list
        attr_list = init_model.attr_list
        attr_to_level = init_model.attr_to_level
        data_num = init_model.data_num
    else:
        graph = None
        measure_list = []
        attr_to_level = None
        data_num = config['data_num']

    if not config['init_marginal'] is  None:
        for marginal in config['init_marginal']:
            measure_list.append(tuple(sorted(marginal)))

    model = MarkovRandomField(data, domain, graph, measure_list, \
        attr_list, attr_to_level, data_num, config, gpu=gpu)
    model.entropy_descent()
    # MarkovRandomField.save_model(model, './temp/' + config['data'] + '_model.mrf')

    # model = MarkovRandomField.load_model('./temp/' + config['data'] + '_model.mrf')
    if config['last_estimation']:
        model.config['convergence_ratio'] = 1.0
        model.config['estimation_iter_num'] = 5000
        model.mirror_descent()
    if config['save_model']:
        MarkovRandomField.save_model(model, './temp/'+model.config['exp_name']+'.mrf')

    # data_list = model.synthetic_data('./out/' + config['exp_name'] + '.csv')

    time_cost = time.time() - start_time
    print('time cost: {:.4f}s'.format(time_cost))

    if not config['print']:
        sys.stdout.close()
        sys.stdout = temp_stream

    os.chdir(cwd)
    

    return model

