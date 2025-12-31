# Copyright 2023 CAI Kuntai

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import reduce
import os
import random


thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num



from . import build_graph
from . import conditional_random_field as crf
import pandas as pd
import networkx as nx
import json
from . import tools
import time
import numpy as np
import multiprocessing as mp
import itertools
import math



default_config = {
    'model_type':               'native',
    'quick_debug':              False,
    'enable_structure_learning': True,
    'delta':                    1e-5,


    "max_clique_size":          1e6,
    "max_parameter_size":       1e7,
    'attr_to_latent_ratio':     5,
    'max_type_num':             625,
    'max_latent_var_size':      25,
    'max_latent_var_num':       2,
    'mixture_group_type_ratio': 2,

    'latent_max_attr_num':      2,
    'marginal_max_attr_num':    4,

    'theta1':                   5,      # marginal theta, 
    'theta2':                   20,     # latent marginal theta2
    'theta3':                   10,     # latent marginal theta3
    'theta4':                   5,      # EM type_size theta
    
    "latent_size":              None,
    # "latent_size":              [22, 22],
    'max_group_size':           5,
    'syn_group_size':           15,
    'EM_group_size':            10,

    'max_TVD_num':              400,
    'max_consider_latent_num':  50,

    'IPUMS':                    False,
    'retype':                   False,
    'random_int_q':             False,
    'size1_type0':              True,
    'q_process_num':            30,
    'syn_process_num':          30,
    'log_likelihood_ratio':     10,
    'EM_type_size':             True,
    'init_EM_step_num':         2,
    'structure_EM_step_num':    2,
    'EM_step_num':              1,
    'last_EM_step_num':         1,
    'get_last_q':               False,
    'only_last_selection':      False,
    'only_selection':           False,
    'latent_total_num':         0.5,
    'marginal_step_loss_ratio': 0.9,
    'init_marginal':            [],
    'build_graph_marginal':     True,

    'group_change_num':         1,
    'tuple_insert_delete_num':  1,

    'block_group_size':         None,
    
    'select_num':               1,
    # 'latent_select_num':        2,
    'latent_select_num':        2,

    'print_interval':           100,
    'ob_iter_num':              1500,
    'iter_num':                 1500,

    'check_marginal_num':           20,
    'check_latent_marginal_num':    15,

    # 'exp_name':             'test_review',
    'exp_name':             'test_IPUMS_2',
    'data_name':            'unspecified',

    'load_edge_score':      False,
    'load_graph':           False,
    'save_model':           True,

    'budget':       None,
}

def get_noisy_block_group_size(group_size, noise, blocks):
    res_group_size = group_size.astype(float)

    res_group_size[:blocks[0][0]] += np.random.normal(scale=noise)

    for i in range(len(blocks)):
        start = blocks[i][0]
        size = blocks[i][1]
        if i == len(blocks) - 1:
            end = len(group_size)
        else:
            end = blocks[i+1][0]

        for j in range(start, end, size):
            res_group_size[j: j+size] = ( np.sum(group_size[j: j+size]) + np.random.normal(scale=noise) ) / len(res_group_size[j: j+size])

    return res_group_size

def get_group_size(group_data, max_size):
    group_size = np.array([min(len(group), max_size) - 1 for group in group_data], dtype=int)
    histogram, _ = np.histogram(group_size, bins=list(range(max_size+1)))
    # histogram = histogram.astype(float)
    return histogram

def get_latent_var_size(config, domain, latent_domain_limit, EM_type_size_noise, noisy_group_num, \
    print_flag=False, latent_var_num=None):
    max_attr_size = max(domain.shape)
    max_latent_var_size = latent_domain_limit/max_attr_size
   
    if print_flag:
        print(domain)
        print('latent_domain_limit: {:.4f},  max_attr_size: {:.4f}'.format(latent_domain_limit, max_attr_size))
        print('max latent size by theta: {:.2f}'.format(max_latent_var_size))

    max_latent_var_size = min(max_latent_var_size, config['max_latent_var_size'])
    max_latent_var_size = max(max_latent_var_size, 2)
    if print_flag:
        print('max latent size: {:.2f}'.format(max_latent_var_size))

    if config['EM_type_size']:
        max_type_num = noisy_group_num / EM_type_size_noise / config['theta4'] / config['EM_group_size']
        print('max_type_num by EM_type_size_noise: {:.2f}'.format(max_type_num))
        max_type_num = min(config['max_type_num'], int(max_type_num))
        max_type_num = max(4, max_type_num)
    else:
        max_type_num = config['max_type_num']
    if print_flag:
        print('max_type_num: {:d}'.format(max_type_num))

    if latent_var_num == None:
        latent_var_num = int(np.log10(max_type_num) / np.log10(max_latent_var_size)) + 1
        # latent_var_num = max(1, latent_var_num)
        # latent_var_num = int(config['max_type_num'] ** (1/max_latent_var_size)) + 1
        if print_flag:
            print('latent var num: {:d}'.format(latent_var_num))

        latent_var_num = min(latent_var_num, config['max_latent_var_num'])
        latent_var_num = max(1, latent_var_num)
        if print_flag:
            print('latent var num by max_latent_var_num: {:d}'.format(latent_var_num))

    latent_size = int(max_type_num ** (1/latent_var_num))
    latent_size = min(latent_size, int(max_latent_var_size))
    latent_size = max(latent_size, 2)
    if print_flag:
        print('final latent var num: {:d}, latent size: {:d}'.format(latent_var_num, latent_size))

    if latent_var_num == 0:
        print('latent_var_num:', latent_var_num)
        print('error: privacy budget is not enough')
        assert(0)
    latent_var_list = [latent_size,] * latent_var_num

    return latent_var_list

def down_sample(group_data, max_group_size):

    length = np.array([len(group) for group in group_data])
    orginal_max_group_size = np.max(length)
    hist, _ = np.histogram(length, bins=list(range(orginal_max_group_size+2)))

    res_group_data = []
    for group in group_data:
        group = group.copy()
        if len(group) > max_group_size:
            np.random.shuffle(group)
            group = group[:max_group_size]
        res_group_data.append(group)

    res_data = np.concatenate(res_group_data, axis=0)
    # res_data = res_data[:, 1:-1]

    res_group_data = np.array(res_group_data, dtype=object)
    total = sum([len(group) for group in group_data])
    print('downsample max_group_size:', max_group_size)
    print('orginal_max_group_size:', orginal_max_group_size)
    print('group size histogram:', hist)
    print('downsample ratio {:.4f}'.format(len(res_data)/total))

    return res_data, res_group_data
    

def cal_noise(config, domain, data, group_data, init_marginal_cnt=0, init_latent_marginal_cnt=0):
    # to support other delta, use tools.cal_privacy_budget to get their privacy budget
    assert(config['delta']==1e-5)

    if not 'beta0' in config:
        assert(not 'beta_y' in config)
        assert(not 'beta_size' in config)
        assert(not 'beta_alpha' in config)
        assert(not 'beta1' in config)
        assert(not 'beta2' in config)
        assert(not 'beta3' in config)
        assert(not 'beta4' in config)
        assert(not 'beta5' in config)
        assert(not 'beta6' in config)

        beta_learn = 0.80
        beta_syn = 0.20

        config['beta0'] = beta_learn * 0.10         # attribute graph
        beta_marginal = beta_learn * 0.90

        beta_model = beta_marginal * 0.80
        beta_ob = beta_marginal * 0.20

        beta_other = beta_model * 0.20
        beta_latent = beta_model * 0.80

        config['beta_alpha'] = beta_other * 0.20    # alpha of each EM step
        config['beta_size'] = beta_other * 0.80     # type_size of each EM step
        config['beta_y'] = beta_latent * 0.10       # latent type marginal of each EM step
        config['beta1'] = beta_latent * 0.80        # latent marginals
        config['beta4'] = beta_latent * 0.10        # select latent marginals

        config['beta2'] = beta_ob * 0.90            # marginals
        config['beta3'] = beta_ob * 0.10            # select marginals

        config['beta5'] = beta_syn * 0.80           # type group size
        config['beta6'] = beta_syn * 0.20           # group size

    temp_sum = 0
    for item in config:
        if item.find('beta') != -1:
            print(item, ': {:.4f}'.format(config[item]))
            temp_sum += config[item]
    assert(abs(temp_sum - 1) < 1e-6)

    # config['marginal_step_num'] = len(domain)
    if not 'marginal_step_num' in config:
        config['marginal_step_num'] = int(len(domain) * 0.8)

    # Note that latent_TVD_sensitivity, type_size_sensitivity, and group_size_sensitivity
    # have different meanings in the native and the mixture model. Should calculate
    # in different ways.

    if config['model_type'] == 'native':
        R_sensitivity = 2 * config['tuple_insert_delete_num']
        latent_marginal_sensitivity = 2*config['max_group_size'] * config['group_change_num']
        marginal_sensitivity = 1 * config['tuple_insert_delete_num']
        latent_TVD_sensitivity = 2 * config['group_change_num']
        type_size_sensitivity = 2 * config['group_change_num']
        group_size_sensitivity = 2 * config['group_change_num']
        alpha_sensitivity = 2 * config['group_change_num']

    elif config['model_type'] == 'mixture':
       raise # legacy

    normal_abs_dev_ratio = (2 / math.pi) ** 0.5
    latent_total_num = int(config['latent_total_num'] * len(domain))


    epsilon = config['epsilon']
    if config['budget'] is None:
        budget = tools.get_privacy_budget(epsilon)
    else:
        budget = config['budget']

    data_num = len(data)
    group_num = len(group_data)

    # beta6
    if config['model_type'] == 'native':
        # a distribution of group sizes

        group_size = get_group_size(group_data, config['syn_group_size'])
        total_size_noise = (group_size_sensitivity**2 / (config['beta6'] * budget)) ** 0.5
        if config['block_group_size'] is None:
            noisy_group_size = group_size + np.random.normal(scale=total_size_noise, size=group_size.shape)
        else:
            noisy_group_size = get_noisy_block_group_size(group_size, total_size_noise, config['block_group_size'])

        print('noisy group size:', noisy_group_size.astype(int))
        print('group size:', group_size.astype(int))

        noisy_group_num = np.sum(noisy_group_size)
        noisy_data_num = sum([(size+1) * noisy_group_size[size] for size in range(len(noisy_group_size))])

        print('data num: {:d}'.format(data_num))
        print('noisy data num: {:.2f}'.format(noisy_data_num))
        print('group num: {:d}'.format(group_num))
        print('noisy group num: {:.2f}'.format(noisy_group_num))

    elif config['model_type'] == 'mixture':
        # a list of sizes of groups
        group_size = np.array([len(group) for group in group_data])
        
        total_size_noise = (group_size_sensitivity**2 / (config['beta6'] * budget)) ** 0.5
        noisy_group_size = group_size + np.random.normal(scale=total_size_noise, size=group_size.shape)
        print('mean group size: {:.4f}'.format(sum(group_size)/len(group_size)))

        noisy_group_num = None
        noisy_data_num = np.sum(noisy_group_size)

        print('data num: {:d}'.format(data_num))
        print('noisy data num: {:.2f}'.format(noisy_data_num))
        print('group num: {:d}'.format(group_num))
        
    print('group size noise: {:.4f}'.format(total_size_noise))
    

    # beta1 
    # latent marginal num of different structure steps
    latent_variable_set = set()
    for attr, value in domain.dict.items():
        if 'latent' in value and value['latent']:
            latent_variable_set.add(attr)

    assert(config['theta2'] >= config['theta3'])

    if config['only_last_selection']:
        config['structure_EM_step_num'] = 0


    if config['enable_structure_learning']:
        estimated_step_cost = init_latent_marginal_cnt * config['init_EM_step_num']
        estimated_step_cost += (latent_total_num+init_latent_marginal_cnt) * config['structure_EM_step_num'] * config['EM_step_num']
        estimated_step_cost += (init_latent_marginal_cnt+latent_total_num) * 1

        estimated_latent_noise = ((latent_marginal_sensitivity ** 2) / (config['beta1'] * budget / estimated_step_cost)) ** 0.5

    else:
        estimated_step_cost = init_latent_marginal_cnt * config['init_EM_step_num']

        estimated_latent_noise = ( latent_marginal_sensitivity**2 / (config['beta1'] * budget / estimated_step_cost)) ** 0.5
    

    print('estimated_step_cost: {:d}'.format(estimated_step_cost))
    print('estimated_latent_noise: {:.4f}'.format(estimated_latent_noise))

    estimated_latent_dom2 = noisy_data_num / normal_abs_dev_ratio / estimated_latent_noise / config['theta2']
    estimated_latent_dom3 = noisy_data_num / normal_abs_dev_ratio / estimated_latent_noise / config['theta3']
    print('estimated_latent_dom2: {:.2f}'.format(estimated_latent_dom2))
    print('estimated_latent_dom3: {:.2f}'.format(estimated_latent_dom3))

    latent_marginal_set2 = set()
    latent_marginal_set3 = set()

    EM_type_size_noise = -1
    if config['enable_structure_learning']:
        total_EM_step_num = config['init_EM_step_num'] + config['structure_EM_step_num']*config['EM_step_num'] + 1
        print('total_EM_step_num: {:d}'.format(total_EM_step_num))

        alpha_noise = ( (alpha_sensitivity ** 2) / (config['beta_alpha'] * budget / total_EM_step_num) )**0.5
        if config['EM_type_size']:
            EM_type_size_noise = ( (type_size_sensitivity ** 2) / (config['beta_size'] * budget / total_EM_step_num) )**0.5
    else:
        total_EM_step_num = config['init_EM_step_num']
        print('total_EM_step_num: {:d}'.format(total_EM_step_num))

        alpha_noise = ( (alpha_sensitivity ** 2) / (config['beta_alpha'] * budget / total_EM_step_num) )**0.5
        if config['EM_type_size']:
            EM_type_size_noise = ( (type_size_sensitivity ** 2) / (config['beta_size'] * budget / total_EM_step_num) )**0.5
        

    estimated_latent_var_list = get_latent_var_size(config, domain, estimated_latent_dom3, EM_type_size_noise, noisy_group_num, print_flag=True)
    estimated_latent_var_size = estimated_latent_var_list[0]
    estimated_latent_dom2 /= estimated_latent_var_size
    estimated_latent_dom3 /= estimated_latent_var_size
    print('estimated_latent_var_size: {:d}'.format(estimated_latent_var_size))


    for attr_num in range(2, config['latent_max_attr_num']+1):
        for marginal in itertools.combinations(domain.attr_list, attr_num):
            dom_size = domain.project(marginal).size()
            # latent var is to be determined
            if len(latent_variable_set.intersection(marginal)) == 0:
                if dom_size < estimated_latent_dom3:
                    latent_marginal_set3.add(marginal)
                    if dom_size < estimated_latent_dom2:
                        latent_marginal_set2.add(marginal)

    # print(latent_marginal_set2)
    # print(latent_marginal_set3)
    if config['enable_structure_learning']:
        if config['only_last_selection']:
            m1 = 0
            m2 = latent_total_num
        elif config['only_selection']:
            m1 = latent_total_num
            m2 = 0
        else:
            m1 = len(latent_marginal_set2) / (len(latent_marginal_set3)+1e-5) * latent_total_num
            m2 = latent_total_num - m1

        print('latent marginals: {:.2f}, {:.2f}'.format(m1, m2))
        print('')

        m1_list = []
        if config['structure_EM_step_num'] > 0:
            m1_list = [int(m1/config['structure_EM_step_num']),] * config['structure_EM_step_num']
            for step in range(int(min(config['structure_EM_step_num'], m1-sum(m1_list)))):
                m1_list[step] += 1
        m2 = latent_total_num - sum(m1_list)
        print('rounded latent marginals:', m1_list)
        print('final step latent marginals:', m2)

        latent_marginal_num = []
        current_latent_num = init_latent_marginal_cnt
        for step in range(config['structure_EM_step_num']):
            current_latent_num += m1_list[step]
            for EM_step in range(config['EM_step_num']):
                latent_marginal_num.append(current_latent_num)

        for EM_step in range(config['last_EM_step_num']-1):
            latent_marginal_num.append(current_latent_num)
        
        print('latent marginal nums:', init_latent_marginal_cnt, latent_marginal_num)


        init_step_cost = init_latent_marginal_cnt * config['init_EM_step_num']
        print('init step cost: {:d}'.format(init_step_cost))

        sum_step_cost = sum(latent_marginal_num)
        print('sum step cost: {:d}'.format(sum_step_cost))

        current_latent_num += m2
        latent_marginal_num.append(current_latent_num)
        last_step_cost = latent_marginal_num[-1] * 1
        print('last step cost: {:d}'.format(last_step_cost))

    else:
        config['structure_EM_step_num'] = 0
        m1 = 0
        m2 = 0
        # m2 = latent_total_num
        m1_list = []
    
        init_step_cost = init_latent_marginal_cnt * config['init_EM_step_num']
        sum_step_cost = 0
        last_step_cost = 0
        # last_step_cost = init_latent_marginal_cnt + latent_total_num
        print('init step cost: {:d}'.format(init_step_cost))
        print('last step cost: {:d}'.format(last_step_cost))
    
    y_noise = ( (latent_marginal_sensitivity ** 2) / (config['beta_y'] * budget / total_EM_step_num) )**0.5 

    sum_step_cost = init_step_cost + sum_step_cost + last_step_cost
    sum_step_cost *= len(estimated_latent_var_list)
    print('final step cost: {:d}\n'.format(sum_step_cost))

    lap_latent_marginal_noise = 1 / (config['beta1'] * config['epsilon'] * 0.8 / sum_step_cost)
    latent_marginal_noise = ((latent_marginal_sensitivity ** 2) / (config['beta1'] * budget / sum_step_cost)) ** 0.5
    print('latent marginal noise: {:.2f}'.format(latent_marginal_noise))
    print('lap_latent_marginal_noise: {:.2f}'.format(lap_latent_marginal_noise))

    latent_domain_limit2 = noisy_data_num / ( latent_marginal_noise * normal_abs_dev_ratio ) / config['theta2']
    print('latent marginal dom limit2: {:.2f}'.format(latent_domain_limit2))

    latent_domain_limit3 = noisy_data_num / ( latent_marginal_noise * normal_abs_dev_ratio ) / config['theta3']
    print('latent marginal dom limit3: {:.2f}\n'.format(latent_domain_limit3))

    # beta5
    type_size_noise = (type_size_sensitivity**2 / (config['beta5'] * budget)) ** 0.5
    if config['model_type'] == 'mixture':
        mean_record_num = noisy_data_num / group_num
        max_type_num = int(mean_record_num/type_size_noise/config['mixture_group_type_ratio'])
        print('max_type_num by signal-to-noise ratio:', max_type_num)
        config['max_type_num'] = min(config['max_type_num'], max_type_num)

    # determine the numebr of latent variables and their sizes.
    latent_size = None
    if config['latent_size'] is None:
        latent_var_list = get_latent_var_size(config, domain, latent_domain_limit3, EM_type_size_noise, \
            noisy_group_num, print_flag=True, latent_var_num=len(estimated_latent_var_list))
        latent_variable = len(domain)
        for latent_size in latent_var_list:
            domain.add_variable(latent_variable, latent_size, {'latent': True})
            latent_variable += 1
    else:
        print('latent variables:', config['latent_size'])

        latent_variable = len(domain)
        for size in config['latent_size']:
            domain.add_variable(latent_variable, size, {'latent': True})
            latent_variable += 1
        latent_size = config['latent_size'][0]
        latent_var_list = config['latent_size']

    # when used to select latent marginals, 
    # we use this to limit the domain size of other attributes
    latent_domain_limit2 /= latent_size
    latent_domain_limit3 /= latent_size
    domain = domain.moveaxis(list(range(len(domain))))

    # beta2
    marginal_num = init_marginal_cnt + config['marginal_step_num'] * config['select_num']
    marginal_noise = (marginal_sensitivity**2 / (config['beta2'] * budget / marginal_num)) ** 0.5
    # marginal_noise = ((config['max_group_size'] ** 2) / (config['beta2'] * budget / marginal_num)) ** 0.5
    print('marginal noise: {:.2f}'.format(marginal_noise))

    marginal_domain_limit = noisy_data_num / ( marginal_noise * normal_abs_dev_ratio ) / config['theta1']
    print('marginal dom limit: {:.2f}'.format(marginal_domain_limit))

    # beta3
    marginal_TVD_num = config['marginal_step_num'] * config['max_TVD_num']
    marginal_TVD_noise = 0
    if marginal_TVD_num > 0:
        marginal_TVD_noise = (((marginal_sensitivity/2) ** 2) / (config['beta3'] * budget / marginal_TVD_num)) ** 0.5
    print('marginal TVD noise: {:.2f}, ratio: {:.4f}'.format(marginal_TVD_noise, marginal_TVD_noise/data_num/2))

    # beta4
    score_num = config['structure_EM_step_num'] * config['max_consider_latent_num']
    score_num = max(score_num, 1e-8)

    # dispersion_noise = ((4 ** 2) / (config['beta4'] * budget / score_num)) ** 0.5
    # print('dispersion noise: {:.2f}'.format(dispersion_noise))

    if config['model_type'] == 'native':
        latent_TVD_noise = (latent_TVD_sensitivity**2 / (config['beta4'] * budget / score_num)) ** 0.5
    elif config['model_type'] == 'mixture':
        latent_TVD_noise = (latent_TVD_sensitivity**2 / (config['beta4'] * budget / score_num)) ** 0.5
    print('latent_TVD_noise: {:.2f}'.format(latent_TVD_noise))
   
    type_size_ratio = group_num / latent_size / config['max_group_size'] / type_size_noise
    print('type size noise: {:.2f}'.format(type_size_noise))
    print('type size ratio: {:.2f}'.format(type_size_ratio))
    print('total size noise: {:.2f}\n'.format(total_size_noise))

    total_latent_domain = 1
    for size in latent_var_list:
        total_latent_domain *= size
    print('total_latent_domain', total_latent_domain)

    if config['model_type'] == 'mixture':
        print('mean_record_num:', int(mean_record_num))
        print('type group signal-to-noise ratio: {:.4f}\n'.format(mean_record_num/type_size_noise/total_latent_domain))

    print('alpha noise: {:.2f}'.format(alpha_noise))
    print('alpha signal-to-noise ratio: {:.2f}\n'.format(group_num/alpha_noise/total_latent_domain))
    if config['EM_type_size']:
        print('EM type size noise: {:.2f}'.format(EM_type_size_noise))
        print('EM type size signal-to-noise ratio: {:.2f}'.format(group_num/EM_type_size_noise/total_latent_domain/config['EM_group_size']))
    

    print('latent marginal of y noise: {:.2f}'.format(y_noise))
    print('latent marginal of y signal-to-noise ratio: {:.2f}'.format(data_num/y_noise/total_latent_domain))

    noise_dict = {
        'data_num': data_num,
        'noisy_data_num': noisy_data_num,
        'group_num': group_num,
        'noisy_group_num': noisy_group_num,
        'latent_marginal_noise': latent_marginal_noise,
        'latent_domain_limit2': latent_domain_limit2,
        'latent_domain_limit3': latent_domain_limit3,
        'marginal_noise': marginal_noise,
        'marginal_domain_limit': marginal_domain_limit,
        'marginal_TVD_noise': marginal_TVD_noise,
        'latent_TVD_noise': latent_TVD_noise,
        'group_size': group_size,
        'noisy_group_size': noisy_group_size,
        'type_size_noise': type_size_noise,
        'm1_list': m1_list,
        'm2': m2,
        'R_sensitivity': R_sensitivity,
        'alpha_noise': alpha_noise,
        'EM_type_size_noise': EM_type_size_noise,
        'y_noise': y_noise,
        'total_size_noise': total_size_noise
    }

    return noise_dict

# caution: group_data contains group_id while data does not
# data: attrs
# group data: id - attrs - (FK id)
def main(p_config, domain, data, group_data):
    start_time = time.time()

    domain = domain.copy()

    epsilon = p_config['epsilon']
    print('main:', p_config['exp_name'])
    print('epsilon: {:.4f}'.format(epsilon))

    # update config
    config = default_config.copy()
    for item in p_config:
        config[item] = p_config[item]


    if config['quick_debug'] == True:
        config['iter_num'] = 150

    if not (config['model_type'] == 'native' \
        or config['model_type'] == 'mixture'):
        print('unsupported model type:', config['model_type'])
        raise

    # print(data[:5])
    # print(group_data[:5])

    # # alternative settings
    # # normal
    # config['get_last_q'] = False
    # config['init_EM_step_num'] = 2
    # config['structure_EM_step_num'] = 2

    # # only last selection
    # config['get_last_q'] = False
    # config['init_EM_step_num'] = 4
    # config['structure_EM_step_num'] = 0
    # config['only_last_selection'] = True
    
    # # only selection
    # config['init_EM_step_num'] = 2
    # config['structure_EM_step_num'] = 2
    # config['only_selection'] = True


    noise_dict = cal_noise(config, domain, data, group_data, init_latent_marginal_cnt=len(domain))

    # init marginals
    latent_marginal_list = []
    latent_var_set = []
    for attr, value in domain.dict.items():
        if 'latent' in value and value['latent']:
            latent_var_set.append(attr)

    for attr, value in domain.dict.items():
        if not ('latent' in value and value['latent']):
            for var in latent_var_set:
                latent_marginal_list.append([attr, var])

    # count query on sampled_data/sampled_group_data has a bounded sensitivity given by max_group_size
    # group level differential privacy
    # print(config['EM_group_size'])
    # print(config['max_group_size'])
    # print(config['syn_group_size'])
    # exit(0)
    sampled_data, sampled_group_data = down_sample(group_data, config['max_group_size'])
    sampled_data = sampled_data[:, 1:-1]

    if config['load_graph']:
        graph = nx.node_link_graph(json.load(open('./temp/'+config['exp_name']+'_graph.json', 'r')))
    else:
        CRF_graph = build_graph.CRFGraph(config, domain)
        graph, init_marginal = CRF_graph.greedy_add(sampled_data, noise_dict['R_sensitivity'], \
            noise_dict['marginal_domain_limit'], noise_dict['noisy_data_num'])
        json.dump(nx.node_link_data(graph), open('./temp/'+config['exp_name']+'_graph.json', 'w'))
    
    tools.print_graph(graph, './temp/graph_'+config['exp_name']+'.png')

    latent_marginal_list.append(sorted(list(latent_var_set)))


    marginal_list = init_marginal.copy()
    marginal_list = [tuple(sorted(item)) for item in marginal_list]
    marginal_list = list(set(marginal_list))
    for marginal in config['init_marginal']:
        marginal_list.append(marginal)
    print('init marginal list')
    print(marginal_list)
    print('init latent marginal list')
    print(latent_marginal_list)


    model = crf.ConditionalRandomField(config, domain, graph,\
        marginal_list, latent_marginal_list, noise_dict)
    model.estimate_parameters(data, group_data, sampled_data, sampled_group_data)

    if config['save_model']:
        print('save model', './temp/'+config['exp_name']+'.pkl')
        crf.ConditionalRandomField.save_model(model, './temp/'+config['exp_name']+'.pkl')
    
    json.dump(model.marginal_list, open('./temp/'+config['exp_name']+'_marginal.json', 'w'))

    print('training model time cost: {:.4f}'.format(time.time()-start_time))

    return model

