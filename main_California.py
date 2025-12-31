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
import numpy as np
import os
import CRF
import PrivMRF
import sys
import pandas as pd
import time
import json
from CRF.domain import Domain

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

def get_domain_by_attrs(dom_dict, columns):
    dom_dict = {attr: dom_dict[attr] for attr in dom_dict}
    dom_dict = {i: dom_dict[columns[i]] for i in range(len(columns))}
    domain = Domain(dom_dict, list(range(len(dom_dict))))
    return domain

def proc_to(data, domain):
    base = 10
    assert(data.shape[1] == len(domain))
    data_list = []
    temp_dict = {}
    new_col = 0
    new_col_list = []
    for col in range(len(domain)):
        if domain.dict[col]['size'] > 20:
            
            # print(col, domain.dict[col]['size'])
            col1, col2 = np.divmod(data[:, col], base)
            data_list.extend([col1, col2])
            temp_dict[new_col] = {'size': np.max(col1)+1}
            new_col += 1
            new_col_list.append((new_col-1, new_col))
            temp_dict[new_col] = {'size': base}
            new_col += 1
        else:
            data_list.append(data[:, col])
            temp_dict[new_col] = {'size': domain.dict[col]['size']}
            new_col += 1

    res_data = np.concatenate([col_data.reshape((-1, 1)) for col_data in data_list], axis=1)
    domain = CRF.domain.Domain(temp_dict, list(range(len(temp_dict))))

    return res_data, domain, new_col_list

def proc_back(data, new_col_list, domain):
    # print(new_col_list)
    base = 10
    res_data = []
    start = 0
    current_col = 0
    for col1, col2 in new_col_list:
        # print('  append', start, '----', col1)
        res_data.append(data[:, start:col1])

        col_data = data[:, col1] * base + data[:, col2]
        current_col += col1 - start
        col_data[col_data >= domain.dict[current_col]['size']] = domain.dict[current_col]['size'] -1
        
        current_col += 1
        start = col2+1
        res_data.append(col_data.reshape((-1,1)))
        # print('  append', col1, col2)
    res_data.append(data[:, start:])
    # print('  append', start)

    res_data = np.concatenate(res_data, axis=1)

    return res_data

if __name__ == '__main__':
    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    # basic config
    epsilon = float(sys.argv[2])
    data_name = sys.argv[3]
    exp_name = sys.argv[1] + '_'+data_name+'_{:.2f}'.format(epsilon)
    i_budget = 0.50
    h_budget = 0.50

    # read data
    i_df = pd.read_csv('./data/'+data_name+'/individual.csv')
    h_df = pd.read_csv('./data/'+data_name+'/household.csv')
    i_attrs = i_df.columns[1:-1]
    h_attrs = h_df.columns[1:]
    i_data = i_df.to_numpy()
    h_data = h_df.to_numpy()

    # # debug
    # for i in range(len(i_data)):
    #     row = i_data[i, -1]
    #     if row >= 2420000:
    #         print(i)
    #         break
    # for i in range(len(h_data)):
    #     row = h_data[i, 0]
    #     if row >= 2420000:
    #         print(i)
    #         break
    # exit(0)

    # i_data = i_data[:24805]
    # h_data = h_data[:10480]

    delta = 1 / len(i_data)
    # total_budget = 0.45
    total_budget = CRF.tools.get_privacy_budget(epsilon, delta)
    print('total_budget: {:.8f}'.format(total_budget))

    # i_data = i_data[:91021]
    # h_data = h_data[:37126]

    i_dom_dict = json.load(open('./data/'+data_name+'/individual_domain.json'))
    h_dom_dict = json.load(open('./data/'+data_name+'/household_domain.json'))
    original_i_dom = get_domain_by_attrs(i_dom_dict, i_df.columns[1:-1])
    h_dom = get_domain_by_attrs(h_dom_dict, h_df.columns[1:])

    # split big domain
    i_data1, i_dom, new_col_list = proc_to(i_data[:, 1:-1], original_i_dom)
    i_data = np.concatenate([i_data[:, [0,]], i_data1, i_data[:, [-1,]]], axis=1)
    print('i_dom:')
    print(i_dom)

    i_data = i_data[np.argsort(i_data[:, -1])]
    i_group_data = CRF.tools.get_group_data(i_data, [-1,])
    i_data, i_group_data = CRF.CRF_main.down_sample(i_group_data, 15)
    h_data = h_data[np.argsort(h_data[:, 0])]
    length = np.array([len(group) for group in i_group_data], dtype=int)

    # get i_h_data, i_h_group_data
    h_data = np.repeat(h_data, length, axis=0)
    assert((h_data[:, 0] == i_data[:, -1]).all())
    i_h_data = np.concatenate([i_data, h_data[:, 1:]], axis=1)
    print('i_h_data:')
    print(i_h_data.shape)
    # INDIVIDUAL, i attrs, HOUSEHOLD, h attrs

    i_h_group_data = CRF.tools.get_group_data(i_h_data, [len(i_dom)+1,])
    print(i_h_group_data.shape)

    i_h_dom_dict = i_dom.dict.copy()
    i_attr_num = len(i_dom)
    for attr in h_dom.dict:
        i_h_dom_dict[attr+i_attr_num] = h_dom.dict[attr]
    i_h_dom = Domain(i_h_dom_dict, list(range(len(i_h_dom_dict))))
    print('i_h_dom:')
    print(i_h_dom)

    # get size 1 households and other size households
    size1_i_h_data = []
    other_i_h_group_data = []
    for group in i_h_group_data:
        if group.shape[0] == 1:
            size1_i_h_data.append(group)
        else:
            other_i_h_group_data.append(group)

    if len(size1_i_h_data) > 0:
        size1_i_h_data = np.array(size1_i_h_data)
        size1_i_h_data = np.concatenate(size1_i_h_data, axis=0)
        print(size1_i_h_data[:10])
        size1_i_h_data = np.concatenate([size1_i_h_data[:, 1:i_attr_num+1], size1_i_h_data[:, 1+i_attr_num+1:]], axis=1)
        print(size1_i_h_data[:10])

    other_i_h_group_data = np.array(other_i_h_group_data, dtype=object)
    other_i_h_data = np.concatenate(other_i_h_group_data, axis=0)

    other_i_group_data = np.array([group[:, :i_attr_num+2] for group in other_i_h_group_data], dtype=object)
    # INDIVIDUAL, i attrs, HOUSEHOLD
    other_i_data = np.concatenate(other_i_group_data, axis=0)
    other_h_data = np.array([group[0, i_attr_num+1:] for group in other_i_h_group_data], dtype=int)
    # HOUSEHOLD, h_attrs, group_size
    print(i_attr_num)
    print(other_i_data[:10])
    print(other_h_data[:10])

    # size1 households: join individual and household table, learn an MRF
    if len(size1_i_h_data) > 0:
        config = {
            'data_name':        data_name,
            'epsilon':          epsilon,
            'exp_name':         exp_name,

            'budget':           total_budget,

            'print_interval':   200,
            'max_measure_attr_num': 5,
            'sensitivity':      6,

            'save_model':       False,
        }
        size1_model = PrivMRF.run(size1_i_h_data, i_h_dom, config['budget'], p_config=config)
        syn_size1_i_h_data = size1_model.synthetic_data(data_len = size1_model.noisy_data_num)

    # other size households, learn CRF and MRF.
    config = {
        'data_name':            data_name,
        'epsilon':              epsilon,
        'exp_name':             exp_name,

        'budget':               total_budget * i_budget,
        
        'group_change_num':         0.5,
        'tuple_insert_delete_num':  6,

        'EM_group_size':        6,
        'max_group_size':       6,
        'syn_group_size':       15,

        'marginal_step_num':    10,

        'enable_structure_learning':    True,
        'init_EM_step_num':     5,

        # 'enable_structure_learning':    False,
        # 'init_EM_step_num':     5,
        'ob_iter_num':          1000,

        'save_model':           False,
        'IPUMS':                True,
        'size1_type0':          False,

        # 'max_latent_var_num':   1,
        # 'max_latent_var_size':  100
    }
    
    i_model = CRF.run(config, i_dom, other_i_data[:, 1:-1], other_i_group_data)
    # i_model = CRF.conditional_random_field.ConditionalRandomField.load_model('./temp/'+config['exp_name']+'.pkl')
    i_latent_var_num = len(i_model.q.shape) - 1

    argmax_q, q_h_data, _, q_h_dom = CRF.tools.concatenate_q_group(
        i_model.q, other_i_group_data, other_h_data, h_dom)
    # q_h_data: HOUSEHOLD, h_attrs, group_type

    config = {
        'data_name':        data_name,
        'epsilon':          epsilon,
        'exp_name':         exp_name,

        'budget':           total_budget * h_budget,

        'print_interval':   200,
        'max_measure_attr_num': 5,

        'save_model':       False,
    }

    print('h_dom:')
    print(h_dom)
    print('q_h_dom:')
    print(q_h_dom)
    print('q_h_data:')
    print(q_h_data.shape)
    print(q_h_data[:30])
    h_model = PrivMRF.run(q_h_data[:, 1:], q_h_dom, config['budget'], p_config=config)

    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    # sample size1 data
    group_id = 0
    if len(size1_i_h_data) > 0:
        syn_size1_i_h_data = size1_model.synthetic_data(data_len = size1_model.noisy_data_num)
        syn_size1_i_data = proc_back(syn_size1_i_h_data[:, :i_attr_num], new_col_list, original_i_dom)
        syn_size1_h_data = syn_size1_i_h_data[:, i_attr_num:]
        group_id = len(syn_size1_i_data)
        syn_size1_i_data = np.concatenate([np.arange(group_id).reshape((-1, 1)), syn_size1_i_data, np.arange(group_id).reshape((-1, 1))], axis=1)
        syn_size1_h_data = np.concatenate([np.arange(group_id).reshape((-1, 1)), syn_size1_h_data], axis=1)
        print('syn_size1_h_data:')
        print(syn_size1_h_data.shape)
        print(syn_size1_h_data)

    # sample other size data
    i_type_in_h = list(range(len(h_dom), len(q_h_dom)))
    i_type_in_i = i_model.domain.get_attr_by({'latent': True})

    syn_other_h_data = h_model.synthetic_data(data_len=h_model.noisy_data_num)
    i_syn_type = syn_other_h_data[:, i_type_in_h]
    i_syn_type_hist, _ = np.histogramdd(i_syn_type, bins=i_model.latent_domain.edge())

    syn_other_i_data = i_model.syn_FK(i_model.observed_domain.attr_list, types=i_syn_type_hist).to_numpy()

    syn_other_i_data = syn_other_i_data[np.lexsort(syn_other_i_data[:, i_type_in_i].T, axis=0)]
    syn_other_h_data = syn_other_h_data[np.lexsort(syn_other_h_data[:, i_type_in_h].T, axis=0)]
    syn_other_i_group_data = CRF.tools.get_group_data(syn_other_i_data, [-1,])
    assert(len(syn_other_i_group_data) == len(syn_other_h_data))
    print('syn_other_i_group_data:')
    print(syn_other_i_group_data.shape)
    print(syn_other_i_group_data)

    h_id = 0
    for i in range(len(syn_other_i_group_data)):
        syn_other_i_group_data[i][:, -1] = h_id + group_id
        h_id += 1

    syn_other_i_data = np.concatenate(syn_other_i_group_data, axis=0)
    syn_other_i_group_id = syn_other_i_data[:, [-1,]]
    syn_other_i_data = proc_back(syn_other_i_data[:, :len(i_model.observed_domain)], new_col_list, original_i_dom)
    syn_other_i_data = np.concatenate([np.arange(len(syn_other_i_data)).reshape((-1, 1)) + group_id, syn_other_i_data, syn_other_i_group_id], axis=1)

    syn_other_h_data = np.concatenate([np.arange(len(syn_other_h_data)).reshape((-1, 1)) + group_id, syn_other_h_data], axis=1)
    print('syn_other_h_data:')
    print(syn_other_h_data.shape)
    print(syn_other_h_data)

    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    if len(size1_i_h_data) > 0:
        syn_i_data = np.concatenate([syn_size1_i_data, syn_other_i_data], axis=0)
        syn_h_data = np.concatenate([syn_size1_h_data, syn_other_h_data[:, :len(h_dom)+1]], axis=0)
    else:
        syn_i_data = syn_other_i_data.copy()
        syn_h_data = syn_other_h_data[:, :len(h_dom)+1]
    print('syn_h_data:')
    print(syn_h_data.shape)
    print(syn_h_data)

    # write data
    col = ['INDIVIDUAL']
    col.extend(i_attrs)
    col.append('HOUSEHOLD')
    i_syn_df = pd.DataFrame(syn_i_data, columns=col)
    print('write', './temp/'+config['exp_name']+'_individual.csv')
    i_syn_df.to_csv('./temp/'+config['exp_name']+'_individual.csv', index=False)

    col = ['HOUSEHOLD']
    col.extend(h_attrs)
    h_syn_df = pd.DataFrame(syn_h_data, columns=col)
    print('write', './temp/'+config['exp_name']+'_household.csv')
    h_syn_df.to_csv('./temp/'+config['exp_name']+'_household.csv', index=False)

    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)