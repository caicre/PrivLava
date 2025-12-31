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
import data_proc
import os
import CRF
import PrivMRF
import sys
import pandas as pd
import time
import TPC_H_proc
import csv
from CRF.domain import Domain

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
thread_num = '16'
os.environ["OMP_NUM_THREADS"] = thread_num
os.environ["OPENBLAS_NUM_THREADS"] = thread_num
os.environ["MKL_NUM_THREADS"] = thread_num
os.environ["VECLIB_MAXIMUM_THREADS"] = thread_num
os.environ["NUMEXPR_NUM_THREADS"] = thread_num

if __name__ == '__main__':
    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    epsilon = float(sys.argv[2])
    exp_name = sys.argv[1] + '_TPC_{:.2f}'.format(epsilon)

    o_budget = 0.6
    o_budget_l = o_budget * 0.33
    o_budget_o = o_budget * 0.33
    o_budget_c = o_budget * 0.33

    # ps_budget = 0.6
    ps_budget = 0.4
    ps_budget_l = ps_budget * 0.4
    ps_budget_ps = ps_budget * 0.6


    config = {}
    config['data_name'] = 'TPC-H'
    config['epsilon'] = epsilon
    config['exp_name'] = exp_name

    ############################################
    # read_data

    # lineitem, attrs - L_ORDERKEY - L_PARTKEY - L_SUPPKEY
    lineitem, lineitem_domain, lineitem_attrs = data_proc.read_table('./data/TPC-H/lineitem.csv', \
        './data/TPC-H/lineitem_domain.json', id_col='LINEITEMKEY', all_attrs=True)
    print('lineitem:', lineitem.shape, len(lineitem_domain))
    print(lineitem[:5])

    # orders, O_ORDERKEY - order_attrs - O_CUSTKEY
    order, order_domain, order_attrs = data_proc.read_table('./data/TPC-H/orders.csv', \
        './data/TPC-H/orders_domain.json', all_attrs=True)
    print('order:', order.shape, len(order_domain))
    print(order_attrs)
    print(order[:5])


    order_date = order[:, [0, 2, 3, 4, 5, 1]]
    order_date_domain = Domain({0: {'size': 7}, 1: {'size': 12}, 2: {'size': 31}, 3: {'size': 5}, 4: {'size': 3}}, [0, 1, 2, 3, 4])
    print(order_date[:5])

    order_attrs = ['O_ORDERKEY', 'O_ORDERDATE1', 'O_ORDERSTATUS', 'O_CUSTKEY']
    order = order[:, [0, 2, 1, 6]]
    order_domain = Domain({0: {'size': 7}, 1: {'size': 3} }, [0, 1])
    print(order_attrs)
    print(order[:5])

    # customer, CUSTKEY - attrs
    customer, customer_domain, customer_attrs = data_proc.read_table('./data/TPC-H/customer.csv', \
        './data/TPC-H/customer_domain.json', all_attrs=True)
    customer = customer[np.argsort(customer[:, 0])]
    print('customer:', customer.shape, len(customer_domain))
    print(customer[:5])

    # part, P_PARTKEY - attrs
    part, part_domain, part_attrs = data_proc.read_table('./data/TPC-H/part.csv', \
        './data/TPC-H/part_domain.json', id_col='P_PARTKEY')
    print('part:', part.shape, len(part_domain))
    print(part[:5])

    # supplier, S_SUPPKEY - attrs
    supplier, supplier_domain, supplier_attrs = data_proc.read_table('./data/TPC-H/supplier.csv', \
        './data/TPC-H/supplier_domain.json', id_col='S_SUPPKEY')
    print('supplier:', supplier.shape, len(supplier_domain))
    print(supplier[:5])

    # partsupp, attrs
    partsupp, partsupp_domain, _ = data_proc.read_table('./data/TPC-H/partsupp.csv', \
        './data/TPC-H/partsupp_domain.json', all_attrs=True)
    assert(np.max(partsupp[:, 1]) == np.max(supplier[:, 0]))
    max_supplier = np.max(partsupp[:, 1]) + 1
    print('partsupp:', partsupp.shape, len(partsupp_domain))
    print(partsupp[:5])
    print('max_supplier', max_supplier)
    print('max_part', np.max(partsupp[:, 0]))
    partsupp_id = partsupp[:, 0] * max_supplier + partsupp[:, 1]
    partsupp = np.concatenate([partsupp_id.reshape((-1, 1)), partsupp[:, 2:]], axis=1)

    delta = 1 / len(lineitem)
    # total_budget = 1e9 # debug
    total_budget = CRF.tools.get_privacy_budget(epsilon, delta)
    # total_budget = 44
    # total_budget = 0.0236

    print('total_budget: {:.8f}'.format(total_budget))

    #######################################################################################
    # lineitem - order
    lineitem_o = lineitem[:, :-2].copy()
    lineitem_o = lineitem_o[np.argsort(lineitem_o[:, -1])]
    
    lineitem_o_group = CRF.tools.get_group_data(lineitem_o, group_id_attrs=[-1,])

    temp_config = config.copy()
    temp_config['budget'] = o_budget_l * total_budget
    temp_config['model_type'] = 'native'
    temp_config['iter_num'] = 500
    temp_config['ob_iter_num'] = 1000
    temp_config['EM_group_size'] = 7
    temp_config['max_group_size'] = 7
    temp_config['syn_group_size'] = 7
    temp_config['marginal_step_num'] = 3
    temp_config['max_type_num'] = 400
    temp_config['max_latent_var_size'] = 20
    temp_config['exp_name'] = exp_name + '_lineitem'
    temp_config['size1_type0'] = False
    temp_config['save_model'] = False

    temp_config['group_change_num'] = 0.5
    temp_config['tuple_insert_delete_num'] = 7

    temp_config['max_clique_size'] = 5e6
    temp_config['max_parameter_size'] = 5e7

    temp_config['enable_structure_learning'] = False
    temp_config['init_EM_step_num'] = 7

    print('learning lineitem-order CRF')

    l_model = CRF.run(temp_config, lineitem_domain, lineitem_o[:, 1:-1], lineitem_o_group)
    # l_model = CRF.conditional_random_field.ConditionalRandomField.load_model('./temp/'+temp_config['exp_name']+'.pkl')
    l_latent_var_num = len(l_model.q.shape) - 1
    l_latent_var_size = l_model.q.shape[1]

    argmax_q, q_o_data, _, q_o_domain = CRF.tools.concatenate_q_group(
        l_model.q, lineitem_o_group, order[:, :-1], order_domain, type_first=True)
    o_type_in_o = list(range(len(q_o_domain) - len(order_domain)))

    q_o_data = np.concatenate([q_o_data, order[:, -1].reshape((-1, 1))], axis=1)
    q_o_data = q_o_data[np.argsort(q_o_data[:, -1])]

    q_o_group = CRF.tools.get_group_data(q_o_data, group_id_attrs=[-1,])

    print('q_o_data:', q_o_data.shape)
    print(q_o_data[:10])
    print('q_o_domain:')
    print(q_o_domain)

    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    ########################################################################################
    # order - customer

    o_budget_o_crf = 0.5
    temp_config = config.copy()
    temp_config['budget'] = o_budget_o * total_budget * o_budget_o_crf
    temp_config['model_type'] = 'native'
    temp_config['ob_iter_num'] = 1000
    temp_config['init_EM_step_num'] = 7
    temp_config['marginal_step_num'] = 5
    temp_config['max_type_num'] = 400
    temp_config['max_latent_var_size'] = 10
    temp_config['enable_structure_learning'] = False
    temp_config['exp_name'] = exp_name + '_o'
    temp_config['max_group_size'] = 10  # waringing: 10 for 2GB, 6 for 1GB
    temp_config['EM_group_size'] = 14   # waringing: 14 for 2GB, 6 for 1GB
    temp_config['theta3'] = 10

    temp_config['group_change_num'] = 1
    temp_config['tuple_insert_delete_num'] = 1
    
    temp_config['syn_group_size'] = 14 # waringing: 14 for 2GB, 6 for 1GB
    temp_config['EM_type_size'] = False
    temp_config['size1_type0'] = False
    temp_config['marginal_step_num'] = 3
    temp_config['iter_num'] = 500
    temp_config['save_model'] = False


    q_o_key_data = q_o_data[:, :1+l_latent_var_num]
    q_o_data = q_o_data[:, 1+l_latent_var_num:]
    q_o_data = np.concatenate([q_o_key_data[:, [0,]], q_o_data], axis=1)
    q_o_group = CRF.tools.get_group_data(q_o_data, group_id_attrs=[-1,])
    q_o_domain_dict = {i: q_o_domain.dict[i+l_latent_var_num].copy() for i in range(len(q_o_domain) - l_latent_var_num)}
    q_o_domain = Domain(q_o_domain_dict, list(range(len(q_o_domain_dict))))

    print('q_o_data:', q_o_data.shape)
    print(q_o_data[:10])
    print('q_o_domain:')
    print(q_o_domain)
    print('order-customer CRF:')

    o_model = CRF.run(temp_config, q_o_domain, q_o_data[:, 1:-1], q_o_group)
    # o_model = CRF.conditional_random_field.ConditionalRandomField.load_model('./temp/'+temp_config['exp_name']+'.pkl')
    o_model.config['IPUMS'] = False
    o_latent_var_num = len(o_model.q.shape) - 1


    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    ########################################################################################
    # customer

    argmax_q, q_c_data, rem_c_data, q_c_domain = CRF.tools.concatenate_q_group(
        o_model.q, q_o_group, customer, customer_domain)

    o_c_config = config.copy()
    o_c_config['print_interval'] = 200
    o_c_config['exp_name'] = exp_name + '_c_mrf'

    o_c_config['theta'] = 4
    o_c_config['sensitivity'] = 2

    print('q_c_domain:')
    print(q_c_domain)
    print('q_c_data:', q_c_data.shape)
    print(q_c_data[:5])
    print('customer MRF:')
    budget = o_budget_c * total_budget
    mrf = PrivMRF.run(q_c_data[:, 1:], q_c_domain, budget, p_config=o_c_config)
    # mrf = PrivMRF.markov_random_field.MarkovRandomField.load_model('./PrivMRF/temp/'+o_c_config['exp_name']+'.mrf')

    c_syn_data = np.full((q_c_data.shape[0], len(q_c_domain)+1), -1, dtype=int)
    c_syn_data[:, :q_c_data.shape[1]] = q_c_data


    c_syn_data = mrf.synthetic_data(c_syn_data[:, 1:], existing_attrs=customer_domain.attr_list, \
        start_clique=mrf.maximal_cliques[0], print_flag=True, existing_error=True)

    c_type_in_c = list(range(len(customer_domain), len(q_c_domain)))
    c_type_hist, _ = np.histogramdd(c_syn_data[:, c_type_in_c], bins=o_model.latent_domain.edge())

    c_type_order = np.lexsort(c_syn_data[:, c_type_in_c].T, axis=0)
    c_syn_data = np.concatenate([q_c_data[:, :-o_latent_var_num], c_syn_data[:, c_type_in_c]], axis=1)
    c_syn_data = c_syn_data[c_type_order]
    print('c_syn_data:')
    print(c_syn_data[:5,])

    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    ########################################################################################
    # synthesize order


    path = './TPC-H_temp/'+config['exp_name']+'_order1.csv'
    order_syn_data = o_model.syn_FK_by(c_type_hist, c_syn_data[:,0])
    col = [order_attrs[0],]
    col.extend(order_attrs[1:-1])
    for i in range(o_latent_var_num):
        col.append('o_group_type'+str(i))
    col.append('O_CUSTKEY')
    print(len(col), col)
    print('order_syn_data:', order_syn_data.shape)
    print(order_syn_data[:10])
    df = pd.DataFrame(order_syn_data, columns=col)

    date_config = config.copy()
    date_config['print_interval'] = 200
    date_config['exp_name'] = exp_name + '_date_mrf'
    date_config['ed_step_num'] = 4
    print('order date:')
    print(order_date_domain)
    print(order_date.shape)
    print(order_date)
    order_date = order_date[np.argsort(order_date[:, 0])]
    q_o_key_data = q_o_key_data[np.argsort(q_o_key_data[:, 0])]
    order_date = np.concatenate([order_date, q_o_key_data[:, 1:]], axis=1)
    for i in range(l_latent_var_num):
        order_date_domain.add_variable(len(order_date_domain), l_latent_var_size)
    print('order date:')
    print(order_date_domain)
    print(order_date.shape)
    print(order_date)
    print('order MRF:')

    budget = o_budget_o * total_budget * (1-o_budget_o_crf)
    mrf = PrivMRF.run(order_date[:, 1:], order_date_domain, budget, p_config=date_config)
    syn_date_data = np.zeros(shape=(len(order_syn_data), len(order_date_domain)), dtype=int)
    syn_date_data[:, 0] = order_syn_data[:, list(df.columns).index('O_ORDERDATE1')]
    syn_date_data = mrf.synthetic_data(syn_date_data, existing_attrs=[0,], \
        start_clique=mrf.maximal_cliques[0], print_flag=True, existing_error=True)
    print('syn_date_data')
    print(syn_date_data.shape)
    print(syn_date_data[:20])

    df['O_ORDERDATE2'] = syn_date_data[:, 1]
    df['O_ORDERDATE3'] = syn_date_data[:, 2]
    df['O_ORDERPRIORITY'] = syn_date_data[:, 3]
    df['O_ORDERSTATUS'] = syn_date_data[:, 4]
    for i in range(l_latent_var_num):
        df['l_group_type'+str(i)] = syn_date_data[:, 4+i+1]
    # for i in range(l_latent_var_num):
    #     cols.insertappend('l_group_type'+str(i))
    
    cols = ['O_ORDERKEY',]
    for i in range(l_latent_var_num):
        cols.append('l_group_type'+str(i))
    cols.extend(['O_ORDERSTATUS', 'O_ORDERDATE1', 'O_ORDERDATE2', 'O_ORDERDATE3', 'O_ORDERPRIORITY'])
    for i in range(o_latent_var_num):
        cols.append('o_group_type'+str(i))
    cols.append('O_CUSTKEY')
    print('write', path)
    df = df[cols]
    df.to_csv(path, index=False)
    order_syn_data = pd.read_csv(path).to_numpy()

    print('order_syn_data:', order_syn_data.shape)
    print(order_syn_data[:10])

    order_syn_data_without_id = order_syn_data[:, 1:]
    # o_type_in_o = list(range(len(order_domain), len(q_o_domain)))
    o_type_hist, _ = np.histogramdd(order_syn_data_without_id[:, o_type_in_o], bins=l_model.latent_domain.edge())
    order_syn_data = order_syn_data[np.lexsort(order_syn_data_without_id[:, o_type_in_o].T, axis=0)]

    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    ########################################################################################
    # synthesize lineitem

    l_col = ['LINEITEMKEY',]
    l_col.extend(lineitem_attrs[:-3])
    for i in range(l_latent_var_num):
        l_col.append('group_type'+str(i))
    l_col.append('L_ORDERKEY')
    
    l_syn_data = l_model.syn_FK_by(o_type_hist, order_syn_data[:,0])
    df = pd.DataFrame(l_syn_data, columns=l_col)
    path = './TPC-H_temp/'+config['exp_name']+'_lineitem1.csv'
    print('write', path)
    df.to_csv(path, index=False)
    l_syn_data = pd.read_csv(path).to_numpy()

    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)
    
    ########################################################################################
    # lineitem partsupp
    path = './TPC-H_temp/'+config['exp_name']+'_lineitem1.csv'
    l_syn_data = pd.read_csv(path).to_numpy()
    # LINEITEMKEY, lineitem attrs, lineitem group type, L_ORDERKEY

    lineitem_price_ps = lineitem[:, [0, 2, 3, 4, 5]].copy()
    partsupp_id = lineitem[:, -2] * max_supplier + lineitem[:, -1]
    assert(np.max(lineitem[:, -1]+1) == max_supplier)
    lineitem_price_ps = np.concatenate([lineitem_price_ps, partsupp_id.reshape((-1, 1))], axis=1)
    lineitem_price_ps = lineitem_price_ps[np.argsort(lineitem_price_ps[:, -1], axis=0)]
    
    lineitem_price_ps_group = CRF.tools.get_group_data(lineitem_price_ps, group_id_attrs=[-1,])
    hist, _ = np.histogram([len(group) for group in lineitem_price_ps_group], bins=list(range(30)))
    print('ps group hist:', len(lineitem_price_ps_group), len(partsupp))
    print(hist)

    ps_l_config = config.copy()
    ps_l_config['budget'] = ps_budget_l * total_budget
    ps_l_config['model_type'] = 'native'
    ps_l_config['iter_num'] = 500
    ps_l_config['ob_iter_num'] = 1000
    ps_l_config['init_EM_step_num'] = 6
    ps_l_config['max_group_size'] = 10
    ps_l_config['syn_group_size'] = 13
    ps_l_config['marginal_step_num'] = 3
    ps_l_config['max_type_num'] = 400
    ps_l_config['EM_type_size'] = False
    ps_l_config['enable_structure_learning'] = False
    ps_l_config['exp_name'] = exp_name + '_ps'
    ps_l_config['size1_type0'] = False
    ps_l_config['save_model'] = True

    ps_l_config['group_change_num'] = 7
    ps_l_config['tuple_insert_delete_num'] = 7
    
    if epsilon < 0.81:
        ps_l_config['max_group_size'] = 7
        ps_l_config['init_EM_step_num'] = 2
        ps_l_config['iter_num'] = 100


    ps_l_config['max_clique_size'] = 5e6
    ps_l_config['max_parameter_size'] = 5e7
    print('learning partsupp CRF')

    ps_l_domain = Domain({0: {'size': 10}, 1: {'size': 10}, 2: {'size': 10}, 3: {'size': 10}}, [0, 1, 2, 3])
    ps_model = CRF.run(ps_l_config, ps_l_domain, lineitem_price_ps[:, 1:-1], lineitem_price_ps_group)
    # ps_model = CRF.conditional_random_field.ConditionalRandomField.load_model('./temp/'+ps_l_config['exp_name']+'.pkl')
    ps_l_domain = ps_model.domain
    ps_latent_var_num = len(ps_model.q.shape) - 1
    type_hist = np.sum(ps_model.noisy_type_size, axis=(-1,))
    print('type_hist:')
    print(type_hist.astype(int))
    print('ps_l_domain:')
    print(ps_l_domain)

    ########################################################################################
    # partsupp

    _, q_ps_data, _, q_ps_domain = CRF.tools.concatenate_q_group(
        ps_model.q, lineitem_price_ps_group, partsupp, partsupp_domain)
    print('q_ps_domain:')
    print(q_ps_domain)
    print('partsupp MRF:')

    ps_ps_config = config.copy()
    ps_ps_config['print_interval'] = 200
    ps_ps_config['exp_name'] = exp_name + '_ps_mrf'
    ps_ps_config['sensitivity'] = 14
    budget = ps_budget_ps * total_budget
    mrf = PrivMRF.run(q_ps_data[:, 1:], q_ps_domain, budget, p_config=ps_ps_config)
    # mrf = PrivMRF.markov_random_field.MarkovRandomField.load_model('./PrivMRF/temp/'+ps_ps_config['exp_name']+'.mrf')

    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    ########################################################################################
    # synthesize partsupp

    # sample part types given part attributes (public ground truth data)
    ps_syn_data = np.full((partsupp.shape[0], len(q_ps_domain)+1), fill_value=-1, dtype=int)
    ps_syn_data[:, :partsupp.shape[1]] = partsupp

    ps_syn_data = mrf.synthetic_data(ps_syn_data[:, 1:], existing_attrs=partsupp_domain.attr_list,\
        start_clique=mrf.maximal_cliques[0], print_flag=True, existing_error=True)

    ps_type_in_ps = list(range(len(partsupp_domain), len(q_ps_domain)))
    # p_type_hist, _ = np.histogramdd(p_syn_data[:, p_type_in_p], bins=p_model.latent_domain.edge())

    # match sampled order, given attributes, order id column
    ps_type_order = np.lexsort(ps_syn_data[:, ps_type_in_ps].T, axis=0)
    ps_syn_type = ps_syn_data[ps_type_order][:, ps_type_in_ps]
    ps_syn_data = np.concatenate([partsupp[:, 0].reshape((-1, 1)), ps_syn_data], axis=1)
    ps_syn_data = ps_syn_data[ps_type_order]


    ########################################################################################
    # synthesize lineitem - partsupp FK
    ps_order, ps_FK = CRF.match_FK_by_model(ps_model, l_syn_data[:, [2, 3, 4, 5]], \
        ps_syn_type, ps_syn_data[:, 0])
    l_syn_data = l_syn_data[ps_order]
    l_syn_data = np.concatenate([l_syn_data, ps_FK.reshape((-1, 1))], axis=1)
    # order_array, expanded_FK_array = CRF.gen_FK.gen_FK(ps_model, l_syn_data[:, 1:-l_latent_var_num-1], \
    #     ps_l_domain, ps_syn_type, ps_syn_data[:, 0], clean_type_size=True)
    # l_syn_data = l_syn_data[order_array]
    # l_syn_data = np.concatenate([l_syn_data, expanded_FK_array.reshape((-1, 1))], axis=1)

    # l_syn_data, lINEITEMKEY - attrs - L_ORDERKEY - L_PARTKEY - L_SUPPKEY
    l_syn_data = l_syn_data[np.argsort(l_syn_data[:, -1])]
    l_syn_group_data =  CRF.tools.get_group_data(l_syn_data, group_id_attrs=[-1,])
    hist, _ = np.histogram([len(group) for group in l_syn_group_data], bins=list(range(30)))
    print('l_syn_group_data', len(l_syn_group_data))
    print(hist)

    part_key, supp_key = np.divmod(l_syn_data[:, -1], max_supplier)
    l_syn_data = np.concatenate([l_syn_data[:, :-1], part_key.reshape((-1, 1)), \
        supp_key.reshape((-1, 1))], axis=1)

    print(l_syn_data.shape)
    print(l_syn_data[:5])

    l_col.extend(['L_PARTKEY', 'L_SUPPKEY'])
    df = pd.DataFrame(l_syn_data, columns=l_col)
    path = './TPC-H_temp/'+config['exp_name']+'_lineitem2.csv'
    print('write', path)
    df.to_csv(path, index=False)

    localtime = time.asctime(time.localtime(time.time()))
    print ("local time:", localtime)

    ########################################################################################
    # postprocess

    reader = csv.reader(open('./input_data/orders.tbl', 'r'), delimiter='|')
    line_list = []
    for line in reader:
        line_list.append(line[:-2])
    gt_data = np.array(line_list)

    path = './TPC-H_temp/'+config['exp_name']+'_order1.csv'
    order_df = pd.read_csv(path)
    order_df = TPC_H_proc.proc_order_back(order_df, gt_data)
    path = './TPC-H_temp/'+config['exp_name']+'_order.csv'
    print('write', path)
    order_df.to_csv(path, index=False)
    order_df = pd.read_csv(path)

    reader = csv.reader(open('./input_data/lineitem.tbl', 'r'), delimiter='|')
    line_list = []
    for line in reader:
        line_list.append(line[:-2])
    gt_data = np.array(line_list)

    path = './TPC-H_temp/'+config['exp_name']+'_lineitem2.csv'
    df = pd.read_csv(path)
    df = TPC_H_proc.proc_lineitem_back(df, order_df, gt_data)
    path = './TPC-H_temp/'+config['exp_name']+'_lineitem.csv'
    print('write', path)
    df.to_csv(path, index=False)



    
