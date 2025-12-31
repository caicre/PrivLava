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
import pandas as pd
import numpy as np

from . import CRF_tools
from multiprocessing import Pool
import itertools
from . import tools
import networkx as nx
from collections import deque
import sys
from .domain import Domain
import random
import time

def get_random_marginal_TVD(data1, data2, domain, size=3, k=300):
    attrs = domain.attr_list
    # print('?')

    marginal_list = list(itertools.combinations(attrs, size))
    # print(attrs, size, k, marginal_list)
    random.shuffle(marginal_list)
    marginal_list = marginal_list[:k]

    # # print(domain)
    # print(data1[:5])
    # print(data2[:5])

    res = 0
    # debug
    # marginal_list = [(3, 12)]
    for marginal in marginal_list:
        hist1 = tools.get_histogram(marginal, data1, domain)
        hist2 = tools.get_histogram(marginal, data2, domain)
        # print(hist1.flatten()[:5])
        # print(hist2.flatten()[:5])
        tvd = tools.get_normalized_TVD(hist1, hist2)
        res += tvd

        # print(marginal)
        # print(hist1.astype(int))
        # print(hist2.astype(int))
        # print('')

        # print('{}, tvd: {:.4f}'.format(magrinal, tvd))

    res /= len(marginal_list)
    return res

class Node:
    def __init__(self, row_list, target_list, attr_val):
        assert(len(row_list) == len(target_list))
        self.row_list = row_list.copy()
        self.target_list = target_list.copy()
        self.attr_val = attr_val
        self.children = {}

    def gen_children(self, domain, data_without_FK, sampled_data, attr, print_flag=False):

        val_idx1 = [ np.array([], dtype=int) for i in range(domain.dict[attr]['size'])]
        val_idx2 = [ np.array([], dtype=int) for i in range(domain.dict[attr]['size'])]
        assignment_idx = [ np.array([], dtype=int) for i in range(domain.dict[attr]['size'])]
        res_idx = [ np.array([], dtype=int) for i in range(domain.dict[attr]['size'])]

        assert(sampled_data.shape[0] == data_without_FK.shape[0])
        if print_flag:
            print('check data')
            print(data_without_FK[self.row_list[:5]])
            print(sampled_data[self.target_list[:5]])

            u, ucnts = np.unique(data_without_FK[:, attr], return_counts=True)
            print('data_without_FK attr uniques:')
            print(u)
            print(ucnts)
            print('')

            u, ucnts = np.unique(sampled_data[:, attr], return_counts=True)
            print('sampled_data attr uniques:')
            print(u)
            print(ucnts)
            print('')

        i_data_without_FK = data_without_FK[self.row_list]
        i_sampled_data = sampled_data[self.target_list]

        if print_flag:
            u, ucnts = np.unique(i_data_without_FK[:, attr], return_counts=True)
            print('i_data_without_FK attr uniques:')
            print(u)
            print(ucnts)
            print('')

            u, ucnts = np.unique(i_sampled_data[:, attr], return_counts=True)
            print('i_sampled_data attr uniques:')
            print(u)
            print(ucnts)
            print('')

        for val in range(domain.dict[attr]['size']):
            idx1 = np.where(i_data_without_FK[:, attr] == val)[0]
            idx2 = np.where(i_sampled_data[:, attr] == val)[0]
            val_idx1[val] = idx1
            val_idx2[val] = idx2
            np.random.shuffle(val_idx2[val])

            if print_flag:
                print(val, len(idx1), len(idx2))
        
            if len(idx1) < len(idx2):
                assignment_idx[val] = idx2[:len(idx1)]
                res_idx[val] = idx2[len(idx1):]
            else:
                assignment_idx[val] = idx2

        # children_list = np.array([len(val_idx1[val]) > 0 for val in range(domain.dict[attr]['size'])])
        # if sum(children_list) <= 1:
        #     return 0

        res_idx = np.concatenate(res_idx)
        res_size = len(res_idx)
        np.random.shuffle(res_idx)
        if print_flag:
            print('res_size:', res_size)
            print('')

        for val in range(domain.dict[attr]['size']):
            size1 = len(val_idx1[val])
            size2 = len(assignment_idx[val])
            if size1 > size2:
                assert(size1-size2 <= len(res_idx))
                assignment_idx[val] = np.concatenate([assignment_idx[val], res_idx[:size1 - size2]])
                res_idx = res_idx[size1 - size2:]

        assert(len(res_idx) == 0)

        debug_set = []
        if print_flag:
            print('attribute values assignments')
        for val in range(domain.dict[attr]['size']):
            # print('attr', attr, 'last val:', self.attr_val, 'val:', val, len(val_idx1[val]), len(assignment_idx[val]), val_idx1[val][:5], assignment_idx[val][:5])
            if print_flag:
                print('attr', attr, 'last val:', self.attr_val, 'val:', val, len(val_idx1[val]), len(assignment_idx[val]))

            # print(val_idx1[val])
            abs_rows = self.row_list[val_idx1[val]]
            abs_targets = self.target_list[assignment_idx[val]]

            if len(val_idx1[val]) > 0:
                self.children[val] = Node(abs_rows, abs_targets, val)
                # print(sampled_data.shape)
                sampled_data[abs_targets, attr] = val

                debug_set.extend(abs_targets)
        
        debug_set = set(debug_set)
        assert(len(debug_set) == len(self.row_list))

        return res_size

def expand_size_by_type(type_size, type_data, latent_domain, domain, clean_type_size):

    group_list = []
    type0 = tuple([0,]*len(latent_domain))

    # print('type_size[type0]', type_size[type0])

    latent_type_list = [list(range(i)) for i in latent_domain.shape]
    for latent_type in itertools.product(*tuple(latent_type_list)):

        if latent_type == type0 and clean_type_size:
            continue

        temp_latent_type = np.zeros(shape=(1, len(domain)), dtype=int)
        temp_latent_type[0, latent_domain.attr_list] = np.array(latent_type)

        size_array = tools.expand_int_prob(type_size[latent_type])
        size_array += 1

        for size in size_array:
            group = np.repeat(temp_latent_type, size, axis=0)
            group_list.append(group)

    if clean_type_size:
        size1_group = np.zeros(shape=(type_size[type0][0], len(domain)), dtype=int)
    else:
        size1_group = np.zeros(shape=(0, len(domain)), dtype=int)

    return group_list, size1_group

def get_data(size1_group, group_list):
    data = np.concatenate(group_list, axis=0)
    data = np.concatenate([size1_group, data], axis=0)

    return data


def get_random_partial_TVD(data1, data2, domain, int_attrs):
    # print(data1.shape, data2.shape, int_attrs)
    temp_data1 = data1[:, int_attrs]
    temp_data2 = data2[:, int_attrs]

    temp_domain = domain.project(int_attrs)
    temp_domain = temp_domain.moveaxis(int_attrs)

    temp_dict = {attr: temp_domain.dict[int_attrs[attr]] for attr in range(len(int_attrs))}
    temp_domain = Domain(temp_dict, list(range(len(int_attrs))))

    tvd = get_random_marginal_TVD(temp_data1, temp_data2, temp_domain, k=10, size=min(3, temp_data1.shape[1]))
    
    return tvd

def generate_next_level(check_list, domain, attr, data_without_FK, group_list, size1_group):

    sampled_data = get_data(size1_group, group_list)
    size1_length = len(size1_group)

    # print('sampled data:')
    # print(sampled_data[:5])
    # print(sampled_data[size1_length-1: size1_length+5])
    # print('')

    level_res_size = 0
    node_idx = 0
    for node in check_list:
        temp_attr_data = sampled_data[:, attr].copy()
        print_flag = False
        # if attr == 3 and node_idx == 212:

        #     marginal = [0,]
        #     hist1 = tools.get_histogram(marginal, data_without_FK, domain)
        #     hist2 = tools.get_histogram(marginal, sampled_data, domain)
        #     print(marginal, 'tvd', tools.get_normalized_TVD(hist1, hist2))
        #     print_flag = True

        #     marginal = [0, 2]
        #     hist1 = tools.get_histogram(marginal, data_without_FK, domain)
        #     hist2 = tools.get_histogram(marginal, sampled_data, domain)
        #     print(marginal, 'tvd', tools.get_normalized_TVD(hist1, hist2))
        #     print_flag = True

        #     marginal = [0, 2, 8]
        #     hist1 = tools.get_histogram(marginal, data_without_FK, domain)
        #     hist2 = tools.get_histogram(marginal, sampled_data, domain)
        #     print(marginal, 'tvd', tools.get_normalized_TVD(hist1, hist2))
        #     print_flag = True

        #     marginal = [0, 2, 3, 8]
        #     hist1 = tools.get_histogram(marginal, data_without_FK, domain)
        #     hist2 = tools.get_histogram(marginal, sampled_data, domain)
        #     print(marginal, 'tvd', tools.get_normalized_TVD(hist1, hist2))
            
        temp_res_size = node.gen_children(domain, data_without_FK, sampled_data, attr, print_flag)
        level_res_size += temp_res_size
        # if attr == 3:
        #     if len(node.row_list) > 48000 and temp_res_size > 48000:
        #         sampled_data[:, attr] = temp_attr_data
        #         print('node', node_idx, node.attr_val, len(node.row_list), 'size', temp_res_size)
        #         temp_res_size = node.gen_children(domain, data_without_FK, sampled_data, attr, print_flag=True)

        # node_idx += 1
    print('level_res_size:', level_res_size)

    print('modified sampled_data')
    print(sampled_data[:5])
    print(sampled_data[size1_length-1: size1_length+5])

    # # # debug, compare them
    # # print(data_without_FK.shape)
    # # print(sampled_data.shape)
    # hist1, _ = np.histogram(data_without_FK[:, attr], bins=list(range(domain.dict[attr]['size']+1)))
    # hist2, _ = np.histogram(sampled_data[:, attr], bins=list(range(domain.dict[attr]['size']+1)))
    # # print(hist1)
    # # print(hist2)
    # print('attr value TVD: {:.4f}'.format(tools.get_normalized_TVD(hist1, hist2)))

    res_check_list = []
    for node in check_list:
        res_check_list.extend(node.children.values())

    idx = len(size1_group)
    res_size1_group = sampled_data[:idx]
    res_group_list = []
    for group in group_list:
        length = len(group)
        res_group_list.append(sampled_data[idx: idx+length])
        idx += length
    
    assert(idx == len(sampled_data))

    return res_check_list, res_group_list, res_size1_group

def sample_next_attr(domain, latent_domain, group_list, size1_group, clique_marginal, cond_attr, target, pool):
    group_list = [pd.DataFrame(group, columns=domain.attr_list) for group in group_list]

    if len(size1_group) > 0:
        size1_group = pd.DataFrame(size1_group, columns=domain.attr_list)

    attr_list  = cond_attr + [target,]
    marginal_value = clique_marginal.project(attr_list)
    # if target == 3:
    #     print('marginal', marginal_value.domain.attr_list)
    #     print(marginal_value.values.shape)
    #     print(marginal_value.project([3]).values)

    #     temp_marginal_value = marginal_value.project([0, 2, 3, 8])
    #     slc = [2, 1, slice(None), 2]
    #     slc = tuple(slc)
    #     print(temp_marginal_value.values.shape)
    #     print(temp_marginal_value.values[slc])
    marginal_value = marginal_value.moveaxis(attr_list)
    # if target == 3:
    #     print('')
    #     print(marginal_value.values.shape)
    #     print(marginal_value.project([3]).values)

    #     temp_marginal_value = marginal_value.project([0, 2, 3, 8])
    #     slc = [2, 1, 2, slice(None)]
    #     slc = tuple(slc)
    #     print(temp_marginal_value.values.shape)
    #     print(temp_marginal_value.values[slc])
    marginal_value = marginal_value.values

    res_list = []
    if len(size1_group) > 0:
        res_list.append(pool.apply_async(
                sample_attr,
                (size1_group, marginal_value, cond_attr, target)
            )
        )

    # this is for the IPUMS dataset, in which there must be exactly one householder
    # in each household.
    if target == 0:
        assert(latent_domain.attr_list == cond_attr)
        marginal_value = marginal_value.copy()
        slc = [slice(None),]*len(cond_attr)
        slc.append(0)
        slc = tuple(slc)
        marginal_value[slc] = 0

        group_list = [group.loc[1:] for group in group_list]

    block_size = int(len(group_list)/40 + 1)
    group_list_list = [group_list[idx: idx+block_size] for idx in range(0, len(group_list), block_size)]
    for block in group_list_list:
        res_list.append(pool.apply_async(
                sample_attr_df_list,
                (block, marginal_value, cond_attr, target)
            )
        )

    res_list = [res.get() for res in res_list]
    if len(size1_group) > 0:
        size1_group = res_list[0].to_numpy()
        res_list = res_list[1:]

    group_list = []
    for res in res_list:
        for group in res:
            group = group.to_numpy()
            # for IPUMS householder
            if target == 0:
                # there should exist latent vars only. simply reuse any row as the householder.
                row = group[0].copy()
                row[target] = 0
                row = row.reshape((1, -1))
                
                group = np.concatenate([row, group], axis=0)
            group_list.append(group)


    # print('size1_group', size1_group[:5, target])
    # print('group_list', group_list[0])
    # print('group_list', group_list[1])
    # print('group_list', group_list[2])
    # print('group_list', group_list[3])

    # if target == 3:
    #     temp_data = np.concatenate(group_list, axis=0)
    #     pos1 = temp_data[:, 0] == 2
    #     pos2 = temp_data[:, 2] == 1
    #     pos3 = temp_data[:, 8] == 2
    #     temp_data = temp_data[pos1 & pos2 & pos3]
    #     print(temp_data[:5])

    #     hist = tools.get_histogram([3], temp_data, domain)
    #     print('sampled hist', hist)

    return group_list, size1_group

def sample_attr_df_list(df_list, marginal_value, cond_attr, target):
    res = []
    for df in df_list:
        df = sample_attr(df, marginal_value, cond_attr, target)
        res.append(df)

    return res

def match_sampled_data_with_FK_types(group_list, size1_group, latent_domain, type_data, FK_array):
    group_num = len(type_data)
    size_group_num = len(group_list)
    assert(len(group_list)+ size1_group.shape[0] == group_num )
    assert(group_num == len(FK_array))

    # size1_group FKs
    type0_pos = np.array([(row == 0).all() for row in type_data])
    size1_FK_array = FK_array[type0_pos]

    assert(np.sum(type0_pos) == size1_group.shape[0])

    # match group_list FKs
    type_data = type_data[~type0_pos]
    FK_array = FK_array[~type0_pos]

    group_type_array = np.array([group[0, latent_domain.attr_list] for group in group_list])

    group_order = np.lexsort(group_type_array.T, axis=0)
    new_group_type_array = group_type_array[group_order]
    new_group_list = group_list[group_order]

    type_order = np.lexsort(type_data.T, axis=0)
    new_type_data = type_data[type_order]
    new_FK_array = FK_array[type_order]

    for i in range(size_group_num):
        assert((new_group_list[i][0, latent_domain.attr_list] == new_type_data[i]).all())

    re_group_order = np.zeros(size_group_num, dtype=int)
    for i in group_order:
        re_group_order[group_order[i]] = i
        
    new_group_list = new_group_list[re_group_order]
    new_type_data = new_type_data[re_group_order]
    new_FK_array = new_FK_array[re_group_order]

    for i in range(size_group_num):
        assert((group_list[i] == new_group_list[i]).all())

    return size1_FK_array, new_FK_array
        

def sample_attr(df, marginal_value, cond_attr, target):
    total = len(df)
    
    if len(cond_attr) == 0:
        assert(0) # never use as the cond_attr contains latent vars at least.
        prob = marginal_value.project(target).values
        df.loc[:, target] = tools.generate_column_data(prob, total)
    else:

        def foo(group):
            idx = group.name
            vals = tools.generate_column_data(marginal_value[idx], group.shape[0])
            group[target] = vals
            return group

        df = df.groupby(list(cond_attr)).apply(foo)

    return df

def match_FK_by_model_naive(model, data_without_FK, type_data, FK_array):
    print('matching FKs')

    type_hist, _ = np.histogramdd(type_data, bins=model.latent_domain.edge())

    group_size = model.noisy_group_size.copy()
    type_size = model.noisy_type_size.copy()
    record_num = len(data_without_FK)

    group_size[group_size<0] = 0
    group_size = CRF_tools.normalize_group_size(group_size, record_num)
    type_size[type_size<0] = 0
    print('size1_type0', model.config['size1_type0'])
    if model.config['size1_type0']:
        type_size = CRF_tools.clean_type_size(type_size, model.latent_domain)
    assert((type_size >= 0).all())
    print('type_size:')
    print(type_size)
    type_size = CRF_tools.normalize_type_size(type_size, group_size, type_data, model.latent_domain, record_num)
    print('type_size:')
    print(type_size)
    assert((type_size >= 0).all())

    syn_data = model.syn_FK_by(type_hist, FK_array, type_size=type_size)

    syn_data = syn_data[np.lexsort(syn_data[:, :data_without_FK.shape[1]].T, axis=0)]
    order = np.lexsort(data_without_FK.T, axis=0)
    expanded_FK = syn_data[:, -1]

    return order, expanded_FK

def match_data(data1, data2):
    assert(data1.shape == data2.shape)

    order1 = np.lexsort(data1.T[np.arange(data1.shape[1]-1, -1, -1)], axis=0)
    data1 = data1[order1]

    order2 = np.lexsort(data2.T[np.arange(data2.shape[1]-1, -1, -1)], axis=0)
    data2 = data2[order2]

    # print('data1')
    # print(data1)
    # print('data2')
    # print(data2)

    node_list = [((0, len(data1)), list(range(len(data2)))),]
    for col in range(data1.shape[1]):
        new_node_list = []
        print('col', col)
        for data1_seg, data2_rows in node_list:
            # print(data1_seg)
            # print('data2_rows', data2_rows)

            start, end = data1_seg
            unique1 = np.unique(data1[start: end, col])
            
            unique2 = np.unique(data2[data2_rows, col])
            val_to_row_list = {u: [] for u in unique2}
            for row in data2_rows:
                val = data2[row, col]
                val_to_row_list[val].append(row)

            child_node_list = []
            res_list = []
            res_start = 0

            val = data1[start, col]
            child_seg_start = start
            child_seg_end = child_seg_start
            while child_seg_end < end:
                temp_val = data1[child_seg_end, col]
                if temp_val != val:
                    node_size = child_seg_end-child_seg_start

                    if val in val_to_row_list:
                        assign = val_to_row_list[val][: node_size]
                        res  = val_to_row_list[val][node_size: ]
                    else:
                        assign = []
                        res = []

                    child_node_list.append(((child_seg_start, child_seg_end), assign))
                    res_list.extend(res)

                    val = temp_val
                    child_seg_start = child_seg_end

                child_seg_end += 1

            node_size = end-child_seg_start
            if val in val_to_row_list:
                assign = val_to_row_list[val][: node_size]
                res  = val_to_row_list[val][node_size: ]
            else:
                assign = []
                res = []

            child_node_list.append(((child_seg_start, child_seg_end), assign))
            res_list.extend(res)

            for val, rows in val_to_row_list.items():
                if val not in unique1:
                    res_list.extend(rows)

            # print('res_list', res_list)
            for node, rows in child_node_list:
                # print(node, rows)
                rows_size = len(rows)
                if rows_size < node[1] - node[0]:
                    size = node[1] - node[0] - rows_size
                    rows.extend(res_list[res_start: res_start+size])
                    res_start = res_start + size

            assert(res_start == len(res_list))

            new_node_list.extend(child_node_list)

        node_list = new_node_list

    res_data2 = []
    data2_row = []
    for node, rows in node_list:
        # print(node, rows)
        res_data2.append(data2[rows])
        data2_row.extend(rows)
    res_data2 = np.concatenate(res_data2, axis=0)

    order2 = order2[data2_row]

    return order1, order2

def match_FK_by_model(model, data_without_FK, type_data, FK_array):
    print('matching FKs')
    
    type_hist, _ = np.histogramdd(type_data, bins=model.latent_domain.edge())
    # print(len(type_data), len(FK_array), np.sum(type_hist))

    group_size = model.noisy_group_size.copy()
    type_size = model.noisy_type_size.copy()
    record_num = len(data_without_FK)

    group_size[group_size<0] = 0
    group_size = CRF_tools.normalize_group_size(group_size, record_num)
    type_size[type_size<0] = 0
    print('size1_type0', model.config['size1_type0'])
    if model.config['size1_type0']:
        type_size = CRF_tools.clean_type_size(type_size, model.latent_domain)
    assert((type_size >= 0).all())
    print('type_size:')
    print(type_size)
    type_size = CRF_tools.normalize_type_size(type_size, group_size, type_data, \
        model.latent_domain, record_num, size1_type0=model.config['size1_type0'])
    assert((type_size >= 0).all())

    # print('final:')
    # print(data_without_FK.shape)
    # print(np.sum(type_hist), len(FK_array), np.sum(type_size))
    # temp = np.sum(type_size, axis=tuple(range(len(type_size.shape) - 1)))
    # print(type_size.shape, tuple(range(len(type_size.shape) - 1)))
    # print(temp)
    # print(CRF_tools.get_group_size_record_num(temp))

    # # debug
    # type_hist = (type_hist/100).astype(int)
    # FK_array = FK_array[: int(len(FK_array)/100)]
    # type_size = (type_size/100).astype(int)
    # print('type_hist')
    # print(type_hist)
    

    syn_data = model.syn_FK_by(type_hist, FK_array, type_size=type_size)
    df = pd.DataFrame(syn_data, columns=list(range(syn_data.shape[1])))
    df.to_csv('./temp/match_syn_data.csv', index=False)
    
    # # debug
    # data_without_FK = data_without_FK[:len(syn_data)]
    # print(data_without_FK[:10])

    # print(syn_data.shape)
    order, order2 = match_data(data_without_FK, syn_data[:, 1:data_without_FK.shape[1]+1])
    syn_data = syn_data[order2]
    expanded_FK = syn_data[:, -1]

    # # debug
    # data_without_FK = data_without_FK[order]
    # data_without_FK = np.concatenate([data_without_FK, expanded_FK.reshape((-1, 1))], axis=1)
    # data_without_FK = data_without_FK[np.argsort(data_without_FK[:, -1])]
    # df = pd.DataFrame(data_without_FK, columns=list(range(data_without_FK.shape[1])))
    # df.to_csv('./temp/match_syn_data2.csv', index=False)

    return order, expanded_FK

def gen_FK(model, data_without_FK, domain, type_data, FK_array, clean_type_size=True):

    # ##############

    # np.random.shuffle(data_without_FK)
    # data_without_FK = data_without_FK[: int(len(data_without_FK)/100)]

    # order = np.arange(len(type_data))
    # np.random.shuffle(order)
    # type_data = type_data[order]
    # print(type_data.shape, FK_array.shape)
    # print(order)
    # FK_array = FK_array[order]

    # type_data = type_data[: int(len(type_data)/100)]
    # order = np.lexsort((type_data[:, 1], type_data[:, 0]))
    # type_data = type_data[order]

    # FK_array = FK_array[: int(len(FK_array)/100)]
    # FK_array = FK_array[order]

    # # print(data_without_FK[:10])
    # pos = 870
    # print(type_data[pos: pos+20])
    # print(FK_array[pos: pos+20])

    # ############

    print('generating foreign keys')
    clique_marginal, partition_func = CRF_tools.belief_propagation(model.message_order, \
            model.potential, total=1)
    clique_marginal.to_cpu()
    pool = Pool(processes=40)

    latent_domain = model.latent_domain
    record_num = len(data_without_FK)
    print('record_num: {:d}'.format(record_num))
    print('group_num: {:d}'.format(len(type_data)))

    group_size = model.noisy_group_size
    type_size = model.noisy_type_size

    group_size[group_size<0] = 0
    group_size = CRF_tools.normalize_group_size(group_size, record_num)
    # # debug
    # print('group_size')
    # print(group_size)
    type_size[type_size<0] = 0
    if clean_type_size:
        type_size = CRF_tools.clean_type_size(type_size, latent_domain)
    assert((type_size >= 0).all())
    type_size = CRF_tools.normalize_type_size(type_size, group_size, type_data, latent_domain, record_num)
    assert((type_size >= 0).all())
    type_size[type_size<0] = 0

    # debug
    print(type_size.shape)
    temp = np.sum(type_size, axis=tuple(range(len(type_size.shape) - 1)))
    print(temp.shape)
    total = CRF_tools.get_group_size_record_num(temp)
    print('total: {:d}'.format(total))
    
    group_list, size1_group = expand_size_by_type(type_size, type_data, latent_domain, domain, clean_type_size)

    print('type_size:')
    print(type_size)
    # print(group_list[5])
    # len_list = np.array([len(group) for group in group_list])
    # print(np.where(len_list == 0))
    # print(len_list[:5])

    total1 = sum([len(group) for group in group_list])
    total2 = len(size1_group)
    total = total1 + total2
    # print(total1, total2)
    print('total: {:d}, record: {:d}'.format(total, record_num))
    assert(total == record_num)
    assert(len(data_without_FK) == record_num)

    root = Node(np.arange(record_num), np.arange(record_num), None)
    check_list = [root,]

    print('sampling attrs')
    # sample data in data_with_type
    for clique in model.maximal_cliques:
        # start form RELATE
        if 0 in clique:
            start_clique = clique
            break
    finished_attr = latent_domain.attr_list.copy()
    cond_attr = finished_attr.copy()
    attr_num = len(domain)

    print('clique:', start_clique)
    for attr in start_clique:
        if attr in finished_attr:
            continue
        print('  {}->{} {}/{}'.format(cond_attr, attr, len(finished_attr), attr_num))
        group_list, size1_group = sample_next_attr(domain, latent_domain, group_list, size1_group, clique_marginal[start_clique], cond_attr, attr, pool)
        check_list, group_list, size1_group = generate_next_level(check_list, domain, attr, data_without_FK, group_list, size1_group)
        cond_attr.append(attr)
        finished_attr.append(attr)

        # # debug
        # sampled_data = get_data(size1_group, group_list)
        # int_attrs = finished_attr.copy()
        # int_attrs.remove(16)
        # int_attrs.remove(17)
        # if len(int_attrs) >= 2:
        #     tvd = get_random_partial_TVD(data_without_FK, sampled_data, domain, int_attrs)
        #     print('marginal TVD: {:.4e}'.format(tvd))

        if len(check_list) == 0:
            break

    if len(check_list) != 0:
        if len(model.maximal_cliques) > 1:
            for start, clique in nx.dfs_edges(model.junction_tree, source=start_clique):
                cond_attr = sorted(list(set(start) & set(clique)))
                print('clique:', clique)
                for attr in clique:
                    if attr in finished_attr:
                        continue
                    if len(finished_attr) > 10:
                        break
                    print('  {}->{} {}/{}'.format(cond_attr, attr, len(finished_attr), attr_num))
                    localtime = time.asctime(time.localtime(time.time()))
                    print ("local time:", localtime)
                    group_list, size1_group = sample_next_attr(domain, latent_domain, group_list, size1_group, clique_marginal[clique], cond_attr, attr, pool)
                    check_list, group_list, size1_group = generate_next_level(check_list, domain, attr, data_without_FK, group_list, size1_group)
                    cond_attr.append(attr)
                    finished_attr.append(attr)

                    # # debug
                    # sampled_data = get_data(size1_group, group_list)
                    # int_attrs = finished_attr.copy()
                    # int_attrs.remove(16)
                    # int_attrs.remove(17)

                    # tvd = get_random_partial_TVD(data_without_FK, sampled_data, domain, int_attrs)
                    # print('marginal TVD: {:.4f}'.format(tvd))

                    if len(check_list) == 0:
                        break
    
    print('generating orders')
    check_queue = deque([root,])
    order_array = np.full(record_num, fill_value=-1)
    while check_queue:
        node = check_queue.popleft()
        if len(node.children) > 0:
            for child in node.children.values():
                check_queue.append(child)
        else:
            order_array[node.target_list] = node.row_list

    group_list = np.array(group_list, dtype=object)
    size1_FK_array, new_FK_array = match_sampled_data_with_FK_types(group_list, \
        size1_group, latent_domain, type_data, FK_array)

    expanded_FK_array = np.full(record_num, fill_value=-1, dtype=int)

    size1_num = len(size1_group)
    expanded_FK_array[:size1_num] = size1_FK_array

    group_len = [len(group) for group in group_list]
    idx = size1_num
    for i in range(len(group_list)):
        length = group_len[i]
        expanded_FK_array[idx: idx+length] = new_FK_array[i]
        idx = idx + length

    return order_array, expanded_FK_array

