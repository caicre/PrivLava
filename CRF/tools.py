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
from scipy import stats
import json
import networkx as nx
import itertools
import csv
import matplotlib.pyplot as plt
import mpmath as mp
import math
from .cp_factor import Factor
import pandas as pd
import time
from .import domain

mp.mp.dps = 1000


def check_data_domain(data, domain):
    print('checking data domain')
    assert(domain.attr_list == list(range(data.shape[1])))
    assert(data.dtype == int)
    for i in range(data.shape[1]):
        min_v = np.min(data[:, i])
        max_v = np.max(data[:, i])
        size = domain.dict[i]['size'] 
        print(i, size, min_v, max_v, end='')
        if min_v == 0 and max_v+1 == size:
            print('\t OK')
        else:
            print('\t -----------')

# calculate/get mutual information from data
def get_mutual_info(MI_map, entropy_map, index_list, data, domain):
    if not isinstance(index_list, tuple):
        index_list = tuple(sorted(index_list))

    if index_list not in MI_map:

        temp_domain = domain.project(index_list)

        MI = -get_entropy(entropy_map, index_list, data, domain)
        for attr in index_list:
            MI += get_entropy(entropy_map, [attr], data, domain)

        MI_map[index_list] = MI

    return MI_map[index_list]

# calculate/get entropy from data
def get_entropy(entropy_map, index_list, data, domain):
    if not isinstance(index_list, tuple):
        index_list = tuple(sorted(index_list))
    if index_list not in entropy_map:
        temp_domain = domain.project(index_list)
        bins = temp_domain.edge()
        size = temp_domain.size()

        if len(index_list) <= 14 and size < 1e7:
            histogram, _= np.histogramdd(data[:, index_list], bins=bins)
            histogram = histogram.flatten()
            entropy = stats.entropy(histogram)
        else:
            value, counts = np.unique(data[:, index_list], return_counts=True, axis=0)
            entropy = stats.entropy(counts)

        entropy_map[index_list] = entropy

    return entropy_map[index_list]


def get_histogram(index_list, data, domain, weights=None):
    temp_domain = domain.project(index_list)
    histogram, _ = np.histogramdd(data[:, index_list], bins=temp_domain.edge(), \
        weights=weights)
    return histogram

def string_2d_low_precision_array(array):
    string = ''
    for row in array:
        row_string = string_low_precision_array(row)
        string += row_string + '\n'
    return string

def string_low_precision_array(array):
    string = ['{:.3f}'.format(item) for item in array]
    string = ', '.join(string)
    return string

# accept only one latent varibale
def get_latent_weighted_histogram(index_list, data, domain, \
    weights, latent_variable_set):
    if tuple(sorted(index_list)) == tuple(sorted(list(latent_variable_set))):
        return np.sum(weights, axis=0)
    # print(index_list)
    # print(q.shape)
    # q[:, :, :] = 0
    # q[:, 0, 1] = 0.4
    # q[:, 1, 2] = 0.2

    temp_domain = domain.project(index_list)
    latent_domain = temp_domain.project(latent_variable_set)


    # merge latent variables of q and get the distribution of the required latent variable
    latent_var_start = min(latent_variable_set)
    axis = [var-latent_var_start+1 for var in latent_domain.attr_list]
    axis = tuple(set(range(1, len(weights.shape))) - set(axis))
    weights = np.sum(weights, axis=axis)
    # print('axis', axis) # debug

    # print('??', np.sum(q))
    # print('??', np.sum(weights))
    # debug_temp = 0

    
    final_histogram = np.zeros(shape=temp_domain.shape, dtype=float)
    # print(final_histogram.shape, weights.shape)

    ob_marginal = tuple(sorted(list(set(index_list) - latent_variable_set)))
    ob_domain = domain.project(ob_marginal)

    # print('ob_marginal', ob_marginal)
    # print(data[:, ob_marginal])
    # print(ob_domain)
    # print(ob_domain.edge())
    # print(ob_marginal)
    # print(np.unique(data[:, 7]))
    # print(np.unique(data[:, 9]))
    # print(np.sum(hist))
    
    # use weights to weight records and get their histogram
    latent_type_list = list(list(range(i)) for i in latent_domain.shape)
    # print(weights)

    # # debug
    # if index_list == (1, 18):
    #     print('get data 10')
    #     print(data[:10])
    #     print(weights[:10])
    #     print(weights[1000:1000+10])
    #     print(latent_type_list)
    for latent_type in itertools.product(*tuple(latent_type_list)):
        # print(latent_type)

        slc = [slice(None),]
        slc.extend(list(latent_type))
        slc = tuple(slc)
        histogram, _ = np.histogramdd(data[:, ob_marginal], bins=ob_domain.edge(), \
            weights=weights[slc])
        # print(slc)
        # print(weights[slc].shape)

        slc = [slice(None),] * len(ob_domain.shape)
        slc.extend(list(latent_type))
        slc = tuple(slc)
        final_histogram[slc] = histogram
        # print(slc)
        # print(histogram.shape)

        # if index_list == (1, 18):
        #     print(slc)
        #     print(latent_type, histogram)

    # if index_list == (1, 18):
    #     print(final_histogram)

    return final_histogram

# collect all the possible values for each attr and sort them by their ferquencies
def collect_domain(np_data, attr_list):
    print(attr_list, len(np_data))

    with open('./temp/attr_list.txt', 'w') as out_file:
        out_file.write(str(attr_list)+'\n')
    domain_dict = {}
    for col in range(len(attr_list)):
        attr = attr_list[col]
        values, cnts = np.unique(np_data[:, col], return_counts=True)
        values_cnts = [(values[i], cnts[i]) for i in range(len(values))]
        values_cnts.sort(key = lambda x: x[1], reverse=True)

        partial_total = sum([item[1] for item in values_cnts[:100]])

        print(col, attr, partial_total)
        print(values_cnts[:100])
        print('')

        domain_dict[attr] = values_cnts
    
    return domain_dict

def get_adaptive_domain(data):
    assert(data.dtype==int)
    dom_dict = {}
    for col in range(data.shape[1]):
        min_v = min(data[:, col])
        max_v = max(data[:, col])
        assert(min_v>=0)
        dom_dict[col] = {'size': max_v+1}
    dom = domain.Domain(dom_dict, list(range(data.shape[1])))
    return dom

def random_data_TVD(data1, data2):
    dom = get_adaptive_domain(data1)
    print('dom:', dom)
    return random_TVD(data1, data2, dom)

def random_TVD(data1, data2, domain, k=3, n=100, normalize=False):
    assert(data1.shape[1] == len(domain))
    assert(data2.shape[1] == len(domain))
    marginal_list = [marginal for marginal in itertools.combinations(domain.attr_list, k) ]
    np.random.shuffle(marginal_list)
    marginal_list = marginal_list[:n]
    mean_TVD = 0
    for marginal in marginal_list:
        hist1 = get_histogram(marginal, data1, domain)
        hist2 = get_histogram(marginal, data2, domain)
        marginal_TVD = get_TVD(hist1, hist2, normalize)
        mean_TVD += marginal_TVD
    return mean_TVD / len(marginal_list)

def triangulate(graph):
    edges = set()
    G = nx.Graph(graph)

    nodes = sorted(graph.degree(), key=lambda x: x[1])
    for node, degree in nodes:
        local_complete_edges = set(itertools.combinations(G.neighbors(node), 2))
        edges |= local_complete_edges

        G.add_edges_from(local_complete_edges)
        G.remove_node(node)
    
    triangulated_graph = nx.Graph(graph)
    triangulated_graph.add_edges_from(edges)

    return triangulated_graph

# randomly round a prob array such that its summation equal to num
def random_round(prob, num, replace=True):
    # assert(len(prob.shape)==1)
    if np.sum(prob) == 0:
        prob += 1
    prob = prob * num/prob.sum()
    # print('prob', prob)
    frac, integral = np.modf(prob)
    integral = integral.astype(int)
    round_number = int(num - integral.sum())
    if frac.sum() == 0:
        return integral
    p = (frac/frac.sum()).flatten()

    # print('integral', integral, 'round_number', round_number)
    # print('frac', frac)

    if round_number > 0:
        # CRF should sample without replacement while MRF should not.
        # this is for keeping intra group structures
        # say, we are sampling a group of 2 records, we have p = [0.49, 0.49, 0.02]
        # sampling without replacement gives [0, 1] with a very high probability
        # while sampling with replacement gives [0, 0] p=0.25, [0, 1]/[1, 0], p=0.5, [1, 1] p=0.25
        # Apparently, we prefer [0, 1] as the group records instead of [0, 0], [1, 1]
        # Although this would detroy attribte correlations if you look into the records
        # For example, sampling a group of 3 records and we have p [0.49, 0.49, 0.02]
        # Sampling without replacemet gives [0, 1, 2] deterministically.

        index = np.random.choice(prob.size, round_number, p=p, replace=replace)
        unique, unique_counts = np.unique(index, return_counts=True)

        # print('unique', unique)
        # print('unique_counts', unique_counts)

        for i in range(len(unique)):
            idx = np.unravel_index(unique[i], prob.shape)
            integral[idx] += unique_counts[i]
    return integral

def expand_int_prob(int_prob, shuffle=True):
    if len(int_prob.shape) > 1:
        data = []
        for idx in np.ndindex(int_prob.shape):
            data.extend([idx,] * int_prob[idx])
        data = np.array(data)
        if shuffle:
            np.random.shuffle(data)
        return data
    else:
        data = np.repeat(np.arange(int_prob.size), int_prob)
        if shuffle:
            np.random.shuffle(data)
    return data

def generate_column_data(prob, num, replace=True):
    if (prob < 0).any():
        print('!!', prob, num)
        exit(0)
    if num < 0:
        print('???', prob, num)

    int_prob = random_round(prob, num, replace=replace)

    if (int_prob < 0).any():
        print('xxx', int_prob, prob, num)
        exit(0)
    return expand_int_prob(int_prob)

def save_np_csv(array, attr_list, path):
    with open(path, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(attr_list)
        for line in array:
            writer.writerow(line)

def print_graph(G, path):
    plt.clf()
    nx.draw(G, with_labels=True, edge_color='b', node_color='g', node_size=20, font_size=4, width=0.5)
    plt.rcParams['figure.figsize'] = (4, 3)
    plt.rcParams['savefig.dpi'] = 600
    # plt.show()
    plt.savefig(path)

def get_TVD_count(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2)) / 2

def get_TVD(hist1, hist2, normalize=False):
    temp = np.sum(hist1)
    if temp == 0:
        return 1
    if normalize:
        hist2 = hist2 * temp /np.sum(hist2)
    return get_TVD_count(hist1, hist2) / temp

def get_normalized_TVD(hist1, hist2):
    hist2 = hist2 * np.sum(hist1)/np.sum(hist2)
    return get_TVD(hist1, hist2)

def get_normalized_TVD_count(hist1, hist2):
    hist2 = hist2 * np.sum(hist1)/np.sum(hist2)
    return get_TVD_count(hist1, hist2)

def split_array_uniformly(array, k):
    if k == 1:
        return [array,]

    flatten_array = array.flatten()
    res_array = np.zeros([k, len(flatten_array)], dtype=int)

    for i in range(len(flatten_array)):
        item = flatten_array[i]
        res_array[:, i] = int(flatten_array[i] / k)
        temp_sum = np.sum(res_array[:, i])

        temp_add = np.zeros(k, dtype=int)
        for j in range(item - temp_sum):
            temp_add[j] += 1
        np.random.shuffle(temp_add)
        # print(res_array[:, i], temp_sum, item)

        res_array[:, i] += temp_add

    res_list = [res_array[i].reshape(array.shape) for i in range(k)]
    
    return res_list

def erf_func(x):
    temp = 2.0/mp.sqrt(mp.pi)
    integral = mp.quad(lambda t: mp.exp(-t**2), [0, x])
    res = temp*integral
    return res

# ref: Data synthesis via Differentially Private Markov Random Field
def cal_privacy_budget(epsilon, error, delta):
    print('calculating privacy budget')
    start = 0
    end = epsilon

    def func(x):
        if x <= 0:
            return - 2*delta
        add1 = erf_func(math.sqrt(x)/2/math.sqrt(2) - epsilon/math.sqrt(2*x))
        add2 = erf_func(math.sqrt(x)/2/math.sqrt(2) + epsilon/math.sqrt(2*x))
        res = add1 + mp.exp(epsilon)*add2 - mp.exp(epsilon) + 1 - 2*delta
        # print(add1, add2)
        return res
    # return mp.findroot(func, start, tol=1e-30)

    # print(func(start), func(end))

    # gradient of func around its root is extemely small (maybe <= 1e-20 depending on epsilon)
    # which makes it is hard to set tol of mp.findroot and mp.mp.dps
    # we simply use binary search to ensure abs error of the root
    if func(start) > 0:
        start_geater = True
        if func(end) > 0:
            print('cant find root in given interval')
            exit(-1)
            return
    else:
        start_geater = False
        if func(end) < 0:
            print('cant find root in given interval')
            exit(-1)
            return
    
    while end - start > error:
        mid = (start + end)/2
        # print(mid)
        if func(mid) > 0:
            if start_geater:
                start = mid
            else:
                end = mid
        else:
            if start_geater:
                end = mid
            else:
                start = mid

    print((start + end)/2)
    return (start + end)/2


def get_privacy_budget(epsilon, delta=1e-5):

    budget = cal_privacy_budget(epsilon, 1e-10, delta)

    return budget

def get_R_score(data, domain, index_list):
    if not isinstance(index_list, tuple):
        index_list = tuple(sorted(index_list))

    domain = domain.project(index_list)
    bins = domain.edge()
    size = domain.size()

    histogram, _= np.histogramdd(data[:, index_list], bins=bins)
    fact1 = Factor(domain, histogram, np)
    # print('!', domain.shape, '!', bins, '!', histogram.shape)

    temp_domain = domain.project([index_list[0]])
    temp_index_list = temp_domain.attr_list
    histogram, _= np.histogramdd(data[:, temp_index_list], bins=temp_domain.edge())
    fact2 = Factor(temp_domain, histogram, np)

    temp_domain = domain.project([index_list[1]])
    temp_index_list = temp_domain.attr_list
    histogram, _= np.histogramdd(data[:, temp_index_list], bins=temp_domain.edge())
    fact3 = Factor(temp_domain, histogram, np)

    data_num = len(data)
    fact4 = fact2.expand(domain) * fact3.expand(domain) / data_num
    R_score = np.sum(np.abs(fact4.values - fact1.values)) / 2

    return R_score

# return h_data with q types in h_data 0-th col order
def concatenate_q_group(q, i_group_data, h_data_with_id, h_domain, type_first=False):
    undetermined_type = np.sum( q[q<0.9] > 1.0/np.prod(q.shape[1:]) )
    if undetermined_type > 0:
        print('warning: too many group types are undermined')
    print('undetermined_type ratio: {:.4f}'.format(undetermined_type/q.size))

    argmax_q = np.argmax(q.reshape((len(q), -1)), axis=1)
    argmax_q = np.unravel_index(argmax_q, shape=q.shape[1:])
    argmax_q = [item.reshape((-1, 1)) for item in argmax_q]
    argmax_q = np.concatenate(argmax_q, axis=1)

    # match household data and the type q
    i_group_FK = [group[0, -1] for group in i_group_data]
    h_to_q = {i_group_FK[i]: argmax_q[i] for i in range(len(i_group_FK))}

    q_h_data = np.zeros(shape=(h_data_with_id.shape[0], \
        h_data_with_id.shape[1]+argmax_q.shape[1]), dtype=int)
    q_h_data[:, :h_data_with_id.shape[1]] = h_data_with_id

    idx_list = []
    idx_list2 = []
    for idx in range(len(h_data_with_id)):
        h_id = h_data_with_id[idx][0]
        if h_id in h_to_q:
            q_h_data[idx, h_data_with_id.shape[1]:] = h_to_q[h_id]
            idx_list.append(idx)
        else:
            idx_list2.append(idx)

    q_h_data = q_h_data[idx_list]

    if len(q_h_data) < len(h_data_with_id):
        print('warning: missing h types of h_data: {} {}'.format(len(q_h_data), len(h_data_with_id)))

    if type_first:
        latent_var_num = len(q.shape) - 1

        q_h_data = np.concatenate([\
            q_h_data[:, [0,]], q_h_data[:, -latent_var_num:], q_h_data[:, 1:-latent_var_num]
            ], axis=1)

        temp_dict = {}
        attr = 0
        for q_size in q.shape[1:]:
            temp_dict[attr] = {'size': q_size}
            attr += 1
        for attr in h_domain.attr_list:
            temp_dict[attr+latent_var_num] = h_domain.dict[attr].copy()
        
        q_h_domain = domain.Domain(temp_dict, list(range(len(temp_dict))))
    else:
        q_h_domain = h_domain.copy()
        q_attr = len(h_domain)
        for q_size in q.shape[1:]:
            q_h_domain.add_variable(q_attr, q_size)
            q_attr += 1

    return argmax_q, q_h_data, h_data_with_id[idx_list2], q_h_domain

def get_group_data(np_data, group_id_attrs=[0,]):

    group_data_list = []
    data_len = len(np_data)
    i = 0
    while i < data_len:
        group = []
        row_id = np_data[i, group_id_attrs]

        while (np_data[i, group_id_attrs] == row_id).all():
            group.append(np_data[i])
            i += 1
            if i >= data_len:
                break
        group = np.array(group)
        group_data_list.append(group)
    group_data_list = np.array(group_data_list, dtype=object)

    return group_data_list

def get_q_h_data(q, i_group_data, h_data_with_id, h_domain):
    argmax_q, q_h_data, q_h_domain = concatenate_q_group(\
        q, i_group_data, h_data_with_id, h_domain)
    # print(h_data_with_id[:, 0].shape)
    # print(q_h_data.shape)
    # print(h_data_with_id.shape)

    return q_h_data, q_h_domain

def get_time():
    return 'time: ' + time.asctime(time.localtime(time.time()))

def get_data_by_FK(i_group_data_with_id, FK_set):
    i_group_list = []
    for i_group in i_group_data_with_id:
        if i_group[0, -1] in FK_set:
            i_group_list.append(i_group)
    return i_group_list

def get_sorted_data_by_FK(i_group_data_with_id, FK_set):
    i_group_list = get_data_by_FK(i_group_data_with_id, FK_set)
    i_data = np.concatenate(i_group_list, axis=0)
    i_data = i_data[np.argsort(i_data[:, -1], axis=0)]
    i_group_data = get_group_data(i_data, -1)
    return i_group_data

def get_domain(col, domain_dict):
    domain_dict = {i: domain_dict[col[i]] for i in range(len(col))}
    dom = domain.Domain(domain_dict, list(range(len(domain_dict))))
    return dom

def dict_add(d1, d2):
    res_dict = {}
    for key, value in d1.items():
        if type(value) == dict:
            res_dict[key] = dict_add(value, d2[key])
        else:
            res_dict[key] = value + d2[key]
    return res_dict

def dict_divide(d1, val):
    res_dict = {}
    for key, value in d1.items():
        if type(value) == dict:
            if type(val) is dict:
                res_dict[key] = dict_divide(value, val[key])
            else:
                res_dict[key] = dict_divide(value, val)
        else:
            if type(val) is dict:
                # print(key, value, val[key])
                # print(value / val[key])
                res_dict[key] = value / val[key]
            else:
                res_dict[key] = value / val
    return res_dict

def down_sample(group_data, max_group_size):

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
    print('downsample ratio {:.4f}'.format(len(res_data)/total))

    return res_data, res_group_data