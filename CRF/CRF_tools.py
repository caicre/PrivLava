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
from dataclasses import dataclass
from . import tools
import numpy as np
import cupy as cp
from .cp_factor import Factor, Potential
import itertools

def print_group(group, attr, q, domain):
    hist = tools.get_histogram((attr,), group, domain)
    hist /= len(group)

    print(hist)
    print(tools.string_2d_low_precision_array(q))
    print(np.argmax(q))

def check_TVD(marginal, hist1, hist2):
    total = cp.sum(hist1).item()
    tvd = cp.sum(cp.abs(hist1 - hist2))/total/2
    tvd = tvd.item()
    # print(marginal, 'TVD: {:.4f}'.format(tvd), \
    #     'total: {:.2f}'.format(total))
    return tvd

def get_log_prob(potential, x, latent_type, latent_domain, partition_func):
    prob = 0
    maximal_cliques = [list(item) for item in potential]
    record = np.concatenate([x, np.zeros(len(latent_domain), dtype=int)])
    record[latent_domain.attr_list] = latent_type
    for clique in maximal_cliques:
        idx = tuple(record[clique])
        prob += potential[tuple(clique)].values[idx]
    prob -= partition_func

    return prob

def get_group_log_prob(potential, group, latent_type, latent_domain, partition_func, log_alpha=None):

    if log_alpha is None:
        res = 0
    else:
        res = log_alpha[latent_type]

    for x in group:
        res += get_log_prob(potential, x, latent_type, latent_domain, partition_func)

    return res

def clean_type_size(type_size, latent_domain):
    
    type_size = type_size.copy()
    debug_type_size = type_size.copy()

    # type=0 <=> size=1, make sure no illegal cell.
    # slc = [slice(None),]*len(self.type_size.shape)

    slc = [0,]*len(type_size.shape)
    
    temp_slc = slc.copy()
    temp_slc[-1] = slice(None)
    temp_slc = tuple(temp_slc)

    temp = np.sum(type_size[temp_slc])
    type_size[temp_slc] = 0

    temp_slc = slc.copy()
    temp_slc = tuple(temp_slc)
    type_size[temp_slc] = temp

    # self.type_size[1:, 1:] += (self.type_size[1:, 0]/(self.config['max_type_size']-1))\
    #     .reshape((-1, 1))

    latent_type_list = [list(range(i)) for i in latent_domain.shape]
    for latent_type in list(itertools.product(*tuple(latent_type_list)))[1:]:

        slc = list(latent_type)
        slc.append(0)

        temp_slc = tuple(slc)
        temp = type_size[temp_slc]
        type_size[temp_slc] = 0

        slc[-1] = slice(1, None, 1)
        temp_slc = tuple(slc)
        type_size[temp_slc] += temp/(type_size.shape[-1]-1)

    # Note that we should have np.sum(self.type_size, axis=1) == self.alpha * group_num
    # Get group size from q directly is against dp
    # Theoretically, we get group size from alpha and q-weighted size distribution


    # error = 0.5 * np.sum(np.abs(debug_type_size-type_size))
    # if error > 50:
    #     print('warning: failed to force groups of size 1 to be of type 0, error: {:.4f}'.format(error))
    #     # print(debug_type_size)
    #     # print(type_size)

    return type_size

def check_type_size(type_size, latent_domain):
    latent_type_list = [list(range(i)) for i in latent_domain.shape]
    for latent_type in itertools.product(*tuple(latent_type_list)):
        if latent_type == (0, 0):
            if not (type_size[latent_type][1:] == 0).all():
                return False
        else:
            if not type_size[latent_type][0] == 0:
                return False
    return True

# calculate margianls (or conditional marginals) of maximal cliques, in exp space by default
# caution: if total is not specified, could overflow
def belief_propagation(message_order, potential, total, log_space=False):
    belief = Potential({clique: potential[clique].copy() for clique in potential})

    sent_message = dict()
    for clique1, clique2 in message_order:
        separator = set(clique1) & set(clique2)
        if (clique2, clique1) in sent_message:
            message = belief[clique1] - sent_message[(clique2, clique1)]
        else:
            message = belief[clique1]
        message = message.logsumexp(separator)
        belief[clique2] += message

        sent_message[(clique1, clique2)] = message

    partition_func = next(iter(belief.values())).logsumexp()

    if log_space:
        return belief, partition_func

    log_total = np.log(total)
    for clique in belief:
        belief[clique] += log_total - partition_func

    for clique in belief:
        belief[clique] = belief[clique].exp()

    return belief, partition_func

# generate a column according to marginal distribution and conditions using pandas
def pandas_generate_cond_column_data(domain, df, clique_factor, cond, target, total, replace=True):
    clique_factor = clique_factor.moveaxis(domain.attr_list)
    if len(cond) == 0:
        prob = clique_factor.project(target).values
        df.loc[:, target] = tools.generate_column_data(prob, total)
    else:
        marginal_value = clique_factor.project(cond + [target])

        attr_list = marginal_value.domain.attr_list.copy()
        attr_list.remove(target)
        cond = attr_list.copy()
        attr_list.append(target)

        marginal_value = marginal_value.moveaxis(attr_list)
        marginal_value = marginal_value.values

        # # debug
        # print(df)
        # print(marginal_value)

        def foo(group):
            idx = group.name
            vals = tools.generate_column_data(marginal_value[idx], group.shape[0], replace=replace)
            group[target] = vals
            return group

        df = df.groupby(list(cond)).apply(foo)

        # print(df)

    return df

def get_group_size_record_num(group_size):
    record_num = 0
    for size in range(len(group_size)):
        record_num += (size + 1) * group_size[size]
    return record_num

def normalize_group_size(group_size, record_num):
    # to do: normalze type_size such that they are consistent with type_data
    # to do: normalize group_size such that its summation equals record_num
    # to do: use group_size to re-distribute type_size

    g_record_num = get_group_size_record_num(group_size)
    print('g_record_num: {:.4f}, record_num: {:d}'.format(g_record_num, record_num)) # debug
    res_group_size = group_size.copy()
    ratio = record_num / g_record_num
    for size in range(len(group_size)):
        res_group_size[size] *= ratio
    res_record_num = get_group_size_record_num(res_group_size)
    print('res_record_num: {:.4f}'.format(res_record_num)) # debug
    return res_group_size

def normalize_type_size_by_group_size(type_size, group_size):
    type_size = type_size.copy()
    type_size[type_size<0] = 0

    # print(group_size)

    syn_group_size = group_size.shape[-1]
    latent_var_num = len(type_size.shape) - 1

    for size in range(syn_group_size):

        slc = [slice(None),] * latent_var_num
        slc.append(size)
        slc = tuple(slc)

        if np.sum(type_size[slc]) > 0:
            type_size[slc] *= group_size[size] / np.sum(type_size[slc])

    # print(type_size.shape)
    # print(group_size.shape)
    # print(np.sum(type_size))
    # print(np.sum(group_size))

    # temp = np.sum(type_size, axis=(0, 1))
    # print(temp)
    # print(group_size)
    return type_size

def normalize_type_size_by_type_dist(type_size, type_dist, latent_domain):
    type_size = type_size.copy()
    latent_type_list = [list(range(i)) for i in latent_domain.shape]
    for latent_type in itertools.product(*tuple(latent_type_list)):
        type_size[latent_type] = tools.random_round(type_size[latent_type],  type_dist[latent_type])
    # type0 = tuple([0] * len(latent_domain))
    # temp = np.sum(type_size[type0])
    # type_size[type0] = 0
    # type_size[type0][0] = temp
    return type_size

def normalize_type_size(type_size, group_size, type_data, latent_domain, record_num, size1_type0=True):

    type_dist = np.zeros(shape=latent_domain.shape, dtype=int)
    for type in type_data:
        type_dist[tuple(type)] += 1

    temp = np.sum(type_size, axis=(0, 1))
    print('original:')
    print('type_size total: {:.4f}'.format(get_group_size_record_num(temp)))
    print('type_size group_num: {:4f}\n'.format(np.sum(type_size)))
    print('type_size1:')
    print(type_size)

    type_size = normalize_type_size_by_group_size(type_size, group_size)
    # print('type_size2:')
    # print(type_size)

    temp = np.sum(type_size, axis=(0, 1))
    print('result:')
    print('type_size total: {:.4f}\n'.format(get_group_size_record_num(temp)))

    type_size = normalize_type_size_by_type_dist(type_size, type_dist, latent_domain)
    # print('type_size3:')
    # print(type_size)

    temp = np.sum(type_size, axis=(0, 1))
    print('result:')
    print('type_size total: {:.4f}'.format(get_group_size_record_num(temp)))
    print('type_size group_num: {:4f}\n'.format(np.sum(type_size)))

    type_size = type_size.astype(int)

    temp = np.sum(type_size, axis=(0, 1))
    total = get_group_size_record_num(temp)
    print('rounded:')
    print('type_size total: {:d}'.format(total))
    print('type_size group_num: {:d}\n'.format(np.sum(type_size)))

    diff = record_num - total
    latent_size = latent_domain.size()
    max_size = type_size.shape[-1]
    print('record_num: {:d}, total: {:d}, diff: {:d}'.format(record_num, total, diff))
    if abs(diff) > 0.01 * record_num:
        print('warning: type_size / type_data are too noisy')
    while diff != 0:

        idx_array = np.random.choice(latent_size, size=abs(diff))
        idx_tuple_array = np.unravel_index(idx_array, latent_domain.shape)
        idx_tuple_array = np.array(idx_tuple_array)
        idx_tuple_array = idx_tuple_array.T

        p = group_size.copy()
        if diff > 0:
            # no size 1 group
            p = p[1:-1]
        else:
            # can not move size 1 to size 0
            # do not move size 2 to size 1
            p = p[2:]
        p = p/np.sum(p)

        for i in range(abs(diff)):
            latent_type = tuple(idx_tuple_array[i])

            if not size1_type0 or latent_type != (0, 0):
                size = np.random.choice(max_size-2, p=p)
                if diff > 0:
                    size += 1
                    if type_size[latent_type][size] > 0:
                        type_size[latent_type][size] -= 1
                        type_size[latent_type][size+1] += 1

                elif diff < 0:
                    size += 2
                    if type_size[latent_type][size] > 0:
                        type_size[latent_type][size] -= 1
                        type_size[latent_type][size-1] += 1

        temp = np.sum(type_size, axis=tuple(range(len(type_size.shape) - 1)))
        total = get_group_size_record_num(temp)
        diff = record_num - total
        assert((type_size >= 0).all())
        print('record_num: {:d}, total: {:d}, diff: {:d}'.format(record_num, total, diff))

    return type_size

def check_group_data_domain(group_data, domain):
    for attr in domain.attr_list:
        size = domain.dict[attr]['size']
        for group in group_data:
            for record in group:
                assert(record[attr] < size)
            assert((group >= 0).all())

def check_data_domain(data, domain):
    for attr in domain.attr_list:
        size = domain.dict[attr]['size']
        for record in data:
            assert(record[attr] < size)
    assert((data >= 0).all())


def get_data_by_FK(i_group_data, h_data):
    FK_array = set(h_data[:, 0])
    assert(len(FK_array) == len(h_data))

    res_i_group = []
    for group in i_group_data:
        h_id = group[0][-1]
        if h_id in FK_array:
            res_i_group.append(group)

    res_i_group = np.array(res_i_group, dtype=object)

    return res_i_group