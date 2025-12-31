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
import json
import pandas as pd
import random
import numpy as np
import time
from multiprocessing import Pool
import sys
import time
from CRF.domain import Domain
from CRF.tools import get_group_data

def read_table(table_path, domain_path, id_col=None, FK_col=None, ratio=1.0, all_attrs=False):
    df = pd.read_csv(table_path)
    # print(df.columns)
    domain_dict = json.load(open(domain_path))

    # print(df.loc[:5])

    attrs = list(domain_dict.keys())

    if id_col is None:
        sorted_cols = []
    else:
        sorted_cols = [id_col,]

    if all_attrs:
        temp_attrs = list(df.columns)
        # print(temp_attrs)
        if not id_col is None:
            temp_attrs.remove(id_col)
        sorted_cols.extend(temp_attrs)
    else:
        sorted_cols.extend(attrs)
        if not FK_col is None:
            sorted_cols.append(FK_col)
    # print(sorted_cols)

    # sort columns in the order of domain keys
    # also let id_col be the first col and FK_col be the last col
    df = df[sorted_cols]

    domain_dict = {i: domain_dict[attrs[i]] for i in range(len(attrs))}
    domain = Domain(domain_dict, list(range(len(attrs))))
    data = df.to_numpy()

    # if id_col is None:
    #     start = 0
    # else:
    #     start = 1
    # for col in range(len(domain)):
    #     print(col, col+start, domain.dict[col], attrs[col])
    #     print(np.sum(data[:, col+start] >= domain.dict[col]['size']))
    #     assert((data[:, col+start] < domain.dict[col]['size']).all())

    if not id_col is None:
        assert(len(np.unique(data[:, 0])) == len(data))
    if FK_col is None:

        if not id_col is None:
            data = data[np.argsort(data[:,sorted_cols.index(id_col)])]

        data_num = int(len(data)*ratio)
        data = data[:data_num]
        if not id_col is None:
            sorted_cols.remove(id_col)

        return data, domain, sorted_cols

    else:
        FK_data = data[:,sorted_cols.index(FK_col)]
        K_data = data[:, 0]

        data = data[np.lexsort((K_data, FK_data))]

        group_data = get_group_data(data, group_id_attrs=[sorted_cols.index(FK_col),])

        group_num = int(len(group_data) * ratio)
        data_num = sum([len(group_data[i]) for i in range(group_num)])

        data = data[:data_num]
        group_data = group_data[:group_num]

        sorted_cols.remove(id_col)
        sorted_cols.remove(FK_col)

        return data, group_data, domain, sorted_cols
    
def get_group_query_list_cnt(group_data, group_size_data, h_data, query_list):
    query_cnt = []
    for query in query_list:
        query_cnt.append(
            get_group_query_cnt(group_data, group_size_data, h_data, query)
        )
    return query_cnt

def get_group_query_cnt(group_data, group_size_data, h_data, query):
    cnt = 0
    for i in range(len(group_data)):
        group = group_data[i]
        h = h_data[i]
        size = group_size_data[i]
        if query.check_sat(group, size, h):
            cnt += 1
    # print('finish', query)
    return cnt

def check_FK(group_data, h_data, FK_col=-1):
    FK_list = []
    for group in group_data:
        if len(group) == 0:
            FK_list.append(-1)
        else:
            FK_list.append(group[0][FK_col])
    FK_list = np.array(FK_list)

    invalid_cnt = np.sum(FK_list == -1)
    print('invalid_cnt:', invalid_cnt)
    mask = FK_list != -1

    # print(FK_list.shape)
    # print(h_data[:, 0].shape)
    return np.equal(FK_list[mask], h_data[:, 0].flatten()[mask]).all()

def cut_group_data(i_group_data, h_data, num):
    # print(num)
    if len(h_data) > num:
        idx = np.arange(len(h_data))
        np.random.shuffle(idx)
        idx = idx[: num]
        idx = np.sort(idx)

        h_data = h_data[idx]

    res_list = []
    length = len(i_group_data)
    idx = 0
    for h_id in h_data[:, 0]:
        while idx < length:
            if i_group_data[idx][0, -1] < h_id:
                idx += 1
            elif i_group_data[idx][0, -1] == h_id:
                res_list.append(i_group_data[idx]) 
                idx += 1
                break
            else:
                res_list.append([])
                break

    i_group_data = np.array(res_list, dtype=object)

    return i_group_data, h_data


def get_group_query_error(group_data1, group_data2, i_domain, h_data1, h_data2, h_domain, \
    query_list, process_num=30, file_name='', base=1):
    # print(domain)

    assert(len(group_data1) == len(h_data1))
    assert(len(group_data2) == len(h_data2))

    query_num = len(query_list)


    # tools.check_group_data_domain(group_data1, i_domain)
    # tools.check_group_data_domain(group_data2, i_domain)
    # tools.check_data_domain(h_data1, h_domain)
    # tools.check_data_domain(h_data2, h_domain)

    group_size_data1 = np.array([len(group) for group in group_data1], dtype=int)
    group_size_data2 = np.array([len(group) for group in group_data2], dtype=int)


    with open('./temp/query_list.txt', 'w') as out_file:
        for query in query_list:
            out_file.write(str(query) + '\n')

    acc_error = 0
    with Pool(processes=process_num) as pool:

        query_block_size = 80

        query_cnt_list1 = []
        query_cnt_list2 = []
        for i in range(0, query_num, query_block_size):
            query_cnt_list1.append(pool.apply_async(
                get_group_query_list_cnt, \
                (group_data1, group_size_data1, h_data1, query_list[i: i+query_block_size])
            ))
            query_cnt_list2.append(pool.apply_async(
                get_group_query_list_cnt, \
                (group_data2, group_size_data2, h_data2, query_list[i: i+query_block_size])
            ))

        query_cnt_list1 = [res.get() for res in query_cnt_list1]
        query_cnt_list2 = [res.get() for res in query_cnt_list2]
        # print('get cnt time cost: {:.4f}'.format(time.time()-start_time))

        query_cnt1 = []
        for item in query_cnt_list1:
            query_cnt1.extend(item)
        query_cnt2 = []
        for item in query_cnt_list2:
            query_cnt2.extend(item)
        
        json.dump(query_cnt1, open('./temp/query_cnt_'+file_name+'_1.json', 'w'))
        json.dump(query_cnt2, open('./temp/query_cnt_'+file_name+'_2.json', 'w'))

        group_num1 = len(group_data1)
        group_num2 = len(group_data2)

        # # debug
        # length = np.array([len(group) for group in group_data1], dtype=int)
        # group_num1 = np.sum(length > 1)
        # length = np.array([len(group) for group in group_data2], dtype=int)
        # group_num2 = np.sum(length > 1)

        # error_base = group_num1 / 200
        # error_base = group_num1 / 1000
        # error_base = group_num1 / 2000
        error_base = group_num1 * base
        # error_base = 300
        error_list = []

        for i in range(len(query_list)):
            query = query_list[i]
            cnt1 = query_cnt1[i]
            cnt2 = query_cnt2[i] * group_num1 / group_num2
            error = abs(cnt1 - cnt2)/max(error_base, cnt1)
            # print(str(query) + ' error: {:.4f}'.format(error))
            # if cnt1 > 20000 and cnt2 < 10:
            #     print(i, cnt1, cnt2, query)
            error_list.append(error)
            acc_error += abs(error)

        json.dump(error_list, open('./temp/error_list.json', 'w'))
        acc_error /= query_num

    # # debug
    # print('group_num1:', group_num1)
    # print('group_num2:', group_num2)
    return acc_error

class IntraGroupQuery:
    def __init__(self):
        self.i_contain_req = []
        self.h_contain_req = []
        self.size_req = None

    def __str__(self):
        str1 = ''
        for req in self.i_contain_req:
            temp_str = [str(attr) + ', '+ str(domain_req) for (attr, domain_req) in req ]
            temp_str = '; '.join(temp_str)
            temp_str = 'individual: '+ temp_str + ' '

            str1 += temp_str

        str2 = [str(attr) + ', '+ str(domain_req) for (attr, domain_req) in self.h_contain_req ]
        str2 = '; '.join(str2)

        return 'i_req: ' + str1 + '; h req: '+ str2 + '; size:'+ json.dumps(self.size_req)

    def i_contain_init(self, domain, domain_ratio, attr_num):

        attr_list = np.random.choice(domain.attr_list, size=attr_num, replace=False)

        for i in range(2):
            req = []
            for attr in attr_list:
                domain_size = domain.dict[attr]['size']
                
                if 'type' in domain.dict[attr] and domain.dict[attr]['type'] == 'continuous':
                    size = int(domain_size * domain_ratio)
                    size = max(1, size)
                    start = int(np.random.random() * (domain_size - size))
                    domain_req = set(list(range(start, start+size)))
                else:
                    domain_req = list(range(domain_size))
                    random.shuffle(domain_req)

                    size = int(domain_size * domain_ratio)
                    size = max(1, size)
                    domain_req = set(domain_req[:size])

                req.append((attr, domain_req))

            self.i_contain_req.append(req)

    # select an additional household attr randomly
    def h_contain_init(self, domain, domain_ratio, attr_num):

        attr_list = np.random.choice(domain.attr_list, size=attr_num, replace=False)

        for attr in attr_list:
            domain_size = domain.dict[attr]['size']

            if 'type' in domain.dict[attr] and domain.dict[attr]['type'] == 'continuous':
                size = int(domain_size * domain_ratio)
                size = max(1, size)
                start = int(np.random.random() * (domain_size - size))
                domain_req = set(list(range(start, start+size)))
            else:
                domain_req = list(range(domain_size))
                random.shuffle(domain_req)

                size = int(domain_size * domain_ratio)
                size = max(1, size)
                domain_req = set(domain_req[:size])

            self.h_contain_req.append((attr, domain_req))

    def size_init(self, size_ratio, max_size):
        size = int(max_size * size_ratio)
        # print('size', size)
        # at least size 1
        # start = int(max_size * random.random()) + 1
        start = int(max_size * random.random()) - size + 2
        end = start + size
        self.size_req = (start, end)

    def check_sat(self, group, group_size, h):

        if len(group) == 0:
            return False

        if self.size_req != None:
            if group_size < self.size_req[0] or group_size >= self.size_req[1]:
                # # debug
                # print('        size: False')
                return False          

        for attr, req in self.h_contain_req:            
            if h[attr] not in req:
                # # debug
                # print('        h: False')
                return False

        sat_records = [[] for item in self.i_contain_req]
        assert(len(sat_records) == 2)

        # # debug
        # attr = [item[0] for item in self.i_contain_req[0]]
        # print(group[:, attr])

        for req_idx in range(2):
            req = self.i_contain_req[req_idx]
            for i in range(group.shape[0]):
                record = group[i]
                req_sat = True
                for attr, domain_req in req:                
                    if not record[attr] in domain_req:
                        req_sat = False
                        break
                
                if req_sat:
                    sat_records[req_idx].append(i)

        # # debug
        # if len(sat_records[0]) > 0:
        #     if len(sat_records[1]) > 0:
        #         print('A')
        #     else:
        #         print('B')
        # else:
        #     if len(sat_records[1]) > 0:
        #         print('C')
        #     else:
        #         print('D')

        if len(sat_records[0]) > 0 and len(sat_records[1]) > 0:
            total = set(sat_records[0]).union(sat_records[1])
            if len(total) >= 2:
                # print('        group: True')
                return True

        # # # debug
        # print('        group: False')
        return False

if __name__ == '__main__':

    i_path          = sys.argv[1]
    i_domain_path   = sys.argv[2]
    h_path          = sys.argv[3]
    h_domain_path   = sys.argv[4]
    test_i_path     = sys.argv[5]
    test_h_path     = sys.argv[6]
    base            = float(sys.argv[7])

    print('evaluating', test_i_path)
    print('evaluating', test_h_path)
    localtime = time.asctime(time.localtime(time.time()))
    start_time = time.time()
    print ("time:", localtime)

    evaluate_num = int(2e5) # for evaluation efficiency
    # evaluate_num = int(1e6)
    # evaluate_num = int(1e5)
    # evaluate_num = 3

    # read ground truth data and test data
    _, gt_i_group, i_domain, i_attrs = read_table(\
        i_path, i_domain_path, id_col='INDIVIDUAL', FK_col='HOUSEHOLD')
    gt_h_data, h_domain, h_attrs = read_table(\
        h_path, h_domain_path, id_col='HOUSEHOLD')

    gt_i_group, gt_h_data = cut_group_data(gt_i_group, gt_h_data, evaluate_num)
    assert(check_FK(gt_i_group, gt_h_data))
    gt_i_group = [group[:, 1:-1] for group in gt_i_group]
    gt_h_data = gt_h_data[:, 1:]
    print('gt household table:', gt_h_data.shape)

    _, test_i_group, i_domain, i_attrs = read_table(\
        test_i_path, i_domain_path, id_col='INDIVIDUAL', FK_col='HOUSEHOLD')
    test_h_data, h_domain, h_attrs = read_table(\
        test_h_path, h_domain_path, id_col='HOUSEHOLD')

    test_i_group, test_h_data = cut_group_data(test_i_group, test_h_data, evaluate_num)
    assert(check_FK(test_i_group, test_h_data))

    res_i_group = []
    for group in test_i_group:
        if len(group) == 0:
            res_i_group.append([])
        else:
            res_i_group.append(group[:, 1:-1])
    test_i_group = np.array(res_i_group, dtype=object)
    # test_i_group = [group[:, 1:-1] for group in test_i_group]
    test_h_data = test_h_data[:, 1:]
    print('test household table:', test_h_data.shape)

    length = np.array([len(group) for group in gt_i_group])
    hist1, _ = np.histogram(length, bins=list(range(max(length)+2)))
    length = np.array([len(group) for group in test_i_group])
    hist2, _ = np.histogram(np.array([len(group) for group in test_i_group]), bins=list(range(max(length)+2)))
    hist1 = hist1 / len(gt_i_group)
    hist2 = hist2 / len(test_i_group)
    print('gt group size:')
    print(hist1)
    print('test group size:')
    print(hist2)

    # i_domain_ratio=0.3
    # h_domain_ratio=0.3
    enable_size=True
    size_ratio=0.2
    max_size=7
    print('enable_size: {}, max_size: {}, size_ratio: {:.2f}'.format(enable_size, max_size, size_ratio))

    for attr_num in range(1, 3):

        h_attr_num, i_attr_num = attr_num, attr_num

        domain_ratio = 0.2 ** ( 1 / (h_attr_num + i_attr_num * 2))
        i_domain_ratio, h_domain_ratio = domain_ratio, domain_ratio
        # print('h_attr_num:', h_attr_num)
        # print('i_attr_num:', i_attr_num)
        # print('domain_ratio:', domain_ratio)
        print('h_attr_num: {}, i_attr_num: {}, domain_ratio: {:.2f}'.format(h_attr_num, i_attr_num, domain_ratio))

        def get_query():
            q  = IntraGroupQuery()
            q.i_contain_init(i_domain, i_domain_ratio, i_attr_num)
            q.h_contain_init(h_domain, h_domain_ratio, h_attr_num)
            if enable_size:
                q.size_init(size_ratio, max_size)
            return q

        query_list = []
        for _ in range(10000):
            query_list.append(get_query())

        # pickle.dump(query_list, open('./temp/query_list.pkl', 'wb'))
        # query_list = pickle.load(open('./temp/query_list.pkl', 'rb'))

        error = get_group_query_error(\
            gt_i_group, test_i_group, i_domain, gt_h_data, test_h_data, h_domain, query_list, \
            process_num=20, file_name=str(i_attr_num)+'and'+str(h_attr_num), base=base)
        print('    error: {:.6f}'.format(error))

        # # debug
        # for i in range(len(gt_h_data)):
        #     print(gt_i_group[i])
        #     print(gt_h_data[i])
        #     print('')
        # for i in range(len(query_list)):
        #     query = query_list[i]
        #     print(i, query)
        #     for i in range(len(gt_h_data)):
        #         res = query.check_sat(gt_i_group[i], len(gt_i_group[i]), gt_h_data[i])
        #         print('   ', res)
        #     print('')

    print('time cost: {:.4f}'.format(time.time()-start_time))