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
import queue
from sklearn.preprocessing import quantile_transform
from .cp_factor import Factor, Potential
import numpy as np
import cupy as cp
import networkx as nx
from . import tools
import time
import itertools
import scipy
from . import CRF_tools
import random
from multiprocessing import Pool
import pandas as pd
import pickle
import os
import math


# build a Conditional Random Filed with a latent variable
class ConditionalRandomField:
    def __init__(self, config, domain, graph, marginal_list, init_latent_marginal_list, noise_dict):
        self.config = config
        self.domain = domain
        self.graph = graph
        self.marginal_list = marginal_list
        self.init_latent_marginal_list = init_latent_marginal_list

        np.set_printoptions(precision=4)

        if not noise_dict is None:
            self.data_num = noise_dict['data_num']
            self.noisy_data_num = noise_dict['noisy_data_num']
            self.group_num = noise_dict['group_num']
            self.noisy_group_num = noise_dict['noisy_group_num']
            self.latent_marginal_noise = noise_dict['latent_marginal_noise']
            self.latent_domain_limit2 = noise_dict['latent_domain_limit2']
            self.latent_domain_limit3 = noise_dict['latent_domain_limit3']
            self.marginal_noise = noise_dict['marginal_noise']
            self.marginal_domain_limit = noise_dict['marginal_domain_limit']
            self.marginal_TVD_noise = noise_dict['marginal_TVD_noise']
            self.latent_TVD_noise = noise_dict['latent_TVD_noise']
            self.group_size = noise_dict['group_size']
            self.noisy_group_size = noise_dict['noisy_group_size']
            self.type_size_noise = noise_dict['type_size_noise']
            self.m1_list = noise_dict['m1_list']
            self.m2 = noise_dict['m2']
            self.alpha_noise = noise_dict['alpha_noise']
            self.EM_type_size_noise = noise_dict['EM_type_size_noise']
            self.y_noise = noise_dict['y_noise']
            self.total_size_noise = noise_dict['total_size_noise']

        # debug
        # self.latent_marginal_noise = 0

        print('domain:')
        print(self.domain)
        print(self.domain.shape)
        print(self.domain.attr_list)

        self.latent_variable_dict = {}
        self.observed_variable_dict = {}
        for attr, value in domain.dict.items():
            if 'latent' in value and value['latent']:
                self.latent_variable_dict[attr] = value['size']
            else:
                self.observed_variable_dict[attr] = value['size']
        
        self.latent_variable_set = set(list(self.latent_variable_dict.keys()))
        self.latent_domain = self.domain.project(self.latent_variable_set)
        self.observed_variable_set = set(list(self.observed_variable_dict.keys()))
        self.observed_domain = self.domain.project(self.observed_variable_set)

        print('latent variable:', self.latent_variable_dict)

        self.maximal_cliques = [tuple(sorted(clique)) for clique in \
            nx.find_cliques(self.graph)]
        size = sum(self.domain.project(clique).size() for clique in \
            self.maximal_cliques)
        print('model size: {:.4e}'.format(size))
        print('maximal cliques:')
        for clique in self.maximal_cliques:
            print(clique, '{:.4e}'.format(self.domain.project(clique).size()))

        self.potential = Potential({clique: Factor.zeros(\
            self.domain.project(clique)) for clique in self.maximal_cliques})
        self.alpha = np.ones(shape=self.latent_domain.shape, dtype=float)
        self.alpha /= np.sum(self.alpha)

        print('init marginal list')
        print(marginal_list)

        # init junction tree
        clique_graph = nx.Graph()
        clique_graph.add_nodes_from(self.maximal_cliques)
        for clique1, clique2 in itertools.combinations(self.maximal_cliques, 2):
            clique_graph.add_edge(clique1, clique2, \
                weight=-len(set(clique1) & set(clique2)))
        self.junction_tree = nx.minimum_spanning_tree(clique_graph)

        # init belief propagation message order
        message_list = [(a, b) for a, b in self.junction_tree.edges()] \
            + [(b, a) for a, b in self.junction_tree.edges()]
        message_edge = []
        for message1 in message_list:
            for message2 in message_list:
                if message1[0] == message2[1] and message2[0] != message1[1]:
                    message_edge.append((message2, message1))
        G = nx.DiGraph()
        G.add_nodes_from(message_list)
        G.add_edges_from(message_edge)
        self.message_order = list(nx.topological_sort(G))

    def estimate_parameters(self, data, group_data, sampled_data, sampled_group_data):
        self.init_parameter_estimation()
        self.structure_EM(sampled_data, sampled_group_data)

        if self.config['model_type'] == 'native':
            self.type_size = get_type_size(group_data, self.q, \
                self.config['syn_group_size'])
            self.noisy_type_size = self.type_size + np.random.normal(\
                scale=self.type_size_noise, size=self.type_size.shape)

    def structure_EM(self, data, group_data):
        data_total = len(data)

        temp = []
        for group in group_data:
            temp.append(group[:, 1:-1])
        self.pure_group_data = np.array(temp, dtype=object)
        assert(self.pure_group_data[0].shape[1] == len(self.observed_domain))

        group_length = np.array([len(item) for item in group_data], dtype=int)

        assert(data.shape[1] == len(self.observed_domain))
        assert((data[-1] == group_data[-1][-1, 1:-1]).all())
        assert((data[0] == group_data[0][0, 1:-1]).all())
        assert(len(data) == sum(group_length))

        # groups of size > 1, used for calculating dispersion
        large_group_indices = []
        for i in range(len(group_data)):
            if len(group_data[i]) > 1:
                large_group_indices.append(i)
        large_group_indices = np.array(large_group_indices, dtype=int)


        # init marginal dist
        # q = np.full(shape=(len(group_length), self.latent_size), fill_value=1/self.latent_size)
        # self.check_and_collect_data_marginal(data, group_length, q)

        # init q
        if self.config['model_type'] == 'native':
            if self.config['random_int_q']:
                q = get_init_argmax_q(len(group_data), self.latent_domain)
            else:
                q = get_init_q(len(group_data), self.latent_domain)
            self.EM_type_size = get_type_size(group_data, q, \
                self.config['EM_group_size'])
            y_dist = get_y_dist(q, group_length)
            y_dist = cp.array(y_dist)
        elif self.config['model_type'] == 'mixture':
            q = get_init_q(len(data), self.latent_domain)
            y_dist = cp.array(np.sum(q, axis=0))
        # print(q.shape)
        # print(y_dist.shape)
        # print(q[:5])
        # print(y_dist[:5])
        # print(cp.sum(y_dist))


        # check marginal TVD
        self.check_and_collect_data_marginal(self.check_marginal_list, \
            self.check_marginal_dict, data, group_length, q)
        
        if len(self.marginal_list) > 0:
            # collect new marginal distributions (with or without the latent variable)
            self.check_and_collect_dp_data_marginal(self.marginal_list, \
                self.marginal_dict, data, group_length, q)

            self.potential, marginal_dict = CRF_gradient_ascent(self.potential, \
                self.marginal_to_clique, self.marginal_dict, self.marginal_list, \
                self.message_order, self.domain, data_total, self.latent_variable_set, \
                self.config['ob_iter_num'], self.config['print_interval'], \
                self.config['marginal_step_loss_ratio']*self.theoretic_loss)

        marginal_step_num = self.config['marginal_step_num']
        for step in range(marginal_step_num):
            print('marginal step {}/{}'.format(step, marginal_step_num))

            # marginal selection step
            new_marginal_list = self.select_marginal(data, self.potential, \
                self.marginal_to_clique, self.alternative_marginal_set, \
                self.message_order, self.domain, q, group_length, self.latent_variable_set, \
                self.cpu_marginal_dict, self.config['select_num'], self.config['max_TVD_num'],\
                self.marginal_TVD_noise)
            self.add_potential(new_marginal_list)

            # collect new marginal distributions (with or without the latent variable)
            self.check_and_collect_dp_data_marginal(self.marginal_list, \
                self.marginal_dict, data, group_length, q)

            self.potential, marginal_dict = CRF_gradient_ascent(self.potential, \
                self.marginal_to_clique, self.marginal_dict, self.marginal_list, \
                self.message_order, self.domain, data_total, self.latent_variable_set, \
                self.config['ob_iter_num'], self.config['print_interval'], \
                self.config['marginal_step_loss_ratio']*self.theoretic_loss, min_it=500)

            self.print_marginal_TVD(data, group_length, q)

        # add init latent marginal list and EM
        # do not add them before the marginal step. In that case, the latent
        # marginal list will be queried automatically. Despite their q is random,
        # they contain information of observed attributes and cause privacy leakage
        self.add_potential(self.init_latent_marginal_list)

        # EM if there exist latent marginals in the initial marginal set
        for marginal in self.marginal_list:
            if len(self.latent_variable_set.intersection(marginal)) != 0:
                q = self.EM(q, group_length, group_data, data, self.config['init_EM_step_num'])
                break

        structure_EM_step_num = self.config['structure_EM_step_num']
        for step in range(structure_EM_step_num):
            print('')
            print('structure EM step {}/{}'.format(step, structure_EM_step_num))

            new_latent_marginal_list = self.select_latent_marginal(group_data, \
                self.alternative_latent_marginal_set2, self.domain, q, \
                self.latent_variable_set, self.m1_list[step], \
                large_group_indices, self.latent_TVD_noise)

            new_latent_marginal_list2 = []
            for marginal in new_latent_marginal_list:
                new_latent_marginal_list2.extend(self.construct_latent_marginal(marginal))
                
            self.add_potential(new_latent_marginal_list2)

            # # each time a new marginal is added, re-init to jump out of the local minimum
            # q = get_init_q(len(group_data), self.latent_size)

            if step == structure_EM_step_num - 1 and self.m2 == 0:
                get_q = self.config['get_last_q']
            else:
                get_q = True
            q = self.EM(q, group_length, group_data, data, self.config['EM_step_num'], get_q)
            
            # print('alpha')
            # print(self.alpha)

            self.print_marginal_TVD(data, group_length, q)

        if self.config['last_EM_step_num'] > 1:
            q = self.EM(q, group_length, group_data, data, self.config['last_EM_step_num']-1, True)

        if self.m2 > 0:
            new_latent_marginal_list = self.select_latent_marginal(group_data, \
                self.alternative_latent_marginal_set3, self.domain, q, \
                self.latent_variable_set, self.m2, \
                large_group_indices, self.latent_TVD_noise)

            new_latent_marginal_list2 = []
            for marginal in new_latent_marginal_list:
                new_latent_marginal_list2.extend(self.construct_latent_marginal(marginal))
                
            self.add_potential(new_latent_marginal_list2)

            q = self.EM(q, group_length, group_data, data, 1, False)
            
        elif self.config['only_selection']:

            q = self.EM(q, group_length, group_data, data, 1, True)


        self.q = q


    def EM(self, q, group_length, group_data, data, EM_step_num, get_q=True):

        latent_type_list = list(list(range(i)) for i in self.latent_domain.shape)
        log_norm_type_size = None
        data_total = len(data)

        marginal, partition_func = CRF_tools.belief_propagation(self.message_order, self.potential, 100, log_space=False)
        temp_clique = self.maximal_cliques[0]
        latent = marginal[temp_clique].project([17, 18]).values
        # print('latent_marginal') # debug
        # print(latent)

        # print('data_marginal')
        # print(self.marginal_dict[(17, 18)].values)

        print('get q:', get_q)
        for step in range(EM_step_num):
            print('  EM step {}/{}'.format(step, EM_step_num))
            # collect new marginal distributions (with or without the latent variable)
            self.check_and_collect_dp_data_marginal(self.marginal_list, \
                self.marginal_dict, data, group_length, q)

            # M step, alpha and MLE + gradient ascent
            self.alpha = np.sum(q, axis=0)
            self.alpha += np.random.normal(scale=self.alpha_noise, size=self.alpha.shape)
            self.alpha[self.alpha<0] = 0


            # set alpha, type 0 <=> group of size 1. set 1e-10 to avoid log 0.
            if self.config['model_type'] == 'native':
                self.alpha[tuple([0,]*len(self.alpha.shape))] = 1e-16
            self.alpha += 1e-16 # avoid log 0
            self.alpha /= np.sum(self.alpha)

            if self.config['model_type'] == 'native':
                y_dist = get_y_dist(q, group_length)
                y_dist = cp.array(y_dist)
            elif self.config['model_type'] == 'mixture':
                y_dist = cp.array(np.sum(q, axis=0))

            self.potential, marginal_dict = CRF_gradient_ascent(self.potential, \
                self.marginal_to_clique, self.marginal_dict, self.marginal_list, \
                self.message_order, self.domain, data_total, self.latent_variable_set, \
                self.config['iter_num'], self.config['print_interval'])

            partition_func = self.get_partition_func()
            

            # E step
            if not self.config['quick_debug']:
                if step < EM_step_num - 1 or get_q:

                    if self.config['model_type'] == 'native':

                        if self.config['EM_type_size']:
                            # truncate noisy_group_size
                            temp_group_size = self.noisy_group_size[:self.EM_type_size.shape[-1]].copy()
                            temp_group_size[-1] += np.sum(self.noisy_group_size[self.EM_type_size.shape[-1]:])

                            # inject noise and denoise
                            log_norm_type_size = self.EM_type_size.copy()
                            log_norm_type_size += np.random.normal(scale=self.EM_type_size_noise, size=self.EM_type_size.shape)
                            log_norm_type_size = CRF_tools.normalize_type_size_by_group_size(log_norm_type_size, temp_group_size)
                            log_norm_type_size[log_norm_type_size<0] = 0
                            log_norm_type_size += 1e-16 # avoid log 0
                            # log_norm_type_size += self.EM_type_size_noise   # avoid impossible cells given by noise, 
                                                                            # which may change classifications completely

                            # log
                            for latent_type in itertools.product(*tuple(latent_type_list)):
                                log_norm_type_size[latent_type] = np.log( log_norm_type_size[latent_type] / np.sum(log_norm_type_size[latent_type]) )
                        else:
                            log_norm_type_size = None

                        # # debug
                        # sum_q = np.sum(self.q, axis=0)
                        # print('sum_q:')
                        # print(sum_q.astype(int))

                        q = get_latent_variable_dist(self.alpha, log_norm_type_size, self.potential, group_data, \
                            self.latent_domain, partition_func, self.config['log_likelihood_ratio'], \
                            self.config['q_process_num'], size1_type0=self.config['size1_type0'], \
                            retype=self.config['retype'])
                        sum_q = np.sum(q, axis=0)
                        print('sum_q:')
                        print(sum_q.astype(int))

                        assert(len(q) == len(group_data))
                    elif self.config['model_type'] == 'mixture':
                        q = get_latent_variable_dist_mixture(self.alpha, self.potential, data,\
                            self.latent_domain, partition_func, self.config['log_likelihood_ratio'], \
                            self.config['q_process_num'])
                        assert(len(q) == sum([len(group) for group in group_data]))
                    else:
                        print('unsupported model type:', self.config['model_type'])
                        raise

            self.EM_type_size = get_type_size(group_data, q, \
                self.config['EM_group_size'])

            type_sum = np.sum(q, axis=0)
            collapsed_num = np.sum(type_sum < 0.1 * np.sum(type_sum) / type_sum.size)
            print('collapsed latent type num: {:d}, ratio: {:4f}'.format(collapsed_num, collapsed_num/type_sum.size))

            # disp_list = get_dispersion_list(group_data, self.check_latent_marginal_list, \
            #     self.domain, q, self.latent_variable_set, large_group_indices)
            # disp_list.sort(key = lambda x: x[1], reverse=True)
            # print('check dispersion list')
            # mean_disp = sum([item[1] for item in disp_list]) / len(disp_list)
            # print('mean disp: {:.4f}'.format(mean_disp))
            # for item in disp_list:
            #     print(item)

        # debug
        if not self.EM_type_size is None:
            for latent_type in list(itertools.product(*tuple(latent_type_list)))[:30]:
                print(latent_type, np.sum(self.EM_type_size[latent_type]), self.EM_type_size[latent_type].astype(int)[:10])

        # marginal, partition_func = CRF_tools.belief_propagation(self.message_order, self.potential, 100, log_space=False)
        # temp_clique = self.maximal_cliques[0]
        # latent = marginal[temp_clique].project([17, 18]).values

        # # debug
        # print('noise scale: {:.4f}'.format(self.latent_marginal_noise))
        # print('data_marginal')
        # print((17, 18))
        # print(self.marginal_dict[(17, 18)].values)
        # for marginal in self.marginal_list:
        #     if len(marginal) == 2 and (17 in marginal or 18 in marginal):
        #         print(marginal)
        #         print(self.marginal_dict[marginal].values.astype(int))

        # print('model marginal')
        # print((17, 18))
        # print(latent)
        

        # for marginal in self.marginal_list:
        #     if len(marginal) == 2 and (17 in marginal or 18 in marginal):
        #         temp_value = marginal_dict[marginal].values
        #         print(np.sum(temp_value, axis=0))
        #         print('marginal', marginal)
        #         print(temp_value.astype(int))

        #         if marginal != (17, 18):
        #             attr = marginal[0]

        #             hist = tools.get_histogram((attr,), group_data[0][:, 1:-1], self.domain)
        #             print('group 0', np.sum(hist), len(group_data[0]))
        #             print(hist)


        return q


    def print_marginal_TVD(self, data, group_length, q):
        if self.config['model_type'] == 'native':
            y_dist = get_y_dist(q, group_length)
            y_dist = cp.array(y_dist)
        elif self.config['model_type'] == 'mixture':
            y_dist = cp.array(np.sum(q, axis=0))

        # self.check_and_collect_data_marginal(self.check_latent_marginal_list, \
        #     self.check_marginal_dict, data, group_length, q)

        check_mu = CRF_latented_weighted_marginal(self.potential, \
            self.marginal_to_clique, self.check_marginal_list, self.message_order,\
                self.noisy_data_num)
        print('check marginal TVD')
        average = 0
        for marginal in self.check_marginal_list:
            average += CRF_tools.check_TVD(marginal, self.check_marginal_dict[marginal].values, \
                check_mu[marginal].values)
        print('average: {:.4f}'.format(average/len(self.check_marginal_list)))

        # check_mu = CRF_latented_weighted_marginal(self.potential, \
        #     self.marginal_to_clique, self.check_latent_marginal_list, self.message_order,\
        #         self.noisy_data_num)
        # average = 0
        # print('check latent marginal TVD')
        # for marginal in self.check_latent_marginal_list:
        #     average += CRF_tools.check_TVD(marginal, self.check_marginal_dict[marginal].values, \
        #         check_mu[marginal].values)
        # print('average: {:.4f}'.format(average/len(self.check_latent_marginal_list)))
    
    def add_potential(self, new_marginal_list):
        for marginal in new_marginal_list:
            print('add potential', marginal)
            marginal = tuple(sorted(marginal))
            self.marginal_list.append(marginal)

            for clique in self.maximal_cliques:
                if set(marginal) <= set(clique):
                    self.marginal_to_clique[marginal] = clique
                    self.clique_to_marginal[clique].add(marginal)

        for marginal in new_marginal_list:
            if len(self.latent_variable_set.intersection(marginal)) != 0:
                marginal = tuple(sorted(list(set(marginal) - self.latent_variable_set)))
                if marginal in self.alternative_latent_marginal_set2:
                    self.alternative_latent_marginal_set2.remove(marginal)
                if marginal in self.alternative_latent_marginal_set3:
                    self.alternative_latent_marginal_set3.remove(marginal)
            else:
                marginal = tuple(sorted(marginal))
                self.alternative_marginal_set.remove(marginal)

    # get partition function (log space)
    def get_partition_func(self):
        belief, partition_func = CRF_tools.belief_propagation(self.message_order, self.potential, None, log_space=True)

        temp_clique = self.maximal_cliques[0]
        belief_domain = belief[temp_clique].domain
        belief = cp.asnumpy(belief[temp_clique].values)

        latent_partition_func = get_partition_func(belief, belief_domain, self.latent_domain)
        print('latent_partition_func:')
        print(latent_partition_func[:2])

        return latent_partition_func

    # find new marginals, get new marginal histograms and latent marginal histograms
    def check_and_collect_data_marginal(self, marginal_list, marginal_dict, data,\
        group_length, q):

        if self.config['model_type'] == 'native':
            weights = np.repeat(q, group_length, axis=0)
        elif self.config['model_type'] == 'mixture':
            weights = q.copy()
        else:
            print('unsupported model type:', self.config['model_type'])
            raise


        for marginal in marginal_list:
            get_data_marginal(data, self.domain, weights, marginal_dict, \
                marginal, self.latent_variable_set)

    # find new marginals, get new marginal histograms and latent marginal histograms
    # These marginals are used to learn the CRF and are noisy.
    def check_and_collect_dp_data_marginal(self, marginal_list, marginal_dict, data, group_length, q):

        if self.config['model_type'] == 'native':
            weights = np.repeat(q, group_length, axis=0)
        elif self.config['model_type'] == 'mixture':
            weights = q.copy()
        else:
            print('unsupported model type:', self.config['model_type'])
            raise

        self.theoretic_loss = 0

        # print(weights) # debug

        # print('data 10')
        # print(data[:10])
        # print(group_length[:5])

        for marginal in marginal_list:

            noise = self.marginal_noise
            if len(self.latent_variable_set.intersection(marginal)) != 0:
                get_data_marginal(data, self.domain, weights, marginal_dict, \
                    marginal, self.latent_variable_set)

                temp_marginal1 = cp.asnumpy(marginal_dict[marginal].values)

                if marginal == tuple(sorted(self.latent_domain.attr_list)):
                    noise = self.y_noise
                else:
                    noise = self.latent_marginal_noise
                xp = cp.get_array_module(marginal_dict[marginal].values)

                # # debug
                # total = xp.sum(marginal_dict[marginal].values)
                # print('total:', total)

                marginal_dict[marginal].values += xp.random.normal(scale=noise, \
                    size=marginal_dict[marginal].values.shape)

                # # debug
                # if marginal == (1, 18) :
                #     print('get (1, 18)')
                #     print(marginal_dict[(1, 18)].values)

                temp_marginal2 = cp.asnumpy(marginal_dict[marginal].values)
                print('latent: {}, noise: {:.2f}, query TVD: {:.4f}, total: {:.2f}'.format(\
                    marginal, noise, tools.get_TVD(temp_marginal1, temp_marginal2), np.sum(temp_marginal2)))


            elif marginal not in marginal_dict:
                
                get_data_marginal(data, self.domain, weights, marginal_dict, \
                    marginal, self.latent_variable_set)

                temp_marginal1 = cp.asnumpy(marginal_dict[marginal].values)

                
                xp = cp.get_array_module(marginal_dict[marginal].values)
                marginal_dict[marginal].values += xp.random.normal(scale=noise, \
                    size=marginal_dict[marginal].values.shape)

                temp_marginal2 = cp.asnumpy(marginal_dict[marginal].values)
                print('marginal: {}, query TVD: {:.4f}, total: {:.2f}'.format(\
                    marginal, tools.get_TVD(temp_marginal1, temp_marginal2), np.sum(temp_marginal2)))
        
            # print(marginal, self.domain.project(marginal).size(), noise, noise**2)
            self.theoretic_loss += self.domain.project(marginal).size() * noise ** 2

            # if (1, 18) in marginal_dict:
            #     print('check (1, 18)')
            #     print(marginal_dict[(1, 18)].values)
        
        self.theoretic_loss /= 2


    # init junction tree, belief propagation
    def init_parameter_estimation(self):


        # generate marginals to be selected
        self.alternative_latent_marginal_set3 = set()
        self.alternative_latent_marginal_set2 = set()
        self.alternative_marginal_set = set()
        for clique in self.maximal_cliques:
            for attr_num in range(1, self.config['latent_max_attr_num']+1):
                for marginal in itertools.combinations(clique, attr_num):
                    # latent var is decided laters
                    dom_size = self.domain.project(marginal).size()
                    if len(self.latent_variable_set.intersection(marginal)) == 0:
                        if dom_size < self.latent_domain_limit3:
                            self.alternative_latent_marginal_set3.add(marginal)
                            if dom_size < self.latent_domain_limit2:
                                self.alternative_latent_marginal_set2.add(marginal)

            for attr_num in range(2, self.config['marginal_max_attr_num']+1):
                for marginal in itertools.combinations(clique, attr_num):
                    if len(self.latent_variable_set.intersection(marginal)) == 0 \
                        and self.domain.project(marginal).size() < self.marginal_domain_limit:
                        self.alternative_marginal_set.add(marginal)

        temp_set = set(self.marginal_list)
        temp_set |= self.alternative_marginal_set
        temp_set |= self.alternative_latent_marginal_set3

        # construct marginals for debugging: showing dispersion and TVD.
        self.alternative_marginal_set = list(self.alternative_marginal_set)
        self.alternative_latent_marginal_set3 = list(self.alternative_latent_marginal_set3)

        random.shuffle(self.alternative_marginal_set)
        random.shuffle(self.alternative_latent_marginal_set3)

        num = min(self.config['check_marginal_num'], len(self.alternative_marginal_set))
        self.check_marginal_list = self.alternative_marginal_set[:num]
        # num = min(self.config['check_latent_marginal_num'], len(self.alternative_latent_marginal_set3))
        # self.check_latent_marginal_list = self.alternative_latent_marginal_set3[:num]

        # construct marginals for selection
        self.alternative_marginal_set = set(self.alternative_marginal_set)
        self.alternative_latent_marginal_set3 = set(self.alternative_latent_marginal_set3)

        # do not add repetitive marginals
        self.alternative_marginal_set -= set(self.marginal_list)
        temp_marginal_set = set([ tuple(sorted(set(marginal) - self.latent_variable_set)) for marginal in self.marginal_list])
        self.alternative_latent_marginal_set3 -= temp_marginal_set
        self.alternative_latent_marginal_set2 -= temp_marginal_set

        print('alternative_marginal_set', len(self.alternative_marginal_set))
        print('alternative_latent_marginal_set2', len(self.alternative_latent_marginal_set2))
        print('alternative_latent_marginal_set3', len(self.alternative_latent_marginal_set3))

        # print(self.alternative_marginal_set)

        # init marginal list assignment
        self.marginal_to_clique = {}
        self.clique_to_marginal = {clique: set() for clique in self.maximal_cliques}
        for marginal in temp_set:

            temp_marginal = set(marginal)
            temp_marginal = temp_marginal.union(self.latent_variable_set)
            temp_marginal = tuple(sorted(list(temp_marginal)))

            for clique in self.maximal_cliques:
                if set(temp_marginal) <= set(clique):
                    self.marginal_to_clique[temp_marginal] = clique
                    self.clique_to_marginal[clique].add(temp_marginal)

                    self.marginal_to_clique[marginal] = clique
                    self.clique_to_marginal[clique].add(marginal)

                self.marginal_to_clique[clique] = clique
                self.clique_to_marginal[clique].add(clique)

        self.marginal_dict = Potential({})
        self.check_marginal_dict = Potential({})
        # store marginals to be selected, the number of which are very large
        self.cpu_marginal_dict = Potential({})
        self.latent_var_to_marginal = {var: [] for var in self.latent_variable_set}

        for marginal in self.marginal_list:
            for var in self.latent_variable_set:
                if var in marginal:
                    self.latent_var_to_marginal[var].append(marginal)
    
    def construct_latent_marginal(self, marginal):
        res = []
        for var in self.latent_variable_set:
            res_marginal = [var,]
            res_marginal.extend(marginal)
            res_marginal.sort()
            res.append(tuple(res_marginal))
        
        return res

    def get_marginal_list_group_TVD(self, q, marginal_list, process_num=32):

        marginal_list = list(marginal_list)
        random.shuffle(marginal_list)
        marginal_list = marginal_list[:self.config['max_consider_latent_num']]

        latent_marginal_list = []
        marginal_to_latent_marginal = {}
        for marginal in marginal_list:
            latent_marginal = list(marginal)
            latent_marginal.extend(self.latent_variable_set)
            latent_marginal.sort()
            latent_marginal = tuple(latent_marginal)

            latent_marginal_list.append(latent_marginal)

            marginal_to_latent_marginal[marginal] = latent_marginal

        # simply generate synthetic data, and compare each marginal TVD
        # could be faster than calculate marginals and scale_along_var for each group
        syn_data = self.synthetic_data_mixture(q, self.pure_group_data,\
            0, self.group_size, process_num=16)
        syn_group_data = tools.get_group_data(syn_data, [-1])

        pool = Pool(processes=process_num)
        res_list = []
        for marginal in marginal_list:
            res_list.append(pool.apply_async(get_marginal_group_TVD,
                    (
                        marginal, self.pure_group_data, syn_group_data, self.domain
                    )                
                )
            )
        res_list = [res.get() for res in res_list]
        res_list = zip(marginal_list, res_list)

        pool.close()
        pool.join()

        return res_list


    def get_marginal_list_group_likelihood(self, group_data, marginal_list, process_num=32):

        # debug
        debug_len = int(3e4)

        idx = np.arange(len(group_data))
        np.random.shuffle(idx)
        idx = idx[:debug_len]

        group_data = group_data[idx]

        noisy_group_size = self.noisy_group_size.copy()
        noisy_group_size[noisy_group_size<0] = 0
        noisy_group_size = noisy_group_size * debug_len / np.sum(noisy_group_size)
        print('noisy_group_size:', tools.string_low_precision_array(noisy_group_size))

        # marginal_list = [(1,)]
        marginal_list = list(marginal_list)
        random.shuffle(marginal_list)
        marginal_list = marginal_list[:self.config['max_consider_latent_num']]

        latent_marginal_list = []
        marginal_to_latent_marginal = {}
        for marginal in marginal_list:
            latent_marginal = list(marginal)
            latent_marginal.extend(self.latent_variable_set)
            latent_marginal.sort()
            latent_marginal = tuple(latent_marginal)

            latent_marginal_list.append(latent_marginal)

            marginal_to_latent_marginal[marginal] = latent_marginal

        latent_marginal_dict = CRF_latented_weighted_marginal(self.potential, \
            self.marginal_to_clique, latent_marginal_list, self.message_order,\
                self.noisy_data_num)
        latent_marginal_dict.to_cpu()

        pool = Pool(processes=process_num)
        mp_results = []

        res_list = []
        for marginal in marginal_list:
            marginal_group_dict = {}
            for group in group_data:
                res = []
                for record in group:
                    res.append(tuple(record[1:][list(marginal)]))
                res.sort()
                res = tuple(res)

                if res in marginal_group_dict:
                    marginal_group_dict[res] += 1
                else:
                    marginal_group_dict[res] = 1
        
            latent_marginal = marginal_to_latent_marginal[marginal]
            mp_results.append(
                pool.apply_async(get_marginal_group_likelihood, \
                    (self.observed_domain, self.latent_domain,\
                    marginal_group_dict, marginal, latent_marginal, \
                    latent_marginal_dict[latent_marginal], self.alpha, noisy_group_size)
                )
            )

        for i in range(len(marginal_list)):
            true_p = mp_results[i].get()
            marginal = marginal_list[i]
            res_list.append((marginal, true_p))

        pool.close()
        pool.join()
    

        return res_list
    
    
    # select latent marginals
    def select_latent_marginal(self, group_data, marginal_set, domain, q, \
        latent_variable_set, select_num, large_group_indices, latent_TVD_noise):
        print('selecting {} from {}'.format(select_num, len(marginal_set)))
        start_time = time.time()
        for marginal in marginal_set:
            assert(len(latent_variable_set.intersection(marginal)) == 0)

        if not self.config['quick_debug']:
            if self.config['model_type'] == 'native':
                # measure the differences berween the data marginal distribution 
                # and the model marginal generation distribution
                res_list = self.get_marginal_list_group_likelihood(group_data, marginal_set)
                res_list = [(marginal, true+np.random.normal(scale=latent_TVD_noise), true)\
                    for marginal, true in res_list]
            elif self.config['model_type'] == 'mixture':
                # measure the TVD between each group and its synthetic version
                res_list = self.get_marginal_list_group_TVD(q, marginal_set)
                res_list = [(marginal, true+np.random.normal(scale=latent_TVD_noise), true)\
                    for marginal, true in res_list]
        else:
            res_list = [(item, -1) for item in marginal_set]


        res_list.sort(key = lambda x: x[1], reverse=True)
        print('select latent marginal, score list')
        for item in res_list[:20]:
            print(item)
        

        select_list = [item[0] for item in res_list[:select_num]]

        print('select latent marginal time cost: {:.4f}'.format(time.time() - start_time))

        return select_list

    def syn_FK(self, attrs, group_data=None, syn_group_num=None, types=None, print_flag=False, type_size=None, replace=True):
        print('generating synthetic data')
        # print(os.environ["OPENBLAS_NUM_THREADS"] )
        start_time = time.time()
        if self.config['model_type'] == 'native':
        
            syn_data = self.synthetic_data(group_num=syn_group_num, \
                types=types, print_flag=print_flag, process_num=self.config['syn_process_num'], \
                IPUMS=self.config['IPUMS'], type_size=type_size, replace=replace)
        else:
            syn_data = self.synthetic_data_mixture(self.q, group_data, \
                self.type_size_noise, self.noisy_group_size,\
                print_flag=False, process_num=16)
        columns = attrs.copy()
        for i in range(len(self.latent_domain)):
            columns.append('group_type_'+str(i))
        columns.append('group_id')
        # print(len(columns), columns)
        # print(syn_data.shape)
        df = pd.DataFrame(syn_data, columns=columns)
        if print_flag:
            print('synthetic data time cost: {:.4f}'.format(time.time()-start_time))
        return df
    
    def syn_FK_by(self, type_hist, group_id, type_size=None):
        syn_data = self.syn_FK(self.observed_domain.attr_list, types=type_hist, type_size=type_size).to_numpy()

        latent_idx = self.domain.get_attr_by({'latent': True})
        syn_data = syn_data[np.lexsort(syn_data[:, latent_idx].T, axis=0)]
        syn_group = tools.get_group_data(syn_data, [-1,])

        for i in range(len(syn_group)):
            syn_group[i][:, -1] = group_id[i]
        syn_data = np.concatenate(syn_group, axis=0)

        syn_data = np.concatenate([np.arange(len(syn_data)).reshape((-1, 1)), syn_data], axis=1)
        return syn_data

    # select marginals without the latent variable
    def select_marginal(self, data, potential, marginal_to_clique, marginal_set, \
        message_order, domain, q, group_length, latent_variable_set, data_marginal_dict, \
        select_num, max_TVD_num, marginal_TVD_noise):
        data_total = len(data)

        if self.config['model_type'] == 'native':
            y_dist = get_y_dist(q, group_length)
            y_dist = cp.array(y_dist)
        elif self.config['model_type'] == 'mixture':
            y_dist = cp.array(np.sum(q, axis=0))

        for marginal in marginal_set:
            assert(len(latent_variable_set.intersection(marginal)) == 0)

        maximal_cliques = list(potential.keys())

        model_clique_marginal = CRF_latented_weighted_marginal(potential, marginal_to_clique,\
            maximal_cliques, message_order, data_total)
        
        temp_list = []
        temp_marginal_list = list(marginal_set)
        random.shuffle(temp_marginal_list)
        temp_marginal_list = temp_marginal_list[:max_TVD_num]

        if self.config['model_type'] == 'native':
            weights = np.repeat(q, group_length, axis=0)
        elif self.config['model_type'] == 'mixture':
            weights = q.copy()
        else:
            print('unsupported model type:', self.config['model_type'])

        for marginal in temp_marginal_list:
            data_marginal = get_data_marginal(data, domain, weights, \
                data_marginal_dict, marginal, latent_variable_set, xp=np)
            model_marginal = model_clique_marginal[marginal_to_clique[marginal]].project(marginal)
            model_marginal.to_cpu()

            dist = np.sum(np.abs((model_marginal - data_marginal).values)) / 2
            true_dist = dist
            dist += np.random.normal(scale=marginal_TVD_noise)

            temp_list.append((marginal, dist, true_dist))

        temp_list.sort(key = lambda x: x[1], reverse=True)

        data_num = len(data)
        print('select new marginal')
        for i in range(min(10, len(marginal_set))):
            print(temp_list[i], 'noisy: {:.4f}, true: {:.4f}'.format(temp_list[i][1]/data_num, temp_list[i][2]/data_num))

        res_list = [item[0] for item in temp_list[:select_num]]

        return res_list

    def synthetic_data_mixture(self, q, group_data, type_size_noise, \
        noisy_group_size, process_num=1, print_flag=False):

        pool = Pool(processes=process_num)
        
        clique_marginal, partition_func = CRF_tools.belief_propagation(self.message_order, \
            self.potential, total=1)
        clique_marginal.to_cpu()

        res_list = []
        start = 0
        for i in range(len(group_data)):
            group = group_data[i]
            end = start + len(group)

            # get proportion of latent variables
            y = np.sum(q[start: end], axis=0)
            y += np.random.normal(scale=type_size_noise, size=y.shape)
            y[y<0] = 0

            # normalize and round y
            y = tools.random_round(y, noisy_group_size[i])

            y_array = tools.expand_int_prob(y)
            data = np.zeros(shape=(len(y_array), len(self.domain)+1), dtype=int)
            # print(y.shape, y_array.shape)
            data[:, self.latent_domain.attr_list] = y_array.reshape(\
                data[:, self.latent_domain.attr_list].shape)
            data[:, -1] = i
            finished_attr = self.latent_domain.attr_list.copy()

            res_list.append(pool.apply_async(generate_syn_data,
                    (len(data), self.maximal_cliques, self.junction_tree,\
                    self.domain, clique_marginal, print_flag, data, finished_attr,\
                    self.maximal_cliques[0]
                    )
                )
            )

            start = end

        res_list = [res.get() for res in res_list]
        data = np.concatenate(res_list, axis=0)

        pool.close()
        pool.join()

        return data

    def synthetic_data(self, group_num=None, types=None, process_num=1, print_flag=False, IPUMS=False, type_size=None, replace=True):
        # to do: given input record_num, normalize group size

        clique_marginal, partition_func = CRF_tools.belief_propagation(self.message_order, \
            self.potential, total=1)
        clique_marginal.to_cpu()

        pool = Pool(processes=process_num)

        if type_size is None:
            if group_num is None:
                if types is None:
                    raise
                else:
                    type_size = self.noisy_type_size.copy()
                    type_size[type_size<0] = 0
                    if self.config['size1_type0']:
                        type_size = CRF_tools.clean_type_size(type_size, self.latent_domain)
                    type_size = CRF_tools.normalize_type_size_by_group_size(type_size, self.noisy_group_size)
                    type_size[type_size<0] = 0

                    latent_type_list = [list(range(i)) for i in self.latent_domain.shape]
                    for latent_type in itertools.product(*tuple(latent_type_list)):

                        if np.sum(type_size[latent_type]) > 0:
                            temp_type_size = type_size[latent_type]
                        else:
                            if types[latent_type] > 0:
                                print('error: sampling invalid types', latent_type, types[latent_type])
                            temp_type_size = self.noisy_group_size[:len(type_size[latent_type])]
                            temp_type_size[0] = 0
                        temp_type_size *= types[latent_type] / np.sum(temp_type_size)
                        temp_type_size = tools.random_round(temp_type_size.flatten(), types[latent_type])

                        type_size[latent_type] = temp_type_size.reshape(type_size[latent_type].shape)
                    
                    temp_sum = np.sum(type_size)
                    type_size = type_size.astype(int)
                    assert(abs(temp_sum - np.sum(type_size)) < 1e-6)
            else:
                type_size = self.noisy_type_size.copy()
                type_size[type_size<0] = 0
                if self.config['size1_type0']:
                    type_size = CRF_tools.clean_type_size(type_size, self.latent_domain)
                type_size = CRF_tools.normalize_type_size_by_group_size(type_size, self.noisy_group_size)
                type_size[type_size<0] = 0
                type_size  *= group_num / np.sum(type_size)
                type_size = tools.random_round(type_size.flatten(), group_num)
                type_size = type_size.reshape(self.noisy_type_size.shape)

                temp_sum = np.sum(type_size)
                type_size = type_size.astype(int)
                assert(abs(temp_sum - np.sum(type_size)) < 1e-6)


        if print_flag:
            print('group size query TVD: {:.4f}'.format(tools.get_normalized_TVD(self.type_size, type_size)))

        # temp = self.type_size*self.config['syn_group_num'] / np.sum(self.type_size)
        # print(temp.astype(int))
        # print(type_size)

        # print('noisy_group_size:')
        # print(self.noisy_group_size)
        print('final type_size:')
        print(type_size)


        latent_type_0 = [0,] * len(self.latent_domain)
        latent_type_0.append(0)
        latent_type_0 = tuple(latent_type_0)

        group0_size = type_size[latent_type_0]
        type_size[latent_type_0] = 0

        print(process_num)
        type_size_list = tools.split_array_uniformly(type_size, process_num-1)

        res_list = []
        group_id = 0

        if print_flag:
            print('noisy group size')
            print(type_size)

            print('group size')
            print(self.type_size.astype(int))

            print('group0 size:', group0_size)

        # print('group0 size:', group0_size)
        # print('type_size_list')
        # for item in type_size_list:
        #     print(item)

        # generate type 0 groups, values of col 0 are exactly type 0
        if group0_size > 0:
            group = np.zeros(shape=(group0_size, len(self.domain)+1), dtype=int)
            finished_attr = self.latent_domain.attr_list.copy()
            res_list.append(
                    pool.apply_async(generate_syn_data,\
                        (group0_size, self.maximal_cliques, self.junction_tree,\
                        self.domain, clique_marginal, print_flag, group, finished_attr, \
                        self.maximal_cliques[0], replace
                        )
                    )
                )
        group_id += group0_size

        # generate groups of other types
        for i in range(0, process_num-1):
            if np.sum(type_size_list[i]) > 0:
                if IPUMS:
                    res_list.append(
                        pool.apply_async(synthetic_IPUMS_household,\
                            (type_size_list[i], self.maximal_cliques, self.junction_tree,\
                            self.latent_domain, self.domain, group_id, clique_marginal, print_flag
                            )
                        )
                    )
                else:
                    res_list.append(
                        pool.apply_async(synthetic_data_worker,\
                            (type_size_list[i], self.maximal_cliques, self.junction_tree,\
                            self.latent_domain, self.domain, group_id, clique_marginal, print_flag, replace
                            )
                        )
                    )
            group_id += np.sum(type_size_list[i])

        res_list = [res.get() for res in res_list]
        if group0_size > 0:
            res_list[0][:, -1] = np.array(list(range(0, group0_size)), dtype=int)

        data = np.concatenate(res_list, axis=0)


        pool.close()
        pool.join()

        return data

    @staticmethod
    def save_model(model, path):
        with open(path, 'wb') as out_file:
            pickle.dump(model, out_file)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as out_file:
            return pickle.load(out_file)

# get partition function (log space) of factor (log space)
def get_partition_func(factor, domain, latent_domain):

    latent_type_list = [list(range(i)) for i in latent_domain.shape]
    partition_func = np.zeros(shape=latent_domain.shape)
    
    for latent_type in itertools.product(*tuple(latent_type_list)):
        slc = [slice(None),] * len(factor.shape)
        axis_list = domain.index_list(latent_domain)
        for i in range(len(axis_list)):
            axis = axis_list[i]
            slc[axis] = latent_type[i]
        slc = tuple(slc)
        # print(slc)
        
        partition_func[latent_type] = scipy.special.logsumexp(factor[slc])

    return partition_func

# groups in group_dict does not contain group_id
def get_marginal_group_likelihood(ob_domain, latent_domain, group_dict, \
    marginal, latent_marginal, conditional_marginal, alpha, group_size):
    ob_attr = ob_domain.attr_list
    latent_attr = latent_domain.attr_list
    log_alpha = np.log(alpha)


    max_group_size = len(group_size) - 1

    model_group_dict = {}
    latent_type_list = list(list(range(i)) for i in latent_domain.shape)

    partition_func = get_partition_func(np.log(conditional_marginal.values), conditional_marginal.domain, latent_domain)
    partition_func = np.exp(partition_func)

    attr_num = max(max(ob_attr), max(latent_attr)) + 1
    error = 0
    for group in group_dict:
        group_p = 0
        for latent_type in itertools.product(*tuple(latent_type_list)):

            type_log_p = 0
            for record in group:

                full_record = -np.ones(attr_num, dtype=int)
                full_record[list(marginal)] = record
                full_record[latent_attr] = latent_type
                marginal_record = tuple(full_record[list(latent_marginal)])

                # normalize
                log_p = np.log(conditional_marginal.values[marginal_record] / partition_func[latent_type])

                log_p *= math.factorial(len(group))

                type_log_p += log_p

            group_p += np.exp(log_alpha[latent_type] + type_log_p)
        
        model_group_count = group_p * group_size[min(len(group)-1, max_group_size)]
        data_goup_count = group_dict[group]

        # print(group)
        # print('cnt1: {:d}, cnt2: {:.2f}'.format(data_goup_count, float(model_group_count)))
        error += max(data_goup_count - model_group_count, 0)

    return float(error)

# force each group has exactly only 1 record whose attr 0 is 0, i.e. householder
def synthetic_IPUMS_household(type_size, maximal_cliques, junction_tree, latent_domain, \
    domain, start_group_id, clique_marginal, print_flag=False):

    group_list = []
    columns = domain.attr_list.copy()
    columns.append('group_id')

    # generate attr RELATE first
    for clique in maximal_cliques:
        if 0 in clique:
            start_clique = clique
            break
    
    latent_type_list = [list(range(i)) for i in latent_domain.shape]

    RELATE_latent_attr_list = [0,]
    RELATE_latent_attr_list.extend(latent_domain.attr_list)

    marginal_value = clique_marginal[start_clique].project(RELATE_latent_attr_list)

    latent_attr = latent_domain.attr_list.copy()


    for latent_type in itertools.product(*tuple(latent_type_list)):
        y_type_size_array = tools.expand_int_prob(type_size[latent_type])
        y_type_size_array += 1 # group size = 0 means actual size of the group is 1

        slc = [slice(None),]
        slc.extend(latent_type)
        slc = tuple(slc)

        relate_dist = marginal_value.values[slc]
        relate_dist[0] = 0

        slc = [slice(None), latent_attr]
        slc = tuple(slc)

        latent_type_array = np.array(latent_type, dtype=int).reshape((1, len(latent_attr)))

        for i in range(len(y_type_size_array)):
            size = y_type_size_array[i]

            group = np.zeros(shape=(size, len(domain)+1), dtype=int)

            group[slc] = latent_type_array
            group[1:, 0] = tools.generate_column_data(relate_dist, size-1)

            finished_attr = [0,]
            finished_attr.extend(latent_attr)
            group = generate_syn_data(size, maximal_cliques, junction_tree, \
                domain, clique_marginal, print_flag, out=group, \
                finished_attr=finished_attr, start_clique=start_clique)
            group[:, -1] = start_group_id

            start_group_id += 1
            group_list.append(group)
    
    if print_flag:
        print('syn group num:', len(group_list), np.sum(type_size))
    return np.concatenate(group_list, axis=0)

# generate groups of size > 1
def synthetic_data_worker(type_size, maximal_cliques, junction_tree, latent_domain, \
    domain, start_group_id, clique_marginal, print_flag=False, replace=True):

    group_list = []
    columns = domain.attr_list.copy()
    columns.append('group_id')

    # generate attr RELATE first
    for clique in maximal_cliques:
        if 0 in clique:
            start_clique = clique
            break

    # print(maximal_cliques[0])
    # print(clique_marginal[maximal_cliques[0]].values[:, :, 1])
    
    latent_type_list = [list(range(i)) for i in latent_domain.shape]

    latent_attr = latent_domain.attr_list.copy()

    for latent_type in itertools.product(*tuple(latent_type_list)):
        y_type_size_array = tools.expand_int_prob(type_size[latent_type])
        y_type_size_array += 1 # group size = 0 means actual size of the group is 1

        slc = [slice(None), latent_attr]
        slc = tuple(slc)

        latent_type_array = np.array(latent_type, dtype=int).reshape((1, len(latent_attr)))

        for i in range(len(y_type_size_array)):
            size = y_type_size_array[i]

            group = np.zeros(shape=(size, len(domain)+1), dtype=int)

            group[slc] = latent_type_array

            finished_attr = latent_attr
            group = generate_syn_data(size, maximal_cliques, junction_tree, \
                domain, clique_marginal, print_flag, out=group, \
                finished_attr=finished_attr, start_clique=start_clique, replace=replace)
            group[:, -1] = start_group_id

            start_group_id += 1
            group_list.append(group)
    
    if print_flag:
        print('syn group num:', len(group_list), np.sum(type_size))
    return np.concatenate(group_list, axis=0)


def generate_syn_data(size, maximal_cliques, junction_tree, domain, \
    clique_marginal, print_flag=False, out=None, finished_attr=[], start_clique=None, replace=True):

    if out is None:
        data = np.zeros(shape=(size, len(domain)), dtype=int)
    else:
        data = out
    df =  pd.DataFrame(data)

    cond_attr = finished_attr.copy()
    finished_attr = set(finished_attr)

    if start_clique is None:
        start_clique = maximal_cliques[0]
    for attr in start_clique:
        if attr in finished_attr:
            continue
        if print_flag:
            print('  cond_attr: {}, attr: {}'.format(cond_attr, attr))

        # # debug
        # print('cond', cond_attr, attr, size)

        df = CRF_tools.pandas_generate_cond_column_data(domain, df, clique_marginal[start_clique], cond_attr, attr, size, replace=replace)

        cond_attr.append(attr)
        finished_attr.add(attr)

    if len(maximal_cliques) > 1:
        for start, clique in nx.dfs_edges(junction_tree, source=start_clique):
            cond_attr = sorted(list(set(start) & set(clique)))
            
            for attr in clique:
                if attr in finished_attr:
                    continue
                if print_flag:
                    print('  cond_attr: {}, attr: {} {}/{}'\
                        .format(cond_attr, attr, len(finished_attr), len(domain.attr_list)))
                df = CRF_tools.pandas_generate_cond_column_data(domain, df, clique_marginal[clique], cond_attr, attr, size, replace=replace)

                cond_attr.append(attr)
                finished_attr.add(attr)

    data = df.to_numpy()
    return data


def get_type_size(group_data, q, max_size):
    group_num = len(group_data)

    shape = q.shape[1:]
    shape = list(shape)
    shape.append(max_size)

    type_size = np.zeros(shape=shape, dtype=float)

    slc = [slice(None),]*len(type_size.shape)

    for i in range(group_num):
        size = len(group_data[i])
        size = min(max_size, size) - 1

        i_slc = slc.copy()
        i_slc[-1] = size
        i_slc = tuple(i_slc)

        type_size[i_slc] += q[i]

    return type_size




def get_dispersion_list(group_data, marginal_set, domain, q, \
    latent_variable_set, large_group_indices, process_num=4):

    print('calculating dispersion list')
    start_time = time.time()
    
    marginal_list = list(marginal_set)
    total_length = len(marginal_set)
    length = int(total_length/process_num)+1
    marginal_list = [marginal_list[i: i+length]\
        for i in range(0, total_length, length)]


    pool = Pool(processes=process_num)
    mp_results = [pool.apply_async(get_dispersion_list_worker, \
                (group_data, marginal_list[i], domain, q, large_group_indices)
            ) for i in range(len(marginal_list))
        ]
    mp_results = [res.get() for res in mp_results]

    disp_list = []
    for item in mp_results:
        disp_list.extend(item)

    pool.close()
    pool.join()

    time_cost = time.time() - start_time
    print('time cost: {:.4f}'.format(time_cost))

    return disp_list
    

# get latent marginal dispersion
def get_dispersion_list_worker(group_data, marginal_set, domain, q, \
    large_group_indices):

    group_data = group_data[large_group_indices]
    q = q[large_group_indices]

    disp_list = []
    for marginal in marginal_set:
        disp = get_dispersion(group_data, marginal, q, domain)
        disp_list.append((marginal, disp))

    return disp_list




def get_dispersion(group_data, marginal, q, domain):

    marginal_domain = domain.project(marginal)
    group_num = len(group_data)

    temp_shape = [len(group_data)]
    temp_shape.extend(marginal_domain.shape)
    group_hist = np.zeros(shape=temp_shape, dtype=float)

    for i in range(group_num):
        # drop group id
        group = group_data[i][:, 1:]
        hist = tools.get_histogram(marginal, group, domain)
        hist /= len(group)
        group_hist[i] = hist

    temp_shape = list(q.shape[1:])
    temp_shape.extend(marginal_domain.shape)
    average_hist = np.zeros(shape=temp_shape, dtype=float)

    latent_type_list = [list(range(i)) for i in q.shape[1:]]

    dist_sum = 0
    for latent_type in itertools.product(*tuple(latent_type_list)):
        slc = [slice(None),]
        slc.extend(list(latent_type))
        slc = tuple(slc)
        q_latent_type = q[slc]

        weight_sum = np.sum(q_latent_type)
        weight_sum += 1e-8

        shape = [1]*len(group_hist.shape)
        shape[0] = -1
        weighted_hist = q_latent_type.reshape(shape)
        weighted_hist = np.multiply(group_hist, weighted_hist)
        # print(y, weight_sum)
        average_hist = np.sum(weighted_hist, axis=0) / weight_sum

        temp_shape = [1]
        temp_shape.extend(marginal_domain.shape)
        average_hist = average_hist.reshape(temp_shape)


        temp_axis = tuple(range(1, len(marginal)+1))
        dist_array = np.sum(np.abs(group_hist - average_hist), axis=temp_axis) / 2

        dist_sum += np.sum(np.multiply(dist_array, q_latent_type))
        # print(temp)

    return dist_sum

# use group_length to weight q to get the distribution of the latent variable
def get_y_dist(q, group_length):

    shape = [1] * len(q.shape)
    shape[0] = -1
    group_length = group_length.reshape(shape)
    y_dist = np.multiply(group_length, q)
    y_dist = np.sum(y_dist, axis=0)
    return y_dist



def get_latent_variable_dist_worker(log_alpha, log_norm_type_size, potential, group_data, \
    latent_domain, partition_func, log_likelihood_ratio, size1_type0=True, retype=False):

    # # debug
    # print('log_likelihood_ratio', log_likelihood_ratio)
    
    default_q = np.zeros(shape=latent_domain.shape, dtype=float)
    default_q[tuple([0,]*len(latent_domain.shape))] = 1

    group_num = len(group_data)
    q_shape = [group_num,]
    q_shape.extend(latent_domain.shape)
    q = np.zeros(shape=q_shape, dtype=float)

    latent_type_list = list(list(range(i)) for i in latent_domain.shape)
    type0 = tuple([0]*len(latent_domain))

    if not log_norm_type_size is None:
        max_size = log_norm_type_size.shape[-1] - 1

    for i in range(group_num):

        size = len(group_data[i]) - 1
        if not log_norm_type_size is None:   
            size = min(max_size, size)

        if size1_type0 and size == 0:
            q[i] = default_q
            continue
        
        for latent_type in itertools.product(*tuple(latent_type_list)):

            pos = [i,]
            pos.extend(latent_type)
            pos = tuple(pos)

            # q[pos] += log_alpha[latent_type]

            if not log_norm_type_size is None:
                q[pos] += log_norm_type_size[tuple(latent_type)][size]

            evidence = 0
            for record in group_data[i]:
                # exp may cause overflow
                evidence += CRF_tools.get_log_prob(potential, record[1:], latent_type, latent_domain, \
                    partition_func[latent_type])
            # evidence *= log_likelihood_ratio
            q[pos] += evidence

            q[pos] *= log_likelihood_ratio
        
        if size1_type0:
            q[i][type0] = np.min(q[i]) - 10 # should not be classified as type0

        # print(i, q[i])

        temp_q = q[i].copy()
        temp_sum = scipy.special.logsumexp(q[i])
        # print(q[i], temp_sum)
        q[i] -= temp_sum
        q[i] = np.exp(q[i])

        

        if np.isnan(q[i]).any():
            print('error: detected nan q')
            print(temp_q, temp_sum)
            exit(0)

    # group_length = [len(group) for group in group_data]
    # q = np.repeat(q, group_length, axis=0)
    if retype:
        pass
    # to do

    return q


# get q (the disttribution of the latent variable) of each group
def get_latent_variable_dist(alpha, log_norm_type_size, potential, group_data, \
    latent_domain, partition_func, log_likelihood_ratio, process_num, size1_type0=True, retype=False):
    print('calculating latent variable')
    print('size1_type0', size1_type0)
    start_time = time.time()
    potential = potential.copy()
    potential.to_cpu()

    # print('alpha', alpha)

    alpha = np.array(alpha)
    alpha = np.log(alpha)


    group_num = len(group_data)

    length = int(group_num/process_num)
    idx_list = [i * length for i in range(process_num)]
    idx_list.append(group_num)
    
    pool = Pool(processes=process_num)
    # print('idx list', idx_list)
    mp_results = [pool.apply_async(get_latent_variable_dist_worker, \
            (alpha, log_norm_type_size, potential, group_data[idx_list[i]: idx_list[i+1]], \
            latent_domain, partition_func, log_likelihood_ratio, size1_type0, retype)
        ) for i in range(process_num)]
    mp_results = [res.get() for res in mp_results]
    q = np.concatenate(mp_results, axis=0)
    # print(q.shape, group_data.shape)
    # assert(len(q) == sum([len(group) for group in group_data]))

    pool.close()
    pool.join()

    time_cost = time.time() - start_time
    print('time cost: {:.4f}'.format(time_cost))

    return q

# MLE + gradient ascent for CRF
def CRF_gradient_ascent(potential, marginal_to_clique, data_mu, marginal_list, \
    message_order, domain, noisy_data_num, latent_variable_set, iter_num, \
    print_interval=100, end_loss=None, min_it=0):

    if end_loss is not None:
        print('end loss: {:.4e}'.format(end_loss))

    temp_potential = potential.copy()

    temp_mu = CRF_latented_weighted_marginal(potential, marginal_to_clique,\
        marginal_list, message_order, noisy_data_num)

    temp_loss, temp_gradient = Potential.l2_marginal_loss(temp_mu, data_mu)

    clique_list = list(potential.keys())
    noisy_data_num = float(data_mu[marginal_list[0]].sum())
    lr = 1.0 / noisy_data_num
    step_size = lambda x: 2.0*lr
    print('learning rate: {:.2e}'.format(lr))

    for it in range(iter_num):
        start_time = time.time()

        potential, mu = temp_potential, temp_mu
        loss, gradient = temp_loss, temp_gradient

        if it % print_interval == 0 or it == iter_num - 1:
            print('    it: {}/{} time: {:.2f} lr: {:.4e} loss: {:.4e}'.format(it, \
                iter_num, time.time()-start_time, lr, temp_loss))
            if end_loss is not None and loss < end_loss and it > min_it:
                break

        lr = step_size(it)
        expanded_gradient = get_expanded_gradient(domain, clique_list, \
            gradient, marginal_to_clique)

        for i in range(25):
            temp_potential = potential - lr * expanded_gradient
            temp_mu = CRF_latented_weighted_marginal(temp_potential, marginal_to_clique,\
                marginal_list, message_order, noisy_data_num)
            temp_loss, temp_gradient = Potential.l2_marginal_loss(temp_mu, data_mu)

            if loss - temp_loss >= 0.5*lr*gradient.dot(mu - temp_mu):
                break
            lr *= 0.5

    return potential, temp_mu

def get_expanded_gradient(domain, clique_list, gradient, marginal_to_clique):
    expanded_gradient = Potential({clique: Factor.zeros(\
        domain.project(clique)) for clique in clique_list})
    for marginal in gradient:
        clique = marginal_to_clique[marginal]
        expanded_gradient[clique] += gradient[marginal]
    return expanded_gradient

# use q to get weighted model marginals and latent model marginals
def CRF_latented_weighted_marginal(potential, marginal_to_clique, marginal_list, \
    message_order, total):
    # print('get marginals')

    marginal_dict = {}

    clique_marginal_dict, partition_func = CRF_tools.belief_propagation(message_order, \
        potential, total)

    for marginal in marginal_list:
        clique =  marginal_to_clique[marginal]

        fact = clique_marginal_dict[clique].project(marginal)
        marginal_dict[marginal] = fact


    return Potential(marginal_dict)

# calculate marginals (or conditional marginals)
def get_model_marginal(potential, marginal_to_clique, marginal_list, \
    message_order, domain, total):
    clique_marginal_dict, partition_func = CRF_tools.belief_propagation(message_order, \
        potential, total)
    marginal_dict = {}
    for marginal in marginal_list:
        clique = marginal_to_clique[marginal]
        marginal_factor = clique_marginal_dict[clique].project(domain.project(marginal))
        marginal_dict[marginal] = marginal_factor
    return marginal_dict

def get_arbitrary_marginal(model, marginal_list, total):
    marginal_dict, _ = belief_propagation(model.message_order, model.potential, total)
    res_list = {}
    for marginal in marginal_list:
        for clique in marginal_dict:
            if set(marginal) <= set(clique):
                res_list[marginal] = marginal_dict[clique].project(model.domain.project(marginal))
                break
    return res_list

# get data marginal and update latent data marginal
def get_data_marginal(data, domain, weights, marginal_dict, marginal, \
    latent_variable_set, xp=cp):

    if len(latent_variable_set.intersection(marginal)) != 0:
        hist = tools.get_latent_weighted_histogram(\
            marginal, data, domain, weights, latent_variable_set)
        # print('get data marginal', xp.sum(hist)) # debug
        fact = Factor(domain.project(marginal), hist, xp)
        marginal_dict[marginal] = fact

    # marginals without latent variables do not change
    elif marginal not in marginal_dict:
        hist = tools.get_histogram(marginal, data, \
                domain)
        fact = Factor(domain.project(marginal), hist, xp)
        marginal_dict[marginal] = fact

        # print(marginal)
        # print(fact.values)

    # # debug
    # if marginal == (1, 18):
    #     print('returned (1, 18)')
    #     print(marginal_dict[(1, 18)].values)
    return marginal_dict[marginal]

def get_init_q(group_num, latent_domain):
    q = np.random.random_sample(size=(group_num, latent_domain.size()))
    q_sum = np.sum(q, axis=1)
    q_sum = q_sum.reshape((-1, 1))
    q = np.divide(q, q_sum)

    shape = [group_num,]
    shape.extend(latent_domain.shape)
    q = q.reshape(shape)


    return q

def get_init_argmax_q(group_num, latent_domain):
    shape = [group_num,]
    shape.extend(latent_domain.shape)
    q = np.zeros(shape=shape, dtype=float)

    types = np.random.choice(latent_domain.size(), size=group_num)
    idx_tuple_array = np.unravel_index(types, latent_domain.shape)
    idx_tuple_array = np.array(idx_tuple_array)
    idx_tuple_array = idx_tuple_array.T

    for i in range(group_num):
        q[i][tuple(idx_tuple_array[i])] = 1

    return q



def get_latent_variable_dist_mixture_worker(log_alpha, potential, data,\
    latent_domain, partition_func, ratio):

    q_list = []
    latent_type_list = list(list(range(i)) for i in latent_domain.shape)

    for i in range(len(data)):
        q = np.zeros(shape=latent_domain.shape, dtype=float)

        for latent_type in itertools.product(*tuple(latent_type_list)):
            q[latent_type] = log_alpha[latent_type]
            temp = CRF_tools.get_log_prob(potential, data[i], latent_type, latent_domain, \
                    partition_func[latent_type])
            # print(latent_type, temp, data[i])
            q[latent_type] += ratio * temp
                
        q -= scipy.special.logsumexp(q)
        q = np.exp(q)
        # print(q)

        if np.isnan(q).any():
            print('Error: detected nan q.')
            print('The model could be crashed.')
            print(q)
        
        q_list.append(q)
    
    q_list = np.array(q_list)
    # print(q_list)

    return q_list

def test(log_alpha, potential, data, latent_domain, partition_func):
    # print(5)
    q_list = []
    latent_type_list = list(list(range(i)) for i in latent_domain.shape)

    for i in range(len(data)):
        q = np.zeros(shape=latent_domain.shape, dtype=float)

        for latent_type in itertools.product(*tuple(latent_type_list)):
            q[latent_type] = log_alpha[latent_type]
            q[latent_type] += CRF_tools.get_log_prob(potential, data[i], latent_type, latent_domain, \
                    partition_func[latent_type])
        
        print(q.shape)
        print(q)
        q -= scipy.special.logsumexp(q)
        
        q_list.append(q)
    
    q_list = np.array(q_list)
    print(q_list)
    return data

def get_marginal_group_TVD(marginal, group_data1, group_data2, domain):
    assert(len(group_data1) == len(group_data2))
    sum_TVD = 0
    for i in range(len(group_data1)):
        hist1 = tools.get_histogram(marginal, group_data1[i], domain)
        hist2 = tools.get_histogram(marginal, group_data2[i], domain)
        TVD = tools.get_normalized_TVD_count(hist1, hist2)

        sum_TVD += TVD
    return sum_TVD


def get_latent_variable_dist_mixture(alpha, potential, data,\
    latent_domain, partition_func, ratio, process_num):
    print('calculating latent variable')
    print('log likelihood ratio:', ratio)
    start_time = time.time()
    potential = potential.copy()
    potential.to_cpu()

    alpha = np.array(alpha)
    log_alpha = np.log(alpha)

    pool = Pool(processes=process_num)
    data_len = len(data)
    length = int(len(data) / process_num + 1)
    mp_data = [data[i: i+length] for i in range(0, data_len, length)]

    mp_results = [pool.apply_async(get_latent_variable_dist_mixture_worker,\
                (log_alpha, potential, mp_data[i], latent_domain, \
                partition_func, ratio)
            ) for i in range(len(mp_data))
        ]
    mp_results = [res.get() for res in mp_results]
    q = np.concatenate(mp_results, axis=0)

    pool.close()
    pool.join()

    print('log_alpha:')
    print(log_alpha.flatten()[0:5])

    print('q:')
    for i in [10, 20, 1000, 5000]:
        print(tools.string_low_precision_array(q[i].flatten()[:10]))

    temp_q = q.reshape((len(q), -1))
    var = np.var(temp_q, axis=1)
    var = float(np.mean(var))

    temp = np.zeros(temp_q[0].shape, dtype=float)
    temp[0] = 1
    max_var = float(np.var(temp))
    print('q mean var: {:.4f}/{:.4f}'.format(var, max_var))

    time_cost = time.time() - start_time
    print('time cost: {:.4f}'.format(time_cost))

    return q

