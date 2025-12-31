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
# algorithms for building running graphs containing latent variables
import itertools
import networkx as nx
from . import tools
import json
import numpy as np
import itertools

def score_clique_limit(graph, edge_dict, \
    max_clique_size, max_parameter_size, domain):
    min_score = -1e7
    
    graph = graph.copy()
    if not nx.is_chordal(graph):
        graph = tools.triangulate(graph)

    sum_size = 0
    clique_list = [tuple(sorted(clique)) for clique in nx.find_cliques(graph)]
    for clique in clique_list:
        clique_size = domain.project(clique).size()
        if clique_size > max_clique_size:
            return min_score, "exceed max clique size" + str(clique_size), -1

        sum_size += clique_size
        if sum_size > max_parameter_size:
            return min_score, "exceed max parameter size" + str(sum_size), -1

    score = 0
    for edge in graph.edges:
        score += edge_dict[edge]

    return score, "succeed", sum_size

def CFS_score(target, marginal, edge_dict):
    numerator = 0
    denominator = len(marginal)

    feature_list = list(marginal)
    feature_list.remove(target)

    for comb in itertools.combinations(feature_list, 2):
        denominator += edge_dict[comb]
    denominator *= 2
    denominator = denominator ** 0.5

    for attr in feature_list:
        comb = tuple(sorted([target, attr]))
        numerator += edge_dict[comb]
    
    return numerator/denominator

class CRFGraph:
    def __init__(self, config, domain):
        """ config:
        max_clique_size:    max size for cliques of the junction tree
        """
        self.config = config
        self.domain = domain
        self.attr_list = domain.attr_list
        print(self.attr_list)
        print(self.domain)

        self.latent_variable_dict = {}
        self.observed_variable_dict = {}

        for attr, value in domain.dict.items():
            if 'latent' in value and value['latent']:
                self.latent_variable_dict[attr] = value['size']
            else:
                self.observed_variable_dict[attr] = value['size']
                
        self.attr_num = len(self.domain)

    def greedy_add(self, data, R_sensitivity, dom_limit, noisy_data_num):
        """ First connect the latent variable to the correlated varibales. Then, 
        randomly enumerate an edge and add it to the graph such that the graph
        contains no large cliques.
        """

        # print(self.domain)
        # print(data[:10])
        # print(graph.nodes)
        # print(graph.edges)

        data_num = len(data)
        ob_attr_num = len(self.observed_variable_dict)

        if self.config['budget'] is None:
            budget = tools.get_privacy_budget(self.config['epsilon'])
        else:
            budget = self.config['budget']

        R_score_num = ob_attr_num * (ob_attr_num - 1) / 2 + 1e-4
        R_score_noise = (R_sensitivity ** 2 / (budget * self.config['beta0'] / R_score_num)) ** 0.5

        # R_score_noise = 1e-8 # debug

        print('R score noise: {:.2f}, ratio: {:.4f}'.format(R_score_noise, R_score_noise/data_num))


        MI_map = {}
        entropy_map = {}
        if self.config['load_edge_score']:
            edge_list = json.load(\
                open('./temp/'+self.config['exp_name']+'_edge_score.json', 'r'))
            temp_edge_list = []
            for edge, score in edge_list:
                temp_edge_list.append((tuple(edge), score))
            edge_list = temp_edge_list
        else:    
            edge_list = []
            for i, j in itertools.combinations(list(self.observed_variable_dict.keys()), 2):
                edge = (i, j)
                # true_edge_score = tools.get_mutual_info(MI_map, entropy_map, edge, \
                #     data, self.domain)

                true_edge_score = tools.get_R_score(data, self.domain, edge)
                noisy_edge_score = true_edge_score + np.random.normal(scale=R_score_noise)

                edge_list.append((edge, noisy_edge_score, true_edge_score))
            edge_list.sort(reverse=True, key=lambda x: x[1])

            json.dump(edge_list, \
                open('./temp/'+self.config['exp_name']+'_edge_score.json', 'w'))

        graph = nx.Graph()
        graph.add_nodes_from(self.domain.attr_list)

        # scores of edges between latent vars and ob vars
        for latent_var in self.latent_variable_dict:
            for observed_var in self.observed_variable_dict:
                # the latent variable should be in every clique since we are going
                # to calculate the conditional marginal probabilities that always contain
                # the latent variable.
                edge_list.append(((observed_var, latent_var), 1e-5, 1e-5))
                graph.add_edge(observed_var, latent_var)

        for var1, var2 in itertools.combinations(list(self.latent_variable_dict.keys()), 2):
            edge_list.append(((var1, var2), 1e-5, 1e-5))
            graph.add_edge(var1, var2)
        edge_dict = {edge: noisy_score for edge, noisy_score, true_score in edge_list}

        new_edge_list = []
        for edge, noisy_score, true_score in edge_list:
            if edge[0] not in graph.adj[edge[1]]:
                new_edge_list.append((edge, noisy_score, true_score))
        edge_list = new_edge_list

        print(edge_list)

        score_func = score_clique_limit
        max_clique_size = self.config['max_clique_size']
        max_parameter_size = self.config['max_parameter_size']

        score, msg, size = score_func(graph, edge_dict, \
            max_clique_size, max_parameter_size, self.domain)
        add_flag = True
        while add_flag:
            add_flag = False
            for edge, noisy_score, true_score in edge_list:

                temp_graph = graph.copy()
                temp_graph.add_edge(*edge)
                temp_score, msg, temp_size = score_func(temp_graph, edge_dict,\
                    max_clique_size, max_parameter_size, self.domain)

                if temp_score > score:
                    score = temp_score
                    graph = temp_graph
                    size = temp_size
                    add_flag = True
                    print('num: {:d}, score: {:.2e}, edge: {}, edge score: {:.2e}, {:.2e}, size: {:.2e},'\
                        .format(nx.classes.function.number_of_edges(graph), \
                        score, edge, noisy_score, true_score, size))
                    break

        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)

        tools.print_graph(graph, './temp/graph_'+self.config['exp_name']+'.png')

        normalized_edge_dict = edge_dict.copy()
        for edge in normalized_edge_dict:
            normalized_edge_dict[edge] = normalized_edge_dict[edge]/noisy_data_num
            # print(edge, normalized_edge_dict[edge])

        if not self.config['build_graph_marginal']:
            return graph

        attr_to_marginal = {attr: (None, -1e9) for attr in self.observed_variable_dict}
        for clique in nx.find_cliques(graph):
            ob_clique = set(clique).intersection(set(self.observed_variable_dict.keys()))
            ob_clique = sorted(ob_clique)
            # print('ob_clique', ob_clique)
            for marginal_size in range(self.config['marginal_max_attr_num']+1):
                for marginal in itertools.combinations(ob_clique, marginal_size):
                    if self.domain.project(marginal).size() > dom_limit:
                        continue
                    for attr in marginal:
                        score = CFS_score(attr, marginal, normalized_edge_dict)
                        if score > attr_to_marginal[attr][1]:
                            attr_to_marginal[attr] = (marginal, score)
        init_marginal = [marginal for marginal, score in attr_to_marginal.values()]
        # print(attr_to_marginal)
        return graph, init_marginal
