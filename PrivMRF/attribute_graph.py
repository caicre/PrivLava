from .utils import tools
import networkx as nx
import numpy as np
import random
import itertools
import math
from .domain import Domain
from networkx.readwrite import json_graph
import json
import pickle
from .factor import Factor

class Status:
    def __init__(self, graph, adj):
        self.graph = graph.copy()
        self.adj = np.copy(adj)

    def get_neighbor_status(self, attr1, attr2, weight):
        neighbor_status = Status(self.graph, self.adj)
        if neighbor_status.adj[attr1, attr2] > 0:
            neighbor_status.adj[attr1, attr2] = 0
            neighbor_status.adj[attr2, attr1] = 0
            neighbor_status.graph.remove_edge(attr1, attr2)
            return neighbor_status
        elif neighbor_status.adj[attr1, attr2] == 0:
            neighbor_status.adj[attr1, attr2] = weight
            neighbor_status.adj[attr2, attr1] = weight
            neighbor_status.graph.add_edge(attr1, attr2)
            return neighbor_status
        return None

class AttributeGraph:
    def __init__(self, data, domain, attr_list, config, data_name):
        self.data = data.copy()
        self.domain = domain
        self.attr_list = attr_list
        self.config = config
        self.data_name = data_name

        self.attr_num = len(domain)
        self.privacy_budget = config['beta1'] * config['budget']

        self.entropy_map = {}
        self.TVD_map = {}
        self.MI_map = {}


        data_num_noise = math.sqrt(1/self.privacy_budget*0.01)
        self.data_num = int(len(self.data) + np.random.normal(scale=data_num_noise))

        # self.config['max_entropy_num'] = (self.attr_num ** 2)*2
        self.config['max_entropy_num'] = (self.attr_num ** 2)/2 + self.attr_num
        self.config['entropy_num_iter'] = self.attr_num*self.attr_num/2
        self.config['search_iter_num'] = int(self.attr_num*(self.attr_num-1))

        # optimal Gaussian Mehchanism
        # self.max_entropy_num = self.config['max_entropy_num']
        # sensitivity = tools.entropy_sensitivity(self.data_num)
        # self.entropy_noise = math.sqrt(self.max_entropy_num * sensitivity * sensitivity / self.privacy_budget)
        # entropy is for debugging and not used to generate synthetic data
        self.entropy_noise = 0

        max_edge_num = self.attr_num * (self.attr_num - 1)/2

        # sensitivity = tools.MI_sensitivity(self.data_num)
        # self.MI_noise = math.sqrt(max_edge_num * sensitivity * sensitivity / self.privacy_budget)

        sensitivity = tools.TVD_sensitivity(self.data_num) * self.config['sensitivity']
        self.TVD_noise = math.sqrt(max_edge_num * sensitivity * sensitivity / self.privacy_budget)

        estimated_noise_scale = 'nan'
        if self.config['beta2'] > 0:
            budget = (1-self.config['beta1']-self.config['beta3']) * self.config['budget']
            estimated_noise_scale = math.sqrt((self.attr_num + self.config['t']*self.attr_num) * self.config['sensitivity'] ** 2 / budget)
            self.max_measure_dom_2way = self.data_num / estimated_noise_scale / config['theta1']
            self.max_measure_dom_high_way = self.data_num / estimated_noise_scale / config['theta2']

        self.min_score = -1e8

        self.max_level = [max(self.attr_list[attr].level_to_size.keys()) for attr in range(self.attr_num)]

        # the best measure for each attr to avoid worst cases
        self.attr_measure = {}
        
        print('total privacy budget:     ', config['budget'])
        print('attribute graph budget:   ', self.privacy_budget)
        print('estimated noise scale:    ', estimated_noise_scale)
        print('max 2way measure dom:     ', self.max_measure_dom_2way)
        print('max high way measure dom: ', self.max_measure_dom_high_way)
        print('max edge number:          ', max_edge_num)
        print('TVD_noise:                ', self.TVD_noise)
        print('data num:                 ', self.data_num)

    def construct_model(self, init_cliques=[]):
        print('construct attribute graph')
        self.graph, entropy = self.local_search(init_cliques)
        # self.graph, entropy = self.pairwise_graph()

        self.attr_to_level = None

        self.get_cliques_marginal()

        # add most valuable measures for attrs
        self.measure_list = []
        for attr in self.attr_measure:
            self.measure_list.append(self.attr_measure[attr][0])
        print('attr measure',  self.measure_list)
        self.measure_list = \
            list(tools.deduplicate_measure_set(tuple(sorted(measure)) \
                for measure in self.measure_list))

        # determine the level of attribute hierarchy
        if self.config['enable_attribute_hierarchy']:
            for measure in self.measure_list:
                if len(measure) == 2:
                    # print(measure, tools.measure_level_size(measure, self.attr_list, self.attr_to_level), self.max_measure_dom_2way)
                    for i in range(3):
                        if tools.measure_level_size(measure, self.attr_list, self.attr_to_level) > self.max_measure_dom_2way:
                            tools.improve_level(measure, self.attr_list, self.attr_to_level, self.config['max_level_gap'])
            
            for attr in self.attr_to_level:
                if self.max_level[attr] > 0:
                    print('  attr: {}, level: {} max_level: {}'.format(attr, self.attr_to_level[attr], self.max_level[attr]))

        attr_flag = [0] * self.attr_num
        for measure in self.measure_list:
            for attr in measure:
                attr_flag[attr] += 1
        for attr in range(self.attr_num):
            if attr_flag[attr] == 0: 
                self.measure_list.append(tuple([attr]))
                print('single attr measure:', attr)


        data = json_graph.node_link_data(self.graph)
        with open('./temp/graph_'+self.config['exp_name']+'.json', 'w') as out_file:
            json.dump(data, out_file)
        with open('./temp/measure_'+self.config['exp_name']+'.json', 'w') as out_file:
            json.dump(self.measure_list, out_file)

        return self.graph, self.measure_list, self.attr_list, self.attr_to_level, entropy

    @staticmethod
    def save_model(model, path):
        with open(path, 'wb') as out_file:
            pickle.dump(model, out_file)

    @staticmethod
    def load_model(path, config=None):
        with open(path, 'rb') as out_file:
            model = pickle.load(out_file)
        if config != None:
            model.config = config
        return model

    def get_all_n_way_measure(self, n):
        measure_list = []
        self.maximal_cliques = list(nx.find_cliques(self.graph))
        for clique in self.maximal_cliques:
            for measure in itertools.combinations(clique, n):
                measure_list.append(measure)
        return measure_list

    # randomly enumerate a next edge to find a graph that minimize KL divergence and get measures
    def local_search(self, init_cliques):
        # with open('./temp/graph_'+self.config['exp_name']+'.json', 'r') as in_file:
        #     graph = json_graph.node_link_graph(json.load(in_file))
        #     return graph, -1
        start_G = nx.Graph()
        start_G.add_nodes_from(list(range(self.attr_num)))
        start_adj = np.zeros(shape=(self.attr_num, self.attr_num), dtype=float)

        for clique in init_cliques:
            for attr1, attr2 in itertools.combinations(clique, 2):
                start_G.add_edge(attr1, attr2)
                start_adj[attr1][attr2] = 1
                start_adj[attr2][attr1] = 1

        data_entropy = tools.dp_entropy({}, self.data, self.domain, list(range(self.attr_num)), 0)[0]
        print('data entropy: {}'.format(data_entropy))

        start_status = Status(start_G, start_adj)
        
        current_status = start_status
        current_score, current_entropy, current_size = self.score(start_status.graph, self.entropy_map)
        print('  init score: {:.4f} query_num: {} size: {:.4e}, edge_num: {} entropy: {:.2f}'\
            .format(current_score, len(self.entropy_map), current_size, \
            current_status.graph.number_of_edges(), current_entropy))

        temp1 = 0
        # temp2 = 0
        for attr1 in range(self.attr_num):
            for attr2 in range(attr1+1, self.attr_num):
                temp1 += tools.dp_TVD(self.TVD_map, self.data, self.domain, [attr1, attr2], self.TVD_noise)[0]
        #         temp2 += tools.dp_mutual_info(self.MI_map, self.entropy_map, self.data, self.domain, [attr1, attr2], self.MI_noise)[0]
        print('average TVD and noise:', temp1/(self.attr_num*(self.attr_num-1)/2), self.TVD_noise)
        # print('average MI and noise:', temp2/self.attr_num**2/2, self.MI_noise)

        # if self.config['score'] == 'pairwsie_TVD':
        #     score_func = self.pairwise_score_TVD
        # elif self.config['score'] == 'pairwsie_MI':
        #     score_func = self.pairwise_score_MI
        # elif self.config['score'] == 'pairwise_entropy':
        #     score_func = self.pairwise_score

        score_func = self.pairwise_score_TVD

        search_iter_num = self.config['search_iter_num']
        check_entropy_map = {}
        for i in range(search_iter_num):

            # generate edge list
            best_score = self.min_score
            best_status = None
            edge_list = []
            for attr1 in range(self.attr_num):
                for attr2 in range(attr1+1, self.attr_num):
                    if current_status.adj[attr1][attr2] == 0:
                        edge_list.append((attr1, attr2))
            random.shuffle(edge_list)

            for attr1, attr2 in edge_list:
                status = current_status.get_neighbor_status(attr1, attr2, 1)
                status_score, mutual_info, size = score_func(status.graph)
                # print('status score:', status_score)
                if status_score > best_score:
                    best_score = status_score
                    best_status = status
                    best_mutual_info = mutual_info
                    current_size = size

            if best_status == None:
                break

            current_status = best_status
            current_score = best_score
            current_mutual_info = best_mutual_info

            # print entropy for debug, which could be very slow as it need to calculate the entropy of
            # large marginals
            # _, entropy, _ = self.score(current_status.graph, check_entropy_map)
            entropy = -1

            print('  iter: {}/{} score: {:.2f} size: {:.2e}, edge_num: {} mutual_info: {:.2f}'\
                .format(i, search_iter_num, current_score, current_size, \
                current_status.graph.number_of_edges(), current_mutual_info))

        graph = current_status.graph
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        tools.print_graph(graph, './temp/graph_'+self.config['exp_name']+'.png')

        return graph, entropy

    def get_cliques_marginal(self):
        self.R_noise = None

        # add budget for constructing inner Bayesian network

        print('construct Bayesian Network for maximal cliques')
        self.maximal_cliques = list(nx.find_cliques(self.graph))
        for i in range(len(self.maximal_cliques)):
            clique = self.maximal_cliques[i]
            print('  {}, {}/{}'.format(clique, i, len(self.maximal_cliques)))
            self.get_marginal(clique)
        

    
    def maximal_parents(self, parents_set, dom):
        if dom < 1:
            return set()
        if len(parents_set) < 1:
            return set([tuple(),])

        # print(parents_set, dom)

        parents_set = parents_set.copy()
        attr = parents_set.pop()
        res1 = self.maximal_parents(parents_set, dom)

        # If there exists a high level subattr satisfying dom limitation
        # It should be considered as levels don't influence the scores of parents
        # debug
        if self.config['enable_attribute_hierarchy']:
            level = max(self.max_level[attr]-2, 0)
        else:
            level = max(self.max_level[attr], 0)
        
        current_attr_size = self.attr_list[attr].level_to_size[level]
        res2 = self.maximal_parents(parents_set, dom/current_attr_size)
        for ps in res2:
            if ps in res1:
                res1.remove(ps)
            temp = list(ps)
            temp.append(attr)
            res1.add(tuple(sorted(temp)))

        return res1
    
    def select_parents(self, target_attr, parents_set):
        attr_parents = []

        dom_limit = self.max_measure_dom_high_way/self.domain.project([target_attr]).size()
        for size in range(1, self.config['max_measure_attr_num']):
            for comb in itertools.combinations(parents_set, size):
                if self.domain.project(comb).size() < dom_limit:
                    attr_parents.append(comb)

        best_parent = []
        best_score = self.min_score
        for comb in attr_parents:
            score = self.attr_parents_score(target_attr, list(comb))
            if score > best_score:
                best_score = score
                best_parent = comb

        marginal = [target_attr]
        marginal.extend(best_parent)
        marginal = tuple(sorted(marginal))

        return (marginal, target_attr), best_score

    # use greedy bayes to find measures
    def get_marginal(self, clique):
        remaining_attributes = list(clique.copy())
        target_attr = random.choice(remaining_attributes)
        remaining_attributes.remove(target_attr)

        measure_list = []
        while len(remaining_attributes) != 0:
            target_attr = remaining_attributes[0]
            remaining_attributes.remove(target_attr)
            parents_set = clique.copy()
            parents_set.remove(target_attr)

            marginal_item, score = self.select_parents(target_attr, parents_set)
            target_attr = marginal_item[1]
            best_parents = marginal_item[0]

            if len(marginal_item[0]) > 1:
                print('    {}<={} score: {}'.format(target_attr, best_parents, score))
                best_parents = tuple(sorted(best_parents))
                measure_list.append(best_parents)

                if target_attr not in self.attr_measure or self.attr_measure[target_attr][1] < score:
                    self.attr_measure[target_attr] = [best_parents, score]
      
        return measure_list
    

    # TVD correlation + correlation-based feature selector
    # ref: Correlation-based Feature Selection for Machine Learning
    def attr_parents_score(self, attr, parents):
        parents = list(parents).copy()
        # If the measure constructed by the only parents of one attribute is too large,
        # it should be add to the model regardless of its size as it provide basic correlation we can get
        # It will aslo be used to determine attribute hierarchy if possible
        # however, if a measure is too large, the noise will also be very large.
        # It will even affect the entire model
        if len(parents) == 1:
            dom_limit = self.max_measure_dom_2way
            if self.config['enable_attribute_hierarchy']:
                for pa in parents:
                    if self.attr_to_level[pa] > 0:
                        dom_limit = self.max_measure_dom_2way * 5
                if self.attr_to_level[attr] > 0:
                    dom_limit = self.max_measure_dom_2way * 5
        else:
            dom_limit = self.max_measure_dom_high_way
        if self.domain.project(parents).size() * self.domain.dict[attr]['size'] > dom_limit:
            return self.min_score
        numerator = 0
        for i in parents:
            # it will reuse the TVD queried before
            if self.config['score'] == 'pairwsie_TVD':
                numerator += tools.dp_TVD(self.TVD_map, self.data, self.domain, [attr, i], self.TVD_noise)[1]
            elif self.config['score'] == 'pairwsie_MI':
                numerator += tools.dp_mutual_info(self.MI_map, self.entropy_map, self.data, self.domain, [attr, i], self.MI_noise)[1]
            else:
                print('score must be pairwsie_TVD or pairwsie_MI')
                exit(-1)
        denominator = len(parents)
        for i in range(len(parents)):
            for j in range(i+1, len(parents)):
                if self.config['score'] == 'pairwsie_TVD':
                    denominator += tools.dp_TVD(self.TVD_map, self.data, self.domain, [i, j], self.TVD_noise)[1]
                elif self.config['score'] == 'pairwsie_MI':
                    numerator += tools.dp_mutual_info(self.MI_map, self.entropy_map, self.data, self.domain, [i, j], self.MI_noise)[1]
        # denominator might be smaller than 0 because of noise. We set it at least 1 as len(parents) >= 1
        if denominator < 1:
            denominator = 1
        return numerator/math.sqrt(denominator)


    def pairwise_score_MI(self, graph):
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        
        # junction tree size
        size = 0
        for clique in nx.find_cliques(graph):
            temp_size = self.domain.project(clique).size()
            # if temp_size > self.config['max_clique_size'] or len(clique) > 15:
            if temp_size > self.config['max_clique_size']:
                return self.min_score, 0, size
            size += temp_size
        if size > self.config['max_parameter_size']:
            return self.min_score, 0, size

        noisy_MI = 0
        for attr1, attr2 in graph.edges:
            noisy_MI += tools.dp_mutual_info(self.MI_map, self.entropy_map, self.data, self.domain, [attr1, attr2], self.MI_noise)[1]
        
        score = noisy_MI - self.config['size_penalty']*size
        return score, noisy_MI, size
    
    def pairwise_score_TVD(self, graph):
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        
        # junction tree size
        size = 0
        for clique in nx.find_cliques(graph):
            temp_size = self.domain.project(clique).size()
            # if temp_size > self.config['max_clique_size'] or len(clique) > 15:
            if temp_size > self.config['max_clique_size']:
                return self.min_score, 0, size
            size += temp_size
        if size > self.config['max_parameter_size']:
            return self.min_score, 0, size

        noisy_TVD = 0
        for attr1, attr2 in graph.edges:
            noisy_TVD += tools.dp_TVD(self.TVD_map, self.data, self.domain, [attr1, attr2], self.TVD_noise)[1]
        
        score = noisy_TVD - self.config['size_penalty']*size
        return score, noisy_TVD, size

    def pairwise_score(self, graph):
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        
        # junction tree size
        size = 0
        for clique in nx.find_cliques(graph):
            temp_size = self.domain.project(clique).size()
            # if temp_size > self.config['max_clique_size'] or len(clique) > 15:
            if temp_size > self.config['max_clique_size']:
                return self.min_score, 0, size
            size += temp_size
        if size > self.config['max_parameter_size']:
            return self.min_score, 0, size

        noisy_mutual_info = 0
        mutual_info = 0
        for attr1, attr2 in graph.edges:
            entropy, noisy_entropy  = tools.dp_entropy(self.entropy_map, self.data, self.domain, [attr1, attr2], self.entropy_noise)
            mutual_info             -= entropy
            noisy_mutual_info       -= noisy_entropy

            entropy, noisy_entropy  = tools.dp_entropy(self.entropy_map, self.data, self.domain, [attr1], self.entropy_noise)
            mutual_info             += entropy
            noisy_mutual_info       += noisy_entropy

            entropy, noisy_entropy  = tools.dp_entropy(self.entropy_map, self.data, self.domain, [attr2], self.entropy_noise)
            mutual_info             += entropy
            noisy_mutual_info       += noisy_entropy

        score = noisy_mutual_info - self.config['size_penalty']*size
        return score, mutual_info, size

    def score(self, graph, entropy_map):
        graph = graph.copy()
        if not nx.is_chordal(graph):
            graph = tools.triangulate(graph)
        
        clique_list = [tuple(sorted(clique)) for clique in nx.find_cliques(graph)]
        clique_graph = nx.Graph()
        clique_graph.add_nodes_from(clique_list)
        for clique1, clique2 in itertools.combinations(clique_list, 2):
            clique_graph.add_edge(clique1, clique2, weight=-len(set(clique1) & set(clique2)))
        junction_tree = nx.minimum_spanning_tree(clique_graph)
        # print('    clique list', len(clique_list), clique_list)

        # junction tree size
        size = 0
        for clique in clique_list:
            temp_size = self.domain.project(clique).size()
            # if temp_size > self.config['max_clique_size'] or len(clique) > 15:
            if temp_size > self.config['max_clique_size']:
                return self.min_score, 0, size
            size += temp_size
        if size > self.config['max_parameter_size']:
            return self.min_score, 0, size

        # KL divergence
        # model entropy is for debugging and can not be used for constructing model as they are not dp
        KL_divergence = 0
        model_entropy = 0
        entropy, noisy_entropy = tools.dp_entropy(entropy_map, self.data, self.domain, clique_list[0], self.entropy_noise)
        KL_divergence += noisy_entropy
        model_entropy += entropy
        for start, clique in nx.dfs_edges(junction_tree, source=clique_list[0]):
            entropy, noisy_entropy = tools.dp_entropy(entropy_map, self.data, self.domain, clique, self.entropy_noise)
            KL_divergence += noisy_entropy
            model_entropy += entropy
            separator = set(start) & set(clique)
            if len(separator) != 0:
                entropy, noisy_entropy = tools.dp_entropy(entropy_map, self.data, self.domain, separator, self.entropy_noise)
                KL_divergence -= noisy_entropy
                model_entropy -= entropy

        # print('KL', KL_divergence, size)
        score = -KL_divergence - self.config['size_penalty']*size

        return score, model_entropy, size