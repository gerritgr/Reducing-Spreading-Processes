import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
import random
plt.clf()
import create_network_ctmc, clustering, compare_ctmc
plt.ion()
os.system('mkdir results')
from collections import Counter

def scatter_plot(x, y, outname='output_diff.pdf', cluster_num_list=None):
    plt.clf()
    plt.scatter(x,y)
    df = {'state_space_size': x, 'error':y }
    if cluster_num_list is not None:
        df = {'state_space_size': x, 'error': y, 'node_partitions':cluster_num_list}
    df = pd.DataFrame.from_records(df)
    df.to_csv(outname + '_data.csv', index=False)
    plt.xlim(xmin=0)
    plt.savefig(outname)
    print('print scatter: ',outname)

# cluster_method takes node_partition, and state
def iter_clusters(contact_network, node_method, cluster_method, rule_system, name, sol_orig=None, initf=None):
    contact_network = nx.convert_node_labels_to_integers(contact_network)
    print('currently evaluating: '+name)
    os.system('mkdir output')
    nodes = contact_network.number_of_nodes()
    diff_list = list()
    state_space_list = list()
    cluster_num_list = list()

    #sol_orig = None
    for cluster_num in range(nodes-1):
        cluster_num += 1
        print('currently evaluating: ' + name, ' with node partitions ', cluster_num)
        node_partition = node_method(contact_network, cluster_num)
        assert(len(set(node_partition)) == cluster_num)
        G, state_space = create_network_ctmc.create_ctmc(contact_network, ['S', 'I'], rule_system, output_name=name+'_'+str(cluster_num))
        if initf is None:
            init = [1.0]*(G.number_of_nodes())
        else:
            init = [initf(s) for s in state_space]
        init_z = np.sum(init)
        init = [i/init_z for i in init]
        ctmc_state_partitioning = clustering.partition_state_space(state_space, contact_network, node_partition, cluster_method)
        diff, G, LG, t, result_lists_1, result_lists_2 = compare_ctmc.compare_with_lumped(G, init, ctmc_state_partitioning, name+'_'+str(cluster_num), sol_orig)
        sol_orig = (t, result_lists_1)
        state_space_size_original = G.number_of_nodes()
        state_sace_size_lumped = LG.number_of_nodes()
        max_diff = np.max(diff)
        diff_list.append(max_diff)
        cluster_num_list.append(cluster_num)
        state_space_list.append(state_sace_size_lumped)
        scatter_plot(state_space_list+[state_space_size_original], diff_list+[0.0], 'results/'+name+'_DIFFERENCE.pdf', cluster_num_list+[nodes])
        print('original state space size: ',state_space_size_original, '   lumped: ', state_sace_size_lumped)
        if (state_space_size_original == state_sace_size_lumped or max_diff < 0.000001) and (range(nodes)[-1] != cluster_num):
            diff_list.append(0.0)
            cluster_num_list.append(nodes)  #append(range(nodes)[-1])
            state_space_list.append(state_space_size_original)
            scatter_plot(state_space_list, diff_list, 'results/' + name + '_DIFFERENCE.pdf', cluster_num_list)
            print('early stop with max_diff: ',max_diff)
            return sol_orig
    return sol_orig


# fix is states = ['S', 'I']
def multi_test(contact_networks, node_methods, cluster_methods, rules, exp_name, initf=None):
    sol_orig = None
    for rule_name, rule_system in rules.items():
        for network_name, g in contact_networks.items():
            for node_methods_name, node_method in node_methods.items():
                for cluster_method_name, cluster_method in cluster_methods.items():
                    try:
                        exp_name_new = '{}_{}_{}_{}_{}'.format(exp_name, network_name, node_methods_name, cluster_method_name, rule_name)
                        sol_orig = iter_clusters(g, node_method, cluster_method, rule_system, exp_name_new, sol_orig=sol_orig, initf=initf)
                    except Exception as e:
                        import traceback
                        sol_orig = None
                        print(traceback.print_exc())
                        print('!!!ERROR!!!   '+str(e))


def oreo():
    rand_graph = nx.Graph([(1,2),(1,3),(3,2),(3,4),(7,5),(4,5),(4,6),(6,7),(7,5),(2,11),(8,9),(9,10),(10,8),(10,11),(11,12),(13,8),(12,13),(13,14)])#,(14,12),(14,15),(15,16),(16,8),(16,9),(16,12)])

    rand_graph = nx.erdos_renyi_graph(13,0.5)

    contact_networks = {'cycle':nx.cycle_graph(10), 'grid': nx.grid_graph(dim=[4,4]), 'rgraph': rand_graph}
    contact_networks = {'rgraph': rand_graph}
    node_methods = {'random': clustering.random_node_partition, 'spectral': clustering.spectral_node_partition, 'degree':clustering.degree_based_node_partition}
    cluster_methods = {'neighbor': clustering.neighbor_based_counting, 'node': clustering.node_based_counting, 'edge':clustering.tuple_based_counting}
    cluster_methods = { 'node': clustering.node_based_counting, 'edge':clustering.tuple_based_counting}
    rules = {'SIS': [('S', 'I', lambda x: x['I'] ), ('I','S',lambda x: 1.1)], 'SI': [('S', 'I', lambda x: x['I'])]}
    rules = {'SIS': [('S', 'I', lambda x: x['I'] ), ('I','S',lambda x: 1.3)]}
    multi_test(contact_networks, node_methods, cluster_methods, rules, 'oreo')





# This is the model used for numerical results
# Random partitioning might vary
oreo()
