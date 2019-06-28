import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import main
plt.ion()


def model_expansion(result_list, cmtc_state_partitioning):
    result_list_exp = [list()]*len(cmtc_state_partitioning)

    for i, y_list in enumerate(result_list):
        exp_positions = [l for l in range(len(cmtc_state_partitioning)) if cmtc_state_partitioning[l] == i]
        #exp_positions = [0,1,2,3]
        y_list_exp = [v/len(exp_positions) for v in y_list]
        for j in exp_positions:
            result_list_exp[j] = y_list_exp

    return result_list_exp

def check_partitioning(ctmc_state_partitioning):
    for p in ctmc_state_partitioning:
        assert('int' in str(type(p)))
        assert(p >= 0)
    max_p = np.max(ctmc_state_partitioning)
    num_partitions = len(set(ctmc_state_partitioning))
    assert(max_p+1 == num_partitions)


def correct_partitioning(ctmc_state_partitioning):
    partition_elems = list(set(ctmc_state_partitioning))
    try:
        partition_elems = sorted(partition_elems)
    except:
        partition_elems = [str(v) for v in partition_elems]
        ctmc_state_partitioning = [str(v) for v in ctmc_state_partitioning]
        partition_elems = sorted(partition_elems)

    new_partitioning = [partition_elems.index(v) for v in ctmc_state_partitioning]
    return new_partitioning


def reduce_initial_state(initial_state, ctmc_state_partitioning):
    ctmc_state_partitioning = correct_partitioning(ctmc_state_partitioning)
    check_partitioning(ctmc_state_partitioning)
    lumped_init = [0.0] * len(set(ctmc_state_partitioning))
    for i, v in enumerate(initial_state):
        p = ctmc_state_partitioning[i]
        lumped_init[p] += v
    return lumped_init

@main.timeit
def lump_ctmc(G, ctmc_state_partitioning, lumped_graph_path = 'lumped_graph.edgelist'):
    G = nx.convert_node_labels_to_integers(G)
    assert(G.number_of_nodes() == len(ctmc_state_partitioning))
    ctmc_state_partitioning = correct_partitioning(ctmc_state_partitioning)
    check_partitioning(ctmc_state_partitioning)

    new_states = sorted(list(set(ctmc_state_partitioning)))
    LG = nx.DiGraph()
    for s in new_states:
        LG.add_node(s)
        LG.nodes[s]['state'] = ''

    for old_node in G.nodes():
        new_node = ctmc_state_partitioning[old_node]
        try:
            LG.nodes[new_node]['state'] += G.nodes[old_node]['state'] + ','
        except:
            LG.nodes[new_node]['state'] += str(old_node) + ','


    for edge in G.edges.data():
        start = edge[0]
        target = edge[1]
        rate = edge[2]['weight']

        start_lumped = ctmc_state_partitioning[start]
        target_lumped = ctmc_state_partitioning[target]
        src_state_count = ctmc_state_partitioning.count(start_lumped)  # number of original ctmc states aggregated in the soure state
        rate_scaled = rate/src_state_count

        if start_lumped == target_lumped:
            continue
        if LG.has_edge(start_lumped, target_lumped):
            LG[start_lumped][target_lumped]['weight'] += rate_scaled
        else:
            LG.add_edge(start_lumped, target_lumped, weight = rate_scaled)


    #if lumped_graph_path is not None:
        #edge_labels = dict()
        #for from_e, to_e, w in LG.edges.data():
        #    edge_labels[(from_e, to_e)] = str(w['weight'])
        #pos = nx.spectral_layout(LG)
        #plt.figure()
        #nx.draw(LG, pos, edge_color='black', width=1, linewidths=1, \
        #        node_size=500, node_color='pink', alpha=0.7, \
        #        labels={node: node for node in LG.nodes()})
        #nx.draw_networkx_edge_labels(LG, pos, edge_labels=edge_labels)
        #plt.axis('off')
        #plt.show()
        #nx.write_edgelist(LG, lumped_graph_path)

    return LG


if __name__ == "__main__":
    G = nx.DiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(0,1, weight=5)
    G.add_edge(1,2, weight=1)
    G.add_edge(2, 0, weight=0.01)
    G.add_edge(1, 3, weight=0.001)

    print(G[0][1])

    ctmc_state_partitioning = [0,1,0,2]

    lump_ctmc(G, ctmc_state_partitioning)
