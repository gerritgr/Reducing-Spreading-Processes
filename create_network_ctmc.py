import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os

import main
import reduce_ctmc
import itertools
plt.ion()

from collections import Counter

output_folder = 'output/'
os.system('mkdir ' + output_folder)

def get_neighborhood(i, state, contact_network, states):
    neighbors = list(contact_network[i])
    neighbor_states = [state[n] for n in neighbors]
    neighbor_dict =  dict(Counter(neighbor_states))
    for s in states:
        if s not in neighbor_dict:
            neighbor_dict[s] = 0
    return neighbor_dict


def get_succ_states(contact_network, states, rules, state):
    succ_dict = dict()
    for i, local_node in enumerate(state):
        for rule in rules:
            if rule[0] != local_node:
                continue
            succ_state = list(state)
            succ_state[i] = rule[1]
            succ_state = tuple(succ_state)
            neighborhood = get_neighborhood(i, state, contact_network, states)
            succ_dict[succ_state] = rule[2](neighborhood)


    return succ_dict

def write_contact_network(contact_network, output_name):
    nx.write_edgelist(contact_network, output_folder + output_name + '_contact_network.edgelist')
    nx.write_gml(contact_network, output_folder + output_name + '_contact_network.gml')
    plt.clf()
    nx.draw(contact_network)
    plt.savefig(output_folder+output_name+'_contact_graph.pdf')

@main.timeit
def create_ctmc(contact_network, states, rules, output_name='ctmc_with_labels'):
    write_contact_network(contact_network, output_name)
    contact_network = nx.convert_node_labels_to_integers(contact_network)
    states = sorted(states)
    number_of_nodes = contact_network.number_of_nodes()
    state_space = list(itertools.product(states, repeat=number_of_nodes))
    print('state space: ', (str(state_space)+'  '*500)[:1000])

    G = nx.DiGraph()
    for state in state_space:
        G.add_node(state)

    for state in state_space:
        succ_states = get_succ_states(contact_network, states, rules, state)
        for succ, rate in succ_states.items():
            G.add_edge(state, succ, weight=rate)

    if output_name is not None:
        G_str = nx.relabel_nodes(G, lambda x: str(x).replace(' ','').replace(',','').replace("'",'').replace('(','').replace(')',''))
        nx.write_edgelist(G_str, output_folder+output_name+'_ctmc.edgelist')
        nx.write_gml(G_str, output_folder + output_name + '_ctmc.gml')


    #convert labels to attributes even if already converted for output
    G_str = nx.relabel_nodes(G, lambda x: str(x).replace(' ','').replace(',','').replace("'",'').replace('(','').replace(')',''))
    for node in G_str:
        G_str.nodes[node]['state'] = node
    G_str = nx.convert_node_labels_to_integers(G_str)


    return G_str, state_space

def state_space_2_node_labels(state_space):
    node_labels = {i: str(state_space[i]).replace(' ','').replace(',','').replace("'",'').replace('(','').replace(')','') for i in range(len(state_space))}
    return node_labels

if __name__ == "__main__":
    G = nx.Graph([(0,1),(1,2),(2,3), (3,0)])
    create_ctmc(G, ['S', 'I'], [('S', 'I', lambda x: x['I']*0.1)])



