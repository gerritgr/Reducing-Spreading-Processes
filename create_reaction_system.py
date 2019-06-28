import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
plt.ion()
plt.clf()

def create_change_vector(species, rule, p):
    change_vector = [0]*len(species)
    for i, (s, p_i) in enumerate(species):
        if p_i == p and s == rule[0]:
            change_vector[i] = -1
        elif p_i == p and s == rule[1]:
            change_vector[i] = 1
    return change_vector


def compute_mean_rate(graph, node, rule, partitioning, species):
    p = partitioning[node]
    current_species = (rule[0], p)
    current_species_i = species.index(current_species)
    nodes_in_p = partitioning.count(p)
    func = '(species[{current_species_i}]/{nodes_in_p})'
    func = func.format(current_species_i=current_species_i, nodes_in_p=nodes_in_p)
    return func


def create_propensity(species, rule, graph, partitioning, p):
    current_species = (rule[0], p)
    current_species_i = species.index(current_species)

    func = '0'
    for node in graph:
        if partitioning[node] != p:
            continue
        mean_rate_current_node = compute_mean_rate(graph, node, rule, partitioning, species)
        func += '+'
        func += mean_rate_current_node

    return func

# TODO prob that node itself is in state

def create_reaction(species, rule, graph, partitioning, p):
    change_vector = create_change_vector(species, rule, p)
    propensity_function = create_propensity(species, rule, graph, partitioning, p)
    return (change_vector, propensity_function)



def write_output(species, reactions, outpath):
    df = {'change_vector': [x[0] for x in reactions], 'propensity': [x[1] for x in reactions]}
    df = pd.DataFrame(df)
    df.to_csv(outpath, sep='\t', index=False)

def main_node_exact(graph, partitioning, states, rules, outpath):
    graph = nx.convert_node_labels_to_integers(graph)
    species = list()
    reactions = list()

    partition_list = sorted(list(set(partitioning)))

    for s in states:
        for p in partition_list:
            species.append((s,p))

    for rule in rules:
        for p in partition_list:
            r = create_reaction(species, rule, graph, partitioning, p)
            reactions.append(r)


    print(species)
    print(reactions)
    write_output(species, reactions, outpath)
    pass



if __name__ == "__main__":
    graph = nx.Graph([(0,1),(1,2),(2,3), (3,0)])
    partitioning = [0,1,2,2]
    states = ['S', 'I']
    rules = [('S', 'I', lambda x: x['I'] * 1.0), ('I','S',lambda x : .5)]
    outpath = 'reaction_system.csv'
    main_node_exact(graph, partitioning, states, rules, outpath)
