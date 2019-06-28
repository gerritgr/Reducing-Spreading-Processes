import reduce_ctmc
import matplotlib
matplotlib.use('Agg')
import numpy as np


#
# meta
#

def partition_state_space(state_space, contact_network, node_partition, cluster_method):
    state_space_partition = [cluster_method(state, contact_network, node_partition) for state in state_space]
    return state_space_partition

def cluster_from_distances(data_points, dist_func, cluster_num):
    from scipy.cluster.hierarchy import linkage, cut_tree
    points = list()
    for d1 in data_points:
        d_list = [dist_func(d1,d2) for d2 in data_points]
        points.append(d_list)
    import scipy.spatial.distance as ssd
    # convert the redundant n*n square matrix form into a condensed nC2 array
    points = ssd.squareform(points)

    Z = linkage(points, 'ward')
    cluster_indicators = cut_tree(Z, n_clusters=cluster_num)
    cluster_indicators = [c[0] for c in cluster_indicators]
    return cluster_indicators

#
# node clustering
#

def random_node_partition(G, int_range):
    import random
    v = list(range(int_range))*100  #todo fix
    v = v[0:G.number_of_nodes()]
    random.shuffle(v)
    return v

def spectral_node_partition(G, cluster_count):
    import numpy as np
    import networkx as nx
    from scipy.cluster.hierarchy import dendrogram, linkage, ward, fcluster, cut_tree

    pos = nx.spectral_layout(G, dim=3, scale=1, center=[0, 0, 0])
    keys = sorted(pos.keys())
    positions_list = [list(pos[k]) for k in keys]
    dist = np.matrix(positions_list)

    # clustering

    Z = linkage(dist, 'ward')
    #cluster_indicators = fcluster(Z, t=cluster_count, criterion='maxclust')
    cluster_indicators = cut_tree(Z, n_clusters=cluster_count)
    cluster_indicators = [c[0] for c in cluster_indicators]
    return cluster_indicators


def degree_based_node_partition(G, cluster_count):
    import random
    from scipy.cluster.hierarchy import dendrogram, linkage, ward, fcluster, cut_tree
    degree_list = [len(G[n])+(random.random()*0.1) for n in G.nodes()]
    #print(degree_list)
    degree_list_trans = [[d**(0.5)] for d in degree_list]
    #print(degree_list_trans)
    Z = linkage(degree_list_trans, 'ward')
    #cluster_indicators = fcluster(Z, t=cluster_count, criterion='maxclust')
    cluster_indicators = cut_tree(Z, n_clusters=cluster_count)
    cluster_indicators = [c[0] for c in cluster_indicators]
    return cluster_indicators

#
# state space clustering
#


def node_based_counting(state, G, node_partition):
    assert (len(state) == len(node_partition))
    part_values = [0] * len(set(node_partition))
    for i, p in enumerate(node_partition):
        if state[i] == 'I':
            part_values[p] += 1
    return tuple(part_values)

def tuple_based_counting(state, G, node_partition):
    assert (len(state) == len(node_partition))
    state = tuple([(state[i], node_partition[i]) for i in range(len(state))])
    edges = G.edges()
    edges_reduced = list()
    for edge in edges:
        s1 = state[edge[0]]
        s2 = state[edge[1]]
        e = tuple(sorted([s1, s2]))
        edges_reduced.append(e)
    edges_reduced = sorted(edges_reduced)
    return tuple(edges_reduced)

def neighbor_based_counting(state, G, node_partition):
    assert (len(state) == len(node_partition))
    state = tuple([(state[i], node_partition[i]) for i in range(len(state))])
    reduced = list()
    for node in G.nodes():
        node_state = state[node]
        neighbor_states = tuple(sorted([state[n] for n in G[node]]))
        node_neighborhood = (node_state, neighbor_states)
        reduced.append(node_neighborhood)
    reduced = sorted(reduced)
    reduced = tuple(reduced)
    #print('neighbor counting', reduced)
    return reduced



if __name__ == "__main__":
    import networkx as nx
    #G = nx.Graph([(1,2),(2,3),(3,4),(4,1),(1,5),(5,1)])
    #print(degree_based_node_partition(G, 3))

    print(cluster_from_distances([1,2,3,4,5,6,7,8,9,10], lambda x,y: np.abs(x-y)/np.max([x, y]), 3))


#
#
#
# def count_tuple_for_partition(node_partition):
#     node_partition = reduce_ctmc.correct_partitioning(node_partition)
#
#     def count_tuple_method(state, contact_network, cluster_number):
#         assert(len(state)==len(node_partition))
#         state = tuple([(state[i], node_partition[i]) for i in range(len(state))])
#         edges = contact_network.edges()
#         edges_reduced = list()
#         for edge in edges:
#             s1 = state[edge[0]]
#             s2 = state[edge[1]]
#             e = tuple(sorted([s1, s2]))
#             edges_reduced.append(e)
#         edges_reduced = sorted(edges_reduced)
#         #print('edges_topology ', edges_reduced)
#         return tuple(edges_reduced)
#
#     return count_tuple_method
#
# def edge_baesd_counting(state, contact_network, cluster_number):
#     edges = contact_network.edges()
#     edges_reduced = list()
#     for edge in edges:
#         s1 = state[edge[0]]
#         s2 = state[edge[1]]
#         e = tuple(sorted([s1,s2]))
#         edges_reduced.append(e)
#     edges_reduced = sorted(edges_reduced)
#     #print('edges ',edges_reduced)
#     return tuple(edges_reduced)
#
# def count_for_partitioning(node_partition):
#     node_partition = reduce_ctmc.correct_partitioning(node_partition)
#
#     def count_method(state, contact_network, cluster_number):
#         assert(len(state)==len(node_partition))
#         part_values = [0] * len(set(node_partition))
#         for i, p in enumerate(node_partition):
#             if state[i] == 'I':
#                 part_values[p] += 1
#
#         return tuple(part_values)
#
#
#     return count_method
#
#
#
# def count(state, contact_network, cluster_number):
#     state = str(state)
#     s1 = state[:int(len(state)/2)]
#     s2 = state[int(len(state) / 2):]
#     #return state
#     #return state.count('I')
#     return (state[:8].count('I'),state[8:19].count('I'),state[19:].count('I'))
#    # return (s1.count('I'), s2.count('I'))
#
#
# def gen_clusters(state_space, contact_network, cluster_number = 3, cluster_method=count):
#     clusters = [cluster_method(v,contact_network,cluster_number) for v in state_space]
#     print('clusters', (str(clusters)+'  '*500)[:1000])
#     return clusters