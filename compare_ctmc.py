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
import clustering
plt.ion()

output_folder = 'output/'
os.system('mkdir ' + output_folder)

def difference(result_lists1, result_lists2):
    dif_all = [0.0] * len(result_lists1[0])
    for i in range(len(result_lists1)):
        y1_i = result_lists1[i]
        y2_i = result_lists2[i]
        diff = [np.abs(y1_i[j] - y2_i[j]) for j in range(len(y1_i))]
        dif_all = [dif_all[i] + diff[i] for i in range(len(y1_i))]
    return dif_all


def compare_ctmcs(ctmc1, init1, ctmc_lumped, init_lumped, ctmc_state_partitioning, output_name='diff_dynamic', sol_orig = None):
    if sol_orig is None:
        t, result_lists_1 =  main.solve_ctmc(init1, ctmc1, output_name=output_name+'_original')
    else:
        t, result_lists_1 = sol_orig
    _, result_lists_2 =  main.solve_ctmc(init_lumped, ctmc_lumped, output_name=output_name+'_lumped')
    result_lists_expanded = reduce_ctmc.model_expansion(result_lists_2, ctmc_state_partitioning)
    diff = difference(result_lists_1, result_lists_expanded)

    if output_name is not None and len(result_lists_expanded) < 30:
        plt.clf()
        plt.plot(t, diff, color='red', lw=1, alpha=0.8)
        for i, y in enumerate(result_lists_expanded):
            plt.plot(t, y, label='y_' + str(i), alpha=0.9, lw=1, ls='--')
        for i, y in enumerate(result_lists_1):
            plt.plot(t, y, label='y_' + str(i), alpha=0.5, lw=3, ls=':')
        plt.ylim((0, 1))
        plt.savefig(output_folder+output_name+'_diffplot.pdf')
        #if len(result_lists_expanded) < 35:
        plt.show()

    df = {'time': t, 'diff':diff }
    df = pd.DataFrame.from_records(df)
    df.to_csv(output_folder+output_name+'_difference.csv', index=False)
    print('diff max: ', np.max(diff), '    diff mean:  ',np.mean(diff))

    #
    # summary
    #
    summary_statistics = {'count_i': lambda x: str(x).count('I')/(str(x).count('I')+str(x).count('S')), 'count_s': lambda x: str(x).count('S')/(str(x).count('I')+str(x).count('S'))}
    main.plot_statistics(result_lists_1,t, summary_statistics, ctmc1, output_folder+output_name+'_summarystats_orig.pdf')
    main.plot_statistics(result_lists_expanded,t, summary_statistics, ctmc1, output_folder+output_name+'_summarystats_lumped.pdf')

    return diff, t, result_lists_1, result_lists_2

@main.timeit
def compare_with_lumped(G, init, ctmc_state_partitioning, output_name='diff_dynamic', sol_orig=None):
    ctmc_state_partitioning_clean = reduce_ctmc.correct_partitioning(ctmc_state_partitioning)
    new_init = reduce_ctmc.reduce_initial_state(init, ctmc_state_partitioning_clean)
    LG = reduce_ctmc.lump_ctmc(G, ctmc_state_partitioning_clean)

    #output
    original_state_labels = [G.nodes[n]['state'] for n in G.nodes()]
    df = pd.DataFrame({'cluster':ctmc_state_partitioning_clean, 'initial_value': init, 'cluster_original':ctmc_state_partitioning, 'original': original_state_labels})
    df.to_csv(output_folder+output_name+'_clusterdata.csv', sep='\t')

    diff, t, result_lists_1, result_lists_2 = compare_ctmcs(G, init, LG, new_init, ctmc_state_partitioning_clean, output_name, sol_orig)
    return diff, G, LG, t, result_lists_1, result_lists_2


def toy_example():
    G = nx.DiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(0, 1, weight=3)
    G.add_edge(1, 0, weight=6)
    G.add_edge(1, 2, weight=0.1)
    G.add_edge(2, 3, weight=5)
    G.add_edge(3, 2, weight=4)

    init = [0.4, 0.4, 0.1, 0.1]
    ctmc_state_partitioning = [0, 0, 1, 1]
    compare_with_lumped(G, init, ctmc_state_partitioning)


if __name__ == "__main__":

    import create_network_ctmc

    contact_network = nx.Graph([(0,1),(1,2),(2,3), (3,0), (3,4), (4,5)])
    rules =  [('S', 'I', lambda x: x['I'] * 10), ('I','S',lambda x : 5)]
    #rules = [('S', 'I', lambda x: x['I'] * 1.0)]

    G, state_space = create_network_ctmc.create_ctmc(contact_network, ['S', 'I'], rules)
    init = [1.0]*G.number_of_nodes()
    print('state space XXYy', state_space)

    ctmc_state_partitioning = clustering.gen_clusters(state_space, contact_network)

    print('ctmc_state_partitioning', ctmc_state_partitioning)

    diff = compare_with_lumped(G, init, ctmc_state_partitioning)




