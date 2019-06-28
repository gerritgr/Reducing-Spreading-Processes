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

graph = None

output_folder = 'output/'
os.system('mkdir ' + output_folder)


def timeit(method):
    import time
    def timed(*args, **kw):
        start = time.time()
        print('start method: '+method.__name__ )
        result = method(*args, **kw)
        end = time.time()
        diff = int(end - start)
        print('end method: ' + method.__name__,' elapsed: '+str(diff)+'sec')
        return result
    return timed

@timeit
def test_timeit(x=None):
    import time
    print('start timeit')
    time.sleep(4)
    print('end timeit')
    return None
#test_timeit(list())


def odeint_adams(f, init, time_points, integrator = 'vode'):  # use vode for a rougher estimate
    from tqdm import tqdm
    def f_ode(t, u):
        return f(u,t)
    from scipy.integrate import ode
    r = ode(f_ode).set_integrator(integrator, with_jacobian=False, method='adams')
    r.set_initial_value(init, 0)
    step = 0
    u = []
    t = []
    if time_points[0] == 0.0:
            time_points = time_points[1:]
            u.append(np.array(init))
    T = len(time_points)
    pbar = tqdm(total=T)
    pbar.set_description('ODE progress')
    while r.successful() and step < T:
            r.integrate(time_points[step])
            step += 1
            pbar.update(1)
            u.append(r.y)
            t.append(r.t)
    u = np.array(u)
    pbar.close()
    return u



def integrate_ctmc(y, _):
    rates = [0.0] * len(y)

    for edge in graph.edges.data():
        start = edge[0]
        target = edge[1]
        rate = edge[2]['weight']

        rates[start] -= rate * y[start]
        rates[target] += rate * y[start]

    return rates

def plot_statistics(result_list, t, stats_dict, graph, output_file_path):
    plt.clf()
    stat_lines = {name: [0.0]*len(t) for name in stats_dict}

    for stat_name in stats_dict:
        for i, prob_traj in enumerate(result_list):
            stat_func = stats_dict[stat_name]
            state = graph.nodes[i]['state']
            x = stat_func(state)
            x_vec = [x*w for w in prob_traj]

            x_old = stat_lines[stat_name]
            x_new = [x_old[i] + x_vec[i] for i in range(len(x_vec)) ]
            stat_lines[stat_name] = x_new


    df = {'time': t }
    for name, y in stat_lines.items():
        df[name] = y
    df = pd.DataFrame.from_records(df)
    df.to_csv(output_file_path + '_sol_summary.csv', index=False)

    for name, y in stat_lines.items():
        plt.plot(t, y, label=name, alpha=0.7, lw=2)
    plt.legend()
    plt.ylim((0, 1))
    plt.xlabel('time')
    plt.ylabel('global statistics')
    plt.savefig(output_file_path)
    #if len(result_list) < 33:
    plt.show()


def plot_evolution(result_list, t, output_file_path):
    if len(result_list) > 30:
        return
    plt.clf()
    fig1, ax1 = plt.subplots()
    for i, y in enumerate(result_list):
        ax1.plot(t, y, label='y_'+str(i), alpha=0.7, lw=2)
    if len(result_list) < 33:
        plt.legend()
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.ylim((0, 1))
    plt.savefig(output_file_path)
    #if len(result_list) < 33:
    plt.show()

@timeit
def solve_ctmc(initial_condition, ctmc_graph, output_name='solved_ctmc', time_horizon=10, node_labels=None, evaluation_points=100):
    # setup
    global graph
    if 'str' in str(type(ctmc_graph)):
        print('read ctmc from file')
        ctmc_graph = nx.read_edgelist(ctmc_graph, nodetype=int, create_using=nx.DiGraph())
    graph = ctmc_graph
    graph = nx.convert_node_labels_to_integers(graph)
    assert(graph.number_of_nodes() == len(initial_condition))

    if output_name is not None:
        plt.clf()
        nx.write_edgelist(graph, output_folder+output_name+'_.edgelist') #careful with unreachable states
        nx.write_gml(graph, output_folder + output_name + '_.gml')  # careful with unreachable states
        with open(output_folder + output_name + '_.edgedata', 'w') as f:
            for n in graph.nodes():
                f.write('id'+str(n)+'-'+str(graph.node[n])+'\n')
                for ne in graph.neighbors(n):
                    f.write('     '+str(graph[n][ne]['weight'])+'  id'+str(ne)+'-'+str(graph.node[ne]) + '\n')

        edge_labels = dict()
        for from_e, to_e, w in graph.edges.data():
            edge_labels[(from_e, to_e)] = str(w['weight'])
        pos = nx.circular_layout(graph)

        if graph.number_of_nodes() < 50:
            plt.figure()
            if node_labels is None:
                node_labels = {node: node for node in graph.nodes()}
                try:
                    node_labels = {node: graph.nodes[node]['state'] for node in graph.nodes()}
                except:
                    pass
            nx.draw(graph, pos, edge_color='black', width=1, linewidths=1, \
                    node_size=500, node_color='pink', alpha=0.7, \
                    labels=node_labels, font_size=6)
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, alpha=0.8)
            plt.axis('off')
            plt.savefig(output_folder+output_name+'.pdf')
            plt.show()


    # normalize input
    initial_condition = list(initial_condition)
    initial_condition = [max(v, 0.0) for v in initial_condition]
    Z = np.sum(initial_condition)
    initial_condition = [v/Z for v in initial_condition]

    # start solver
    t = np.linspace(0, time_horizon, evaluation_points)
    #y = odeint(integrate_ctmc, initial_condition, t)
    y = odeint_adams(integrate_ctmc, initial_condition, t)
    result_lists = [y[:,i] for i in range(y.shape[1])]

    # output
    plot_evolution(result_lists,t, output_folder+output_name+'_evol.pdf')

    df = {'time': t }
    int_length = len(str(len(initial_condition)))
    for i, y in enumerate(result_lists):
        df['y_'+str(i).zfill(int_length)] = y
    df = pd.DataFrame.from_records(df)
    df.to_csv(output_folder+output_name+'_sol.csv', index=False)


    return t, result_lists


if __name__ == "__main__":
    #G = nx.DiGraph()
    #G.add_node(0)
    #G.add_node(1)
    #G.add_node(2)
    #G.add_edge(0,1, weight=5)
    #G.add_edge(1,2, weight=1)
    #G.add_edge(2, 0, weight=0.01)
    #t, results = solve_ctmc([1.0,0.0,0.0], G)
    #t, results = solve_ctmc([1.0, 0.0, 0.0], 'graph_file.edgelist')
    #sum = np.sum([y[-1] for y in results])
    #assert(sum > 0.999999 and sum < 1.000000001)
    import create_network_ctmc

    G, state_space = create_network_ctmc.create_ctmc(nx.Graph([(0,1),(1,2),(2,3), (3,0)]), ['S', 'I'], [('S', 'I', lambda x: x['I'] * 1.0), ('I','S',lambda x : .5)])
    t, results = solve_ctmc([1.0]*G.number_of_nodes(), G, time_horizon=15)




