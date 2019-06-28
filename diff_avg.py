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
plt.ion()
import glob


import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys, os, glob, subprocess, random, time
from IPython.display import display, HTML
mpl.rc('xtick', labelsize=13)
mpl.rc('ytick', labelsize=13)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import networkx as nx
print('-'*100)

sns.set_context('paper', font_scale = 2.7, rc={"lines.linewidth": 2, 'xtick.labelsize':23, 'ytick.labelsize':23})
sns.set_style('white')



def get_mean_max(filename_filter):
    meanlist_node = list()
    maxlist_node = list()
    odecountlist_node = list()

    meanlist_edge = list()
    maxlist_edge = list()
    odecountlist_edge = list()

    for path in glob.glob('./output/*oreo*diff*.csv'):
        cpath = path.replace('_difference.csv', '_clusterdata.csv')
        os.system('cp {} {}'.format(path, path.replace('output/','output/out2/')))
        os.system('cp {} {}'.format(cpath, cpath.replace('output/', 'output/out2/')))

        continue

        if filename_filter not in path:
            continue

        #odecount
        clusterdata = path.replace('_difference.csv', '_clusterdata.csv')
        dfc = pd.read_csv(clusterdata, sep='\t')
        ode_count = len(set(dfc['cluster']))
        #ode_count = int(path.split('_SIS_')[1].split('_diff')[0])
        df = pd.read_csv(path, sep=',')
        mean = np.mean(df['diff'])
        max = np.max(df['diff'])
        if '_node_' in path:
            meanlist_node.append(mean)
            maxlist_node.append(max)
            odecountlist_node.append(ode_count)
        else:
            meanlist_edge.append(mean)
            maxlist_edge.append(max)
            odecountlist_edge.append(ode_count)
        print(path, mean, max)
        plt.clf()


    #plt.scatter(odecountlist_node,meanlist_node, alpha=0.5, c=(0, 77/255, 154/255), marker='^', s=130, label='Node avg', lw=0)
    #plt.scatter(odecountlist_node, maxlist_node, alpha=0.5, c=(0, 77/255, 154/255), marker='v', s=130, label='Node max', lw=0)
    plt.scatter(odecountlist_node, maxlist_node, alpha=0.5, edgecolor=(0, 77/255, 154/255), marker='v', s=120, label='Node max', lw=2.3, c='none')
    plt.scatter(odecountlist_node,meanlist_node, alpha=0.5, edgecolor=(0, 77/255, 154/255), marker='^', s=120, label='Node avg', lw=2.3, c='none')

    plt.scatter(odecountlist_edge, maxlist_edge, alpha=0.7, c='none', marker='v', edgecolor=(154/255, 0, 77/255), lw=1.3,s=120, label='Edge max')
    plt.scatter(odecountlist_edge,meanlist_edge, alpha=0.7, c='none', marker='^', edgecolor=(154/255, 0, 77/255), lw=1.3,s=120, label='Edge avg')


    plt.gca().set(xlabel='Number of CTMC states', ylabel='Accuracy')
    plt.gca().set_ylim(ymin=0.0, ymax=0.5)
    leg = plt.legend()
    plt.savefig('meanmaxplot/'+filename_filter+'_meanmaxplot.pdf', dpi=300, bbox_inches="tight")



#node
#s=130, marker='^', alpha=0.5, c=(0, 77/255, 154/255)
# edge
# s=130, marker='v', alpha=0.5, c=(154/255, 0, 77/255))

get_mean_max('oreo_rgraph_random')
get_mean_max('oreo_rgraph_degree')
get_mean_max('oreo_rgraph_spectral')