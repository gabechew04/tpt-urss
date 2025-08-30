# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 10:17:28 2025

@author: trund
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_reactive_distributions(n, N, A, B, reactive_distributions, prob_current):
    
    for t in range(N):
        distribution_nx = reactive_distributions[t].reshape(n, n)
        G = nx.grid_2d_graph(n, n, create_using=nx.DiGraph())
        pos = {(i, j): (j, -i) for i, j in G.nodes()} 
        labels = {}
        for i in A:
            labels[(i // n, i % n)] = 'A'
    
        for i in B:
            labels[(i // n, i % n)] = 'B'
        
    
        # Create a mapping from (i, j) coords to integer indices and vice versa
        node_idx = {node: i for i, node in enumerate(sorted(G.nodes()))}
        idx_node = {i: node for node, i in node_idx.items()}
    
        edge_widths = []
        edges_to_draw = []
    
        for u, v in G.edges():
            i, j = node_idx[u], node_idx[v]
            weight = prob_current[t, i, j]
            if weight > 0:
                edge_widths.append(weight * 500)  # scale for visibility
                edges_to_draw.append((u, v))  # only include edges with nonzero current

        vmax = reactive_distributions.max()
        vmin = reactive_distributions.min()
    
        plt.figure(figsize=(4,4))
        nodes_draw = nx.draw_networkx_nodes(G, pos, node_size=200, node_color=distribution_nx,cmap=plt.cm.coolwarm, edgecolors='black', vmax=vmax,
                                            vmin = vmin)
        edges_draw = nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color='black', connectionstyle= 'arc3,rad=0.3', width=edge_widths)
        labels_draw = nx.draw_networkx_labels(G, pos, labels=labels, font_color='lime')
    
        plt.colorbar(nodes_draw)
        plt.axis('off')
        plt.title(r'Probability of (state) $\cap$ (reactive) at $T = $' + f'${t}$')
        #plt.savefig(f'PT{t}.png')
        plt.show()
    return

def rates_visualisation(n, N, A, B, rates):
    
    for t in range(N):
        rate_AB = rates[t].reshape(n, n)
        G = nx.grid_2d_graph(n, n)
        pos = {(i, j): (j, -i) for i, j in G.nodes()}
        labels = {}
    
        for i in A:
            labels[(i // n, i % n)] = 'A'
    
        for i in B:
            labels[(i // n, i % n)] = 'B'
        
        vmax = rates.max()
        vmin = rates.min()
        
        plt.figure(figsize=(4,4))
        nodes = nx.draw_networkx_nodes(G, pos, 
                                       node_size=200, node_color=rate_AB, cmap=plt.cm.coolwarm, edgecolors='black', 
                                       vmax= vmax, vmin=vmin)
        labels_draw = nx.draw_networkx_labels(G, pos, labels=labels)

        plt.title(f'Transition rates for A = {A} and B = {B} at $T = {t}$')
        plt.colorbar(nodes)
        plt.axis('off')
        #plt.savefig(f'Rates at time {t}.png')
        plt.show()
    return
    