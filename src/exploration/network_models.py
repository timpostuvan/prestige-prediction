import networkx as nx
import numpy as np


def original_graph_statistics(G: nx.Graph):
    feats = {}
    feats["num_nodes"] = G.number_of_nodes()
    feats["num_edges"] = G.size()
    feats["average_degree"] = sum([d for _, d in G.degree()]) / feats["num_nodes"]
    feats["average_shortest_path"] = nx.average_shortest_path_length(G)
    feats["clustering_coefficient"] = nx.average_clustering(G)
    return feats

def ER_graph_statistics(G: nx.Graph):
    num_nodes = G.number_of_nodes()
    num_edges = G.size()
    p = num_edges / (num_nodes * (num_nodes - 1) / 2)

    feats = {}
    feats["num_nodes"] = num_nodes
    feats["num_edges"] = p * feats["num_nodes"] * (feats["num_nodes"] - 1) / 2
    feats["average_degree"] = p * (feats["num_nodes"] - 1)
    feats["average_shortest_path"] = np.log(feats["num_nodes"]) / np.log(feats["average_degree"])
    feats["clustering_coefficient"] = feats["average_degree"] / feats["num_nodes"]
    return feats

def WS_graph_statistics(G: nx.Graph):
    num_nodes = G.number_of_nodes()
    num_edges = G.size()
    k = round(2 * num_edges / num_nodes)
    G_ws = nx.watts_strogatz_graph(n=num_nodes, k=k, p=0.0)

    feats = {}
    feats["num_nodes"] = G_ws.number_of_nodes()
    feats["num_edges"] = G_ws.size()
    feats["average_degree"] = sum([d for _, d in G_ws.degree()]) / feats["num_nodes"]
    feats["average_shortest_path"] = nx.average_shortest_path_length(G_ws)
    feats["clustering_coefficient"] = nx.average_clustering(G_ws)
    return feats

def BA_graph_statistics(G: nx.Graph):
    num_nodes = G.number_of_nodes()
    num_edges = G.size()
    m = int(np.round(num_edges / num_nodes))

    feats = {}
    feats["num_nodes"] = num_nodes
    feats["num_edges"] = feats["num_nodes"] * m
    feats["average_degree"] = 2 * m
    feats["average_shortest_path"] = np.log(feats["num_nodes"]) / np.log(np.log(feats["num_nodes"]))
    feats["clustering_coefficient"] =  np.log(feats["num_nodes"])**2 / feats["num_nodes"]
    return feats
