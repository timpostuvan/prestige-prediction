import networkx as nx

def process_features(G: nx):
    # extract some vanilla features from a graph G

    feats = {}
    feats["num_nodes"] = G.number_of_nodes()
    feats["num_edges"] = G.size()
    feats["weighted_num_edges"] = G.size(weight="weight")

    feats["density"] = nx.density(G)
    feats["average_degree"] = sum([d for _, d in G.degree()]) / feats["num_nodes"]
    feats["weighted_average_degree"] = sum([d for _, d in G.degree(weight="weight")]) / feats["num_nodes"]

    feats["clustering_coefficient"] = nx.average_clustering(G)
    feats["weighted_clustering_coefficient"] = nx.average_clustering(G, weight="weight")
    feats["average_shortest_path"] = nx.average_shortest_path_length(G)
    feats["diameter"] = nx.diameter(G)

    feats["algebraic_connectivity"] = nx.algebraic_connectivity(G)
    feats["weighted_algebraic_connectivity"] = nx.algebraic_connectivity(G, weight="weight")
    return feats
