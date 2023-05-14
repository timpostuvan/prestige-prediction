import networkx as nx

def process_features(G: nx):
    # extract some vanilla features from a graph G
    
    feats = {}

    # size, edges

    # TODO fix the weights here 
    feats["num_edges"] = G.size(weight="weight")
    
    feats["num_nodes"] = G.number_of_nodes()
    feats["average_degree_connectivity"] = nx.average_degree_connectivity(G)

    feats["clustering_coeff"] = nx.average_clustering(G, weight="weight")
    feats["average_shortest_path"] = nx.average_shortest_path_length(G, weight="weight")

    feats["density"] = nx.density(G)

    return feats