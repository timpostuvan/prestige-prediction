import networkx as nx
import numpy as np
import pygsp
import sklearn.cluster


def process_features(G: nx):
    #  extract some vanilla features from a graph G

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


def compute_laplacian(adjacency: np.ndarray, normalize: str):
    """ normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    # Your solution here ###########################################################
    eps = 1e-10
    degrees = adjacency.sum(axis=-1)
    D = np.diag(degrees)
    W = adjacency

    if normalize is None:
        # combinatorial Laplacian
        L = D - W
        return L

    elif normalize == "sym":
        # symmetric normalized Laplacian
        I = np.eye(D.shape[0])
        D_12 = np.diag(1 / np.sqrt(degrees + eps))
        L_sym = I - D_12 @ W @ D_12
        return L_sym

    elif normalize == "rw":
        # random walk Laplaican
        I = np.eye(D.shape[0])
        D_11 = np.diag(1 / (degrees + eps))
        L_rw = I - D_11 @ W
        return L_rw
    raise ValueError(f'Unknown normalization: {normalize}')


def compute_number_connected_components(lamb: np.array, threshold: float):
    """ lamb: array of eigenvalues of a Laplacian
        Return:
        n_components (int): number of connected components.
    """
    return np.count_nonzero(lamb < threshold)


def spectral_decomposition(laplacian: np.ndarray):
    """ Return:
        lamb (np.array): eigenvalues of the Laplacian
        U (np.ndarray): corresponding eigenvectors.
    """
    lamb, U = np.linalg.eig(laplacian)
    return lamb, U


def get_laplacians_pygsp(G: nx.Graph):
    A = nx.adjacency_matrix(G).toarray().astype(float)
    G_pygsp = pygsp.graphs.Graph(A)

    G_pygsp.compute_laplacian("combinatorial")
    laplacian_comb_true = G_pygsp.L.toarray().astype(float)

    G_pygsp.compute_laplacian("normalized")
    laplacian_norm_true = G_pygsp.L.toarray().astype(float)

    return G_pygsp, laplacian_comb_true, laplacian_norm_true


class SpectralClustering:
    def __init__(self, n_classes: int, normalize: str):
        self.n_classes = n_classes
        self.normalize = normalize
        # Your solution here ###########################################################
        self.clustering_method = sklearn.cluster.KMeans(n_clusters=n_classes, random_state=0)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def fit_predict(self, G: nx.Graph):
        """ Your code should be correct both for the combinatorial
            and the symmetric normalized spectral clustering.
            Return:
            y_pred (np.ndarray): cluster assignments.
        """
        # Your solution here ###########################################################
        A = nx.adjacency_matrix(G).toarray().astype(float)
        laplacian = compute_laplacian(A, normalize=self.normalize)

        lamb, U = spectral_decomposition(laplacian)
        lamb = np.real(lamb)
        print("U.shape", U.shape)
        smallest_eigenvalues_ind = np.argsort(lamb)[:self.n_classes]
        U = np.real(U[:, smallest_eigenvalues_ind])

        if self.normalize == "sym":
            U_norm = np.linalg.norm(U, ord=2, axis=-1, keepdims=True)
            U = U / U_norm

        predictions = self.clustering_method.fit_predict(U)
        return predictions
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
