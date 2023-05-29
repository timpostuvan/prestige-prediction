import pickle
from os.path import join, exists
from pathlib import Path

import torch
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx

from .import_graph import AcademicGraph


class PyGAcademicGraph(Dataset):
    def __init__(
        self,
        data_dir: str = "../data",
        split: str = "train",
        setting: str = "transductive"
    ):
        self.setting = setting
        
        dataset_path = join(data_dir, setting)
        if not exists(dataset_path):
            graphs = self.create_dataset()
            Path(dataset_path).mkdir(parents=True)

            with open(join(dataset_path, "graphs.pkl"), "wb") as f:
                pickle.dump(graphs, f)

        
        with open(join(dataset_path, "graphs.pkl"), "rb") as f:
            self.graphs = pickle.load(f)

        for graph in self.graphs:
            graph.mask = getattr(graph, f"{split}_mask")


    def create_dataset(self):
        networkx_G = AcademicGraph()
        domains, nx_graphs, _ = networkx_G.segmented_graphs()
        
        # rename "PrestigeRank" to "y"
        for graph in nx_graphs:
            for _, data in graph.nodes(data=True):
                data["y"] = data.pop("PrestigeRank")
        
        # add topological features
        for graph in nx_graphs:
            nx.set_node_attributes(graph, nx.betweenness_centrality(graph), "betweenness_centrality")
            nx.set_node_attributes(graph, nx.eigenvector_centrality(graph), "eigenvector_centrality")
            nx.set_node_attributes(graph, nx.clustering(graph), "clustering_coefficient")
            nx.set_node_attributes(graph, nx.degree_centrality(graph), "degree_centrality")
            nx.set_node_attributes(graph, nx.centrality.closeness_centrality(graph), "closeness_centrality")

        # create PyG graphs
        node_features = ["NonAttritionEvents", "AttritionEvents", "ProductionRank"]
        topological_features = [
            "betweenness_centrality", "eigenvector_centrality", "clustering_coefficient", 
            "degree_centrality", "closeness_centrality"
        ]
        edge_features = ["weight", "Men", "Women"]
        graphs = [
            from_networkx(
                G=graph,
                group_node_attrs=node_features + topological_features,
                group_edge_attrs=edge_features
            )
            for graph in nx_graphs
        ]
        # convert numbers to floats
        for graph in graphs:
            graph.x = graph.x.float()
            graph.y = graph.y.float()

        for idx, domain in enumerate(domains):
            graphs[idx].domain = domain

        # generate masks for different splits
        if self.setting == "transductive":
            for graph in graphs:
                train_mask = torch.zeros(graph.num_nodes, dtype=bool)
                val_mask = torch.zeros(graph.num_nodes, dtype=bool)
                test_mask = torch.zeros(graph.num_nodes, dtype=bool)

                indices = torch.randperm(graph.num_nodes)
                split_l = int(0.7 * graph.num_nodes)
                split_r = int((0.7 + 0.1) * graph.num_nodes)

                train_mask[indices[:split_l]] = True
                val_mask[indices[split_l:split_r]] = True
                test_mask[indices[split_r:]] = True

                assert torch.all((train_mask & test_mask) == 0)
                assert torch.all((train_mask & val_mask) == 0)
                assert torch.all((val_mask & test_mask) == 0)

                graph.train_mask = train_mask
                graph.val_mask = val_mask
                graph.test_mask = test_mask
        
        elif self.setting == "inductive":
            indices = torch.randperm(len(graphs))
            split_l = int(0.7 * len(graphs))
            split_r = int((0.7 + 0.1) * len(graphs))

            train_graphs = torch.zeros(len(graphs), dtype=bool)
            val_graphs = torch.zeros(len(graphs), dtype=bool)
            test_graphs = torch.zeros(len(graphs), dtype=bool)
            
            train_graphs[indices[:split_l]] = True
            val_graphs[indices[split_l:split_r]] = True
            test_graphs[indices[split_r:]] = True

            assert torch.all((train_graphs & test_graphs) == 0)
            assert torch.all((train_graphs & val_graphs) == 0)
            assert torch.all((val_graphs & test_graphs) == 0)

            for idx, graph in enumerate(graphs):
                train_mask = torch.zeros(graph.num_nodes, dtype=bool)
                val_mask = torch.zeros(graph.num_nodes, dtype=bool)
                test_mask = torch.zeros(graph.num_nodes, dtype=bool)

                if train_graphs[idx] == True:
                    train_mask[:] = True
                elif val_graphs[idx] == True:
                    val_mask[:] = True
                elif test_graphs[idx] == True:
                    test_mask[:] = True
                else:
                    ValueError("Something is wrong with splits.")

                graph.train_mask = train_mask
                graph.val_mask = val_mask
                graph.test_mask = test_mask
        else:
            raise ValueError("Not a supported setting.")

        return graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
