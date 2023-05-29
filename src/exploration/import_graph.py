### let's use this for graph import and creation

import pandas as pd
from config import edge_list, data_dir
import networkx as nx
import os


class AcademicGraph:
    def __init__(self):
        self.df_edges = pd.read_csv(edge_list)
        self.df_edges = self.df_edges.rename(columns={"Total": "weight"})

        self.df_nodes = pd.read_csv(os.path.join(data_dir, "institution-stats.csv"))

    def segmented_graphs(
        self, 
        node_attributes: list = ["NonAttritionEvents", "AttritionEvents", "ProductionRank", "PrestigeRank"],
        edge_attributes: list = ["weight", "Men", "Women"]
    ):
        filtered_edges = self.df_edges[(self.df_edges["TaxonomyLevel"] == "Domain") | (self.df_edges["TaxonomyLevel"] == "Academia")]
        filtered_nodes = self.df_nodes[(self.df_nodes["TaxonomyLevel"] == "Domain") | (self.df_nodes["TaxonomyLevel"] == "Academia")]
        
        groups = filtered_edges.groupby("TaxonomyValue")  #  group by the domains

        graph_list, features, graph_labels = [], [], []
        for name, group_edges in groups:
            print(f"Preparing domain: {name}")

            # all hires, men, women
            total, men, women = group_edges["weight"].sum(), group_edges["Men"].sum(), group_edges["Women"].sum()
            features.append({"Total": total, "Men": men, "Women": women})

            graph_labels.append(name)
            
            group_nodes = filtered_nodes[filtered_nodes["TaxonomyValue"] == name]
            group_ranked_institutions = set(group_nodes["InstitutionId"])

            # keep edges only between ranked institutions
            keep_edges = (
                group_edges["InstitutionId"].isin(group_ranked_institutions) & 
                group_edges["DegreeInstitutionId"].isin(group_ranked_institutions)
            )
            group_edges = group_edges[keep_edges]


            # load edges
            graph = nx.from_pandas_edgelist(
                group_edges,
                source="DegreeInstitutionId",
                target="InstitutionId",
                edge_attr=edge_attributes,
                create_using=nx.DiGraph()
            )

            # load node attributes
            for attr in node_attributes:
                attr_values = {row[0]:row[1] for _, row in group_nodes[["InstitutionId", attr]].iterrows()}
                nx.set_node_attributes(graph, attr_values, attr)

            graph_list.append(graph)

        return graph_labels, graph_list, features
