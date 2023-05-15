### let's use this for graph import and creation

import pandas as pd 
import numpy as np 
from config import edge_list, data_dir
import networkx as nx
import os

class AcademicGraph:
    def __init__(self):
        df = pd.read_csv(edge_list)
        df["weight"] = df["Total"]

        # connect to locations
        country_pairings = pd.read_csv(os.path.join(data_dir, "university_country_pairing.csv"))
        us_unis = country_pairings[country_pairings["CountryName"] == "United States"]
        non_us_unis = country_pairings[country_pairings["CountryName"] != "United States"]

        self.us_unis = us_unis
        self.non_us_unis = non_us_unis 

        self.df = df.merge(country_pairings[["DegreeInstitutionId", "CountryName"]], on="DegreeInstitutionId", how="left")
        self.df["location"] = self.df["CountryName"].apply(lambda x: 1 if x == "United States" else 0)  # add a 1 for US locations, used later.

    def full_graph(self) -> nx.Graph:
        G = nx.from_pandas_edgelist(
            self.df, 
            source="DegreeInstitutionId",
            targer="InstitutionId",
            edge_attr=["TaxonomyLevel", "TaxonomyValue", "InstitutionName", "DegreeInstitutionName", "weight", "Men", "Women"],
            create_using=nx.DiGraph()
        )
        return G

    def segmented_graphs(self, limit_to_US: bool = True): 
        temp = self.df.copy()
        temp = temp[(temp["TaxonomyLevel"] == "Domain") | (temp["TaxonomyLevel"] == "Academia")]
        groups = temp.groupby("TaxonomyValue")          # groupby the domains

        graph_list, features, graph_labels = [], [], []

        for name, group in groups:
            print(f"Preparing domain: {name}")
            feat_dict = {}

            # all hires, men, women, percentage from US
            total, men, women, location = group["Total"].sum(), group["Men"].sum(), group["Women"].sum(), group["location"].mean()         

            features.append({"Total": total, "Men": men, "Women": women, "Prop_US": location})
            graph_labels.append(name)
            
            # limit to only US relation
            if limit_to_US: 
                group = group[group["location"] == 1]

            graph_list.append(
                nx.from_pandas_edgelist(
                    group, 
                    source="DegreeInstitutionId", 
                    target="InstitutionId", 
                    edge_attr=["TaxonomyLevel", "TaxonomyValue", "InstitutionName", "DegreeInstitutionName", "weight", "Men", "Women"],
                    create_using=nx.DiGraph()
                )
            )

        return graph_labels, graph_list, features
