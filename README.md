# From Pedagogy to Prestige: Predicting Prestige Using Academic Hiring Networks
Tim Postuvan and Veniamin Veselovsky

This is the code base for the paper `From Pedagogy to Prestige:\\Predicting Prestige Using Academic Hiring Networks`. This paper defines a few ML models for predicting prestige from a [faculty hiring graph](https://github.com/LarremoreLab/us-faculty-hiring-networks). More information about the dataset and intial paper is available [here](https://www.nature.com/articles/s41586-022-05222-x).

## Requirements
You can install all the required packages using the following command:

```pip install -r requirements.txt```

## Jupyter Notebook description
To re-run the experiments in the paper we list how the Jupyter Notebooks are structured. Note that for files starting with 1, 2, 3 we have two versions on for transductive and the other for inductive. 

1. In `0_exploration.ipynb` we conduct the exploration found in our paper. This includes extracting the clusters, random graph approximations, domain-level features, and variance graphs.
2. In `1_average_baseline_*` we measure the basic average baseline for our models.
2. In `2_linear_regression_*` we run the linear regression on the node and topological features. 
2. In `3_gnn_*` we train our GNN from scratch and evaluate it on our test set. 

## Main code description
* `exploration/import_graph.py` we extract the graph and conduct initial filterings. 
* `exploration/graph_features.py` we extract the topological features. 
* The PyTorch Geometric dataset for all analyses that proceed is created in `exploration/dataset.py`. In the dataset we provide two splitting techniques for the graph: (1) transductive and (2) inductive. 
* Additionally, the models used for training are included in `exploitation/models.py`. 

