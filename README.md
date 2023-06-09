# From Pedagogy to Prestige: Predicting Prestige Using Academic Hiring Networks
Tim Postuvan and Veniamin Veselovsky

This is the code base for the paper `From Pedagogy to Prestige: Predicting Prestige Using Academic Hiring Networks`. This paper defines a few ML models for predicting the prestige of institutions in [US faculty hiring dataset](https://github.com/LarremoreLab/us-faculty-hiring-networks). More information about the dataset can be found in the initial paper [here](https://www.nature.com/articles/s41586-022-05222-x).

## Requirements
You can install all the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Jupyter Notebook description
To re-run the experiments in the paper, we list how the Jupyter Notebooks are structured. Note that files starting with `1_*`, `2_*`, and `3_*`  have two versions: one for the transductive and one for the inductive experimental setting. 

1. In `0_exploration.ipynb` we conduct the initial exploration of the dataset. This includes extracting the graph-level features for each domain, clustering of institutions, approximations with network models, and exploring variance in prestige.
2. In `1_average_baseline_*` we obtain the basic average baseline for our models.
3. In `2_linear_regression_*` we run the linear regression on the node and topological features. 
4. In `3_gnn_*` we train our GNN models from scratch and evaluate them on our test set. 

## Main code description
* In `exploration/import_graph.py` we extract the graphs and perform initial filtering. 
* In `exploration/graph_features.py` we extract the topological features. 
* The PyTorch Geometric dataset used in all analyses is created in `exploration/dataset.py`. In the dataset, we provide two approaches to create train/validation/test splits: (1) transductive, and (2) inductive. 
* Additionally, the models that we conduct experiments on are included in `exploitation/models.py`. 

You can read about experiments in our paper [here](https://github.com/vminvsky/nml-project/blob/main/paper.pdf).