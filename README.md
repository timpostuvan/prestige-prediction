### NML Project

Node level features
* Rank
* Field
* Presitge
* Production Rank? 

Staff make-up:
* Attrition
* Up Hires / Down Hires

Edge features
* University (from -> to)



Initial analyses:
* Cleaning the graph. 
* Add additional features. Degree distribution, clustering coefficents on domains. 
    * Number of nodes, number of edges, clustering coefficent, average degree, graph density, average distance, percentage male vs. female, gini coefficent, % international.
* Most international universities.
* Remove non-US universities. 
* How does diversity affect presitge? 
* 8 domains + academia. 
* Clustering coefficents / 


Sample tasks:
* Predict university rank given hiring decisions
* Predict faculty gender / race based on geography / taxonomy

Exploration:
* Can geography affect hiring decisions. Are there more conservative regions than others?
* Consider the US institutions. Add a field for hires outside the United States. How much they hire by specific country. 

Why we cut density
* The graph is highly dense and we don't want over-smoothing. We use a PMI method to only keep edge weights for institutions that are highly important for institution. 

For GNN
1. Only ConvGNN. With some simple cutoff, with mutual information.
2. For message propogation be careful about the directions. flow="source_to_target".
3. Normalize the data.
