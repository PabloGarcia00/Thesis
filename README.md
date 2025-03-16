# Thesis
Title: Gene representations for gene regulatory network inference

Gene Regulatory Networks (GRN) describe the relationships between Transcription Factors (TFs) and
target genes. Complete GRNs are important for understanding the molecular and functional processes. As
experimental GRN inference remains laborious, various computational methods have been developed for GRN
inference. Supervised link-prediction models learned from a prior GRN to predict missing links, the prior GRN
is often attributed with a secondary data sources that describe the genes in the network. Link-prediction
models learn node representations based on their topology in the network and the secondary data source. As
topology constitutes structural features such as node connectivity and centrality, it is not clear from what
component the performance is derived, topology or the secondary data source. This project investigates
expression, coexpression and functional gene representations for GRN inference demonstrated in Arabidopsis
thaliana. It was found that the gene representations improved GRN inference, however, a great reliance
on topological information was observed. By testing model performance on unseen genes using various
datasplits that constituted transductive, semi-transductive and inductive test settings, it was found that the
gene representations varied in their utility for GRN inference. Furthermore, results show that testing on genes
seen during training (transductive) positively impacts performance compared to unseen genes (inductive),
due to exploitation of the topological information. Thus the inductive test setting provides a more difficult
prediction problem, and a better evaluation method for gene representations. Furthermore, key areas of
improvement are proposed to enhance the learning of robust gene representations for inductive prediction.
Key words: Gene Regulatory Network, Arabidopsis thaliana, deep learning, inductive, transductive, link-prediction

# Code
Contains scripts and notebooks that were used for generating and analysing the data.

- BinDouble.py: used for analysing the performance of paired representations.
- BinSingle.py: used for analysising the performance of individual representations.
- BinTriple.py: used for analysing the performance of combining all representations
- Data.py: containes classes for creating the iterable-style datasets.
- DataProcessor.py: contains code for partitioning the label data.
- Experiments.ipynb: contains the representation, vector-length, base-line, GO evidence code, and network analysis
- GO.ipynb: Code used to construct the GO representations.
- Go.py: code used for analysing the various GO representations.
- Model.py: contains the neural network model architectures. 
- Optimization.py: code for hyperparameter optimization using random search.
- Plot.py: code used for plotting the data.
- Plotting.py: contains the code for analysing the GO evidance code, and embedding space.
- Utils.py: contains code for initailizing weights, early stopping, and sampling the config space.
- VectorLength.py: code used for analysing the effect of vector length.
- benchmark.py: code used to benchmark the scGREAT model (Wang2024, et al) across the datasplits.
- check.py: code to evaluate the distributions of the datasplits.
- data\_prep.ipynb: code used for pre-procssing the label.
- demo.py: contains code used to validate model performance and contstruct the various GO representations.
- Gene2vec.ipynb: code used to construct the harvest coexpression neighbors.
- Gensim.ipynb: Code used to construct the coexpression representation. 
- optimization.ipynb: code used to analyse the hyperparameter random search results.
- scgreat.ipynb: code used to analyse the performance and network of scGREAT.

# DATA
contains the label data that was use during training.

- BaseLine Folder contains the data that was used for evaluating all proprietary models.
- scGREAT contains the data that was used to evaluate the scGREAT model.

