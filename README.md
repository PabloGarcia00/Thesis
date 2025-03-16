# Thesis
Msc thesis work

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

