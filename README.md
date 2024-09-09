# Machine Learning for Graphs and Sequential Data

This repository contains several projects exploring machine learning techniques for graphs and sequential data. Each project focuses on different aspects of machine learning, such as Hidden Markov Models, word embeddings, neural temporal point processes, spectral clustering, and graph neural networks.

## Projects Overview

### 1. Hidden Markov Models (HMMs)
Implement inference and sampling techniques for HMMs and use them to classify sequence data. The project involves filling in the code for inference and classification in Python files (`classification.py`, `generator.py`) based on the instructions in the `task_01_hidden_markov_model.ipynb` notebook.

### 2. Word2Vec
Develop vector representations for words using the Skip-Gram model, trained on a dataset of restaurant reviews. The implementation focuses on functionality in Python files (`data.py`, `model.py`, `train.py`, `analogies.py`). The `word2vec.ipynb` notebook provides an integrated framework to train and evaluate the model.

### 3. Neural Temporal Point Process (TPP)
Implement an autoregressive neural TPP, including utility functions for handling event sequences, an RNN-based encoder, a conditional distribution parameterized in PyTorch, and log-likelihood computation for training. This project enhances understanding of neural modeling for temporal data.

### 4. Spectral Clustering
Apply spectral clustering techniques to graph data to categorize users based on their restaurant reviews. This project requires implementing clustering functionality in `clustering.py` and understanding spectral embeddings using methods like `scipy.sparse.linalg.eigsh` as detailed in `spectral_clustering.ipynb`.

### 5. Graph Neural Networks (GNNs)
Gain hands-on experience with graph machine learning by implementing various GNN models to analyze graph-structured data.
