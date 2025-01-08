## How to Run the Experiment

To run the experiment, execute the experiment script (typically `run_compareExp.py`). This script runs a series of tests with various combinations of model names, clustering methods, and dimensionality reduction methods.

### 1. Set up the Experiment:
- Import the `Exp` class from the `modeling` module.
- Initialize an instance of the `Exp` class.

### 2. Define Experiment Loops:
- The experiment will loop through multiple iterations (`for i in range(10)`), testing the models with different configurations.
- The models are tested for different seasons (`summer`, `winter`).
- Each model (`ann_lin_dev`, `cnn_nn_dev`, `nn_lstm_nn_dev`, `DLinear`, `Autoformer`, `LSTNet`) is tested under both clustered and non-clustered settings.

### 3. Running the Script:
- When executed, the script will automatically run all the configurations and generate results for each test.
- Output will include performance metrics for each model under the given conditions.

## Parameters:

### `model_name`
- **Description**: Specifies the model to test.
- **Options**:
  - `ann_lin_dev`: A type of Artificial Neural Network (ANN).
  - `cnn_nn_dev`: Convolutional Neural Network (CNN) based model.
  - `nn_lstm_nn_dev`: Neural Network with LSTM.
  - `DLinear`: A deep learning-based model.
  - `Autoformer`: A transformer-based model for time series forecasting.
  - `LSTNet`: A model combining LSTM with temporal convolution.

### `n_cluster`
- **Description**: The number of clusters to use for clustering.
- **Options**:
  - `1`: No clustering applied.
  - `2`: Use 2 clusters.
  - `3`: Use 3 clusters.
  - `4`: Use 4 clusters.

### `cluster_method`
- **Description**: The method to use for clustering.
- **Options**:
  - `kmeans`: K-means clustering.
  - `None`: No clustering method (i.e., single group).

### `season`
- **Description**: The seasonality of the data.
- **Options**:
  - `summer`: Seasonality related to summer.
  - `winter`: Seasonality related to winter.

### `cluster_reduce_method`
- **Description**: The method to use for reducing dimensions during clustering.
- **Options**:
  - `pca`: Principal Component Analysis, reduces dimensionality by projecting data onto the principal components.
  - `tsne`: t-Distributed Stochastic Neighbor Embedding, reduces dimensions while preserving local relationships.
  - `umap`: Uniform Manifold Approximation and Projection, preserves both local and global structure, faster than t-SNE.
  - `None`: No dimensionality reduction.

### `cluster_reduce_dim`
- **Description**: The number of dimensions for the clustering reduction method.
- **Options**:
  - `2`: Reduce to 2 dimensions.
  - `None`: No dimensionality reduction.
