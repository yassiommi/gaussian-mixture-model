# Gaussian Mixture Model (GMM) in Python

This project contains an implementation of a Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm in Python. 

## Files

### `gmm.py`

The `gmm.py` file contains the implementation of the `GaussianMixtureModel` class, which includes functionalities for fitting the model to the data, making predictions, and clustering based on the GMM.

### `test.py`

The `test.py` file generates synthetic data using `make_blobs` from `scikit-learn`, fits the GMM to the data using the implemented GaussianMixtureModel class, and visualizes the clusters based on the predicted labels.

## How to Use

### Requirements

- `numpy`
- `scipy`

### Usage

1. **Initialization:** Import the `GaussianMixtureModel` class from `gmm.py`.
2. **Data Preparation:** Prepare your dataset.
3. **Instantiate GMM:** Create an instance of `GaussianMixtureModel` by specifying the number of components.
4. **Fit GMM:** Use the `fit` method to fit the GMM to your data.
5. **Predict:** Use the `predict` method to predict the clusters for new data points.

### Example

example of usage:
```
, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
gmm = GaussianMixtureModel(n_components=4)
gmm.fit(X)
predicted_labels = gmm.predict(X)
```

output:

