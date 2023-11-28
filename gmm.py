import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components, max_iterations=100, tol=1e-6):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = [np.cov(X.T) for _ in range(self.n_components)]

        log_likelihood = 0
        for iteration in range(self.max_iterations):
            # E-step: Expectation step
            gamma = self.expectation_step(X)

            # M-step: Maximization step
            self.maximization_step(X, gamma)

            # check for convergence
            new_log_likelihood = self.compute_log_likelihood(X)
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

    def expectation_step(self, X):
        n_samples = X.shape[0]
        gamma = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            gamma[:, k] = self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])

        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def maximization_step(self, X, gamma):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        for k in range(self.n_components):
            Nk = np.sum(gamma[:, k], axis=0)
            self.weights[k] = Nk / n_samples
            self.means[k] = np.sum(X * gamma[:, k].reshape(-1, 1), axis=0) / Nk
            diff = X - self.means[k]
            self.covariances[k] = np.dot((diff * gamma[:, k].reshape(-1, 1)).T, diff) / Nk

    def compute_log_likelihood(self, X):
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
        return np.sum(np.log(log_likelihood))

    def predict(self, X):
        n_samples = X.shape[0]
        gamma = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            gamma[:, k] = self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])

        return np.argmax(gamma, axis=1)
