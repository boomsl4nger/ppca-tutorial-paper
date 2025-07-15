"""
Demonstration of modelling and applications for probabilistic principal component analysis (PPCA), 
using a sample dataset. Accompanies my tutorial paper, "Linear Factor Models - Probabilistic PCA".
For STAT3007 Deep Learning at UQ, 2024 Sem 1.

Author: Angus Scroggie
Email: a.scroggie@student.uq.edu.au
Date: 04/05/2024

References:
Bishop (2006), Pattern Recognition and Machine Learning
https://bsc-iitm.github.io/ML_Handbook/pages/Probabilistic_PCA.html
https://cs.nyu.edu/~roweis/code.html#empca
https://github.com/PRML/PRMLT/tree/master/chapter12
https://github.com/cangermueller/ppca
https://github.com/ymcdull/ppca
https://www.tensorflow.org/probability/examples/Probabilistic_PCA
https://www.youtube.com/watch?v=lJ0cXPoEozg
"""

## Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme()
rng = np.random.default_rng(seed=88)


## Construct dataset
n_samples = 40
X = rng.multivariate_normal((2, 1), [[5, 3], [3, 4]], size=n_samples)


## Part 1: EM algorithm for PPCA
def ppca_em(X, m=2, s=1.0, maxiter=20, tol=1e-6):
    """Perform EM algorithm to maximize likelihood of prob. PCA model.

    Args:
        X (array-like): Input data matrix (d, n), where n is the number of samples and d is the 
            original data dimension.
        m (int, optional): Dimension of latent space. Defaults to 2.
        s (float, optional): Prior sigma^2 value. Defaults to 1.0.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-6.

    Returns:
        W (array-like): Weight matrix (d, m)
        mu (array-like): Mean vector (d, 1)
        beta (float): Inverse of variance
        llh (array-like): Log-likelihood estimates at each iteration (n_iter, 1)
    """
    d, n = X.shape
    mu = np.mean(X, axis=1, keepdims=True)
    X = X - mu # Mean centering

    # EM estimate initialization
    llh = np.zeros(maxiter)
    llh[0] = -np.inf
    W = np.random.randn(d, m)
    R = X.dot(X.T) / n
    
    for iter in range(1, maxiter):
        M = W.T.dot(W) + s * np.eye(m)
        U = np.linalg.inv(M.dot(W.T).dot(R).dot(W) + s * np.eye(m))
        C = W.dot(W.T) + s * np.eye(m)

        # Likelihood computation
        llh[iter] = -n * (d * np.log(2 * np.pi) + np.log(np.linalg.det(C)) + np.trace(C)) / 2

        if abs(llh[iter] - llh[iter - 1]) < tol * abs(llh[iter - 1]):
            break

        # Update parameters
        W = R.dot(W).dot(U)
        s = np.trace(R - R.dot(W).dot(M).dot(W.T)) / d

    return W, mu, s, llh[1:iter]

# Apply EM algorithm to generated dataset
m = 1
W_hat, mu, sigma2_hat, llhs = ppca_em(X.T, m, s=0.2)

# Plot log-likelihood over iterations
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(1, len(llhs)+1), y=llhs, marker='o', color='blue')
plt.xlabel('Iteration #')
plt.ylabel('Loglikelihood')
plt.xticks(range(1, len(llhs)+1, 2))
plt.title('Loglikelihood Convergence')
plt.grid(True)
plt.show()


## Part 2: PCA vs. PPCA dim-reduction
# Using sklearn's PCA object
pca = PCA(n_components=1, random_state=42)
X_r = pca.fit_transform(X)
X_pca_projects = pca.inverse_transform(X_r)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], s=20, alpha=0.8)
sns.scatterplot(x=X_pca_projects[:, 0], y=X_pca_projects[:, 1])
plt.plot(
    [X[:, 0], X_pca_projects[:, 0]], [X[:, 1], X_pca_projects[:, 1]], 
    color="red", linestyle="-", linewidth=1, alpha=0.5)
sns.scatterplot
plt.title("Conventional PCA projections")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot projections via probabilistic PCA
M_hat = W_hat.T @ W_hat + sigma2_hat * np.eye(m)
Z = np.linalg.solve(M_hat, W_hat.T.dot((X.T - mu)))
X_ppca_projects = W_hat @ Z + mu
X_ppca_projects = X_ppca_projects.T

plt.subplot(1, 2, 2)
sns.scatterplot(x=X[:, 0], y=X[:, 1], s=20, alpha=0.8)
sns.scatterplot(x=X_ppca_projects[:, 0], y=X_ppca_projects[:, 1])
plt.plot(
    [X[:, 0], X_ppca_projects[:, 0]], [X[:, 1], X_ppca_projects[:, 1]], 
    color="red", linestyle="-", linewidth=1, alpha=0.5)
plt.title("Probabilistic PCA projections")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()


## Part 3: Application to missing samples
# NOTE: This code is not implemented properly and hence was not used in the tutorial paper.
# Omit samples from the generated dataset
missing_ratio = 0.3
n_missing = int(n_samples * missing_ratio)
missing_indices = np.random.choice(n_samples, n_missing, replace=False)
X_missing = X.copy()
X_missing[missing_indices] = np.nan
known_mask = np.logical_not(np.isin(np.arange(len(X)), missing_indices))
W_hat, mu, sigma2_hat, llhs = ppca_em(X[known_mask].T, m=1, s=1.0)

# Impute missing values by reconstruction
Z_missing = np.linalg.solve(
    np.dot(W_hat.T, W_hat) + sigma2_hat * np.eye(m), 
    W_hat.T.dot(X[missing_indices].T - mu))
noise = rng.normal(0, np.sqrt(sigma2_hat), size=X[missing_indices].T.shape)
X_imputed = W_hat.dot(Z_missing) + mu + noise
X_imputed = X_imputed.T

# Plot to compare original and imputed values
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], color="blue", label="Original Data")
sns.scatterplot(
    x=X[missing_indices][:, 0], y=X[missing_indices][:, 1], 
    color="red", marker="x", linewidth=2, label="Missing Values")
plt.title("Original Data with Missing Values")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Imputed data
plt.subplot(1, 2, 2)
sns.scatterplot(x=X[known_mask][:, 0], y=X[known_mask][:, 1], color="blue", label="Original Data")
sns.scatterplot(
    x=X_imputed[:, 0], y=X_imputed[:, 1], 
    color="orange", marker="o", label="Imputed Values")
plt.title("Imputed Data using PPCA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()