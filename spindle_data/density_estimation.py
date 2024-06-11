import pickle

import numpy as np
from sklearn.neighbors import KernelDensity

validation_embedding_path = r'C:\Users\lucas\PycharmProjects\02466-spindle\extract_embeddings\kornum\validate_embeddings.pkl'
with open(validation_embedding_path, 'rb') as file:
    validation_embeddings = pickle.load(file)

# Load the embeddings from the provided .pkl file
test_embedding_path = r'C:\Users\lucas\PycharmProjects\02466-spindle\extract_embeddings\kornum\test_embeddings.pkl'
with open(test_embedding_path, 'rb') as file:
    test_embeddings = pickle.load(file)

# Fit KDE on the embeddings
kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(validation_embeddings)

# Evaluate density at specific points (example: first 10 points)
log_density = kde.score_samples(test_embeddings[:10])
# if below threshold, then check if it's an artifact or not and then compute the accuracy.

density = np.exp(log_density)  # Convert log density to actual density
print('Density at specific points:', density)

# Extract the first dimension of the embeddings
first_dimension = train_embeddings[:, 2]

# Perform Kernel Density Estimation on the first dimension
kde = gaussian_kde(first_dimension)

# Create a range of values for the first dimension
x_range = np.linspace(first_dimension.min(), first_dimension.max(), 1000)

# Evaluate the KDE on this range
kde_values = kde(x_range)

# Plot the KDE
plt.figure(figsize=(10, 6))
plt.plot(x_range, kde_values, label='KDE')
plt.title('Kernel Density Estimation of the First Dimension of Embeddings')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


param_grid = {
    "n_components": range(3, 10),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
, n_jobs=-1)
grid_search.fit(validation_embeddings)
