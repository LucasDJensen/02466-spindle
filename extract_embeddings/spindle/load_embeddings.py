import pickle

import numpy as np

validation_embedding_path = 'embeddings/validation_embeddings.pkl'
with open(validation_embedding_path, 'rb') as file:
    validation_embeddings = pickle.load(file)

# Load the embeddings from the provided .pkl file
test_embedding_path = 'embeddings/test_embeddings.pkl'
with open(test_embedding_path, 'rb') as file:
    test_embeddings = pickle.load(file)

# Compute mean and standard deviation of the validation embeddings
mean = np.mean(validation_embeddings, axis=0)
std = np.std(validation_embeddings, axis=0)
# Normalize test embeddings using the norms computed from the validation embeddings
epsilon = 1e-8
normalized_validation_embeddings = np.divide(validation_embeddings - mean, std + epsilon)
normalized_test_embeddings = np.divide(test_embeddings - mean, std + epsilon)

file = 'embeddings/true_labels.npy'
true_labels = np.asarray(np.load(file), dtype=np.int32)
label_list_test_embeddings: np.ndarray = np.array([1 if x[-1] == 1 else 0 for x in true_labels])
