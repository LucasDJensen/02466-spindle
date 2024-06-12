from load_embeddings import normalized_validation_embeddings, normalized_test_embeddings, label_list_test_embeddings
label_list_test_embeddings = label_list_test_embeddings[:len(normalized_test_embeddings)]
print(normalized_validation_embeddings.shape)  # (140102, 1000)
print(normalized_test_embeddings.shape)  # (176465, 1000)
print(label_list_test_embeddings.shape)  # (176465,)
print(label_list_test_embeddings.sum())  # 495

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

# Step 2: Fit a density estimation model (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=1, covariance_type='full')
gmm.fit(normalized_validation_embeddings)

# Step 3: Detect anomalies
# Score samples: lower score indicates higher likelihood of being an outlier
test_scores = gmm.score_samples(normalized_test_embeddings)

# Choose a threshold for anomaly detection
threshold = np.percentile(test_scores, 5)  # This is an arbitrary choice; you might need to tune it

# Classify samples based on the threshold
predicted_labels = test_scores < threshold

# Step 4: Evaluate the results
# Assuming label_list_test_sample has 1 for artifacts and 0 for normal data
accuracy = accuracy_score(label_list_test_embeddings, predicted_labels)

print(f"Accuracy: {accuracy}")

