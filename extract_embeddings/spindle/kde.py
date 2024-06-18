import os

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KernelDensity

from _globals import cur_dir
from load_embeddings import normalized_validation_embeddings, normalized_test_embeddings, label_list_test_embeddings

# Load the embeddings
label_list_test_embeddings = label_list_test_embeddings[:len(normalized_test_embeddings)]
print(normalized_validation_embeddings.shape)  # (5955, 1000)
print(normalized_test_embeddings.shape)  # (43192, 1000)
print(label_list_test_embeddings.shape)  # (43192,)
print(label_list_test_embeddings.sum())  # 11696
model_filename = os.path.join(cur_dir, 'kde_model.joblib')

# Hypertune n_components
bandwidths = np.divide(range(2, 12, 2), 10)
best_bandwidth = 1
best_roc_auc = 0

for bandwidth in bandwidths:
    gmm = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    gmm.fit(normalized_validation_embeddings)

    # Compute the log probabilities on the validation set
    test_log_prob = gmm.score_samples(normalized_test_embeddings)

    # Compute the ROC AUC score
    roc_auc = roc_auc_score(label_list_test_embeddings, test_log_prob)
    print(f'Bandwidth: {bandwidth}, ROC AUC: {roc_auc:.4f}')

    # Update the best n_components if the current model is better
    if roc_auc > best_roc_auc:
        best_bandwidth = bandwidth
        best_roc_auc = roc_auc

print(f'Best bandwidth: {best_bandwidth}, Best ROC AUC: {best_roc_auc:.4f}')

# # Check if model file exists and load the model if it does, otherwise fit a new model
# if os.path.exists(model_filename):
#     kde = joblib.load(model_filename)
#     print("Loaded KDE model from disk.")
# else:
# Fit a Kernel Density Estimation (KDE)
kde = KernelDensity(kernel='gaussian', bandwidth=best_bandwidth)
kde.fit(normalized_validation_embeddings)
joblib.dump(kde, model_filename)
print("Fitted KDE model and saved to disk.")

# Compute the log probabilities
kde_log_prob = kde.score_samples(normalized_test_embeddings)

# Compute the ROC AUC score
roc_auc_kde = roc_auc_score(label_list_test_embeddings, kde_log_prob)
print(f'ROC AUC score for KDE: {roc_auc_kde:.4f}')

# Compute the ROC curve
fpr_kde, tpr_kde, _ = roc_curve(label_list_test_embeddings, kde_log_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_kde, tpr_kde, label=f'KDE (AUC = {roc_auc_kde:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Save the plot
plt.savefig('roc_curve_kde.png', dpi=500)
# plt.show()
print('Done!')
