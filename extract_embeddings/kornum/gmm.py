import os

import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.mixture import GaussianMixture

from load_embeddings import normalized_validation_embeddings, normalized_test_embeddings, label_list_test_embeddings

# Step 1: Load the embeddings
label_list_test_embeddings = label_list_test_embeddings[:len(normalized_test_embeddings)]
print(normalized_validation_embeddings.shape)  # (140102, 1000)
print(normalized_test_embeddings.shape)  # (176465, 1000)
print(label_list_test_embeddings.shape)  # (176465,)
print(label_list_test_embeddings.sum())  # 495

model_filename = 'gmm_model.joblib'

# Step 2: Check if model file exists and load the model if it does, otherwise fit a new model
if os.path.exists(model_filename):
    gmm = joblib.load(model_filename)
    print("Loaded KDE model from disk.")
else:
    # Fit a Kernel Density Estimation (KDE)
    gmm = GaussianMixture(n_components=2, covariance_type='full', verbose=1)
    gmm.fit(normalized_validation_embeddings)
    joblib.dump(gmm, model_filename)
    print("Fitted KDE model and saved to disk.")

# Step 3: Compute the log probabilities
gmm_log_prob = gmm.score_samples(normalized_test_embeddings)

# Step 4: Compute the ROC AUC score
roc_auc_gmm = roc_auc_score(label_list_test_embeddings, gmm_log_prob)
print(f'ROC AUC score for GMM: {roc_auc_gmm:.4f}')

# Step 5: Compute the ROC curve
fpr_gmm, tpr_gmm, _ = roc_curve(label_list_test_embeddings, gmm_log_prob)

# Step 6: Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_gmm, tpr_gmm, label=f'GMM (AUC = {roc_auc_gmm:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Step 7: Save the plot
plt.savefig('roc_curve_gmm.png', dpi=500)
plt.show()
