import os

import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.mixture import GaussianMixture

from load_embeddings import normalized_validation_embeddings, normalized_test_embeddings, label_list_test_embeddings

# Step 1: Load the embeddings
label_list_test_embeddings = label_list_test_embeddings[:len(normalized_test_embeddings)]
print(normalized_validation_embeddings.shape)  # (5955, 1000)
print(normalized_test_embeddings.shape)  # (43192, 1000)
print(label_list_test_embeddings.shape)  # (43192,)
print(label_list_test_embeddings.sum())  # 11696

model_filename = 'gmm_pca_model.joblib'
pca_model_filename = 'pca_model.joblib'

delete_old = True
if delete_old:
    if os.path.exists(model_filename):
        os.remove(model_filename)
    if os.path.exists(pca_model_filename):
        os.remove(pca_model_filename)

# Step 2: Check if model file exists and load the model if it does, otherwise fit a new model
if os.path.exists(model_filename):
    gmm = joblib.load(model_filename)
    pca = joblib.load(pca_model_filename)
    print("Loaded KDE model from disk.")
else:
    # Fit a Kernel Density Estimation (KDE)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=100)
    pca.fit(normalized_validation_embeddings)
    joblib.dump(pca, pca_model_filename)
    normalized_validation_embeddings = pca.transform(normalized_validation_embeddings)

    gmm = GaussianMixture(n_components=3, covariance_type='full', verbose=1)
    gmm.fit(normalized_validation_embeddings)
    joblib.dump(gmm, model_filename)
    print("Fitted KDE model and saved to disk.")


normalized_test_embeddings = pca.transform(normalized_test_embeddings)

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
