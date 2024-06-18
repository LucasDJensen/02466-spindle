import os

import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KernelDensity

from _globals import cur_dir
from load_embeddings import normalized_validation_embeddings, normalized_test_embeddings, label_list_test_embeddings

# Step 1: Load the embeddings
label_list_test_embeddings = label_list_test_embeddings[:len(normalized_test_embeddings)]
print(normalized_validation_embeddings.shape)  # (140102, 1000)
print(normalized_test_embeddings.shape)  # (176465, 1000)
print(label_list_test_embeddings.shape)  # (176465,)
print(label_list_test_embeddings.sum())  # 495

model_filename = os.path.join(cur_dir, 'kde_model.joblib')

# Step 2: Check if model file exists and load the model if it does, otherwise fit a new model
if os.path.exists(model_filename):
    kde = joblib.load(model_filename)
    print("Loaded KDE model from disk.")
else:
    # Fit a Kernel Density Estimation (KDE)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(normalized_validation_embeddings)
    joblib.dump(kde, model_filename)
    print("Fitted KDE model and saved to disk.")

# Step 3: Compute the log probabilities
kde_log_prob = kde.score_samples(normalized_test_embeddings)

# Step 4: Compute the ROC AUC score
roc_auc_kde = roc_auc_score(label_list_test_embeddings, kde_log_prob)
print(f'ROC AUC score for KDE: {roc_auc_kde:.4f}')

# Step 5: Compute the ROC curve
fpr_kde, tpr_kde, _ = roc_curve(label_list_test_embeddings, kde_log_prob)

# Step 6: Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_kde, tpr_kde, label=f'KDE (AUC = {roc_auc_kde:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Step 7: Save the plot
plt.savefig('roc_curve_kde.png', dpi=500)
plt.show()
