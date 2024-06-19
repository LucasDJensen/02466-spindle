import numpy as np

print("Running GMM script...")
import os

import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from _globals import cur_dir
from load_embeddings import normalized_validation_embeddings, normalized_test_embeddings, label_list_test_embeddings
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import pandas as pd
import seaborn as sns

# Load the embeddings
label_list_test_embeddings = label_list_test_embeddings[:len(normalized_test_embeddings)]
print(normalized_validation_embeddings.shape)  # (5955, 1000)
print(normalized_test_embeddings.shape)  # (43200, 1000)
print(label_list_test_embeddings.shape)  # (43200,)
print(label_list_test_embeddings.sum())  # 11696
model_filename = os.path.join(cur_dir, 'gmm_model.joblib')
use_existing_model = True

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


def train_gmm():
    param_grid = {
        "n_components": [1, 2, 3, 4, 5, 10, 15, 20],
        "covariance_type": ["full"],
    }

    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score, n_jobs=-1, cv=5
    )
    grid_search.fit(normalized_validation_embeddings)
    # Print the best number of components and the best model
    print(f"Best number of components: {grid_search.best_params_['n_components']}")
    print(f"Best BIC score: {grid_search.best_score_}")
    # Best model
    best_gmm = grid_search.best_estimator_
    joblib.dump(best_gmm, model_filename)
    print(best_gmm)

    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )
    print(df.sort_values(by="BIC score").head())

    sns.catplot(
        data=df,
        kind="bar",
        x="Number of components",
        y="BIC score",
        hue="Type of covariance",
    )
    plt.savefig(os.path.join(cur_dir, "gmm_bic_score.png"), dpi=500)
    # plt.show()
    return best_gmm


def compute_probs(normalized_test_embeddings, label_list_test_embeddings, gmm):
    # Compute the log probabilities
    # Multiply by -1 because model is trained on sleep stages and we predict artifacts.
    # Log Likelihood will be high for sleep stages and low for artifacts.
    gmm_log_prob = np.array(gmm.score_samples(normalized_test_embeddings)) * -1

    # Compute the ROC AUC score
    roc_auc_gmm = roc_auc_score(label_list_test_embeddings, gmm_log_prob)
    print(f'ROC AUC score for GMM: {roc_auc_gmm:.4f}')

    # Compute the ROC curve
    fpr_gmm, tpr_gmm, thresholds = roc_curve(label_list_test_embeddings, gmm_log_prob)
    print(f'Thresholds: {thresholds}')
    return roc_auc_gmm, fpr_gmm, tpr_gmm, thresholds


def plot(fpr_gmm, tpr_gmm, roc_auc_gmm):
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gmm, tpr_gmm, label=f'GMM (AUC = {roc_auc_gmm:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    # Save the plot
    plt.savefig(os.path.join(cur_dir, 'roc_curve_gmm.png'), dpi=500)
    # plt.show()


# Check if model file exists and load the model if it does, otherwise fit a new model with best n_components
if os.path.exists(model_filename) and use_existing_model:
    print("Loading GMM model from disk...")
    gmm = joblib.load(model_filename)
    print("Loaded GMM model from disk.")
else:
    print("Training GMM model...")
    gmm = train_gmm()
    print("Trained GMM model.")

roc_auc_gmm, fpr_gmm, tpr_gmm, thresholds = compute_probs(normalized_test_embeddings, label_list_test_embeddings, gmm)
plot(fpr_gmm, tpr_gmm, roc_auc_gmm)


# Function to find the best threshold based on ROC curve data
def find_best_threshold(fpr, tpr, thresholds):
    # Find the index where the difference between TPR and FPR is minimized
    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]
    return best_threshold


# Function to generate predictions based on the chosen threshold
def generate_predictions(log_prob, threshold):
    # Since we use negative log probabilities, we classify as positive if -log_prob >= threshold
    predictions = (log_prob <= -threshold).astype(int)
    return predictions


# Function to compute and print the confusion matrix
def compute_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    return cm


# Assuming gmm_log_prob has been computed as in the provided script
gmm_log_prob = gmm.score_samples(normalized_test_embeddings)

# Find the best threshold
best_threshold = find_best_threshold(fpr_gmm, tpr_gmm, thresholds)
print(f'Best threshold: {best_threshold}')

# Generate predictions based on the best threshold
predictions = generate_predictions(gmm_log_prob, best_threshold)

# Compute and print the confusion matrix
confusion_matrix_result = compute_confusion_matrix(label_list_test_embeddings, predictions)

# Optionally, visualize the confusion matrix using seaborn heatmap

plt.figure(figsize=(8, 6))
labels = ["not art.", "art."]
sns.heatmap(confusion_matrix_result, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(cur_dir, 'confusion_matrix_gmm.png'), dpi=500)
# plt.show()


print("GMM script completed successfully.")
