# 8. **Cross-Validation and Model Evaluation**
#    - Main: Implement k-fold cross-validation
#    - Extensions:
#      a) Implement stratified k-fold for imbalanced datasets
#      b) Create functions for various evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
#      c) Implement learning curves to diagnose bias-variance tradeoff
#      d) Discuss cross-validation strategies for time series data


import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


def k_fold_cross_validation(data, k=5):
    """
    Perform vanilla k-fold cross-validation.

    Args:
        data (list): The dataset to be split into folds.
        k (int): Number of folds (default: 5).

    Returns:
        list: A list of k folds, where each fold is a subset of the data.

    This function:
    1. Splits the data into k folds.
    2. Shuffles the folds randomly.
    3. Returns the shuffled folds.
    """
    # Split the data into k folds
    fold_size = len(data) // k
    folds = [data[i * fold_size:(i + 1) * fold_size] for i in range(k)]

    # Shuffle the folds randomly
    random.shuffle(folds)
    
    return folds

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The accuracy score.

    Accuracy is the ratio of correctly predicted instances to the total number of instances.
    """
    return np.mean(np.array(y_true) == np.array(y_pred))

def precision(y_true, y_pred):
    """
    Calculate the precision of predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The precision score.

    Precision is the ratio of true positives to the total number of predicted positives.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def recall(y_true, y_pred):
    """
    Calculate the recall of predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The recall score.

    Recall is the ratio of true positives to the total number of actual positives.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

def f1_score(y_true, y_pred):
    """
    Calculate the F1 score of predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The F1 score.

    F1 Score is the harmonic mean of precision and recall.
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

# Example usage of cross-validation
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 0, 1, 1, 1])

# Split the data into 3 folds
folds = k_fold_cross_validation(list(zip(X, y)), k=3)

# Perform cross-validation
for i, fold in enumerate(folds, 1):
    # Separate the current fold into test set
    test_data = fold
    # Combine all other folds into training set
    train_data = [item for f in folds if f != fold for item in f]
    
    # Split features and labels
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)
    
    # Train and evaluate the model (assuming KNN is defined elsewhere)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Print evaluation metrics for this fold
    print(f"Fold {i} Results:")
    print(f"  Accuracy:  {accuracy(y_test, y_pred):.4f}")
    print(f"  Precision: {precision(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print()

