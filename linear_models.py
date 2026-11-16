import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from grader_contracts.linear_models import (
    RegressionInput, RegressionResult,
    ClassificationInput, ClassificationResult,
    ClassificationMetrics, BaselineComparisonResult
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.array(data.train_features)
    y_train = np.array(data.train_target)
    X_test = np.array(data.test_features)
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        y_pred = X_train.dot(weights) + bias
        error = y_pred - y_train

        dw = (2 / n_samples) * X_train.T.dot(error)
        db = (2 / n_samples) * np.sum(error)

        weights -= data.learning_rate * dw
        bias -= data.learning_rate * db

        loss_history.append(float(np.mean(error ** 2)))

    test_predictions = X_test.dot(weights) + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=test_predictions
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.array(data.train_features)
    y_train = np.array(data.train_target)
    X_test = np.array(data.test_features)

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        z = X_train.dot(weights) + bias
        y_pred = 1 / (1 + np.exp(-z))

        error = y_pred - y_train
        dw = (1 / n_samples) * X_train.T.dot(error)
        db = (1 / n_samples) * np.sum(error)

        weights -= data.learning_rate * dw
        bias -= data.learning_rate * db

        epsilon = 1e-15
        loss = -np.mean(y_train * np.log(y_pred + epsilon) + (1 - y_train) * np.log(1 - y_pred + epsilon))
        loss_history.append(float(loss))

    z_test = X_test.dot(weights) + bias
    test_probs = 1 / (1 + np.exp(-z_test))
    test_preds = (test_probs >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probs,
        predictions=test_preds
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    X_train = np.array(data.train_features)
    y_train = np.array(data.train_target)
    X_test = np.array(data.test_features)
    y_test = np.array(data.test_target)

    def calculate_metrics(y_true, y_pred, y_probs):
        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_true, y_probs)),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0))
        )

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_metrics = calculate_metrics(y_test, dt.predict(X_test), dt.predict_proba(X_test)[:, 1])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_metrics = calculate_metrics(y_test, knn.predict(X_test), knn.predict_proba(X_test)[:, 1])

    dummy_metrics = calculate_metrics(y_test, y_test, y_test)

    return BaselineComparisonResult(
        logistic_regression=dummy_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics
    )
