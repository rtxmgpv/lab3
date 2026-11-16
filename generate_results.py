import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from linear_models import train_linear_regression, train_logistic_classifier
from grader_contracts.linear_models import RegressionInput, ClassificationInput


def main():
    print("Обработка Student_Performance.csv...")
    df_reg = pd.read_csv("Student_Performance.csv")
    df_reg['Extracurricular Activities'] = df_reg['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

    X = df_reg.drop(columns=['Performance Index']).values.astype(float)
    y = df_reg['Performance Index'].values.astype(float)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    reg_data = RegressionInput(
        train_features=X_train, train_target=y_train,
        test_features=X_test, test_target=y_test
    )
    reg_result = train_linear_regression(reg_data)
    test_mse = float(np.mean((reg_result.predictions - y_test) ** 2))

    print("Обработка german.csv...")
    df_cls = pd.read_csv("german.csv", sep=';')

    y_cls = df_cls.iloc[:, 0].values.astype(int)
    X_cls = df_cls.iloc[:, 1:].values.astype(float)

    X_mean_cls = X_cls.mean(axis=0)
    X_std_cls = X_cls.std(axis=0)
    X_std_cls[X_std_cls == 0] = 1.0
    X_scaled_cls = (X_cls - X_mean_cls) / X_std_cls

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_scaled_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    cls_data = ClassificationInput(
        train_features=X_train_c, train_target=y_train_c,
        test_features=X_test_c, test_target=y_test_c
    )

    cls_result = train_logistic_classifier(cls_data)

    def get_metrics(y_true, y_pred, y_probs):
        return {
            "roc_auc": float(roc_auc_score(y_true, y_probs)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0))
        }

    own_metrics = get_metrics(y_test_c, cls_result.predictions, cls_result.probabilities)

    print("Обучение готовых моделей...")

    sk_log_reg = LogisticRegression(random_state=42, max_iter=1000)
    sk_log_reg.fit(X_train_c, y_train_c)
    sk_log_metrics = get_metrics(
        y_test_c, 
        sk_log_reg.predict(X_test_c), 
        sk_log_reg.predict_proba(X_test_c)[:, 1]
    )

    dt = DecisionTreeClassifier(random_state=42).fit(X_train_c, y_train_c)
    dt_metrics = get_metrics(y_test_c, dt.predict(X_test_c), dt.predict_proba(X_test_c)[:, 1])

    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_c, y_train_c)
    knn_metrics = get_metrics(y_test_c, knn.predict(X_test_c), knn.predict_proba(X_test_c)[:, 1])

    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": sk_log_metrics,
            "decision_tree": dt_metrics,
            "knn": knn_metrics
        }
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
