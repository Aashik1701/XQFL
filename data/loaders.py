import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_breast_cancer_data(test_size=0.2, random_state=42):
    """Load and scale Wisconsin Breast Cancer dataset.

    Returns X_train, X_test, y_train, y_test as numpy arrays (float32).
    Scaler is fit on train set only to prevent data leakage.
    """
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test
