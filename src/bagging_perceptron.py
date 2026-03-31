import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from perceptron import Perceptron
from bagging import BaggingClassifier


def run_comparison(dataset_loader, dataset_name):
    """
    Loads a dataset, trains and evaluates a simple Perceptron and a Bagging Perceptron,
    and prints their accuracy scores.

    :param dataset_loader: A function from sklearn.datasets that loads a dataset.
    :param dataset_name: The name of the dataset for display purposes.
    """
    print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")

    # Load and prepare data
    data = dataset_loader()
    X, y = data.data, data.target

    # For simplicity in the Perceptron, we'll focus on binary classification.
    # We'll take the first two classes if there are more than two.
    classes = np.unique(y)
    if len(classes) > 2:
        print(
            f"Dataset has more than 2 classes. Using classes {classes[0]} and {classes[1]}."
        )
        mask = np.isin(y, classes[:2])
        X = X[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 1. Simple Perceptron
    print("\n--- Training Simple Perceptron ---")
    simple_perceptron = Perceptron(learning_rate=0.1, max_iter=1000, random_state=123)
    simple_perceptron.fit(X_train, y_train)
    y_pred_simple = simple_perceptron.predict(X_test)
    accuracy_simple = accuracy_score(y_test, y_pred_simple)
    print(f"Simple Perceptron Test Accuracy: {accuracy_simple * 100:.2f}%")

    # 2. Bagging with Perceptron
    print("\n--- Training Bagging Perceptron ---")
    bagging_perceptron = BaggingClassifier(
        estimator_factory=lambda: Perceptron(learning_rate=0.1, max_iter=1000),
        n_estimators=15,  # Using 15 perceptrons in the ensemble
        random_state=42,
    )
    bagging_perceptron.fit(X_train, y_train)
    y_pred_bagging = bagging_perceptron.predict(X_test)
    accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
    print(f"Bagging Perceptron Test Accuracy: {accuracy_bagging * 100:.2f}%")

    print(f"{'=' * (42 + len(dataset_name))}")


def main():
    """Main function to run the comparisons."""
    # Compare models on different datasets
    run_comparison(load_iris, "Iris Dataset")
    run_comparison(load_wine, "Wine Dataset")
    run_comparison(load_breast_cancer, "Breast Cancer Dataset")


if __name__ == "__main__":
    main()
