import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, random_state=None, max_iter=100) -> None:
        """
        Initialize a Perceptron classifier.

        :param learning_rate: Step size used in the weight and bias update rule.
        :param random_state: Seed for random weight initialization. If None,
            initialization is non-deterministic.
        :param max_iter: Maximum number of training epochs.
        """
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_iter = max_iter
        self._bias = 0.0
        self._weights: np.ndarray | None = None
        self.errors_per_epoch = []

    def _f(self, x: np.ndarray) -> np.ndarray:
        """
        Element-wise step activation function.
        :param x: Input features.
        :return: A binary array where values greater than 0 are mapped to 1, otherwise 0.
        """
        return np.where(x > 0, 1, 0)

    def _predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Compute binary predictions from input features using current weights and bias."""
        if self._weights is None:
            raise ValueError("Model weights are not initialized.")

        linear_out = np.dot(X, self._weights) + self._bias
        return self._f(linear_out)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the training data.
        :param X: Training data features, shape (n_samples, n_features).
        :param y: Target labels, shape (n_samples,).
        """
        if len(X) != len(y):
            raise ValueError("The number of samples and labels must be the same.")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("Binary classification is required. Found classes: {}".format(classes))
        y_bin = np.where(y == classes[0], 0, 1)

        rng = np.random.RandomState(self.random_state)
        self._bias = 0.0
        self._weights = rng.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.errors_per_epoch = []

        for _ in range(self.max_iter):
            errors = 0  # Errores durante la iteración
            for x_i, y_i in zip(X, y_bin):
                # For each training sample, calculate the linear output and apply the step function
                y_predicted = self._predict_binary(x_i)

                # Perceptron update
                update = self.learning_rate * (y_i - y_predicted)
                self._weights += update * x_i
                self._bias += update

                errors += int(update != 0)

            self.errors_per_epoch.append(errors)

            if errors == 0:
                break

    def predict(self, X: np.ndarray):
        """
        Predict class labels for samples in X.
        :param X: Input features, shape (n_samples, n_features).
        :return: Predicted class labels, shape (n_samples,).
        """
        return self._predict_binary(X)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load_iris(return_X_y=True)
    X = X[y != 2]
    y = y[y != 2]
    X = X[:, :2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = Perceptron(learning_rate=0.1, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=" * 60)
    print("PERCEPTRON TRAINING RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Epochs trained: {len(clf.errors_per_epoch)}")
    print("\nErrors per epoch (evolution):")
    print(clf.errors_per_epoch)
    print("=" * 60)
