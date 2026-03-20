import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, random_state=None, max_iter=100) -> None:
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.max_iter = max_iter
        self._bias = 0.0
        self._weights = None
        self.miss_classified_examples = None

    def _f(self, x: np.ndarray) -> np.ndarray:
        """
        Element-wise step activation function.
        :param x: Input features.
        :return: A binary array where values greater than 0 are mapped to 1, otherwise 0.
        """
        return np.where(x > 0, 1, 0)

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
        self._weights = rng.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.miss_classified_examples = []

        for _ in range(self.max_iter):
            errors = 0  # Errores durante la iteración
            for x_i, y_i in zip(X, y_bin):
                # For each training sample, calculate the linear output and apply the step function
                linear_out = np.dot(x_i, self._weights) + self._bias
                y_predicted = self._f(linear_out)

                # Perceptron update
                update = self.learning_rate * (y_i - y_predicted)
                self._weights += update * x_i
                self._bias += update

                errors += int(update != 0)

            self.miss_classified_examples.append(errors)

    def predict(self, X: np.ndarray):
        """
        Predict class labels for samples in X.
        :param X: Input features, shape (n_samples, n_features).
        :return: Predicted class labels, shape (n_samples,).
        """
        if self._weights is None:
            raise ValueError("The model has not been trained yet.")

        linear_out = np.dot(X, self._weights) + self._bias
        y_predicted = self._f(linear_out)
        return y_predicted
