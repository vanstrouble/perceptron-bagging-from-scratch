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


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from matplotlib import pyplot as plt

    def plot_decision_boundary(ax, clf, X_data, y_data, title):
        """Plot 2D data points and the perceptron decision boundary."""
        x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
        y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = clf.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5], cmap="coolwarm")
        ax.scatter(
            X_data[y_data == 0, 0],
            X_data[y_data == 0, 1],
            color="tab:blue",
            edgecolor="k",
            label="Class 0",
            s=45,
        )
        ax.scatter(
            X_data[y_data == 1, 0],
            X_data[y_data == 1, 1],
            color="tab:red",
            edgecolor="k",
            label="Class 1",
            s=45,
        )

        ax.set_title(title)
        ax.set_xlabel("Feature 1 (sepal length)")
        ax.set_ylabel("Feature 2 (sepal width)")
        ax.legend(loc="best")

    try:
        X, y = load_iris(return_X_y=True)
        # Seleccionar solo dos clases para clasificación binaria
        X = X[y != 2]
        y = y[y != 2]
        # Usar dos características para poder visualizar en 2D
        X = X[:, :2]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = Perceptron(learning_rate=0.1, random_state=42, max_iter=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))

        # Visualizar cómo separa el modelo y dónde se equivocó en test
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        plot_decision_boundary(axes[0], clf, X_train, y_train, "Train set + decision boundary")
        plot_decision_boundary(axes[1], clf, X_test, y_test, "Test set + decision boundary")

        wrong_mask = y_pred != y_test
        if np.any(wrong_mask):
            axes[1].scatter(
                X_test[wrong_mask, 0],
                X_test[wrong_mask, 1],
                facecolors="none",
                edgecolors="yellow",
                linewidths=2,
                s=140,
                label="Misclassified",
            )
            axes[1].legend(loc="best")

        plt.show()
    except Exception as e:
        print(f"{e}")
