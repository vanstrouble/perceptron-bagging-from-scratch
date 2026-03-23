import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from perceptron import Perceptron


def build_dataset(kind: str, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create a small 2D binary dataset with controllable separability."""
    if kind == "separable":
        X, y = make_blobs(
            n_samples=220,
            centers=((-2.5, -2.0), (2.2, 2.0)),
            cluster_std=(0.75, 0.75),
            return_centers=False,
            random_state=random_state,
        )
    elif kind == "semi_separable":
        X, y = make_blobs(
            n_samples=220,
            centers=((-1.8, -1.6), (1.8, 1.6)),
            cluster_std=(1.2, 1.2),
            return_centers=False,
            random_state=random_state,
        )
    elif kind == "non_separable":
        X, y = make_blobs(
            n_samples=220,
            centers=((-1.0, -1.0), (1.0, 1.0)),
            cluster_std=(1.8, 1.8),
            return_centers=False,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
    return X, y


def train_and_score(
    X_train, X_test, y_train, y_test, learning_rate, max_iter, random_state=0
):
    clf = Perceptron(
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, y_pred, acc


def plot_boundary(ax, clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 220),
        np.linspace(y_min, y_max, 220),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.2, levels=[-0.5, 0.5, 1.5], cmap="coolwarm")
    ax.scatter(
        X[y == 0, 0], X[y == 0, 1], c="tab:blue", edgecolor="k", s=35, label="Class 0"
    )
    ax.scatter(
        X[y == 1, 0], X[y == 1, 1], c="tab:red", edgecolor="k", s=35, label="Class 1"
    )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


if __name__ == "__main__":
    configs = [
        {"learning_rate": 0.01, "max_iter": 10},
        {"learning_rate": 0.01, "max_iter": 100},
        {"learning_rate": 0.1, "max_iter": 20},
        {"learning_rate": 0.1, "max_iter": 100},
        {"learning_rate": 0.5, "max_iter": 20},
    ]

    dataset_specs = [
        {"key": "separable", "label": "Easy (linearly separable)"},
        {"key": "semi_separable", "label": "Intermediate (almost separable)"},
        {"key": "non_separable", "label": "Hard (not separable)"},
    ]
    dataset_kinds = [spec["key"] for spec in dataset_specs]
    dataset_labels = {spec["key"]: spec["label"] for spec in dataset_specs}
    results = {}
    best_models = {}

    for kind in dataset_kinds:
        X, y = build_dataset(kind, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        run_results = []
        for cfg in configs:
            clf, y_pred, acc = train_and_score(
                X_train,
                X_test,
                y_train,
                y_test,
                learning_rate=cfg["learning_rate"],
                max_iter=cfg["max_iter"],
                random_state=7,
            )
            run_results.append(
                {
                    "cfg": cfg,
                    "acc": acc,
                    "clf": clf,
                    "pred": y_pred,
                    "X_test": X_test,
                    "y_test": y_test,
                }
            )

        run_results.sort(key=lambda r: r["acc"], reverse=True)
        results[kind] = run_results
        best_models[kind] = run_results[0]

    print("=" * 70)
    print("Perceptron comparison by difficulty: Easy -> Intermediate -> Hard")
    print("=" * 70)
    for kind in dataset_kinds:
        print(f"\nDataset: {dataset_labels[kind]}")
        for r in results[kind]:
            lr = r["cfg"]["learning_rate"]
            max_iter = r["cfg"]["max_iter"]
            epochs = len(r["clf"].errors_per_epoch)
            print(
                f"  lr={lr:<4} | max_iter={max_iter:<3} | epochs={epochs:<3} | acc={r['acc']:.3f}"
            )

    fig, axes = plt.subplots(
        len(dataset_kinds), 2, figsize=(11, 13), constrained_layout=True
    )

    for row, kind in enumerate(dataset_kinds):
        best = best_models[kind]
        clf = best["clf"]
        X_test = best["X_test"]
        y_test = best["y_test"]
        y_pred = best["pred"]

        cfg = best["cfg"]
        title = (
            f"{dataset_labels[kind]} (best: lr={cfg['learning_rate']}, max_iter={cfg['max_iter']})\n"
            f"acc={best['acc']:.3f}"
        )
        plot_boundary(axes[row, 0], clf, X_test, y_test, title)

        wrong = y_pred != y_test
        if np.any(wrong):
            axes[row, 0].scatter(
                X_test[wrong, 0],
                X_test[wrong, 1],
                facecolors="none",
                edgecolors="yellow",
                linewidths=1.8,
                s=120,
                label="Misclassified",
            )
        axes[row, 0].legend(loc="best", fontsize=8)

        labels = [
            f"lr={r['cfg']['learning_rate']}\niter={r['cfg']['max_iter']}"
            for r in results[kind]
        ]
        scores = [r["acc"] for r in results[kind]]
        axes[row, 1].bar(labels, scores, color="slateblue", alpha=0.8)
        axes[row, 1].set_ylim(0.0, 1.05)
        axes[row, 1].set_title(f"Accuracies by configuration ({dataset_labels[kind]})")
        axes[row, 1].set_ylabel("accuracy")
        axes[row, 1].tick_params(axis="x", labelrotation=25)

    plt.show()
