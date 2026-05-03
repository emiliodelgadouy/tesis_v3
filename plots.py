import math

import matplotlib.pyplot as plt


def _base_metric_name(metric_name):
    return metric_name.removeprefix("val_").removeprefix("test_")


def plot_history(history, test_metrics=None, cols=2):
    hist = history.history
    test_metrics = test_metrics or {}
    normalized_test_metrics = {
        _base_metric_name(key): value
        for key, value in test_metrics.items()
    }
    metric_names = sorted(
        {_base_metric_name(key) for key in hist.keys()}
        | set(normalized_test_metrics.keys())
    )
    epochs = range(1, len(next(iter(hist.values()))) + 1)
    rows = math.ceil(len(metric_names) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, metric_names):
        for prefix, label in [("", "train"), ("val_", "val"), ("test_", "test")]:
            key = f"{prefix}{metric}" if prefix else metric
            if key in hist:
                ax.plot(epochs, hist[key], marker="o", label=label)

        if metric in normalized_test_metrics:
            ax.plot(
                epochs,
                [normalized_test_metrics[metric]] * len(epochs),
                linestyle="--",
                label="test",
            )

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    for ax in axes[len(metric_names):]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()