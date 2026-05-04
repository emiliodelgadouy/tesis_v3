import math

import matplotlib.pyplot as plt


def _base_metric_name(metric_name):
    return metric_name.removeprefix("val_").removeprefix("test_")


def _as_history_list(history):
    if isinstance(history, (list, tuple)):
        return history
    return [history]


def _join_histories(histories):
    hist = {}
    for history in histories:
        for key, values in history.history.items():
            hist.setdefault(key, []).extend(values)
    return hist


def _stage_offsets(histories):
    offsets = []
    offset = 0
    for history in histories:
        offsets.append(offset)
        offset += len(next(iter(history.history.values())))
    return offsets


def plot_history(
        history,
        test_metrics=None,
        best_metric_epochs=None,
        best_metric_name="pr_auc",
        best_auc_epochs=None,
        cols=2,
        backbone_name=None,
):
    histories = _as_history_list(history)
    hist = _join_histories(histories)
    test_metrics = test_metrics or {}
    if best_metric_epochs is None:
        best_metric_epochs = best_auc_epochs or []
        if best_auc_epochs is not None:
            best_metric_name = "auc"
    normalized_test_metrics = {
        _base_metric_name(key): value
        for key, value in test_metrics.items()
    }
    metric_names = sorted(
        {_base_metric_name(key) for key in hist.keys()}
        | set(normalized_test_metrics.keys())
    )
    epochs = range(1, len(next(iter(hist.values()))) + 1)
    stage_offsets = _stage_offsets(histories)
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

        if metric == best_metric_name:
            for index, best_epoch in enumerate(best_metric_epochs):
                if best_epoch is None or index >= len(histories):
                    continue
                stage_hist = histories[index].history
                val_metric_name = f"val_{best_metric_name}"
                if val_metric_name not in stage_hist:
                    continue

                x = stage_offsets[index] + best_epoch
                y = stage_hist[val_metric_name][best_epoch - 1]
                ax.scatter(x, y, s=90, facecolors="none", edgecolors="black")

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    for ax in axes[len(metric_names):]:
        ax.axis("off")

    if backbone_name:
        fig.suptitle(backbone_name, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()
    plt.show()