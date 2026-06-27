import math

import matplotlib.pyplot as plt

_DARK_FIG_FACE = "black"
_DARK_AX_FACE = "black"
_DARK_TEXT = "white"
_DARK_TICK = "#cccccc"
_DARK_GRID = "#333333"
_DARK_SPINE = "#555555"
_DARK_LEGEND_FACE = "#1a1a1a"
_DARK_LEGEND_EDGE = "#555555"
_LINE_COLORS = ("#4FC3F7", "#FFB74D", "#81C784", "#E57373", "#CE93D8", "#FFF176")


def _apply_dark_theme(ax):
    ax.set_facecolor(_DARK_AX_FACE)
    ax.tick_params(colors=_DARK_TICK)
    ax.xaxis.label.set_color(_DARK_TEXT)
    ax.yaxis.label.set_color(_DARK_TEXT)
    ax.title.set_color(_DARK_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_DARK_SPINE)
    legend = ax.get_legend()
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor(_DARK_LEGEND_FACE)
        frame.set_edgecolor(_DARK_LEGEND_EDGE)
        for text in legend.get_texts():
            text.set_color(_DARK_TEXT)


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
        values = list(history.history.values())
        offset += len(values[0]) if values else 0
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

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(7 * cols, 4 * rows),
        squeeze=False,
        facecolor=_DARK_FIG_FACE,
    )
    axes = axes.flatten()

    for ax, metric in zip(axes, metric_names):
        color_idx = 0
        for prefix, label in [("", "train"), ("val_", "val"), ("test_", "test")]:
            key = f"{prefix}{metric}" if prefix else metric
            if key in hist:
                ax.plot(
                    epochs,
                    hist[key],
                    marker="o",
                    label=label,
                    color=_LINE_COLORS[color_idx % len(_LINE_COLORS)],
                )
                color_idx += 1

        if metric in normalized_test_metrics:
            ax.plot(
                epochs,
                [normalized_test_metrics[metric]] * len(epochs),
                linestyle="--",
                label="test",
                color=_LINE_COLORS[color_idx % len(_LINE_COLORS)],
            )
            color_idx += 1

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
                ax.scatter(
                    x,
                    y,
                    s=90,
                    facecolors="none",
                    edgecolors=_DARK_TEXT,
                    linewidths=2,
                )

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, color=_DARK_GRID, alpha=0.85)
        ax.legend()
        _apply_dark_theme(ax)

    for ax in axes[len(metric_names):]:
        ax.axis("off")
        ax.set_facecolor(_DARK_AX_FACE)

    if backbone_name:
        fig.suptitle(
            backbone_name,
            fontsize=14,
            fontweight="bold",
            color=_DARK_TEXT,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()
    plt.show()