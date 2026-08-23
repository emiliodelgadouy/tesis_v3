"""Diagnóstico de localización para una guía patch → ABMIL.

La evaluación recorre todos los tiles de la grilla MIL. La ROI no participa en
la inferencia: se usa solamente después para comprobar si el patch head ordenó
primero una región que contiene el hallazgo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.patches import Rectangle
from sklearn.metrics import average_precision_score, roc_auc_score

from src.dataset_provider import (
    _montage_bag,
    _roi_norms_from_row,
    as_tf_dataset,
    build_dataset_provider,
)


@dataclass
class PatchGuideLocalizationResult:
    """Resultados agregados, por imagen y ejemplos visuales del diagnóstico."""

    summary: pd.DataFrame
    per_image: dict[str, pd.DataFrame]
    examples: dict[str, list[dict[str, Any]]]
    bag_grid: tuple[int, int]

    def show_examples(self, split: str = "val", *, max_examples: int = 6):
        """Muestra probabilidades patch, atención guía y ROI sobre bags reales."""
        if split not in self.examples:
            available = ", ".join(sorted(self.examples))
            raise KeyError(f"Split {split!r} ausente; disponibles: {available}")
        selected = self.examples[split][:max_examples]
        if not selected:
            raise ValueError(f"No se guardaron ejemplos para el split {split!r}")

        cols = min(3, len(selected))
        rows = int(np.ceil(len(selected) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.6 * rows))
        axes = np.atleast_1d(axes).reshape(-1)
        grid_rows, grid_cols = self.bag_grid

        for ax, example in zip(axes, selected):
            bag = np.asarray(example["bag"])
            canvas = _montage_bag(bag, self.bag_grid).astype(np.float32)
            if canvas.max() > 1.0:
                canvas /= 255.0
            canvas = np.clip(canvas, 0.0, 1.0)
            canvas_h, canvas_w = canvas.shape[:2]
            tile_h, tile_w = canvas_h / grid_rows, canvas_w / grid_cols

            ax.imshow(canvas)
            for row in range(1, grid_rows):
                ax.axhline(row * tile_h - 0.5, color="cyan", linewidth=1)
            for col in range(1, grid_cols):
                ax.axvline(col * tile_w - 0.5, color="cyan", linewidth=1)

            probabilities = example["patch_probabilities"]
            attention = example["guide_attention"]
            for index, (probability, weight) in enumerate(zip(probabilities, attention)):
                row, col = divmod(index, grid_cols)
                ax.text(
                    (col + 0.5) * tile_w,
                    (row + 0.5) * tile_h,
                    f"p={probability:.2f}\na={weight:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
                )

            roi = example["roi"]
            if roi is not None:
                xmin, ymin, xmax, ymax = roi
                ax.add_patch(
                    Rectangle(
                        (xmin * canvas_w, ymin * canvas_h),
                        (xmax - xmin) * canvas_w,
                        (ymax - ymin) * canvas_h,
                        edgecolor="red",
                        facecolor="none",
                        linewidth=2.5,
                    )
                )

            ax.set_title(
                f"{example['category']} | y={int(example['label'])} | "
                f"top={int(example['top_tile_index'])}\n{example['name']}"
            )
            ax.axis("off")

        for ax in axes[len(selected) :]:
            ax.axis("off")
        fig.suptitle(
            "Rojo: ROI (solo evaluación) · p: probabilidad patch · a: atención guía",
            fontsize=12,
        )
        fig.tight_layout()
        return fig


def _stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _tile_roi_overlap(
    roi: tuple[float, float, float, float] | None,
    bag_grid: tuple[int, int],
) -> np.ndarray:
    rows, cols = bag_grid
    overlap = np.zeros(rows * cols, dtype=np.float64)
    if roi is None:
        return overlap

    xmin, ymin, xmax, ymax = (float(np.clip(value, 0.0, 1.0)) for value in roi)
    if xmax <= xmin or ymax <= ymin:
        return overlap

    for index in range(rows * cols):
        row, col = divmod(index, cols)
        tile_xmin, tile_xmax = col / cols, (col + 1) / cols
        tile_ymin, tile_ymax = row / rows, (row + 1) / rows
        overlap_w = max(0.0, min(tile_xmax, xmax) - max(tile_xmin, xmin))
        overlap_h = max(0.0, min(tile_ymax, ymax) - max(tile_ymin, ymin))
        overlap[index] = overlap_w * overlap_h
    return overlap


def _predict_patch_logits(model, tiles: np.ndarray, *, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        stop = min(start + batch_size, len(tiles))
        logits = model.predict_on_batch(tiles[start:stop])
        outputs.append(np.asarray(logits, dtype=np.float64).reshape(-1))
    return np.concatenate(outputs, axis=0)


def _safe_binary_metric(metric_fn, labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels.astype(int)).size < 2:
        return float("nan")
    return float(metric_fn(labels, scores))


def _select_examples(records: list[dict[str, Any]], max_per_category: int) -> list[dict[str, Any]]:
    positive_hits = [
        record
        for record in records
        if record["label"] >= 0.5 and record["roi_valid"] and record["top1_roi_hit"]
    ][:max_per_category]
    positive_misses = [
        record
        for record in records
        if record["label"] >= 0.5 and record["roi_valid"] and not record["top1_roi_hit"]
    ][:max_per_category]
    negative_high = sorted(
        (record for record in records if record["label"] < 0.5),
        key=lambda record: record["max_patch_probability"],
        reverse=True,
    )[:max_per_category]

    selected: list[dict[str, Any]] = []
    for category, category_records in (
        ("positivo_acierto", positive_hits),
        ("positivo_error", positive_misses),
        ("negativo_score_alto", negative_high),
    ):
        for record in category_records:
            example = dict(record)
            example["category"] = category
            selected.append(example)
    return selected


def _summarize_split(split: str, frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.float64)
    max_probabilities = frame["max_patch_probability"].to_numpy(dtype=np.float64)
    positives = frame[frame["label"] >= 0.5]
    negatives = frame[frame["label"] < 0.5]
    localized = positives[positives["roi_valid"]]

    def _mean(column: str) -> float:
        return float(localized[column].astype(float).mean()) if len(localized) else float("nan")

    return {
        "split": split,
        "bags": len(frame),
        "positives": len(positives),
        "negatives": len(negatives),
        "positives_with_roi": len(localized),
        "bag_max_roc_auc": _safe_binary_metric(roc_auc_score, labels, max_probabilities),
        "bag_max_pr_auc": _safe_binary_metric(average_precision_score, labels, max_probabilities),
        "roi_top1_hit_rate": _mean("top1_roi_hit"),
        "roi_top3_hit_rate": _mean("top3_roi_hit"),
        "roi_top1_max_overlap_rate": _mean("top1_max_overlap_hit"),
        "roi_mean_best_rank": _mean("best_roi_rank"),
        "positive_max_probability_median": (
            float(positives["max_patch_probability"].median()) if len(positives) else float("nan")
        ),
        "negative_max_probability_median": (
            float(negatives["max_patch_probability"].median()) if len(negatives) else float("nan")
        ),
        "negative_max_probability_p90": (
            float(negatives["max_patch_probability"].quantile(0.90)) if len(negatives) else float("nan")
        ),
        "mean_effective_tiles": float(frame["effective_tiles"].mean()),
    }


def evaluate_patch_guide_localization(
    config: dict[str, Any],
    patch_builder,
    *,
    val: pd.DataFrame,
    test: pd.DataFrame | None = None,
    top_k: int = 3,
    patch_batch_size: int | None = None,
    guide_temperature: float | None = None,
    max_examples_per_category: int = 2,
    show_examples: bool = True,
) -> PatchGuideLocalizationResult:
    """Evalúa el patch head sobre todos los tiles de bags reales.

    El modelo nunca recibe la ROI. Las métricas de localización se calculan
    comparando posteriormente el ranking de logits con el solapamiento ROI/tile.
    """
    if patch_builder is None or getattr(patch_builder, "model", None) is None:
        raise ValueError("patch_builder debe conservar el modelo patch entrenado")
    if top_k <= 0:
        raise ValueError("top_k debe ser mayor que cero")

    bag_grid = tuple(int(value) for value in (config.get("FULL") or {}).get("BAG_GRID", (3, 3)))
    bag_size = bag_grid[0] * bag_grid[1]
    top_k = min(int(top_k), bag_size)
    patch_batch_size = int(patch_batch_size or config["GENERAL"]["BATCH_SIZE"])
    guide_temperature = float(
        guide_temperature
        if guide_temperature is not None
        else config["MIL"].get("GUIDED_ATTENTION_TEMPERATURE", 1.0)
    )
    if guide_temperature <= 0:
        raise ValueError("guide_temperature debe ser mayor que cero")

    provider = build_dataset_provider(
        config,
        tuple(patch_builder.IMG_SIZE),
        config["MIL"]["BATCH_SIZE"],
        lateralize=True,
        mode="abmil",
        bag_keras_tiling=False,
        cache_dataset=False,
    )

    per_image: dict[str, pd.DataFrame] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    split_tables = {"val": val}
    if test is not None:
        split_tables["test"] = test

    for split, source_table in split_tables.items():
        table = source_table.reset_index(drop=True)
        dataset = provider.build_eval(table, name=f"{split}_guide_localization")
        records: list[dict[str, Any]] = []
        offset = 0

        for bags_tensor, labels_tensor in as_tf_dataset(dataset):
            bags = np.asarray(bags_tensor)
            labels = np.asarray(labels_tensor, dtype=np.float64).reshape(-1)
            batch_count, tile_count = bags.shape[:2]
            if tile_count != bag_size:
                raise ValueError(
                    f"El dataset produjo {tile_count} tiles pero BAG_GRID={bag_grid} requiere {bag_size}"
                )

            flat_tiles = bags.reshape((-1, *bags.shape[2:]))
            logits = _predict_patch_logits(
                patch_builder.model,
                flat_tiles,
                batch_size=patch_batch_size,
            ).reshape(batch_count, tile_count)
            probabilities = tf.math.sigmoid(logits).numpy()
            guide_attention = _stable_softmax(logits, guide_temperature)

            for local_index in range(batch_count):
                row = table.iloc[offset + local_index]
                roi = _roi_norms_from_row(
                    row,
                    provider.config,
                    provider.lateralize_flip_side,
                )
                overlap = _tile_roi_overlap(roi, bag_grid)
                ranking = np.argsort(-logits[local_index], kind="stable")
                relevant_tiles = overlap > 0.0
                roi_valid = bool(np.any(relevant_tiles))
                top_tile = int(ranking[0])
                best_roi_rank = (
                    int(np.flatnonzero(relevant_tiles[ranking])[0] + 1)
                    if roi_valid
                    else float("nan")
                )
                top1_roi_hit = bool(relevant_tiles[top_tile]) if roi_valid else False
                top3_roi_hit = bool(np.any(relevant_tiles[ranking[:top_k]])) if roi_valid else False
                max_overlap = float(overlap.max()) if roi_valid else 0.0
                top1_max_overlap_hit = (
                    bool(overlap[top_tile] >= max_overlap - 1e-12) if roi_valid else False
                )
                attention = guide_attention[local_index]
                entropy = float(-np.sum(attention * np.log(np.clip(attention, 1e-12, 1.0))))
                path = str(row.get(provider.config.path_column, ""))

                records.append(
                    {
                        "path": path,
                        "name": path.rsplit("/", 1)[-1],
                        "image_id": row.get("image_id", offset + local_index),
                        "label": float(labels[local_index]),
                        "roi": roi,
                        "roi_valid": roi_valid,
                        "top_tile_index": top_tile,
                        "top1_roi_hit": top1_roi_hit,
                        "top3_roi_hit": top3_roi_hit,
                        "top1_max_overlap_hit": top1_max_overlap_hit,
                        "best_roi_rank": best_roi_rank,
                        "max_patch_probability": float(probabilities[local_index].max()),
                        "guide_entropy": entropy,
                        "effective_tiles": float(np.exp(entropy)),
                        "patch_logits": logits[local_index].copy(),
                        "patch_probabilities": probabilities[local_index].copy(),
                        "guide_attention": attention.copy(),
                        "tile_roi_overlap": overlap,
                        "bag": bags[local_index].copy(),
                    }
                )
            offset += batch_count

        if offset != len(table):
            raise RuntimeError(
                f"El split {split!r} produjo {offset} bags para una tabla de {len(table)} filas"
            )

        frame = pd.DataFrame(
            [
                {key: value for key, value in record.items() if key != "bag"}
                for record in records
            ]
        )
        per_image[split] = frame
        examples[split] = _select_examples(records, max_examples_per_category)
        summaries.append(_summarize_split(split, frame))

    result = PatchGuideLocalizationResult(
        summary=pd.DataFrame(summaries).set_index("split"),
        per_image=per_image,
        examples=examples,
        bag_grid=bag_grid,
    )
    print("\nDiagnóstico de localización del patch head sobre bags completos:")
    print(result.summary.round(4).to_string())
    if show_examples:
        result.show_examples("val")
        plt.show()
    return result
