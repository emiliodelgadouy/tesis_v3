from __future__ import annotations
import os
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
import tensorflow as tf

from src.modes import is_mil_mode, normalize_mode, resolve_mode_kwargs

SplitName = Literal["train", "val", "test"]
PatchSampling = Literal["uniform", "normal"]
PatchCropStrategy = Literal["uniform", "roi", "normal", "avoid_roi"]

DEFAULT_PATCH_CROP_BY_LABEL: dict[float, PatchCropStrategy] = {
    0.0: "avoid_roi",
    1.0: "roi",
}


def _roll_tf_random_seed() -> int:
    """Nueva semilla TF por preview estocastico (evita repetir con set_seed global fijo)."""
    seed = int.from_bytes(os.urandom(4), "big") % (2**31 - 1)
    tf.random.set_seed(seed)
    return seed


def _decode_fallback(raw: tf.Tensor) -> tf.Tensor:
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img


def _clahe_np_uint8(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    import cv2

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = arr.astype(np.uint8).reshape(arr.shape[0], arr.shape[1])
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid), int(tile_grid)),
    )
    eq = clahe.apply(gray)
    return np.repeat(eq[..., None], 3, axis=-1).astype(np.float32)


def apply_clahe_tf(img: tf.Tensor, clip_limit: float = 2.0, tile_grid: int = 8) -> tf.Tensor:
    """CLAHE sobre luminancia; salida RGB float32 en [0, 255]."""
    img_u8 = tf.cast(tf.clip_by_value(tf.cast(img, tf.float32), 0.0, 255.0), tf.uint8)
    out = tf.numpy_function(
        func=lambda x: _clahe_np_uint8(x, clip_limit, tile_grid),
        inp=[img_u8],
        Tout=tf.float32,
    )
    out.set_shape([None, None, 3])
    return out


def decode_image(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    lower = tf.strings.lower(path)
    is_jpeg = tf.strings.regex_full_match(lower, ".*\\.jpe?g")
    is_png = tf.strings.regex_full_match(lower, ".*\\.png")

    def _jpeg():
        img = tf.image.decode_jpeg(raw, channels=3, dct_method="INTEGER_FAST")
        img.set_shape([None, None, 3])
        return img

    def _png():
        img = tf.image.decode_png(raw, channels=3)
        img.set_shape([None, None, 3])
        return img

    img = tf.cond(
        is_jpeg,
        _jpeg,
        lambda: tf.cond(is_png, _png, lambda: _decode_fallback(raw)),
    )
    img = tf.image.convert_image_dtype(img, tf.float16) * tf.cast(255.0, tf.float16)
    return img


def _images_to_display(images: np.ndarray) -> np.ndarray:
    """Normaliza un batch de imagenes para imshow (float [0,1] o uint8)."""
    x = np.asarray(images)
    if x.dtype in (np.float16, np.float32, np.float64):
        if x.max() > 1.5:
            x = x / 255.0
        return np.clip(x.astype(np.float32), 0.0, 1.0)
    return x


def _montage_bag(bag: np.ndarray, grid: tuple[int, int] | None = None) -> np.ndarray:
    """Une las instancias (K, H, W, C) de un bag en una unica imagen-grilla."""
    bag = np.asarray(bag)
    k = bag.shape[0]
    if grid is not None:
        rows, cols = int(grid[0]), int(grid[1])
    else:
        cols = int(np.ceil(np.sqrt(k)))
        rows = int(np.ceil(k / cols))
    ph, pw = bag.shape[1], bag.shape[2]
    channels = bag.shape[3] if bag.ndim == 4 else 1
    canvas = np.zeros((rows * ph, cols * pw, channels), dtype=bag.dtype)
    for idx in range(k):
        r, c = divmod(idx, cols)
        canvas[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = bag[idx]
    return canvas


def _read_image_hw(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(str(path)) as im:
        w, h = im.size
    return h, w


def _scaled_hw(h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if h >= ph and w >= pw:
        return h, w
    scale = max(ph / h, pw / w)
    return int(np.ceil(h * scale)), int(np.ceil(w * scale))


def _flip_roi_norm_x_np(xmin: float, xmax: float) -> tuple[float, float]:
    return 1.0 - xmax, 1.0 - xmin


def _path_hash_bucket(path: str) -> int:
    return int(tf.strings.to_hash_bucket_fast(str(path), 2**31 - 1).numpy())


def _crop_strategy_for_label(
    config: TfDatasetConfig,
    label: float,
) -> PatchCropStrategy:
    if config.patch_crop_strategy is not None:
        return config.patch_crop_strategy
    by_label = config.patch_crop_by_label or DEFAULT_PATCH_CROP_BY_LABEL
    key = 1.0 if label >= 0.5 else 0.0
    return by_label.get(key, "uniform")


def _deterministic_crop_offset(
    *,
    config: TfDatasetConfig,
    label: float,
    roi: tuple[float, float, float, float],
    img_h: int,
    img_w: int,
    path: str,
) -> tuple[int, int, int, int]:
    """Offset de crop determinista y dimensiones tras upscale minimo (y0, x0, h, w)."""
    ph, pw = _normalize_size(config.image_size)
    h, w = _scaled_hw(img_h, img_w, ph, pw)
    roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi
    max_y = max(h - ph, 0)
    max_x = max(w - pw, 0)
    strategy = _crop_strategy_for_label(config, label)

    if strategy == "roi":
        cx = (roi_xmin * w + roi_xmax * w) / 2.0
        cy = (roi_ymin * h + roi_ymax * h) / 2.0
        y0 = int(np.round(np.clip(cy - ph / 2.0, 0, max_y)))
        x0 = int(np.round(np.clip(cx - pw / 2.0, 0, max_x)))
    elif strategy == "avoid_roi":
        if roi_xmin >= roi_xmax or roi_ymin >= roi_ymax:
            bucket = _path_hash_bucket(path)
            y0 = bucket % (max_y + 1) if max_y >= 0 else 0
            x0 = (bucket // 7) % (max_x + 1) if max_x >= 0 else 0
        else:
            xmin, ymin, xmax, ymax = (
                roi_xmin * w,
                roi_ymin * h,
                roi_xmax * w,
                roi_ymax * h,
            )
            max_attempts = config.patch_avoid_roi_max_attempts
            y0, x0 = 0, 0
            for attempt in range(max_attempts):
                seed = (_path_hash_bucket(path) + attempt * 9973) % (2**31 - 1)
                cand_y = seed % (max_y + 1) if max_y >= 0 else 0
                cand_x = (seed // 17) % (max_x + 1) if max_x >= 0 else 0
                if not (
                    cand_y + ph > ymin
                    and cand_y < ymax
                    and cand_x + pw > xmin
                    and cand_x < xmax
                ):
                    y0, x0 = cand_y, cand_x
                    break
            else:
                bucket = _path_hash_bucket(path)
                y0 = bucket % (max_y + 1) if max_y >= 0 else 0
                x0 = (bucket // 7) % (max_x + 1) if max_x >= 0 else 0
    elif strategy == "normal":
        y0 = int(np.round(max_y * config.patch_bias_y))
        x0 = int(np.round(max_x * config.patch_bias_x))
    else:
        bucket = _path_hash_bucket(path)
        y0 = bucket % (max_y + 1) if max_y >= 0 else 0
        x0 = (bucket // 7) % (max_x + 1) if max_x >= 0 else 0
    return y0, x0, h, w


def _roi_norms_from_row(
    row: pd.Series,
    config: TfDatasetConfig,
    lateralize_flip_side: Literal["L", "R"],
) -> tuple[float, float, float, float] | None:
    cols = config.patch_roi_norm_columns
    try:
        xmin, ymin, xmax, ymax = (float(row[c]) for c in cols)
    except (KeyError, TypeError, ValueError):
        return None
    if any(np.isnan(v) for v in (xmin, ymin, xmax, ymax)) or xmax <= xmin or ymax <= ymin:
        return None
    flip = (
        config.lateralize
        and str(row.get(config.laterality_column, "")) == lateralize_flip_side
    )
    if flip:
        xmin, xmax = _flip_roi_norm_x_np(xmin, xmax)
    return xmin, ymin, xmax, ymax


def _roi_rect_in_patch_pixels(
    *,
    config: TfDatasetConfig,
    row: pd.Series,
    lateralize_flip_side: Literal["L", "R"],
    crop_y0: float | None = None,
    crop_x0: float | None = None,
    crop_h: float | None = None,
    crop_w: float | None = None,
) -> tuple[float, float, float, float] | None:
    """Rectangulo ROI en pixeles del parche (xmin, ymin, xmax, ymax).

    Si se pasan crop_y0/x0/h/w (del pipeline TF), coinciden con crops aleatorios.
    """
    norms = _roi_norms_from_row(row, config, lateralize_flip_side)
    if norms is None:
        return None
    xmin, ymin, xmax, ymax = norms
    ph, pw = _normalize_size(config.image_size)

    if not config.patch_mode:
        return xmin * pw, ymin * ph, xmax * pw, ymax * ph

    if crop_y0 is not None and crop_x0 is not None and crop_h is not None and crop_w is not None:
        y0, x0, h, w = crop_y0, crop_x0, crop_h, crop_w
    else:
        path = str(row[config.path_column])
        label = float(row[config.label_column])
        img_h, img_w = _read_image_hw(path)
        y0, x0, h, w = _deterministic_crop_offset(
            config=config,
            label=label,
            roi=(xmin, ymin, xmax, ymax),
            img_h=img_h,
            img_w=img_w,
            path=path,
        )
        y0, x0, h, w = float(y0), float(x0), float(h), float(w)

    pxmin = xmin * w - x0
    pymin = ymin * h - y0
    pxmax = xmax * w - x0
    pymax = ymax * h - y0
    pxmin = max(0.0, min(float(pw), pxmin))
    pymin = max(0.0, min(float(ph), pymin))
    pxmax = max(0.0, min(float(pw), pxmax))
    pymax = max(0.0, min(float(ph), pymax))
    if pxmax <= pxmin or pymax <= pymin:
        return None
    return pxmin, pymin, pxmax, pymax


def _plot_roi_overlay(
    ax: Any,
    rect: tuple[float, float, float, float] | None,
    *,
    linewidth: float = 2.0,
    color: str = "red",
) -> None:
    """Dibuja el borde de la ROI original en rojo (solo visualizacion)."""
    if rect is None or not ax.images:
        return
    from matplotlib.patches import Rectangle

    xmin, ymin, xmax, ymax = rect
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
    )


_FINDING_TITLE_COLUMNS: tuple[str, ...] = (
    "Mass",
    "Suspicious_Lymph_Node",
    "Nipple_Retraction",
    "Skin_Retraction",
    "Skin_Thickening",
    "Suspicious_Calcification",
    "Architectural_Distortion",
    "Asymmetry",
    "Focal_Asymmetry",
    "Global_Asymmetry",
    "No_Finding",
)


def _finding_text_from_row(row: pd.Series) -> str:
    if "finding_categories" in row.index:
        raw = row["finding_categories"]
        if pd.notna(raw):
            text = str(raw).strip()
            if text.startswith("["):
                import ast

                try:
                    items = ast.literal_eval(text)
                    if isinstance(items, (list, tuple)) and items:
                        return ", ".join(str(item) for item in items)
                except (ValueError, SyntaxError):
                    pass
            return text.strip("[]'\" ")
    active = [
        name
        for name in _FINDING_TITLE_COLUMNS
        if name in row.index and row[name] == 1
    ]
    if active:
        return ", ".join(active)
    if "cls" in row.index and pd.notna(row["cls"]):
        return "finding" if float(row["cls"]) >= 0.5 else "no finding"
    return ""


def _sample_title_from_row(
    row: pd.Series,
    *,
    path_column: str = "path",
) -> str:
    parts: list[str] = []
    finding = _finding_text_from_row(row)
    if finding:
        parts.append(finding)

    meta: list[str] = []
    if "laterality" in row.index and pd.notna(row["laterality"]):
        meta.append(str(row["laterality"]).strip())
    if "view" in row.index and pd.notna(row["view"]):
        meta.append(str(row["view"]).strip())
    if meta:
        parts.append(" ".join(meta))

    lines: list[str] = []
    if parts:
        lines.append(" | ".join(parts))
    if path_column in row.index and pd.notna(row[path_column]):
        lines.append(Path(str(row[path_column])).name)
    return "\n".join(lines)


def _plot_sample_grid(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    deterministic: bool,
    dpi: float = 100.0,
    sample_titles: list[str] | None = None,
    roi_rects: list[tuple[float, float, float, float] | None] | None = None,
) -> None:
    """Muestra parches a resolucion nativa (1 px imagen = 1 px en pantalla)."""
    import matplotlib.pyplot as plt

    n = len(labels)
    cols = min(n, 4) if n else 1
    rows = int(np.ceil(n / cols)) if n else 1
    if n:
        patch_h, patch_w = images[0].shape[:2]
    else:
        patch_h, patch_w = 1, 1

    title_row_in = 0.62
    fig_w_in = cols * patch_w / dpi
    fig_h_in = rows * (patch_h / dpi + title_row_in)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_w_in, fig_h_in),
        dpi=dpi,
        facecolor="black",
    )
    axes = np.atleast_1d(axes).reshape(-1)

    for idx in range(rows * cols):
        ax = axes[idx]
        ax.set_facecolor("black")
        if idx < n:
            img = images[idx]
            h, w = img.shape[:2]
            ax.imshow(img, interpolation="nearest", aspect="equal", origin="upper")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_box_aspect(h / w)
            if roi_rects is not None:
                _plot_roi_overlay(ax, roi_rects[idx])
            if sample_titles is not None:
                ax.set_title(sample_titles[idx], color="white", fontsize=9)
            else:
                label = int(labels[idx] >= 0.5)
                ax.set_title(f"label={label}", color="white")
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.12, hspace=0.55)
    suffix = " [crop fijo]" if deterministic else " [crop aleatorio]"
    roi_suffix = " + ROI" if roi_rects is not None else ""
    fig.suptitle(f"{title}{roi_suffix}{suffix}", color="white", y=0.995)
    plt.show()


def _collect_batches(
    dataset: tf.data.Dataset,
    *,
    max_batches: int | None,
    with_crop_meta: bool = False,
) -> (
    tuple[list[np.ndarray], list[np.ndarray]]
    | tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    images_batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    y0_batches: list[np.ndarray] = []
    x0_batches: list[np.ndarray] = []
    h_batches: list[np.ndarray] = []
    w_batches: list[np.ndarray] = []
    for batch_idx, batch in enumerate(dataset):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if with_crop_meta:
            images, labels, y0, x0, crop_h, crop_w = batch
            y0_batches.append(np.asarray(y0).reshape(-1))
            x0_batches.append(np.asarray(x0).reshape(-1))
            h_batches.append(np.asarray(crop_h).reshape(-1))
            w_batches.append(np.asarray(crop_w).reshape(-1))
        else:
            images, labels = batch
        images_batches.append(np.asarray(images))
        labels_batches.append(np.asarray(labels).reshape(-1))
    if with_crop_meta:
        return (
            images_batches,
            labels_batches,
            np.concatenate(y0_batches),
            np.concatenate(x0_batches),
            np.concatenate(h_batches),
            np.concatenate(w_batches),
        )
    return images_batches, labels_batches


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    y = labels.reshape(-1)
    pos = int((y >= 0.5).sum())
    return {"positivos": pos, "negativos": int(len(y) - pos), "total": int(len(y))}


def _image_value_stats(images: np.ndarray) -> dict[str, float]:
    x = np.asarray(images, dtype=np.float32)
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def _normalize_size(size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(size, int):
        return (int(size), int(size))
    h, w = size
    return (int(h), int(w))


def _resize_preserving_dtype(img: tf.Tensor, size: tf.Tensor | list[int]) -> tf.Tensor:
    """tf.image.resize promueve a float32; restauramos el dtype de entrada."""
    return tf.cast(tf.image.resize(img, size), img.dtype)


def _ensure_min_spatial(img: tf.Tensor, min_h: int, min_w: int) -> tf.Tensor:
    """Escala la imagen solo si es mas chica que el parche objetivo."""
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    needs_upscale = tf.logical_or(h < min_h, w < min_w)

    def _upscale() -> tf.Tensor:
        scale = tf.maximum(
            tf.cast(min_h, tf.float32) / tf.cast(h, tf.float32),
            tf.cast(min_w, tf.float32) / tf.cast(w, tf.float32),
        )
        new_h = tf.cast(tf.math.ceil(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.math.ceil(tf.cast(w, tf.float32) * scale), tf.int32)
        return _resize_preserving_dtype(img, [new_h, new_w])

    return tf.cond(needs_upscale, _upscale, lambda: img)


def _random_patch_offset(
    max_offset: tf.Tensor,
    *,
    sampling: PatchSampling,
    mean_frac: float,
    sigma_frac: float,
) -> tf.Tensor:
    """Offset de crop en [0, max_offset]; normal truncada sesgada o uniforme."""
    zero = tf.constant(0, dtype=tf.int32)

    def _sample() -> tf.Tensor:
        max_f = tf.cast(max_offset, tf.float32)
        if sampling == "uniform":
            offset = tf.random.uniform(()) * max_f
        else:
            mean = tf.cast(mean_frac, tf.float32) * max_f
            std = tf.maximum(tf.cast(sigma_frac, tf.float32) * max_f, 1.0)
            offset = tf.random.normal(()) * std + mean
            offset = tf.clip_by_value(offset, 0.0, max_f)
        return tf.cast(tf.round(offset), tf.int32)

    return tf.cond(max_offset > 0, _sample, lambda: zero)


def _deterministic_patch_offset(max_offset: tf.Tensor, mean_frac: float) -> tf.Tensor:
    return tf.cast(
        tf.round(tf.cast(max_offset, tf.float32) * tf.cast(mean_frac, tf.float32)),
        tf.int32,
    )


def _uniform_patch_offsets(max_y: tf.Tensor, max_x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    y0 = _random_patch_offset(
        max_y, sampling="uniform", mean_frac=0.5, sigma_frac=0.0
    )
    x0 = _random_patch_offset(
        max_x, sampling="uniform", mean_frac=0.5, sigma_frac=0.0
    )
    return y0, x0


def _roi_is_valid(
    roi_xmin: tf.Tensor,
    roi_ymin: tf.Tensor,
    roi_xmax: tf.Tensor,
    roi_ymax: tf.Tensor,
) -> tf.Tensor:
    return tf.logical_and(
        tf.less(roi_xmin, roi_xmax),
        tf.logical_and(
            tf.less(roi_ymin, roi_ymax),
            tf.math.is_finite(roi_xmin + roi_ymin + roi_xmax + roi_ymax),
        ),
    )


def _roi_box_pixels(
    roi_xmin: tf.Tensor,
    roi_ymin: tf.Tensor,
    roi_xmax: tf.Tensor,
    roi_ymax: tf.Tensor,
    h: tf.Tensor,
    w: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    h0 = tf.cast(h, tf.float32)
    w0 = tf.cast(w, tf.float32)
    return (
        roi_xmin * w0,
        roi_ymin * h0,
        roi_xmax * w0,
        roi_ymax * h0,
    )


def _patch_intersects_roi(
    y0: tf.Tensor,
    x0: tf.Tensor,
    ph: int,
    pw: int,
    roi_xmin_px: tf.Tensor,
    roi_ymin_px: tf.Tensor,
    roi_xmax_px: tf.Tensor,
    roi_ymax_px: tf.Tensor,
) -> tf.Tensor:
    y0f = tf.cast(y0, tf.float32)
    x0f = tf.cast(x0, tf.float32)
    phf = tf.cast(ph, tf.float32)
    pwf = tf.cast(pw, tf.float32)
    separated = tf.logical_or(
        tf.logical_or(y0f + phf <= roi_ymin_px, y0f >= roi_ymax_px),
        tf.logical_or(x0f + pwf <= roi_xmin_px, x0f >= roi_xmax_px),
    )
    return tf.logical_not(separated)


def _offsets_avoiding_roi(
    max_y: tf.Tensor,
    max_x: tf.Tensor,
    ph: int,
    pw: int,
    roi_xmin: tf.Tensor,
    roi_ymin: tf.Tensor,
    roi_xmax: tf.Tensor,
    roi_ymax: tf.Tensor,
    h: tf.Tensor,
    w: tf.Tensor,
    *,
    random_patch: bool,
    max_attempts: int,
    path: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Muestrea offsets de parche que no intersectan la ROI (rejection sampling)."""
    max_y_i = tf.maximum(max_y, 0)
    max_x_i = tf.maximum(max_x, 0)
    n = max_attempts

    if random_patch:
        # Enteros en [0, max_*_i]; evitar uniform+round con maxval=max+1 (puede dar max+1).
        y0_cand_i = tf.random.uniform([n], maxval=max_y_i + 1, dtype=tf.int32)
        x0_cand_i = tf.random.uniform([n], maxval=max_x_i + 1, dtype=tf.int32)
    else:
        bucket = tf.cast(tf.strings.to_hash_bucket_fast(path, 2**31 - 1), tf.int32)
        idx = tf.cast(tf.range(n), tf.int32)
        seeds = bucket + idx * 9973
        y0_cand_i = seeds % tf.maximum(max_y_i + 1, 1)
        x0_cand_i = (seeds // 17) % tf.maximum(max_x_i + 1, 1)
    xmin, ymin, xmax, ymax = _roi_box_pixels(roi_xmin, roi_ymin, roi_xmax, roi_ymax, h, w)

    y0f = tf.cast(y0_cand_i, tf.float32)
    x0f = tf.cast(x0_cand_i, tf.float32)
    phf = tf.cast(ph, tf.float32)
    pwf = tf.cast(pw, tf.float32)
    intersects = tf.logical_not(
        tf.logical_or(
            tf.logical_or(y0f + phf <= ymin, y0f >= ymax),
            tf.logical_or(x0f + pwf <= xmin, x0f >= xmax),
        )
    )
    valid = tf.logical_not(intersects)
    has_valid = tf.reduce_any(valid)
    first_idx = tf.argmax(tf.cast(valid, tf.int32))
    y0_pick = y0_cand_i[first_idx]
    x0_pick = x0_cand_i[first_idx]
    y0_fb, x0_fb = _scan_disjoint_offset(
        max_y, max_x, ph, pw, xmin, ymin, xmax, ymax, stride=8
    )
    return (
        tf.cond(has_valid, lambda: y0_pick, lambda: y0_fb),
        tf.cond(has_valid, lambda: x0_pick, lambda: x0_fb),
    )


def _scan_disjoint_offset(
    max_y: tf.Tensor,
    max_x: tf.Tensor,
    ph: int,
    pw: int,
    roi_xmin_px: tf.Tensor,
    roi_ymin_px: tf.Tensor,
    roi_xmax_px: tf.Tensor,
    roi_ymax_px: tf.Tensor,
    *,
    stride: int = 8,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Busqueda en grilla (fallback) de un offset sin interseccion con la ROI."""
    max_y_i = tf.maximum(max_y, 0)
    max_x_i = tf.maximum(max_x, 0)
    y0s = tf.range(0, max_y_i + 1, stride)
    x0s = tf.range(0, max_x_i + 1, stride)
    y0_grid, x0_grid = tf.meshgrid(y0s, x0s, indexing="ij")
    y0_flat = tf.reshape(y0_grid, [-1])
    x0_flat = tf.reshape(x0_grid, [-1])
    y0f = tf.cast(y0_flat, tf.float32)
    x0f = tf.cast(x0_flat, tf.float32)
    phf = tf.cast(ph, tf.float32)
    pwf = tf.cast(pw, tf.float32)
    disjoint = tf.logical_or(
        tf.logical_or(y0f + phf <= roi_ymin_px, y0f >= roi_ymax_px),
        tf.logical_or(x0f + pwf <= roi_xmin_px, x0f >= roi_xmax_px),
    )
    valid = tf.reshape(disjoint, [-1])
    has_valid = tf.reduce_any(valid)
    first_idx = tf.argmax(tf.cast(valid, tf.int32))
    y0_pick = y0_flat[first_idx]
    x0_pick = x0_flat[first_idx]
    zero = tf.constant(0, dtype=tf.int32)
    return (
        tf.cond(has_valid, lambda: y0_pick, lambda: zero),
        tf.cond(has_valid, lambda: x0_pick, lambda: zero),
    )


def _flip_roi_norm_x(
    xmin: tf.Tensor,
    xmax: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    return 1.0 - xmax, 1.0 - xmin


def _offset_from_center(
    center: tf.Tensor,
    patch_size: int,
    max_offset: tf.Tensor,
    *,
    random_jitter: bool,
    sigma_frac: float,
) -> tf.Tensor:
    max_f = tf.cast(max_offset, tf.float32)
    ideal = tf.clip_by_value(
        center - tf.cast(patch_size, tf.float32) / 2.0,
        0.0,
        max_f,
    )
    if random_jitter:
        std = tf.maximum(tf.cast(sigma_frac, tf.float32) * max_f, 1.0)
        offset = tf.random.normal([]) * std + ideal
        return tf.cast(tf.round(tf.clip_by_value(offset, 0.0, max_f)), tf.int32)
    return tf.cast(tf.round(ideal), tf.int32)


def _labels_positive_mask(labels: tf.Tensor) -> tf.Tensor:
    """Mascara de positivos: binaria (>=0.5) o multilabel (cualquier canal >=0.5).

    Asume primer eje = batch (como en ``dataset.batch``).
    """
    labels = tf.cast(labels, tf.float32)
    batch_size = tf.shape(labels)[0]
    flat = tf.reshape(labels, [batch_size, -1])
    return tf.reduce_any(flat >= 0.5, axis=1)


def _expand_mixup_lambda(lam: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """Expande lambda [B] al rank de `target` (imagen o etiqueta)."""
    lam = tf.reshape(lam, [-1])
    target_rank = tf.rank(target)
    broadcast_shape = tf.concat(
        [[tf.shape(lam)[0]], tf.ones([target_rank - 1], dtype=tf.int32)],
        axis=0,
    )
    return tf.reshape(lam, broadcast_shape)


def positive_only_mixup_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    *,
    alpha: float = 0.1,
    probability: float = 0.5,
) -> tuple[tf.Tensor, tf.Tensor]:
    """MixUp solo entre muestras positivas del batch (Beta(alpha, alpha), prob p).

    Pensado para ejecutarse en ``tf.data`` despues del ``map`` con flip/crop
    y antes de augmentaciones fotometricas/geometricas del modelo.
    Sin ``tf.cond`` (compatible con Autograph en ``tf.data.map``).
    """
    images_f32 = tf.cast(images, tf.float32)
    labels_f32 = tf.cast(labels, tf.float32)
    batch_size = tf.shape(images_f32)[0]
    pos_mask = _labels_positive_mask(labels_f32)
    n_pos = tf.reduce_sum(tf.cast(pos_mask, tf.int32))
    mix_gate = tf.cast(
        tf.logical_and(
            tf.less(tf.random.uniform([], dtype=tf.float32), probability),
            tf.greater_equal(n_pos, 2),
        ),
        tf.float32,
    )

    pos_indices = tf.boolean_mask(tf.range(batch_size), pos_mask)
    n = tf.shape(pos_indices)[0]
    order = tf.random.shuffle(tf.range(n))
    partner_order = tf.math.floormod(order + 1, tf.maximum(n, 1))
    partners = tf.gather(pos_indices, partner_order)
    partner_at = tf.tensor_scatter_nd_update(
        tf.range(batch_size),
        tf.expand_dims(pos_indices, axis=1),
        partners,
    )

    lam_a = tf.random.gamma([batch_size], alpha, beta=1.0, dtype=tf.float32)
    lam_b = tf.random.gamma([batch_size], alpha, beta=1.0, dtype=tf.float32)
    lam = lam_a / (lam_a + lam_b + 1e-8)

    partner_images = tf.gather(images_f32, partner_at)
    partner_labels = tf.gather(labels_f32, partner_at)

    lam_img = _expand_mixup_lambda(lam, images_f32)
    lam_lbl = _expand_mixup_lambda(lam, labels_f32)
    mixed_images = lam_img * images_f32 + (1.0 - lam_img) * partner_images
    mixed_labels = lam_lbl * labels_f32 + (1.0 - lam_lbl) * partner_labels

    pos_mask_f = tf.cast(pos_mask, tf.float32)
    pos_mask_img = _expand_mixup_lambda(pos_mask_f, images_f32)
    pos_mask_lbl = _expand_mixup_lambda(pos_mask_f, labels_f32)
    mixed_images = pos_mask_img * mixed_images + (1.0 - pos_mask_img) * images_f32
    mixed_labels = pos_mask_lbl * mixed_labels + (1.0 - pos_mask_lbl) * labels_f32

    out_images = (1.0 - mix_gate) * images_f32 + mix_gate * mixed_images
    out_labels = (1.0 - mix_gate) * labels_f32 + mix_gate * mixed_labels
    return out_images, out_labels


def _dataset_deterministic_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Orden estable para eval, TTA y alineacion con tablas fuente."""
    opts = tf.data.Options()
    if hasattr(opts, "deterministic"):
        opts.deterministic = True
    elif hasattr(opts, "experimental_deterministic"):
        opts.experimental_deterministic = True
    return ds.with_options(opts)


def _dataset_perf_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    opts = tf.data.Options()
    if hasattr(opts, "deterministic"):
        opts.deterministic = False
    elif hasattr(opts, "experimental_deterministic"):
        opts.experimental_deterministic = False
    return ds.with_options(opts)


@dataclass
class TfDatasetConfig:
    image_size: tuple[int, int]
    batch_size: int
    seed: int = 42
    path_column: str = "path"
    label_column: str = "cls"
    cache_dataset: bool = True
    cache_filename: str | Path | None = None
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    lateralize: bool = False
    laterality_column: str = "laterality"
    patch_mode: bool = False
    patch_sampling: PatchSampling = "uniform"
    patch_bias_x: float = 0.75
    patch_bias_y: float = 0.5
    patch_bias_sigma: float = 0.2
    patch_crop_strategy: PatchCropStrategy | None = None
    patch_crop_by_label: dict[float, PatchCropStrategy] | None = None
    patch_roi_norm_columns: tuple[str, str, str, str] = (
        "pad_resized_xmin_norm",
        "pad_resized_ymin_norm",
        "pad_resized_xmax_norm",
        "pad_resized_ymax_norm",
    )
    patch_roi_sigma_frac: float = 0.1
    patch_avoid_roi_max_attempts: int = 64
    return_crop_offset: bool = False
    positive_mixup: bool = False
    positive_mixup_alpha: float = 0.1
    positive_mixup_probability: float = 0.5
    # Modo MIL (ABMIL): cada imagen es un "bag" troceado en parches.
    mode: str = "simple"
    bag_grid: tuple[int, int] = (3, 3)
    bag_keras_tiling: bool = False

    def needs_roi_columns(self) -> bool:
        if is_mil_mode(self.mode):
            return False
        if not self.patch_mode:
            return False
        if self.patch_crop_strategy in ("roi", "avoid_roi"):
            return True
        by_label = self.patch_crop_by_label or DEFAULT_PATCH_CROP_BY_LABEL
        return by_label.get(1.0) == "roi" or by_label.get(0.0) == "avoid_roi"


@dataclass
class DatasetProviderConfig:
    """Configuracion de alto nivel; el notebook arma variantes con `with_overrides`."""

    image_size: tuple[int, int] | int
    batch_size: int
    seed: int = 42
    path_column: str = "path"
    label_column: str = "cls"
    cache_dataset: bool = True
    cache_filename: str | Path | None = None
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    lateralize: bool = False
    laterality_column: str = "laterality"
    lateralize_flip_side: Literal["L", "R"] = "R"
    patch_mode: bool = False
    patch_sampling: PatchSampling = "uniform"
    patch_bias_x: float = 0.75
    patch_bias_y: float = 0.5
    patch_bias_sigma: float = 0.2
    patch_crop_strategy: PatchCropStrategy | None = None
    patch_crop_by_label: dict[float, PatchCropStrategy] | None = None
    patch_roi_norm_columns: tuple[str, str, str, str] = (
        "pad_resized_xmin_norm",
        "pad_resized_ymin_norm",
        "pad_resized_xmax_norm",
        "pad_resized_ymax_norm",
    )
    patch_roi_sigma_frac: float = 0.1
    patch_avoid_roi_max_attempts: int = 64
    return_crop_offset: bool = False
    positive_mixup: bool = False
    positive_mixup_alpha: float = 0.1
    positive_mixup_probability: float = 0.5
    mode: str = "simple"
    bag_grid: tuple[int, int] = (3, 3)
    bag_keras_tiling: bool = False

    def __post_init__(self) -> None:
        self.image_size = _normalize_size(self.image_size)
        self.bag_grid = (int(self.bag_grid[0]), int(self.bag_grid[1]))
        self.mode = normalize_mode(self.mode)

    def with_overrides(self, **kwargs: Any) -> DatasetProviderConfig:
        return replace(self, **kwargs)

    def to_tf_config(self) -> TfDatasetConfig:
        return TfDatasetConfig(
            image_size=self.image_size,
            batch_size=self.batch_size,
            seed=self.seed,
            path_column=self.path_column,
            label_column=self.label_column,
            cache_dataset=self.cache_dataset,
            cache_filename=self.cache_filename,
            use_clahe=self.use_clahe,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid=self.clahe_tile_grid,
            lateralize=self.lateralize,
            laterality_column=self.laterality_column,
            patch_mode=self.patch_mode,
            patch_sampling=self.patch_sampling,
            patch_bias_x=self.patch_bias_x,
            patch_bias_y=self.patch_bias_y,
            patch_bias_sigma=self.patch_bias_sigma,
            patch_crop_strategy=self.patch_crop_strategy,
            patch_crop_by_label=self.patch_crop_by_label,
            patch_roi_norm_columns=self.patch_roi_norm_columns,
            patch_roi_sigma_frac=self.patch_roi_sigma_frac,
            patch_avoid_roi_max_attempts=self.patch_avoid_roi_max_attempts,
            return_crop_offset=self.return_crop_offset,
            positive_mixup=self.positive_mixup,
            positive_mixup_alpha=self.positive_mixup_alpha,
            positive_mixup_probability=self.positive_mixup_probability,
            mode=self.mode,
            bag_grid=self.bag_grid,
            bag_keras_tiling=self.bag_keras_tiling,
        )


def _preview_from_dataframe(
    df: pd.DataFrame,
    config: DatasetProviderConfig,
    *,
    deterministic: bool = True,
    show_roi: bool = False,
    dpi: float = 100.0,
    title: str | None = None,
    **config_overrides: Any,
) -> None:
    if config_overrides:
        config = config.with_overrides(**config_overrides)

    preview_deterministic = deterministic
    preview_config = config.with_overrides(
        cache_dataset=False,
        cache_filename=None,
    )
    if show_roi:
        preview_config = preview_config.with_overrides(return_crop_offset=True)
    table = df.reset_index(drop=True)
    if table.empty:
        raise ValueError("No hay filas para previsualizar.")
    preview = build_dataset_provider(config=preview_config).build(
        table,
        shuffle=False,
        random_patch=not preview_deterministic,
        name="preview",
    )
    preview.show_samples(
        deterministic=preview_deterministic,
        show_roi=show_roi,
        dpi=dpi,
        title=title or f"preview ({len(table)} filas)",
    )


def _dataframe_show_samples(
    self: pd.DataFrame,
    config: DatasetProviderConfig,
    *,
    deterministic: bool = True,
    show_roi: bool = False,
    dpi: float = 100.0,
    title: str | None = None,
    **config_overrides: Any,
) -> None:
    _preview_from_dataframe(
        self,
        config,
        deterministic=deterministic,
        show_roi=show_roi,
        dpi=dpi,
        title=title,
        **config_overrides,
    )


def _register_dataframe_show_samples() -> None:
    if getattr(pd.DataFrame, "show_samples", None) is _dataframe_show_samples:
        return
    pd.DataFrame.show_samples = _dataframe_show_samples  # type: ignore[attr-defined]


def split_pos_neg(
    df: pd.DataFrame,
    *,
    label_column: str = "cls",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = df[label_column].astype(float)
    pos = df[labels >= 0.5].copy()
    neg = df[labels < 0.5].copy()
    return pos, neg


def hard_negatives_from_positives(
    pos_df: pd.DataFrame,
    *,
    label_column: str = "cls",
) -> pd.DataFrame:
    """Mismas imagenes con hallazgo, etiqueta 0 y parches aleatorios (hard negatives)."""
    hard = pos_df.copy()
    hard[label_column] = 0.0
    return hard


def merge_inspect_datasets(
    *parts: InspectDataset,
    weights: list[float] | None = None,
    name: str = "merged",
) -> InspectDataset:
    if not parts:
        raise ValueError("merge_inspect_datasets requiere al menos un dataset")
    merged = tf.data.Dataset.sample_from_datasets(
        [part.dataset for part in parts],
        weights=weights,
    )
    row_count = sum(part.row_count or 0 for part in parts)
    return InspectDataset(
        merged,
        name=name,
        config=parts[0].config,
        row_count=row_count or None,
    )


class InspectDataset:
    """Proxy sobre `tf.data.Dataset` con utilidades de inspeccion para notebooks."""

    def __init__(
        self,
        dataset: tf.data.Dataset,
        *,
        name: str,
        config: TfDatasetConfig,
        row_count: int | None = None,
        ordered_dataset: tf.data.Dataset | None = None,
        source_table: pd.DataFrame | None = None,
        lateralize_flip_side: Literal["L", "R"] = "R",
    ):
        self._dataset = dataset
        self._ordered_dataset = ordered_dataset or dataset
        self.name = name
        self.config = config
        self.row_count = row_count
        self.source_table = source_table
        self.lateralize_flip_side = lateralize_flip_side

    @property
    def dataset(self) -> tf.data.Dataset:
        return self._dataset

    def ordered(self) -> tf.data.Dataset:
        """Misma pipeline sin shuffle; siempre el mismo orden de muestras."""
        return self._ordered_dataset

    def unwrap(self) -> tf.data.Dataset:
        return self._dataset

    def __iter__(self) -> Iterator:
        return iter(self._dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def __repr__(self) -> str:
        rows = f", rows={self.row_count}" if self.row_count is not None else ""
        return f"InspectDataset(name={self.name!r}{rows})"

    def sample_batches(
        self,
        n_batches: int = 1,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        source = self._ordered_dataset if deterministic else self._dataset
        images_batches, labels_batches = _collect_batches(
            source,
            max_batches=n_batches,
        )
        if not images_batches:
            raise ValueError(f"El dataset {self.name!r} no produjo batches.")
        return np.concatenate(images_batches, axis=0), np.concatenate(labels_batches, axis=0)

    def print_stats(self, *, max_batches: int | None = 10) -> dict[str, Any]:
        images_batches, labels_batches = _collect_batches(
            self._dataset,
            max_batches=max_batches,
        )
        if not images_batches:
            print(f"[{self.name}] sin batches")
            return {}

        images = np.concatenate(images_batches, axis=0)
        labels = np.concatenate(labels_batches, axis=0)
        counts = _label_counts(labels)
        img_stats = _image_value_stats(images)
        batch_size = int(images.shape[0] // max(len(images_batches), 1))

        stats: dict[str, Any] = {
            "name": self.name,
            "row_count": self.row_count,
            "batches_seen": len(images_batches),
            "samples_seen": int(len(labels)),
            "batch_size_config": self.config.batch_size,
            "batch_size_observed": batch_size,
            "image_shape": tuple(int(d) for d in images.shape[1:]),
            "image_dtype": str(images.dtype),
            "labels": counts,
            "positivo_pct": 100.0 * counts["positivos"] / counts["total"],
            "image_values": img_stats,
            "use_clahe": self.config.use_clahe,
            "lateralize": self.config.lateralize,
            "patch_mode": self.config.patch_mode,
            "bag_mode": is_mil_mode(self.config.mode),
            "mode": self.config.mode,
            "bag_grid": self.config.bag_grid if is_mil_mode(self.config.mode) else None,
            "image_size": self.config.image_size,
        }

        print(f"--- {self.name} ---")
        if self.row_count is not None:
            print(f"  filas en tabla: {self.row_count}")
        print(f"  batches leidos: {stats['batches_seen']} ({stats['samples_seen']} muestras)")
        print(f"  batch_size (config): {stats['batch_size_config']}")
        print(f"  shape imagen: {stats['image_shape']}  dtype: {stats['image_dtype']}")
        print(
            f"  etiquetas: {counts['positivos']} pos / {counts['negativos']} neg "
            f"({stats['positivo_pct']:.1f}% positivos)"
        )
        print(
            f"  pixeles: min={img_stats['min']:.2f} max={img_stats['max']:.2f} "
            f"mean={img_stats['mean']:.2f} std={img_stats['std']:.2f}"
        )
        if is_mil_mode(self.config.mode):
            rows, cols = self.config.bag_grid
            if self.config.bag_keras_tiling:
                print(
                    f"  modo bag (MIL, tiling Keras): imagen completa {rows * self.config.image_size[0]}"
                    f"x{cols * self.config.image_size[1]} -> BagTiling en modelo -> "
                    f"{rows * cols} tiles de {self.config.image_size}"
                )
            else:
                print(
                    f"  modo bag (MIL): grilla {rows}x{cols} = {rows * cols} instancias "
                    f"de {self.config.image_size} por imagen"
                )
        if self.config.patch_mode:
            if self.config.patch_crop_strategy is not None:
                crop_desc = f"crop={self.config.patch_crop_strategy}"
            else:
                by_label = self.config.patch_crop_by_label or DEFAULT_PATCH_CROP_BY_LABEL
                crop_desc = f"crop_by_label={by_label}"
            print(f"  modo parche: {self.config.image_size} ({crop_desc})")
        if max_batches is not None:
            print(f"  (estadisticas sobre los primeros {max_batches} batches)")
        return stats

    def show_samples(
        self,
        *,
        deterministic: bool = False,
        show_roi: bool = False,
        dpi: float = 100.0,
        title: str | None = None,
    ) -> None:
        if show_roi and self.source_table is None:
            raise ValueError(
                f"show_roi en {self.name!r} requiere tabla fuente; "
                "construye el dataset con build() desde un DataFrame."
            )
        if not deterministic:
            _roll_tf_random_seed()

        use_crop_meta = show_roi and self.config.return_crop_offset
        if show_roi:
            table_for_roi = (
                self.source_table.reset_index(drop=True)
                if self.source_table is not None
                else None
            )
            if table_for_roi is None:
                raise ValueError(
                    f"show_roi en {self.name!r} requiere tabla fuente alineada con iloc."
                )
            inspect_config = replace(
                self.config,
                return_crop_offset=True,
                cache_dataset=False,
                cache_filename=None,
            )
            inspect = DatasetProvider(
                inspect_config,
                lateralize_flip_side=self.lateralize_flip_side,
            ).build(
                table_for_roi,
                shuffle=False,
                random_patch=not deterministic,
                name=f"{self.name}_inspect",
            )
            source = inspect._ordered_dataset
            use_crop_meta = True
        else:
            source = self._ordered_dataset if deterministic else self._dataset

        collected = _collect_batches(
            source,
            max_batches=None,
            with_crop_meta=use_crop_meta,
        )
        if use_crop_meta:
            images_batches, labels_batches, crop_y0, crop_x0, crop_h, crop_w = collected
        else:
            images_batches, labels_batches = collected
            crop_y0 = crop_x0 = crop_h = crop_w = None

        if not images_batches:
            raise ValueError(f"El dataset {self.name!r} no produjo batches.")
        images = _images_to_display(np.concatenate(images_batches, axis=0))
        if images.ndim == 5:
            grid = self.config.bag_grid if is_mil_mode(self.config.mode) else None
            images = np.stack([_montage_bag(bag, grid) for bag in images])
        labels = np.concatenate(labels_batches, axis=0).reshape(-1)
        n = len(labels)

        sample_titles: list[str] | None = None
        roi_rects: list[tuple[float, float, float, float] | None] | None = None
        if self.source_table is not None:
            table = self.source_table.reset_index(drop=True)
            if len(table) < n:
                raise ValueError(
                    f"La tabla fuente tiene {len(table)} filas pero el dataset devolvio {n} muestras."
                )
            sample_titles = [
                _sample_title_from_row(
                    table.iloc[i],
                    path_column=self.config.path_column,
                )
                for i in range(n)
            ]
            if show_roi:
                roi_rects = [
                    _roi_rect_in_patch_pixels(
                        config=self.config,
                        row=table.iloc[i],
                        lateralize_flip_side=self.lateralize_flip_side,
                        crop_y0=None if crop_y0 is None else float(crop_y0[i]),
                        crop_x0=None if crop_x0 is None else float(crop_x0[i]),
                        crop_h=None if crop_h is None else float(crop_h[i]),
                        crop_w=None if crop_w is None else float(crop_w[i]),
                    )
                    for i in range(n)
                ]

        _plot_sample_grid(
            images,
            labels,
            title=title or f"{self.name} (n={n})",
            deterministic=deterministic,
            dpi=dpi,
            sample_titles=sample_titles,
            roi_rects=roi_rects,
        )


def as_tf_dataset(dataset: tf.data.Dataset | InspectDataset) -> tf.data.Dataset:
    """Keras solo acepta `tf.data.Dataset`; desenvuelve `InspectDataset` si hace falta."""
    if isinstance(dataset, InspectDataset):
        return dataset.unwrap()
    return dataset


@dataclass
class DatasetSplits:
    train: InspectDataset
    val: InspectDataset
    test: InspectDataset

    def __iter__(self) -> Iterator[InspectDataset]:
        yield self.train
        yield self.val
        yield self.test

    def __getitem__(self, name: SplitName) -> InspectDataset:
        return getattr(self, name)

    def items(self) -> tuple[tuple[str, InspectDataset], ...]:
        return (("train", self.train), ("val", self.val), ("test", self.test))

    def unwrap(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self.train.unwrap(), self.val.unwrap(), self.test.unwrap()

    def print_stats(
        self,
        *,
        max_batches: int | None = 10,
        splits: tuple[SplitName, ...] = ("train", "val", "test"),
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name in splits:
            out[name] = self[name].print_stats(max_batches=max_batches)
        return out

    def show_samples(
        self,
        *,
        deterministic: bool = False,
        show_roi: bool = False,
        dpi: float = 100.0,
        splits: tuple[SplitName, ...] = ("train", "val", "test"),
    ) -> None:
        for name in splits:
            self[name].show_samples(
                deterministic=deterministic,
                show_roi=show_roi,
                dpi=dpi,
            )


class DatasetProvider:
    """Construye `tf.data.Dataset` a partir de tablas pandas con paths y etiquetas."""

    def __init__(
        self,
        config: TfDatasetConfig,
        *,
        lateralize_flip_side: Literal["L", "R"] = "R",
    ):
        self.config = config
        self.lateralize_flip_side = lateralize_flip_side
        self._height, self._width = _normalize_size(config.image_size)

    def _uniform_patch_offsets_branch(
        self,
        img: tf.Tensor,
        path: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if random_patch:
            return _uniform_patch_offsets(max_y, max_x)
        return self._extract_patch_legacy(
            img, path, random_patch=False, max_y=max_y, max_x=max_x
        )

    def _extract_patch_legacy(
        self,
        img: tf.Tensor,
        path: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if random_patch:
            y0 = _random_patch_offset(
                max_y,
                sampling=self.config.patch_sampling,
                mean_frac=self.config.patch_bias_y,
                sigma_frac=self.config.patch_bias_sigma,
            )
            x0 = _random_patch_offset(
                max_x,
                sampling=self.config.patch_sampling,
                mean_frac=self.config.patch_bias_x,
                sigma_frac=self.config.patch_bias_sigma,
            )
            return y0, x0
        if self.config.patch_sampling == "normal":
            return (
                _deterministic_patch_offset(max_y, self.config.patch_bias_y),
                _deterministic_patch_offset(max_x, self.config.patch_bias_x),
            )
        bucket = tf.cast(tf.strings.to_hash_bucket_fast(path, 2**31 - 1), tf.int32)
        return bucket % (max_y + 1), (bucket // 7) % (max_x + 1)

    def _offsets_for_label_strategy(
        self,
        strategy: PatchCropStrategy,
        img: tf.Tensor,
        path: tf.Tensor,
        label: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
        ph: int,
        pw: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if strategy == "roi":
            return self._extract_patch_roi(
                img,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
                max_y=max_y,
                max_x=max_x,
                ph=ph,
                pw=pw,
            )
        if strategy == "avoid_roi":
            return self._extract_patch_avoid_roi(
                img,
                path,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
                max_y=max_y,
                max_x=max_x,
                ph=ph,
                pw=pw,
            )
        if strategy == "normal":
            return self._extract_patch_legacy(
                img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
            )
        return self._uniform_patch_offsets_branch(
            img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
        )

    def _extract_patch_avoid_roi(
        self,
        img: tf.Tensor,
        path: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
        ph: int,
        pw: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        valid = _roi_is_valid(roi_xmin, roi_ymin, roi_xmax, roi_ymax)

        def _sample_avoiding() -> tuple[tf.Tensor, tf.Tensor]:
            return _offsets_avoiding_roi(
                max_y,
                max_x,
                ph,
                pw,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                h,
                w,
                random_patch=random_patch,
                max_attempts=self.config.patch_avoid_roi_max_attempts,
                path=path,
            )

        return tf.cond(
            valid,
            _sample_avoiding,
            lambda: self._uniform_patch_offsets_branch(
                img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
            ),
        )

    def _extract_patch_roi(
        self,
        img: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
        ph: int,
        pw: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        h0 = tf.cast(tf.shape(img)[0], tf.float32)
        w0 = tf.cast(tf.shape(img)[1], tf.float32)
        xmin = roi_xmin * w0
        ymin = roi_ymin * h0
        xmax = roi_xmax * w0
        ymax = roi_ymax * h0
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        return (
            _offset_from_center(
                cy,
                ph,
                max_y,
                random_jitter=random_patch,
                sigma_frac=self.config.patch_roi_sigma_frac,
            ),
            _offset_from_center(
                cx,
                pw,
                max_x,
                random_jitter=random_patch,
                sigma_frac=self.config.patch_roi_sigma_frac,
            ),
        )

    def _extract_patch(
        self,
        img: tf.Tensor,
        path: tf.Tensor,
        label: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        *,
        random_patch: bool,
    ) -> tf.Tensor:
        ph, pw = self._height, self._width
        img = _ensure_min_spatial(img, ph, pw)
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        max_y = tf.maximum(h - ph, 0)
        max_x = tf.maximum(w - pw, 0)
        strategy = self.config.patch_crop_strategy

        if strategy == "roi":
            y0, x0 = self._extract_patch_roi(
                img,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
                max_y=max_y,
                max_x=max_x,
                ph=ph,
                pw=pw,
            )
        elif strategy == "uniform":
            y0, x0 = self._uniform_patch_offsets_branch(
                img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
            )
        elif strategy == "avoid_roi":
            y0, x0 = self._extract_patch_avoid_roi(
                img,
                path,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
                max_y=max_y,
                max_x=max_x,
                ph=ph,
                pw=pw,
            )
        elif strategy == "normal":
            y0, x0 = self._extract_patch_legacy(
                img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
            )
        else:
            by_label = self.config.patch_crop_by_label or DEFAULT_PATCH_CROP_BY_LABEL
            is_positive = label >= 0.5

            def _positive_offsets() -> tuple[tf.Tensor, tf.Tensor]:
                return self._offsets_for_label_strategy(
                    by_label[1.0],
                    img,
                    path,
                    label,
                    roi_xmin,
                    roi_ymin,
                    roi_xmax,
                    roi_ymax,
                    random_patch=random_patch,
                    max_y=max_y,
                    max_x=max_x,
                    ph=ph,
                    pw=pw,
                )

            def _negative_offsets() -> tuple[tf.Tensor, tf.Tensor]:
                return self._offsets_for_label_strategy(
                    by_label[0.0],
                    img,
                    path,
                    label,
                    roi_xmin,
                    roi_ymin,
                    roi_xmax,
                    roi_ymax,
                    random_patch=random_patch,
                    max_y=max_y,
                    max_x=max_x,
                    ph=ph,
                    pw=pw,
                )

            y0, x0 = tf.cond(is_positive, _positive_offsets, _negative_offsets)
        y0 = tf.minimum(y0, max_y)
        x0 = tf.minimum(x0, max_x)
        img = tf.image.crop_to_bounding_box(img, y0, x0, ph, pw)
        img.set_shape([ph, pw, 3])
        if self.config.return_crop_offset:
            return (
                img,
                tf.cast(y0, tf.float32),
                tf.cast(x0, tf.float32),
                tf.cast(h, tf.float32),
                tf.cast(w, tf.float32),
            )
        return img

    def _make_bag(self, img: tf.Tensor) -> tf.Tensor:
        """Redimensiona y, opcionalmente, trocea en una grilla (rows x cols).

        Con bag_keras_tiling=False (defecto): devuelve (K, ph, pw, 3) —
        el tiling ocurre aqui en tf.data, la augmentacion se aplica por tile.

        Con bag_keras_tiling=True: devuelve (rows*ph, cols*pw, 3) —
        el tiling lo realiza la capa BagTiling dentro del modelo Keras,
        permitiendo que la augmentacion se aplique sobre la imagen completa (1x).
        """
        rows, cols = self.config.bag_grid
        ph, pw = self._height, self._width
        full = _resize_preserving_dtype(img, [rows * ph, cols * pw])
        if self.config.use_clahe:
            full = apply_clahe_tf(
                full,
                self.config.clahe_clip_limit,
                self.config.clahe_tile_grid,
            )
        full.set_shape([rows * ph, cols * pw, 3])
        if self.config.bag_keras_tiling:
            return full
        tiles = tf.reshape(full, [rows, ph, cols, pw, 3])
        tiles = tf.transpose(tiles, [0, 2, 1, 3, 4])
        bag = tf.reshape(tiles, [rows * cols, ph, pw, 3])
        bag.set_shape([rows * cols, ph, pw, 3])
        return bag

    def _roi_norm_tensors(self, tbl: pd.DataFrame) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        xmin_c, ymin_c, xmax_c, ymax_c = self.config.patch_roi_norm_columns
        missing = [
            column
            for column in self.config.patch_roi_norm_columns
            if column not in tbl.columns
        ]
        if missing:
            raise KeyError(
                "Las columnas ROI son necesarias para crop en hallazgos: "
                f"{missing}"
            )
        return (
            tf.constant(tbl[xmin_c].astype(np.float32).values),
            tf.constant(tbl[ymin_c].astype(np.float32).values),
            tf.constant(tbl[xmax_c].astype(np.float32).values),
            tf.constant(tbl[ymax_c].astype(np.float32).values),
        )

    def _process(
        self,
        path: tf.Tensor,
        label: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        flip_lateral: tf.Tensor | None = None,
        *,
        random_patch: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        img = decode_image(path)
        if flip_lateral is not None:
            flip = flip_lateral

            def _flip_image() -> tf.Tensor:
                return tf.image.flip_left_right(img)

            img = tf.cond(flip, _flip_image, lambda: img)
            if self.config.needs_roi_columns():
                flipped_xmin, flipped_xmax = _flip_roi_norm_x(roi_xmin, roi_xmax)
                roi_xmin = tf.where(flip, flipped_xmin, roi_xmin)
                roi_xmax = tf.where(flip, flipped_xmax, roi_xmax)
        if is_mil_mode(self.config.mode):
            return self._make_bag(img), label
        crop_meta: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor] | None = None
        if self.config.patch_mode:
            patch_out = self._extract_patch(
                img,
                path,
                label,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
            )
            if self.config.return_crop_offset:
                img, y0, x0, crop_h, crop_w = patch_out
                crop_meta = (y0, x0, crop_h, crop_w)
            else:
                img = patch_out
        else:
            img = _resize_preserving_dtype(img, self.config.image_size)
            if self.config.return_crop_offset:
                crop_meta = (
                    tf.constant(0.0, tf.float32),
                    tf.constant(0.0, tf.float32),
                    tf.cast(tf.shape(img)[0], tf.float32),
                    tf.cast(tf.shape(img)[1], tf.float32),
                )
        if self.config.use_clahe:
            img = apply_clahe_tf(
                img,
                self.config.clahe_clip_limit,
                self.config.clahe_tile_grid,
            )
            img.set_shape([self._height, self._width, 3])
        if self.config.return_crop_offset:
            assert crop_meta is not None
            return img, label, *crop_meta
        return img, label

    def _base_dataset(self, tbl: pd.DataFrame) -> tf.data.Dataset:
        paths = tf.constant(tbl[self.config.path_column].values)
        labels = tf.constant(tbl[self.config.label_column].values.astype(np.float32))
        roi_tensors: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor] | tuple[()] = ()
        if self.config.needs_roi_columns():
            roi_tensors = self._roi_norm_tensors(tbl)
        elements: tuple[Any, ...] = (paths, labels, *roi_tensors)
        if not self.config.lateralize:
            return tf.data.Dataset.from_tensor_slices(elements)

        flip_lateral = tf.constant(
            tbl[self.config.laterality_column].astype(str).values
            == self.lateralize_flip_side,
            dtype=tf.bool,
        )
        return tf.data.Dataset.from_tensor_slices((*elements, flip_lateral))

    def _wrap(
        self,
        dataset: tf.data.Dataset,
        *,
        name: str,
        row_count: int,
        ordered_dataset: tf.data.Dataset | None = None,
        source_table: pd.DataFrame | None = None,
    ) -> InspectDataset:
        return InspectDataset(
            dataset,
            name=name,
            config=self.config,
            row_count=row_count,
            ordered_dataset=ordered_dataset,
            source_table=source_table,
            lateralize_flip_side=self.lateralize_flip_side,
        )

    def _make_process_fn(self, *, random_patch: bool):
        with_roi = self.config.needs_roi_columns()

        if self.config.lateralize:

            def process_with_laterality(
                path: tf.Tensor,
                label: tf.Tensor,
                *extra: tf.Tensor,
            ) -> tuple[tf.Tensor, tf.Tensor]:
                if with_roi:
                    roi_xmin, roi_ymin, roi_xmax, roi_ymax, flip_lateral = extra
                    return self._process(
                        path,
                        label,
                        roi_xmin,
                        roi_ymin,
                        roi_xmax,
                        roi_ymax,
                        flip_lateral=flip_lateral,
                        random_patch=random_patch,
                    )
                flip_lateral = extra[0]
                return self._process(
                    path,
                    label,
                    tf.constant(0.0, tf.float32),
                    tf.constant(0.0, tf.float32),
                    tf.constant(0.0, tf.float32),
                    tf.constant(0.0, tf.float32),
                    flip_lateral=flip_lateral,
                    random_patch=random_patch,
                )

            return process_with_laterality

        def process(path: tf.Tensor, label: tf.Tensor, *extra: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            if with_roi:
                roi_xmin, roi_ymin, roi_xmax, roi_ymax = extra
            else:
                roi_xmin = roi_ymin = roi_xmax = roi_ymax = tf.constant(0.0, tf.float32)
            return self._process(
                path,
                label,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
            )

        return process

    def build(
        self,
        tbl: pd.DataFrame,
        *,
        shuffle: bool,
        random_patch: bool | None = None,
        name: str = "dataset",
    ) -> InspectDataset:
        if random_patch is None:
            random_patch = shuffle

        base = self._base_dataset(tbl)
        processed = base.map(
            self._make_process_fn(random_patch=random_patch),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.config.cache_filename is not None:
            cache_path = Path(self.config.cache_filename)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            processed = processed.cache(str(cache_path))
        elif self.config.cache_dataset:
            processed = processed.cache()

        if self.config.patch_mode and shuffle:
            ordered_processed = base.map(
                self._make_process_fn(random_patch=False),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ordered_processed = processed

        ordered_batched = _dataset_deterministic_options(
            ordered_processed.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        )

        if shuffle:
            processed = processed.shuffle(
                len(tbl),
                seed=self.config.seed,
                reshuffle_each_iteration=True,
            )
            batched = _dataset_perf_options(
                processed.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
            )
        else:
            batched = ordered_batched

        if shuffle and self.config.positive_mixup:
            alpha = float(self.config.positive_mixup_alpha)
            probability = float(self.config.positive_mixup_probability)

            if self.config.return_crop_offset:

                def _mixup_map_with_crop(
                    images: tf.Tensor,
                    labels: tf.Tensor,
                    y0: tf.Tensor,
                    x0: tf.Tensor,
                    crop_h: tf.Tensor,
                    crop_w: tf.Tensor,
                ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
                    images, labels = positive_only_mixup_batch(
                        images,
                        labels,
                        alpha=alpha,
                        probability=probability,
                    )
                    return images, labels, y0, x0, crop_h, crop_w

                mixup_fn = _mixup_map_with_crop
            else:

                def _mixup_map(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                    return positive_only_mixup_batch(
                        images,
                        labels,
                        alpha=alpha,
                        probability=probability,
                    )

                mixup_fn = _mixup_map

            batched = batched.map(mixup_fn, num_parallel_calls=tf.data.AUTOTUNE)
            batched = _dataset_perf_options(batched.prefetch(tf.data.AUTOTUNE))

        return self._wrap(
            batched,
            name=name,
            row_count=len(tbl),
            ordered_dataset=ordered_batched,
            source_table=tbl,
        )

    def build_train(self, tbl: pd.DataFrame) -> InspectDataset:
        return self.build(tbl, shuffle=True, name="train")

    def build_eval(self, tbl: pd.DataFrame, *, name: str = "eval") -> InspectDataset:
        return self.build(tbl, shuffle=False, name=name)

    def build_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> DatasetSplits:
        return DatasetSplits(
            train=self.build_train(train),
            val=self.build_eval(val, name="val"),
            test=self.build_eval(test, name="test"),
        )


def build_dataset_provider(
    image_size: tuple[int, int] | int | None = None,
    batch_size: int | None = None,
    *,
    config: DatasetProviderConfig | None = None,
    **kwargs: Any,
) -> DatasetProvider:
    if config is None:
        if image_size is None or batch_size is None:
            raise ValueError(
                "Indica `config=DatasetProviderConfig(...)` o los argumentos "
                "image_size y batch_size."
            )
        kwargs = resolve_mode_kwargs(kwargs)
        config = DatasetProviderConfig(
            image_size=image_size,
            batch_size=batch_size,
            **kwargs,
        )
    elif kwargs:
        raise ValueError("Pasa solo `config` o kwargs sueltos, no ambos.")
    return DatasetProvider(
        config.to_tf_config(),
        lateralize_flip_side=config.lateralize_flip_side,
    )


def build_tf_dataset(
    df: pd.DataFrame,
    image_size: tuple[int, int] | int | None = None,
    batch_size: int | None = None,
    *,
    shuffle: bool = True,
    name: str = "dataset",
    config: DatasetProviderConfig | None = None,
    as_batched: bool = True,
    **kwargs: Any,
) -> InspectDataset | tf.data.Dataset:
    """Arma un `tf.data.Dataset` batched desde un DataFrame pandas.

    El notebook controla las filas (positivos, hard negatives, etc.) en `df`.
    Crop por fila segun `cls` si `patch_crop_by_label` es None (default:
    cls=0 evita ROI, cls=1 centrado en ROI). Para forzar un crop en todo el df:
    `patch_crop_strategy="roi"`, `"uniform"` o `"avoid_roi"`.

    Returns
    -------
    InspectDataset
        Por defecto; expone `.dataset` (tf.data batched) y utilidades de inspeccion.
    tf.data.Dataset
        Si `as_batched=False`, el pipeline sin batch (raro en entrenamiento).
    """
    if df.empty:
        raise ValueError("El DataFrame esta vacio.")

    if config is None:
        if kwargs:
            if image_size is None or batch_size is None:
                raise ValueError(
                    "Indica `config=DatasetProviderConfig(...)` o image_size y batch_size."
                )
            config = DatasetProviderConfig(
                image_size=image_size,
                batch_size=batch_size,
                **kwargs,
            )
        else:
            raise ValueError(
                "Indica image_size/batch_size o `config=DatasetProviderConfig(...)`."
            )
    elif kwargs or image_size is not None or batch_size is not None:
        raise ValueError("Pasa solo `config` o kwargs sueltos, no ambos.")

    provider = build_dataset_provider(config=config)
    built = provider.build(df, shuffle=shuffle, name=name)
    if as_batched:
        return built
    return built.dataset


_register_dataframe_show_samples()
