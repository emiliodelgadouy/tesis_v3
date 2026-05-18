from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
import tensorflow as tf

SplitName = Literal["train", "val", "test"]


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


def _collect_batches(
    dataset: tf.data.Dataset,
    *,
    max_batches: int | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    images_batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    for batch_idx, (images, labels) in enumerate(dataset):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images_batches.append(np.asarray(images))
        labels_batches.append(np.asarray(labels).reshape(-1))
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
    ):
        self._dataset = dataset
        self._ordered_dataset = ordered_dataset or dataset
        self.name = name
        self.config = config
        self.row_count = row_count

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
        if max_batches is not None:
            print(f"  (estadisticas sobre los primeros {max_batches} batches)")
        return stats

    def show_samples(
        self,
        n: int = 4,
        *,
        n_batches: int = 1,
        deterministic: bool = False,
        figsize: tuple[float, float] = (3.0, 3.0),
        title: str | None = None,
    ) -> None:
        images, labels = self.sample_batches(
            n_batches=n_batches,
            deterministic=deterministic,
        )
        n = min(n, len(labels))
        images = _images_to_display(images[:n])

        import matplotlib.pyplot as plt

        cols = min(n, 4)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(figsize[0] * cols, figsize[1] * rows),
            facecolor="black",
        )
        axes = np.atleast_1d(axes).reshape(-1)

        for idx in range(rows * cols):
            ax = axes[idx]
            ax.set_facecolor("black")
            if idx < n:
                ax.imshow(images[idx])
                label = int(labels[idx] >= 0.5)
                ax.set_title(f"label={label}", color="white")
            ax.axis("off")

        suffix = " [orden fijo]" if deterministic else ""
        fig.suptitle(title or f"{self.name} (n={n}){suffix}", color="white")
        plt.tight_layout()
        plt.show()


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
        n: int = 4,
        *,
        n_batches: int = 1,
        deterministic: bool = False,
        figsize: tuple[float, float] = (3.0, 3.0),
        splits: tuple[SplitName, ...] = ("train", "val", "test"),
    ) -> None:
        for name in splits:
            self[name].show_samples(
                n=n,
                n_batches=n_batches,
                deterministic=deterministic,
                figsize=figsize,
            )


class DatasetProvider:
    """Construye `tf.data.Dataset` a partir de tablas pandas con paths y etiquetas."""

    def __init__(self, config: TfDatasetConfig):
        self.config = config
        self._height, self._width = int(config.image_size[0]), int(config.image_size[1])

    def _process(
        self,
        path: tf.Tensor,
        label: tf.Tensor,
        flip_lateral: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        img = decode_image(path)
        if flip_lateral is not None:
            img = tf.cond(
                flip_lateral,
                lambda: tf.image.flip_left_right(img),
                lambda: img,
            )
        img = tf.image.resize(img, self.config.image_size)
        if self.config.use_clahe:
            img = apply_clahe_tf(
                img,
                self.config.clahe_clip_limit,
                self.config.clahe_tile_grid,
            )
            img.set_shape([self._height, self._width, 3])
        return img, label

    def _base_dataset(self, tbl: pd.DataFrame) -> tf.data.Dataset:
        paths = tf.constant(tbl[self.config.path_column].values)
        labels = tf.constant(tbl[self.config.label_column].values.astype(np.float32))
        if not self.config.lateralize:
            return tf.data.Dataset.from_tensor_slices((paths, labels))

        flip_lateral = tf.constant(
            tbl[self.config.laterality_column].astype(str).values == "L",
            dtype=tf.bool,
        )
        return tf.data.Dataset.from_tensor_slices((paths, labels, flip_lateral))

    def _wrap(
        self,
        dataset: tf.data.Dataset,
        *,
        name: str,
        row_count: int,
        ordered_dataset: tf.data.Dataset | None = None,
    ) -> InspectDataset:
        return InspectDataset(
            dataset,
            name=name,
            config=self.config,
            row_count=row_count,
            ordered_dataset=ordered_dataset,
        )

    def build(self, tbl: pd.DataFrame, *, shuffle: bool, name: str = "dataset") -> InspectDataset:
        if self.config.lateralize:
            process_fn = lambda path, label, flip: self._process(
                path, label, flip_lateral=flip
            )
        else:
            process_fn = self._process

        ds = self._base_dataset(tbl).map(
            process_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.config.cache_filename is not None:
            cache_path = Path(self.config.cache_filename)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ds = ds.cache(str(cache_path))
        elif self.config.cache_dataset:
            ds = ds.cache()

        ordered_batched = _dataset_perf_options(
            ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        )

        if shuffle:
            ds = ds.shuffle(
                len(tbl),
                seed=self.config.seed,
                reshuffle_each_iteration=True,
            )
            batched = _dataset_perf_options(
                ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
            )
        else:
            batched = ordered_batched

        return self._wrap(
            batched,
            name=name,
            row_count=len(tbl),
            ordered_dataset=ordered_batched,
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
    image_size: tuple[int, int],
    batch_size: int,
    *,
    seed: int = 42,
    path_column: str = "path",
    label_column: str = "cls",
    cache_dataset: bool = True,
    cache_filename: str | Path | None = None,
    use_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
    lateralize: bool = False,
    laterality_column: str = "laterality",
) -> DatasetProvider:
    config = TfDatasetConfig(
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        path_column=path_column,
        label_column=label_column,
        cache_dataset=cache_dataset,
        cache_filename=cache_filename,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
        lateralize=lateralize,
        laterality_column=laterality_column,
    )
    return DatasetProvider(config)
