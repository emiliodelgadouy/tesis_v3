import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
import time
from pathlib import Path
from tensorflow import keras


def _decode_fallback(raw):
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img


# CLAHE no tiene op nativa en TF: se ejecuta en CPU via cv2 dentro de tf.numpy_function.
# Como es deterministico por imagen, conviene combinarlo con .cache() para amortizar el costo.
def _clahe_np_uint8(image, clip_limit, tile_grid):
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


def apply_clahe_tf(img, clip_limit=2.0, tile_grid=8):
    """CLAHE sobre luminancia, salida RGB en float32 con rango [0, 255].

    Espera entrada HxWx3 en [0, 255] (cualquier dtype). El backbone se encarga
    despues del preprocess_input correspondiente.
    """
    img_u8 = tf.cast(tf.clip_by_value(tf.cast(img, tf.float32), 0.0, 255.0), tf.uint8)
    out = tf.numpy_function(
        func=lambda x: _clahe_np_uint8(x, clip_limit, tile_grid),
        inp=[img_u8],
        Tout=tf.float32,
    )
    out.set_shape([None, None, 3])
    return out


def decode_image(path):
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


def _dataset_perf_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    opts = tf.data.Options()
    if hasattr(opts, "deterministic"):
        opts.deterministic = False
    elif hasattr(opts, "experimental_deterministic"):
        opts.experimental_deterministic = False
    return ds.with_options(opts)

def stratified_split(
    ds,
    val_split=0.20,
    seed=42,
    split_column="split",
    train_split_value="training",
    test_split_value="test",
    label_column="cls",
    group_column="patient_id",
):
    tbl_training_full = ds[ds[split_column] == train_split_value].reset_index(drop=True)
    tbl_test = ds[ds[split_column] == test_split_value].reset_index(drop=True)

    n_splits = round(1 / val_split)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(
        sgkf.split(
            tbl_training_full,
            y=tbl_training_full[label_column],
            groups=tbl_training_full[group_column],
        )
    )

    tbl_train = tbl_training_full.iloc[train_idx].reset_index(drop=True)
    tbl_val = tbl_training_full.iloc[val_idx].reset_index(drop=True)
    return tbl_train, tbl_val, tbl_test


def to_tf_dataset(
    tbl,
    IMAGE_SIZE,
    batch_size,
    shuffle,
    seed,
    *,
    mil=False,
    mil_num_instances=8,
    mil_pool_size=None,
    cache_dataset=True,
    cache_filename=None,
    use_clahe=False,
    clahe_clip_limit=2.0,
    clahe_tile_grid=8,
):
    """Si mil=True, cada elemento es una bolsa (K, H, W, 3) con la misma etiqueta de imagen.

    Se reescala la imagen a mil_pool_size (por defecto 2x el parche) y se extraen K parches
    de tamano IMAGE_SIZE. Train usa random_crop; val/test (shuffle=False) usa cortes
    reproducibles por ruta (stateless_random_crop).

    cache_dataset: cache en RAM despues del primer epoch (desactivar si no cabe en memoria).
    cache_filename: ruta opcional para cache en disco (persistente entre corridas); crea el directorio padre.

    use_clahe: aplica CLAHE (en cv2) sobre la luminancia y replica a 3 canales antes de cachear.
    Como es deterministico por imagen, queda en cache y no penaliza epochs posteriores.
    """
    paths = tf.constant(tbl["path"].values)
    labels = tf.constant(tbl["cls"].values.astype(np.float32))
    ds_tf = tf.data.Dataset.from_tensor_slices((paths, labels))

    ph, pw = int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])
    train_mode = shuffle

    if mil:
        if mil_pool_size is None:
            pool_hw = (ph * 2, pw * 2)
        else:
            pool_hw = (int(mil_pool_size[0]), int(mil_pool_size[1]))
        if pool_hw[0] < ph or pool_hw[1] < pw:
            raise ValueError(
                f"mil_pool_size {pool_hw} debe ser >= tamano de parche ({ph}, {pw})"
            )
        mil_k = int(mil_num_instances)
        if mil_k < 1:
            raise ValueError("mil_num_instances debe ser >= 1")

        def process(path, label):
            img = decode_image(path)
            img = tf.image.resize(img, pool_hw)
            if use_clahe:
                img = apply_clahe_tf(img, clahe_clip_limit, clahe_tile_grid)
                img.set_shape([pool_hw[0], pool_hw[1], 3])
            hpath = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
            crops = []
            for i in range(mil_k):
                if train_mode:
                    c = tf.image.random_crop(img, [ph, pw, 3])
                else:
                    seed_pair = tf.stack(
                        [
                            tf.cast(hpath + i, tf.int64),
                            tf.cast(hpath + 1000 * i + 911, tf.int64),
                        ],
                        axis=0,
                    )
                    c = tf.image.stateless_random_crop(
                        tf.cast(img, tf.float32), [ph, pw, 3], seed=seed_pair
                    )
                    c = tf.cast(c, img.dtype)
                crops.append(c)
            bag = tf.stack(crops, axis=0)
            return bag, label

    else:

        def process(path, label):
            img = decode_image(path)
            img = tf.image.resize(img, IMAGE_SIZE)
            if use_clahe:
                img = apply_clahe_tf(img, clahe_clip_limit, clahe_tile_grid)
                img.set_shape([ph, pw, 3])
            return img, label

    ds_tf = ds_tf.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if cache_filename is not None:
        Path(cache_filename).parent.mkdir(parents=True, exist_ok=True)
        ds_tf = ds_tf.cache(cache_filename)
    elif cache_dataset:
        ds_tf = ds_tf.cache()

    if shuffle:
        ds_tf = ds_tf.shuffle(len(tbl), seed=seed, reshuffle_each_iteration=True)

    out = ds_tf.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return _dataset_perf_options(out)

class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.total_elapsed_times = []
        self._fit_start = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        total = time.perf_counter() - self._fit_start
        self.total_elapsed_times.append(total)
        if logs is not None:
            logs["epoch_time_seconds"] = elapsed
            logs["total_elapsed_seconds"] = total
