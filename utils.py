import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold


def make_patient_stratified_split(
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


def make_binary_dataset(tbl, batch_size, shuffle, seed):
    paths = tf.constant(tbl["path"].values)
    labels = tf.constant(tbl["cls"].values.astype(np.float32))
    ds_tf = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds_tf = ds_tf.shuffle(len(tbl), seed=seed, reshuffle_each_iteration=True)

    ds_tf = ds_tf.map(
        lambda path, label: (decode_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds_tf.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def decode_image(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32) * 255.0
    return img
