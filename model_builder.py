from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from src.dataset_provider import as_tf_dataset
from src.utils import EpochTimer


class ModelBuilder:
    def __init__(
        self,
        IMG_SIZE,
        backbone,
        preprocess_input,
        backbone_trainable=False,
        top_dense=256,
        dropout=0.4,
        learning_rate=1e-3,
        focal_alpha=0.90,
        focal_gamma=2.0,
        metric_to_maximize="pr_auc",
        checkpoint_monitor=None,
        monitor_mode="max",
        early_stopping_patience=8,
        reduce_lr_patience=4,
        reduce_lr_factor=0.5,
        min_lr=1e-7,
        aggressive_augmentation=False,
        initial_bias=None,
    ):
        self.IMG_SIZE = IMG_SIZE
        self.backbone = backbone
        self.preprocess_input = preprocess_input
        self.backbone.trainable = backbone_trainable
        self.input_shape = backbone.input_shape[1:3]
        self.aggressive_augmentation = bool(aggressive_augmentation)
        self.top_dense = top_dense
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        if checkpoint_monitor is not None:
            self.checkpoint_monitor = checkpoint_monitor
            self.metric_to_maximize = (
                checkpoint_monitor.removeprefix("val_")
                if checkpoint_monitor.startswith("val_")
                else checkpoint_monitor
            )
        else:
            m = str(metric_to_maximize)
            if m.startswith("val_"):
                self.checkpoint_monitor = m
                self.metric_to_maximize = m.removeprefix("val_")
            else:
                self.metric_to_maximize = m
                self.checkpoint_monitor = f"val_{m}"
        self.monitor_mode = monitor_mode
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.min_lr = min_lr
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_path = self.checkpoint_dir / "best_checkpoint.weights.h5"
        self.fit_number = 0
        self.best_checkpoints = []
        self.initial_bias = initial_bias
        self.loss_from_logits = True

    def top_mlp(self, x):
        x = layers.Dense(self.top_dense, activation="relu", name="dense")(x)
        x = layers.Dropout(self.dropout, name="dropout")(x)
        return x

    def head(self, x):
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        return self.top_mlp(x)

    def resize(self, x):
        x = layers.Resizing(self.input_shape[0], self.input_shape[1], crop_to_aspect_ratio=True, name="resize")(x)
        return x

    def augmentation_seq(self):
        if self.aggressive_augmentation:
            return keras.Sequential(
                [
                    layers.RandomFlip("horizontal", name="aug_flip_h"),
                    layers.RandomRotation(0.14, fill_mode="reflect", name="aug_rot"),
                    layers.RandomZoom(
                        height_factor=(0.0, 0.22),
                        width_factor=(0.0, 0.22),
                        fill_mode="reflect",
                        name="aug_zoom",
                    ),
                    layers.RandomTranslation(
                        height_factor=0.14,
                        width_factor=0.14,
                        fill_mode="reflect",
                        name="aug_translate",
                    ),
                    layers.RandomContrast(0.25, name="aug_contrast"),
                    layers.RandomBrightness(
                        0.25,
                        value_range=(0.0, 255.0),
                        name="aug_brightness",
                    ),
                ],
                name="augmentation_aggressive",
            )
        return keras.Sequential(
            [
                layers.RandomContrast(0.08),
                layers.RandomBrightness(0.08, value_range=(0.0, 255.0)),
            ],
            name="augmentation",
        )

    def augmentation(self, x):
        return self.augmentation_seq()(x)

    def inputs(self):
        return keras.Input(shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3), name="image")

    def output(self, x):
        # Keep classifier head in float32 for numeric stability under mixed precision.
        bias_init = (
            tf.keras.initializers.Constant(self.initial_bias)
            if self.initial_bias is not None
            else "zeros"
        )
        x = layers.Dense(1, dtype="float32", bias_initializer=bias_init, name="output")(x)
        if self.loss_from_logits:
            return x
        return layers.Activation("sigmoid", dtype="float32", name="output_sigmoid")(x)

    def preprocess(self, x):
        return layers.Lambda(self.preprocess_input, name="preprocess_input")(x)

    def optimizer(self):
        return keras.optimizers.Adam(learning_rate=self.learning_rate)

    def loss(self):
        return self.focal_loss()

    def focal_loss(self):
        return keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            from_logits=self.loss_from_logits,
        )

    def metrics(self):
        threshold = 0.0 if self.loss_from_logits else 0.5
        return [
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
            keras.metrics.AUC(name="auc", from_logits=self.loss_from_logits),
            keras.metrics.AUC(curve="PR", name="pr_auc", from_logits=self.loss_from_logits),
            keras.metrics.Precision(name="precision", thresholds=threshold),
            keras.metrics.Recall(name="recall", thresholds=threshold),
        ]

    def keep_batch_norm_frozen(self):
        for layer in self.backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    def make_backbone_trainable(self, trainable=True, learning_rate=None, train_batch_norm=False):
        self.backbone.trainable = trainable
        if trainable and not train_batch_norm:
            self.keep_batch_norm_frozen()
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.compile()

        return self

    def make_backbone_partially_trainable(
        self,
        trainable_fraction=0.30,
        learning_rate=None,
        train_batch_norm=False,
    ):
        total_layers = len(self.backbone.layers)
        trainable_layers = max(1, round(total_layers * trainable_fraction))
        freeze_until = total_layers - trainable_layers

        self.backbone.trainable = True
        for layer in self.backbone.layers[:freeze_until]:
            layer.trainable = False
        for layer in self.backbone.layers[freeze_until:]:
            layer.trainable = True

        if not train_batch_norm:
            self.keep_batch_norm_frozen()

        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.compile()

        return self

    def compile(self):
        # jit_compile suele caer a False en modelos con augment capas custom (warning Keras); evita intentos XLA inútiles.
        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.focal_loss(),
            metrics=self.metrics(),
            jit_compile=False,
        )
        return self

    def build(self):
        inputs = self.inputs()
        # x = self.resize(inputs)
        x = self.augmentation(inputs)
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.head(x)
        outputs = self.output(x)
        self.model = keras.Model(inputs, outputs, name="simple_validation_vgg")
        self.compile()
        return self

    def summary(self):
        return self.model.summary()

    def checkpoint_filepath(self, epoch):
        checkpoint_path = Path(self.checkpoint_path)

        suffix = ".weights.h5"
        checkpoint_name = checkpoint_path.name
        if checkpoint_name.endswith(suffix):
            checkpoint_name = checkpoint_name[: -len(suffix)]

        return checkpoint_path.with_name(f"{checkpoint_name}_epoch{epoch:02d}{suffix}")

    def checkpoint_files(self):
        checkpoint_path = Path(self.checkpoint_path)
        suffix = ".weights.h5"
        checkpoint_name = checkpoint_path.name
        if checkpoint_name.endswith(suffix):
            checkpoint_name = checkpoint_name[: -len(suffix)]

        return checkpoint_path.parent.glob(f"{checkpoint_name}_epoch*{suffix}")

    def checkpoint_info(self, checkpoint_path):
        checkpoint_name = Path(checkpoint_path).name
        epoch_text = checkpoint_name.rsplit("_epoch", 1)[1].removesuffix(".weights.h5")
        return int(epoch_text)

    def monitor_improved(self, current, best):
        if best is None:
            return True
        if self.monitor_mode == "min":
            return current < best
        return current > best

    def checkpoint_callback(self):
        monitor = self.checkpoint_monitor
        stage = self.fit_number
        best_value = {monitor: None}

        def on_epoch_end(epoch, logs=None):
            logs = logs or {}
            current = logs.get(monitor)
            if current is None:
                return

            current = float(current)
            if not self.monitor_improved(current, best_value[monitor]):
                return

            checkpoint_path = self.checkpoint_filepath(epoch + 1)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_weights(str(checkpoint_path))

            for previous_checkpoint_path in self.checkpoint_files():
                if previous_checkpoint_path != checkpoint_path:
                    previous_checkpoint_path.unlink()

            best_value[monitor] = current
            self.best_checkpoints = [info for info in self.best_checkpoints if info["stage"] != stage]
            self.best_checkpoints.append(
                {
                    "stage": stage,
                    "epoch": epoch + 1,
                    "monitor": monitor,
                    "value": current,
                    "path": checkpoint_path,
                }
            )
            print(f"\nEpoch {epoch + 1}: {monitor} improved to {current:.4f}. Saved {checkpoint_path}")

        return keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

    def early_stopping_callback(self):
        return keras.callbacks.EarlyStopping(
            monitor=self.checkpoint_monitor,
            mode=self.monitor_mode,
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )

    def reduce_lr_callback(self):
        return keras.callbacks.ReduceLROnPlateau(
            monitor=self.checkpoint_monitor,
            mode=self.monitor_mode,
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
            verbose=1,
        )

    def callbacks(self):
        return [
            self.checkpoint_callback(),
            self.early_stopping_callback(),
            self.reduce_lr_callback(),
            EpochTimer(),
        ]

    def fit(self, train_ds, val_ds, epochs=5, callbacks=None):
        self.fit_number += 1
        self.checkpoint_path = self.checkpoint_dir / f"stage_{self.fit_number}.weights.h5"
        callbacks = self.callbacks() + list(callbacks or [])
        return self.model.fit(
            as_tf_dataset(train_ds),
            validation_data=as_tf_dataset(val_ds),
            epochs=epochs,
            callbacks=callbacks,
        )

    def load_best_checkpoint(self):
        stage = self.fit_number
        checkpoint_info = next(
            (info for info in self.best_checkpoints if info["stage"] == stage),
            None,
        )

        if checkpoint_info is not None:
            checkpoint_path = checkpoint_info["path"]
            epoch = checkpoint_info["epoch"]
        else:
            checkpoint_files = list(self.checkpoint_files())
            if not checkpoint_files:
                raise FileNotFoundError(f"No existen checkpoints para: {self.checkpoint_path}")

            checkpoint_path = max(checkpoint_files, key=lambda path: path.stat().st_mtime)
            epoch = self.checkpoint_info(checkpoint_path)

        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"No existen checkpoints para: {self.checkpoint_path}")

        self.model.load_weights(str(checkpoint_path))
        return epoch

    def load_best_global_checkpoint(self):
        if not self.best_checkpoints:
            raise FileNotFoundError("No existen checkpoints registrados en esta corrida.")

        checkpoint_info = min(self.best_checkpoints, key=lambda info: info["value"]) if self.monitor_mode == "min" else max(self.best_checkpoints, key=lambda info: info["value"])
        checkpoint_path = checkpoint_info["path"]
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"No existe el checkpoint global: {checkpoint_path}")

        self.model.load_weights(str(checkpoint_path))
        return checkpoint_info

    def evaluate(self, test_ds, return_dict=True):
        return self.model.evaluate(as_tf_dataset(test_ds), return_dict=return_dict)

    def predict(self, test_ds, verbose=1):
        return self.model.predict(as_tf_dataset(test_ds), verbose=verbose)