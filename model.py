from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers

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
            checkpoint_monitor="val_pr_auc",
    ):
        self.IMG_SIZE = IMG_SIZE
        self.backbone = backbone
        self.preprocess_input = preprocess_input
        self.backbone.trainable = backbone_trainable
        self.input_shape = backbone.input_shape[1:3]
        self.top_dense = top_dense
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_path = self.checkpoint_dir / "best_checkpoint.weights.h5"
        self.fit_number = 0
        self.best_checkpoints = []

    def head(self, x):
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(self.top_dense, activation="relu", name="dense")(x)
        x = layers.Dropout(self.dropout, name="dropout")(x)
        return x

    def resize(self, x):
        x = layers.Resizing(self.input_shape[0], self.input_shape[1], crop_to_aspect_ratio=True, name="resize")(x)
        return x

    def augmentation(self, x):
        return keras.Sequential(
            [
                layers.RandomContrast(0.08),
                layers.RandomBrightness(0.08, value_range=(0.0, 255.0)),
            ],
            name="augmentation",
        )(x)

    def inputs(self):
        x = keras.Input(shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3), name="image")
        return x

    def output(self, x):
        return layers.Dense(1, activation="sigmoid", dtype="float32", name="cls")(x)

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
        )

    def metrics(self):
        return [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
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
        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.focal_loss(),
            metrics=self.metrics(),
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
            checkpoint_name = checkpoint_name[:-len(suffix)]

        return checkpoint_path.with_name(
            f"{checkpoint_name}_epoch{epoch:02d}{suffix}"
        )

    def checkpoint_files(self):
        checkpoint_path = Path(self.checkpoint_path)
        suffix = ".weights.h5"
        checkpoint_name = checkpoint_path.name
        if checkpoint_name.endswith(suffix):
            checkpoint_name = checkpoint_name[:-len(suffix)]

        return checkpoint_path.parent.glob(
            f"{checkpoint_name}_epoch*{suffix}"
        )

    def checkpoint_info(self, checkpoint_path):
        checkpoint_name = Path(checkpoint_path).name
        epoch_text = checkpoint_name.rsplit("_epoch", 1)[1].removesuffix(
            ".weights.h5"
        )
        return int(epoch_text)

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
            if best_value[monitor] is not None and current <= best_value[monitor]:
                return

            checkpoint_path = self.checkpoint_filepath(epoch + 1)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_weights(str(checkpoint_path))

            for previous_checkpoint_path in self.checkpoint_files():
                if previous_checkpoint_path != checkpoint_path:
                    previous_checkpoint_path.unlink()

            best_value[monitor] = current
            self.best_checkpoints = [
                info for info in self.best_checkpoints
                if info["stage"] != stage
            ]
            self.best_checkpoints.append(
                {
                    "stage": stage,
                    "epoch": epoch + 1,
                    "monitor": monitor,
                    "value": current,
                    "path": checkpoint_path,
                }
            )
            print(
                f"\nEpoch {epoch + 1}: {monitor} improved to {current:.4f}. "
                f"Saved {checkpoint_path}"
            )

        return keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

    def callbacks(self):
        return [
            self.checkpoint_callback(),
            EpochTimer(),
        ]

    def fit(self, train_ds, val_ds, epochs=5, callbacks=None):
        self.fit_number += 1
        self.checkpoint_path = self.checkpoint_dir / f"stage_{self.fit_number}.weights.h5"
        callbacks = self.callbacks() + list(callbacks or [])
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

    def load_best_checkpoint(self):
        stage = self.fit_number
        checkpoint_info = next(
            (
                info for info in self.best_checkpoints
                if info["stage"] == stage
            ),
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

        checkpoint_info = max(self.best_checkpoints, key=lambda info: info["value"])
        checkpoint_path = checkpoint_info["path"]
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"No existe el checkpoint global: {checkpoint_path}")

        self.model.load_weights(str(checkpoint_path))
        return checkpoint_info

    def evaluate(self, test_ds, return_dict=True):
        return self.model.evaluate(test_ds, return_dict=return_dict)
