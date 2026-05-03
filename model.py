from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


class ModelBuilder:
    def __init__(self,IMG_SIZE, backbone, preprocess_input, backbone_trainable=False, top_dense=256, dropout=0.4, learning_rate=1e-3):
        self.IMG_SIZE = IMG_SIZE
        self.backbone = backbone
        self.preprocess_input = preprocess_input
        self.backbone.trainable = backbone_trainable
        self.input_shape = backbone.input_shape[1:3]
        self.top_dense = top_dense
        self.dropout = dropout
        self.learning_rate = learning_rate

    def head(self, x):
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(self.top_dense, activation="relu", name="dense")(x)
        x = layers.Dropout(self.dropout, name="dropout")(x)
        return x

    def resize(self, x):
        x = layers.Resizing(self.input_shape[0], self.input_shape[1], crop_to_aspect_ratio=True, name="resize")(x)
        return x

    def augmentation(self,x):
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
        return keras.losses.BinaryCrossentropy()

    def metrics(self):
        return [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    def make_backbone_trainable(self, trainable=True, learning_rate=None):
        self.backbone.trainable = trainable
        self.learning_rate = learning_rate
        self.compile()

        return self
    
    def compile(self):
        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.loss(),
            metrics=self.metrics(),
        )
        return self

    def build(self):
        inputs = self.inputs()
        x = self.resize(inputs)
        x = self.augmentation(x)
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.head(x)
        outputs = self.output(x)
        self.model = keras.Model(inputs, outputs, name="simple_validation_vgg")
        self.compile()
        return self

    
    def summary(self):
        return self.model.summary()

    def fit(self, train_ds, val_ds, epochs=5, callbacks=None):
        return self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    def evaluate(self, test_ds, return_dict=True):
        return self.model.evaluate(test_ds, return_dict=return_dict)
    