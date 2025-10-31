"""
CNN Model definition using DenseNet121 with transfer learning
"""

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, regularizers

class CNNModel(object):
    """
    Initializes a DenseNet121 model with a custom classification head
    """

    def __init__(self, input_shape=(50, 50, 3), freeze_upto=100, l2=0.01, dropout=0.5):
        self.input_shape = input_shape
        self.freeze_upto = freeze_upto
        self.l2 = l2
        self.dropout = dropout
        self.model = None

    def build(self):
        """
        Build and compile the model (transfer learning + weighted loss ready)
        """
        # Load pre-trained DenseNet without top layer
        base = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            pooling='avg'
        )

        # Freeze initial layers (transfer learning)
        for layer in base.layers[:self.freeze_upto]:
            layer.trainable = False

        # Custom classification head
        inputs = layers.Input(shape=self.input_shape)
        x = base(inputs, training=False)
        x = layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(self.l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)

        # Custom weighted loss: penalize FN more
        def weighted_loss(y_true, y_pred):
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            weight = tf.where(y_true == 1, 2.5, 1.0)  # Higher penalty for positives
            return tf.reduce_mean(weight * bce)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=weighted_loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        self.model = model
        return self.model

if __name__ == "__main__":
    # Minimal sanity-check (shape only)
    cnn = CNNModel()
    m = cnn.build()
    m.summary()