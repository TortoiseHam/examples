from fastestimator.estimator.estimator import Estimator
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.estimator.trace import Accuracy
import tensorflow as tf
import numpy as np

class Network:
    def __init__(self):
        self.model = LeNet()
        self.optimizer = tf.optimizers.Adam()
        self.loss = tf.losses.SparseCategoricalCrossentropy()

    def train_op(self, batch):
        with tf.GradientTape() as tape:
            predictions = self.model(batch["x"])
            loss = self.loss(batch["y"], predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return predictions, loss

    def eval_op(self, batch):
        predictions = self.model(batch["x"], training=False)
        loss = self.loss(batch["y"], predictions)
        return predictions, loss

def pad(list, padding_size, padding_value):
    return list + [padding_value] * abs((len(list)-padding_size))

def get_estimator(epochs=2, batch_size=64, optimizer="adam"):

    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.imdb.load_data(maxlen=300)
    x_train = np.array([pad(x, 300, 0) for x in x_train])
    x_eval = np.array([pad(x, 300, 0) for x in x_eval])


    pipeline = Pipeline(batch_size=batch_size,
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        validation_data={"x": x_eval, "y": y_eval},
                        transform_train= [[], []])

    traces = [Accuracy(feature_true="y")]

    estimator = Estimator(network= Network(),
                          pipeline=pipeline,
                          epochs= epochs,
                          traces= traces)
    return estimator