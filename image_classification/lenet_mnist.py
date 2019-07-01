from fastestimator.pipeline.static.preprocess import Minmax
from fastestimator.estimator.estimator import Estimator
from fastestimator.pipeline.pipeline import Pipeline
from fastestimator.architecture.lenet import LeNet
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

def get_estimator(epochs=2, batch_size=32, optimizer="adam"):

    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, -1)
    x_eval = np.expand_dims(x_eval, -1)

    pipeline = Pipeline(batch_size=batch_size,
                        feature_name=["x", "y"],
                        train_data={"x": x_train, "y": y_train},
                        validation_data={"x": x_eval, "y": y_eval},
                        transform_train= [[Minmax()], []])

    traces = [Accuracy(y_true_key="y")]

    estimator = Estimator(network= Network(),
                          pipeline=pipeline,
                          epochs= epochs,
                          traces= traces)
    return estimator