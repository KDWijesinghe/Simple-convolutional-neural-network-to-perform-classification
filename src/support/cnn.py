from pathlib import Path

import numpy as np

from .util import ModelIO


class CNN:
    def __init__(self, input_shape=None, num_classes=10, random_state=None):
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.input_shape = self.make_shape(input_shape)
        self.num_classes = int(num_classes)
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.history = {"loss": [], "accuracy": []}

        if self.input_shape:
            self.initialize(int(np.prod(self.input_shape)))

    @staticmethod
    def softmax(scores):
        scores = np.asarray(scores, dtype=float)
        single_row = scores.ndim == 1

        if single_row:
            scores = scores.reshape(1, -1)

        if scores.ndim != 2:
            raise ValueError("scores must be one- or two-dimensional")

        scores = scores - scores.max(axis=1, keepdims=True)
        values = np.exp(scores)
        probabilities = values / values.sum(axis=1, keepdims=True)

        if single_row:
            return probabilities[0]

        return probabilities

    @staticmethod
    def one_hot(labels, num_classes=None):
        labels = np.asarray(labels)

        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be an integer vector")

        if len(labels) == 0:
            raise ValueError("labels cannot be empty")

        if num_classes is None:
            num_classes = labels.max() + 1

        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError("label outside the class range")

        return np.eye(num_classes)[labels]

    @staticmethod
    def cross_entropy(targets, probabilities):
        targets = np.asarray(targets)
        probabilities = np.asarray(probabilities)

        if targets.shape != probabilities.shape:
            raise ValueError("targets and probabilities must have equal shapes")

        probabilities = np.clip(probabilities, 1e-12, 1.0)
        return float(-np.mean(np.sum(targets * np.log(probabilities), axis=1)))

    def fit(
        self,
        X,
        y,
        learning_rate=0.1,
        max_epochs=1000,
        tolerance=1e-4,
        batch_size=None,
        verbose=False,
    ):
        features = self.get_features(X)
        targets, labels = self.get_targets(y, len(features))

        if learning_rate <= 0 or max_epochs < 1 or tolerance < 0:
            raise ValueError("invalid training parameters")

        if self.weights is None:
            self.input_shape = tuple(np.asarray(X).shape[1:])
            self.initialize(features.shape[1])

        if features.shape[1] != self.weights.shape[0]:
            raise ValueError("X does not match the model input shape")

        if batch_size is None:
            batch_size = len(features)

        batch_size = int(batch_size)

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        random = np.random.default_rng(self.random_state)
        self.history = {"loss": [], "accuracy": []}
        previous_loss = None

        for epoch in range(max_epochs):
            if batch_size < len(features):
                order = random.permutation(len(features))
            else:
                order = np.arange(len(features))

            for start in range(0, len(features), batch_size):
                indexes = order[start : start + batch_size]
                batch_x = features[indexes]
                batch_y = targets[indexes]
                scores = batch_x @ self.weights + self.bias
                error = self.softmax(scores) - batch_y
                self.weights -= learning_rate * batch_x.T @ error / len(indexes)
                self.bias -= learning_rate * error.mean(axis=0)

            probabilities = self.predict_proba(features, already_flat=True)
            loss = self.cross_entropy(targets, probabilities)
            predictions = probabilities.argmax(axis=1)
            accuracy = float(np.mean(predictions == labels))
            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)

            if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
                print(f"epoch {epoch + 1}: loss={loss:.4f}, accuracy={accuracy:.2%}")

            if previous_loss is not None and abs(previous_loss - loss) <= tolerance:
                break

            previous_loss = loss

        return self.history

    def train(self, X, y, **options):
        return self.fit(X, y, **options)

    def predict_proba(self, X, already_flat=False):
        self.check_fitted()

        if already_flat:
            features = np.asarray(X, dtype=float)
        else:
            features = self.get_features(X)

        if features.shape[1] != self.weights.shape[0]:
            raise ValueError("X does not match the model input shape")

        return self.softmax(features @ self.weights + self.bias)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X, y):
        y = np.asarray(y)

        if y.ndim == 2:
            labels = y.argmax(axis=1)
        else:
            labels = y.astype(int)

        probabilities = self.predict_proba(X)
        targets = self.one_hot(labels, self.num_classes)
        predictions = probabilities.argmax(axis=1)

        return {
            "loss": self.cross_entropy(targets, probabilities),
            "accuracy": float(np.mean(predictions == labels)),
        }

    def save(self, path):
        self.check_fitted()
        arrays = {"weights": self.weights, "bias": self.bias}
        return ModelIO.save(
            path,
            arrays,
            input_shape=list(self.input_shape),
            num_classes=self.num_classes,
            random_state=self.random_state,
        )

    @classmethod
    def load(cls, path):
        arrays, details = ModelIO.load(path)

        if "weights" not in arrays or "bias" not in arrays:
            raise ValueError("invalid model file")

        input_shape = details.get("input_shape", arrays["weights"].shape[0])
        num_classes = details.get("num_classes", len(arrays["bias"]))
        random_state = details.get("random_state")
        model = cls(input_shape, num_classes, random_state)
        model.weights = arrays["weights"].astype(float)
        model.bias = arrays["bias"].astype(float)
        return model

    @staticmethod
    def conv2d(image, kernels, stride=1, padding=0):
        image = np.asarray(image, dtype=float)
        kernels = np.asarray(kernels, dtype=float)

        if image.ndim == 2:
            image = image[:, :, None]

        if kernels.ndim == 2:
            kernels = kernels[:, :, None, None]
        elif kernels.ndim == 3:
            kernels = kernels[:, :, :, None]

        if image.ndim != 3 or kernels.ndim != 4:
            raise ValueError("invalid image or kernel shape")

        if kernels.shape[2] != image.shape[2]:
            raise ValueError("image and kernel channels do not match")

        if stride < 1 or padding < 0:
            raise ValueError("invalid stride or padding")

        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))
        kernel_height, kernel_width, channels, filters = kernels.shape
        output_height = (image.shape[0] - kernel_height) // stride + 1
        output_width = (image.shape[1] - kernel_width) // stride + 1

        if output_height < 1 or output_width < 1:
            raise ValueError("kernel is larger than image")

        output = np.empty((output_height, output_width, filters))

        for row in range(output_height):
            for column in range(output_width):
                top = row * stride
                left = column * stride
                area = image[top : top + kernel_height, left : left + kernel_width]
                output[row, column] = np.sum(
                    area[:, :, :, None] * kernels,
                    axis=(0, 1, 2),
                )

        return output

    @staticmethod
    def max_pool2d(feature_map, pool_size=2, stride=None):
        feature_map = np.asarray(feature_map)
        remove_channel = feature_map.ndim == 2

        if remove_channel:
            feature_map = feature_map[:, :, None]

        if stride is None:
            stride = pool_size

        if feature_map.ndim != 3 or pool_size < 1 or stride < 1:
            raise ValueError("invalid pooling arguments")

        output_height = (feature_map.shape[0] - pool_size) // stride + 1
        output_width = (feature_map.shape[1] - pool_size) // stride + 1

        if output_height < 1 or output_width < 1:
            raise ValueError("pool window is larger than feature map")

        output = np.empty(
            (output_height, output_width, feature_map.shape[2]),
            dtype=feature_map.dtype,
        )

        for row in range(output_height):
            for column in range(output_width):
                top = row * stride
                left = column * stride
                area = feature_map[top : top + pool_size, left : left + pool_size]
                output[row, column] = area.max(axis=(0, 1))

        if remove_channel:
            return output[:, :, 0]

        return output

    def initialize(self, feature_count):
        self.weights = np.zeros((feature_count, self.num_classes))
        self.bias = np.zeros(self.num_classes)

    @staticmethod
    def make_shape(shape):
        if shape is None:
            return None

        if isinstance(shape, (int, np.integer)):
            return (int(shape),)

        return tuple(int(value) for value in shape)

    @staticmethod
    def get_features(X):
        X = np.asarray(X, dtype=float)

        if X.ndim < 2 or len(X) == 0:
            raise ValueError("X must be a non-empty batch")

        return X.reshape(len(X), -1)

    def get_targets(self, y, sample_count):
        y = np.asarray(y)

        if len(y) != sample_count:
            raise ValueError("X and y must have equal lengths")

        if y.ndim == 1:
            labels = y.astype(int)

            if not np.array_equal(labels, y):
                raise ValueError("labels must be integers")

            return self.one_hot(labels, self.num_classes), labels

        if y.ndim == 2 and y.shape[1] == self.num_classes:
            return y.astype(float), y.argmax(axis=1)

        raise ValueError("y must contain labels or one-hot rows")

    def check_fitted(self):
        if self.weights is None or self.bias is None:
            raise RuntimeError("model is not fitted")


cnn = CNN

__all__ = ["CNN", "cnn"]
