from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.support.cnn import CNN


class CNNRegressionTests(unittest.TestCase):
    def test_training_and_saving(self):
        X = np.array(
            [
                [-2.0, -1.0],
                [-1.5, -2.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [1.5, 2.0],
                [2.0, 1.0],
            ]
        )
        y = np.array([0, 0, 0, 1, 1, 1])

        model = CNN(input_shape=(2,), num_classes=2, random_state=7)
        history = model.fit(
            X,
            y,
            learning_rate=0.2,
            max_epochs=200,
            tolerance=1e-8,
            batch_size=3,
        )

        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertEqual(model.evaluate(X, y)["accuracy"], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "toy_model")
            loaded_model = CNN.load(path)

        np.testing.assert_allclose(
            loaded_model.predict_proba(X),
            model.predict_proba(X),
        )
        np.testing.assert_array_equal(loaded_model.predict(X), y)

    def test_convolution_and_pooling(self):
        image = np.arange(1, 17).reshape(4, 4)
        kernel = np.array([[1, 0], [0, -1]])

        result = CNN.conv2d(image, kernel)
        expected = np.full((3, 3, 1), -5.0)
        np.testing.assert_array_equal(result, expected)

        result = CNN.max_pool2d(image, pool_size=2, stride=2)
        expected = np.array([[6, 8], [14, 16]])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
