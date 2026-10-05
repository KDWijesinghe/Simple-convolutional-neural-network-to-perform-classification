import json
from pathlib import Path

import numpy as np


class DatasetLoader:
    @staticmethod
    def prepare_images(images, normalize=True, flatten=False, dtype=np.float32):
        images = np.asarray(images, dtype=dtype)

        if images.ndim < 2:
            raise ValueError("images must contain a batch")

        if normalize and images.size > 0 and images.max() > 1:
            images = images / 255.0

        if flatten:
            images = images.reshape(len(images), -1)

        return images

    @staticmethod
    def save(path, images, labels):
        path = npz_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=np.asarray(images), y=np.asarray(labels))
        return path

    @staticmethod
    def load(path):
        with np.load(npz_path(path), allow_pickle=False) as data:
            return data["X"], data["y"]


class ImageUtils:
    @staticmethod
    def show(image, title=None, cmap="gray"):
        import matplotlib.pyplot as plt

        image = np.asarray(image)
        plt.imshow(image, cmap=cmap if image.ndim == 2 else None)

        if title:
            plt.title(title)

        plt.axis("off")
        plt.show()

    @staticmethod
    def save(image, path, title=None, cmap="gray"):
        import matplotlib.pyplot as plt

        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.asarray(image)
        figure, axis = plt.subplots()
        axis.imshow(image, cmap=cmap if image.ndim == 2 else None)

        if title:
            axis.set_title(title)

        axis.axis("off")
        figure.savefig(path, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        return path

    @staticmethod
    def save_grid(images, path, columns=5, titles=None, cmap="gray"):
        import matplotlib.pyplot as plt

        images = list(images)

        if not images or columns < 1:
            raise ValueError("provide images and at least one column")

        if titles is not None and len(titles) != len(images):
            raise ValueError("titles and images must have equal lengths")

        rows = (len(images) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, squeeze=False)

        for index, axis in enumerate(axes.flat):
            axis.axis("off")

            if index < len(images):
                image = np.asarray(images[index])
                axis.imshow(image, cmap=cmap if image.ndim == 2 else None)

                if titles is not None:
                    axis.set_title(str(titles[index]))

        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        return path


class ModelIO:
    @staticmethod
    def save(path, arrays, **metadata):
        path = npz_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {}

        for name, value in arrays.items():
            values[name] = np.asarray(value)

        values["__metadata__"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(path, **values)
        return path

    @staticmethod
    def load(path):
        with np.load(npz_path(path), allow_pickle=False) as data:
            arrays = {}

            for name in data.files:
                if name != "__metadata__":
                    arrays[name] = data[name].copy()

            metadata = {}

            if "__metadata__" in data:
                metadata = json.loads(str(data["__metadata__"].item()))

        return arrays, metadata


class Utils:
    datasets = DatasetLoader
    images = ImageUtils
    models = ModelIO


def npz_path(path):
    path = Path(path).expanduser()

    if path.suffix:
        return path

    return path.with_suffix(".npz")


__all__ = ["DatasetLoader", "ImageUtils", "ModelIO", "Utils"]
