"""
Simple MNIST experiments without using existing ML training libraries.
Implements naive data download/parsing plus a few basic classifiers.
"""
from __future__ import annotations

import gzip
import math
import os
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


@dataclass
class Dataset:
    images: np.ndarray  # (N, 28*28)
    labels: np.ndarray  # (N,)


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "mnist-basic"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _read_idx(filename: Path) -> np.ndarray:
    with gzip.open(filename, "rb") as f:
        data = f.read()
    magic, dims = struct.unpack(">II", data[:8])
    if magic == 2051:  # images
        count, rows, cols = dims, *struct.unpack(">II", data[8:16])
        offset = 16
        arr = np.frombuffer(data, dtype=np.uint8, offset=offset)
        return arr.reshape((count, rows * cols)).astype(np.float32) / 255.0
    elif magic == 2049:  # labels
        count = dims
        offset = 8
        return np.frombuffer(data, dtype=np.uint8, offset=offset)
    raise ValueError(f"Unexpected magic number {magic}")


def load_mnist(data_dir: Path) -> Tuple[Dataset, Dataset]:
    for key, url in MNIST_URLS.items():
        fname = data_dir / f"{key}.gz"
        _download_file(url, fname)
    train_images = _read_idx(data_dir / "train_images.gz")
    train_labels = _read_idx(data_dir / "train_labels.gz")
    test_images = _read_idx(data_dir / "test_images.gz")
    test_labels = _read_idx(data_dir / "test_labels.gz")
    return Dataset(train_images, train_labels), Dataset(test_images, test_labels)


def train_test_slice(dataset: Dataset, train_size: int, test_size: int) -> Tuple[Dataset, Dataset]:
    return (
        Dataset(dataset.images[:train_size], dataset.labels[:train_size]),
        Dataset(dataset.images[-test_size:], dataset.labels[-test_size:]),
    )


class QDFClassifier:
    def __init__(self):
        self.means = None
        self.covs = None
        self.priors = None

    def fit(self, images: np.ndarray, labels: np.ndarray) -> None:
        classes = np.unique(labels)
        means = []
        covs = []
        priors = []
        for c in classes:
            mask = labels == c
            Xc = images[mask]
            means.append(np.mean(Xc, axis=0))
            cov = np.cov(Xc, rowvar=False) + np.eye(Xc.shape[1]) * 1e-3
            covs.append(cov)
            priors.append(len(Xc) / len(labels))
        self.means = np.stack(means)
        self.covs = covs
        self.priors = np.array(priors)
        self.classes_ = classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = []
        for mean, cov, prior in zip(self.means, self.covs, self.priors):
            inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
            diff = X - mean
            term = -0.5 * np.sum(diff @ inv * diff, axis=1)
            log_prob = term - 0.5 * math.log(det) + math.log(prior)
            scores.append(log_prob)
        scores = np.stack(scores, axis=1)
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]


def knn_predict(train_X: np.ndarray, train_y: np.ndarray, X: np.ndarray, k: int = 3) -> np.ndarray:
    preds = []
    for x in X:
        dists = np.linalg.norm(train_X - x, axis=1)
        idx = np.argsort(dists)[:k]
        vals, counts = np.unique(train_y[idx], return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)


def pca_transform(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    X_center = X - np.mean(X, axis=0)
    cov = np.cov(X_center, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order[:k]]
    return X_center @ vecs, vecs


def lda_transform(X: np.ndarray, y: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    mean_total = np.mean(X, axis=0)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros_like(Sw)
    for c in classes:
        Xc = X[y == c]
        mean_c = np.mean(Xc, axis=0)
        Sw += (Xc - mean_c).T @ (Xc - mean_c)
        diff = (mean_c - mean_total).reshape(-1, 1)
        Sb += len(Xc) * (diff @ diff.T)
    vals, vecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order[:k]].real
    X_center = X - mean_total
    return X_center @ vecs, vecs


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(pred == labels))


def main() -> None:
    data_dir = Path(os.environ.get("MNIST_HOME", "~/.cache/mnist_basic")).expanduser()
    train, test = load_mnist(data_dir)

    # Keep sizes small to finish quickly.
    train_small, test_small = train_test_slice(train, 4000, 500)

    # QDF on PCA features
    pca_train, pca_vecs = pca_transform(train_small.images, 30)
    qdf = QDFClassifier()
    qdf.fit(pca_train, train_small.labels)
    test_proj = (test_small.images - np.mean(train_small.images, axis=0)) @ pca_vecs
    qdf_pred = qdf.predict(test_proj)
    print(f"QDF-PCA accuracy: {accuracy(qdf_pred, test_small.labels):.3f}")

    # Simple KNN baseline on raw pixels.
    knn_pred = knn_predict(train_small.images, train_small.labels, test_small.images, k=3)
    print(f"KNN accuracy: {accuracy(knn_pred, test_small.labels):.3f}")

    # LDA dimensionality reduction + QDF
    lda_train, lda_vecs = lda_transform(train_small.images, train_small.labels, k=9)
    qdf_lda = QDFClassifier()
    qdf_lda.fit(lda_train, train_small.labels)
    test_lda = (test_small.images - np.mean(train_small.images, axis=0)) @ lda_vecs
    pred_lda = qdf_lda.predict(test_lda)
    print(f"QDF-LDA accuracy: {accuracy(pred_lda, test_small.labels):.3f}")


if __name__ == "__main__":
    main()
