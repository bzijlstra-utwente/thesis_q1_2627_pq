import numpy as np


def standardize(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=0)

    std = np.std(data, axis=0, ddof=1)
    std[std == 0] = 1.0

    return (data - mean) / std


def pca(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Standardize the columns
    Z = standardize(data)
    # Get the covariance matrix
    C = np.cov(Z.T, ddof=1)

    # Get the eigenvectors and -values
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Swap to order from low->high to high->low
    eigenvectors = eigenvectors[:, ::-1]
    eigenvalues = eigenvalues[::-1]

    return C, eigenvalues, eigenvectors
