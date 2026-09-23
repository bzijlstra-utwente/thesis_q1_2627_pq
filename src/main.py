import numpy as np
import pandas as pd

from plot_functions import cov_matrix_plot

DATA_PATH = "data/example_data.xlsx"
THRESHOLD = 0.9


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


def main():
    df = pd.read_excel(DATA_PATH)

    # Drop not important columns
    df = df.iloc[:, 0:14]

    cov_matrix, _, _ = pca(df.to_numpy())

    cov_matrix_plot(cov_matrix, list(df.columns))

if __name__ == "__main__":
    main()
