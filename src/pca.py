import numpy as np
from scipy.stats import pearsonr


def standardize(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=0)

    std = np.std(data, axis=0, ddof=1)
    std[std == 0] = 1.0

    return (data - mean) / std


def calculate_explained_variance(eigenvalues: np.ndarray, threshold: float = 90) -> tuple[np.ndarray, np.ndarray, int]:
    # Calculate variance explained from the eigenvalues
    explained_variance = eigenvalues / np.sum(eigenvalues) * 100
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    # Find number of components to reach the threshold
    n_components = int(np.argmax(cumulative_variance >= threshold) + 1)

    return explained_variance, cumulative_variance, n_components


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


def get_significant_principal_components(
    eigenvectors: np.ndarray, eigenvalues: np.ndarray
) -> np.ndarray:
    _, _, n_components = calculate_explained_variance(eigenvalues)
    significant_pc = eigenvectors[:, :n_components]

    return significant_pc


def compute_correlation_metric(
    data: np.ndarray, significant_pc: np.ndarray, eigenvalues: np.ndarray
) -> np.ndarray:
    # Project measurment data to PCA space
    Z = standardize(data)  # Why use standardized data for projection?
    projected_data = Z @ significant_pc
    # Calculate Pearson correlation between all parameters and PCs
    pearson_correlation = np.empty((data.shape[1], projected_data.shape[1]))
    for i, parameter in enumerate(data.T):
        for j, pc in enumerate(projected_data.T):
            result = pearsonr(parameter, pc)
            pearson_correlation[i][j] = result.correlation
    # Multiply the Pearson correlation factor (r) with the explained varaince of each PC
    explained_variance, _, _ = calculate_explained_variance(eigenvalues)
    correlation_metric = np.empty(pearson_correlation.shape)
    for i, col in enumerate(pearson_correlation.T):
        correlation_metric[:, i] = col * explained_variance[i]

    return correlation_metric
