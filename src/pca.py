import numpy as np
from scipy.stats import pearsonr


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Standardizes the input data by subtracting the mean and dividing by the standard deviation.
    The sample standard deviation is used.

    Parameters
    ----------
    data : np.ndarray
        Data that is measured by multiple PQMs and combined into one array. Dimensions: (n, mk)

    Returns
    -------
    np.ndarray
        The standardized data. Dimensions: (n, mk)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    std[std == 0] = 1.0

    return (data - mean) / std


def calculate_explained_variance(
    eigenvalues: np.ndarray, threshold: float = 90
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    This function calculates the explained variance. The cumulative is also calculated from this explained variance.
    Finally the number of components (PCs) needed to reach a percentage of explained variance is determined.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues that belong to the eigenvectors of the covariance matrix in descending order.

        Dimension: (mk,)
    threshold : float, optional
        The percentage of explained variance that needs to be reached, by default 90 (%).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        - Explained variance (dim: (mk,)): The explained variance for each eigenvalue
        - Cumulative variance (dim: (mk,)): e.g. (50, 60, 75, ..., 100)
        - Number of components: The number of PCs that are significant (e.g. 7)

    """
    # Calculate explained variance for each eigenvalue
    explained_variance = eigenvalues / np.sum(eigenvalues) * 100
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    # Find number of components to reach the threshold
    n_components = int(np.argmax(cumulative_variance >= threshold) + 1)

    return explained_variance, cumulative_variance, n_components


def pca(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perfoms the core of the Principal Component Analysis (PCA) algorithm.

    First the data is standardized. Then the covariance matrix is calculated.

    The eigenvectors of the covariance matrix are the Principal Components (PC).
    The eigenvalues determine the significance of each PC.

    The explained variance is calculated with the eigenvalues.
    As well as the number of significan PCs.

    Parameters
    ----------
    data : np.ndarray
        Data that is measured by multiple PQMs and combined into one array.

        Dimension: (n, mk)
                - n: The data collected at each timestamp. The rows of the array.
                - m: The number of PQMs used in the measurement campaign in a sub-system
                - k: The number of parameters a single PQM measures.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - Explained variance (dim: (mk,)): The explained variance for each eigenvalue
        - Significant PCs (dim: (mk, n_c)): The n_c most significant PCs
    """
    # Standardize the columns
    Z = standardize(data)
    # Get the covariance matrix
    C = np.cov(Z, rowvar=False)

    # Get the eigenvalues and -vectors
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Swap the order from low->high to high->low
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Determine the significant PCs
    explained_variance, _, n_components = calculate_explained_variance(eigenvalues)
    significant_pc = eigenvectors[:, :n_components]

    return explained_variance, significant_pc


def compute_correlation_metric(
    data: np.ndarray, significant_pc: np.ndarray, explained_variance: np.ndarray
) -> np.ndarray:
    """
    The correlation metric that is calculated the Pearson correlation factor (r) 
    between each measured parameter and PC multiplied by the explained variance of that PC.

    Parameters
    ----------
    data : np.ndarray
        Data that is measured by multiple PQMs and combined into one array. Dimensions: (n, mk)
    significant_pc : np.ndarray
        The most significant PCs, which are determined by PCA. Dimensions: (mk, n_c) 
    explained_variance : np.ndarray
        The explained variance for each eigenvalue. Dimensions: (mk,)

    Returns
    -------
    np.ndarray
        A matrix of the calculated correlation metric calculated between each parameter and PC.
        Dimensions: (mk, n_c)
    """
    # Project measurment data to PCA space
    Z = standardize(data)  # Why use standardized data for projection?
    projected_data = Z @ significant_pc
    # Calculate Pearson correlation between all parameters and PCs
    pearson_correlation_factors = np.empty((data.shape[1], projected_data.shape[1]))
    for i, parameter in enumerate(data.T):
        for j, pc in enumerate(projected_data.T):
            result = pearsonr(parameter, pc)
            pearson_correlation_factors[i][j] = result.correlation
    # Multiply the Pearson correlation factor (r) with the explained variance of each PC
    correlation_metric = np.empty(pearson_correlation_factors.shape)
    for i, col in enumerate(pearson_correlation_factors.T):
        correlation_metric[:, i] = col * explained_variance[i]

    return correlation_metric
