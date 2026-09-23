import numpy as np

from src.pca import pca


def get_test_data():
    return np.array(
        [
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 4.0],
            [3.0, 3.0, 3.0],
            [4.0, 5.0, 2.0],
            [5.0, 1.0, 1.0],
        ]
    )


def test_eigenvalues_are_descending():
    X = get_test_data()

    _, eigenvalues, _ = pca(X)

    assert np.all(np.diff(eigenvalues) <= 0)


def test_eigenvectors_are_orthonormal():
    X = get_test_data()

    _, _, eigenvectors = pca(X)

    np.testing.assert_allclose(
        eigenvectors.T @ eigenvectors, np.eye(X.shape[1]), atol=1e-12
    )


def test_eigenvectors_are_correct():
    X = get_test_data()

    _, eigenvalues, eigenvectors = pca(X)

    # Reconstruct covariance matrix
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    C = np.cov(Z.T, ddof=1)

    for i in range(len(eigenvalues)):
        np.testing.assert_allclose(
            C @ eigenvectors[:, i], eigenvalues[i] * eigenvectors[:, i], atol=1e-12
        )


def test_total_variance():
    X = get_test_data()

    _, eigenvalues, _ = pca(X)

    np.testing.assert_allclose(eigenvalues.sum(), X.shape[1], atol=1e-12)
