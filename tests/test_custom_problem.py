import numpy as np

from src.main import pca


def test_custom_problem():
    X = np.array(
        [
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3],
        ]
    )

    C, eig_val, eig_vec = pca(X)

    np.testing.assert_equal(C, np.array([[1, 0.6], [0.6, 1]]))

    np.testing.assert_equal(eig_val, np.array([1.6, 0.4]))

    np.testing.assert_equal(eig_vec, (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]]))
