import numpy as np

from src.main import standardize


def test_standardize_mean():
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]
    )

    Z = standardize(X)

    np.testing.assert_allclose(np.mean(Z, axis=0), 0, atol=1e-12)


def test_standardize_std():
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]
    )

    Z = standardize(X)

    np.testing.assert_allclose(np.std(Z, axis=0, ddof=1), 1, atol=1e-12)
