import numpy as np
import pandas as pd

DATA_PATH = "data/example_data.xlsx"

def main():
    df = pd.read_excel(DATA_PATH)

    # Drop not important columns
    df = df.iloc[:, 0:14]

    # Do the data processing in NumPy
    X = df.to_numpy()

    # Standardize the columns
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    std[std==0] = 1.0
    Z = (X - mean) / std

    # Get the covariance matrix
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)

    print(Vt)


if __name__ == "__main__":
    main()
