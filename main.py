import numpy as np
import pandas as pd

DATA_PATH = "data/example_data.xlsx"
THRESHOLD = 0.9

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

    # Get the number of PCs to get a contained variance of >= 0.9
    contained_variance = 0
    num_pc = 0
    while contained_variance < 0.9:
        num_pc += 1
        contained_variance = sum(S[:num_pc]) / sum(S)
       
    # print(contained_variance)
    # print(num_pc)

    print(Vt.shape)


if __name__ == "__main__":
    main()
