import pandas as pd
import numpy as np

DATA_PATH = "data/example_data.xlsx"

def standardize(data_matrix: np.ndarray) -> np.ndarray:
    # Equivalent to data_matrix.mean(0)
    mean = data_matrix.sum(0) / len(data_matrix)
    # Equivalent to data_matrix.std(0)
    variance = ((data_matrix - mean)**2).sum(0) / len(data_matrix)
    std = np.sqrt(variance)
    # Standardize
    return (data_matrix - mean) / std
    
def covariance_matrix(standardized_matrix: np.ndarray) -> np.ndarray:
    transposed = standardized_matrix.transpose()

    return (1 / (len(standardized_matrix) - 1)) * np.dot(transposed, standardized_matrix)
    


if __name__ == "__main__":
    df = pd.read_excel(DATA_PATH)
    
    # Drop not important columns
    df = df.iloc[:, 0:14]

    # Do the data processing in NumPy
    X = df.to_numpy()

    Z = standardize(X)
    C = covariance_matrix(Z)

    