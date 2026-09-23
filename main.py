import pandas as pd

from src.pca import pca
from src.plot_functions import cov_matrix_plot

DATA_PATH = "data/example_data.xlsx"
THRESHOLD = 0.9


def main():
    df = pd.read_excel(DATA_PATH)

    # Drop not important columns
    df = df.iloc[:, 0:14]

    cov_matrix, _, _ = pca(df.to_numpy())

    cov_matrix_plot(cov_matrix, list(df.columns))


if __name__ == "__main__":
    main()
