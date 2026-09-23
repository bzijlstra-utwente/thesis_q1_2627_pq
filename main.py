import pandas as pd

from src.pca import pca, standardize
from src.plot_functions import (
    cov_matrix_plot,
    plot_distributions,
    plot_signals_over_time,
    scree_plot,
)

DATA_PATH = "data/example_data.xlsx"
THRESHOLD = 0.9


def main():
    df = pd.read_excel(DATA_PATH)

    # Get the timestamp for each measurement
    epoch_times = df.iloc[:, 18].to_numpy()
    # Get the pararmeter columns from the df
    df_parameters = df.iloc[:, 0:14]
    data = df_parameters.to_numpy()

    cov_matrix, eigenvalues, eigenvectors = pca(data)

    # cov_matrix_plot(cov_matrix, list(df.columns))
    # scree_plot(eigenvalues)
    # plot_parameters_over_time(data, epoch_times, list(df_parameters.columns))

    # # BlaBla
    # Z = standardize(data)
    # pca_dimensions = Z @ eigenvectors
    # plot_signals_over_time(
    #     pca_dimensions, epoch_times, [f"PC {i}" for i in range(1, 15)]
    # )

    plot_distributions(data.T)


if __name__ == "__main__":
    main()
