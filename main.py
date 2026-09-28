import pandas as pd

from src.pca import (
    compute_correlation_metric,
    get_significant_principal_components,
    pca,
)
from src.plot_functions import (
    cov_matrix_plot,
    plot_distributions,
    plot_signals_over_time,
    scree_plot,
)

DATA_PATH = "data/example_data.xlsx"
THRESHOLD = 0.9


def algorithm():
    df = pd.read_excel(DATA_PATH)

    # Get the timestamp for each measurement
    epoch_times = df.iloc[:, 18].to_numpy()
    # Get the pararmeter columns from the df
    df_parameters = df.iloc[:, 0:14]
    data = df_parameters.to_numpy()

    _, eigenvalues, eigenvectors = pca(data)

    principal_components = get_significant_principal_components(
        eigenvectors, eigenvalues
    )

    compute_correlation_metric(data, principal_components, eigenvalues)

if __name__ == "__main__":
    algorithm()
