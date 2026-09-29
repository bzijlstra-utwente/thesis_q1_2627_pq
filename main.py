import pandas as pd

from src.pca import (
    compute_correlation_metric,
    pca,
)
from src.plot_functions import (
    cov_matrix_plot,
    plot_distributions,
    plot_multiple_number_grids,
    plot_number_grid,
    plot_signals_over_time,
    scree_plot,
)

DATA_PATH = "data/example_data.xlsx"
THRESHOLD = 0.9

ABBREVATIONS_COLUMN_NAMES = {
    "Current[A]": "I",
    "Voltage[mV]": "V",
    "Active Power[W]": "P",
    "Reactive Power[var]": "Q",
    "Frequency[Hz]": "f",
    "Power factor[/1000]": "PF",
    "Phase Angle[0.1deg]": "ϕ",
    "Mean Apparent Power[VA]": "S",
    "Forward active energy[0.1pulse]": r"$E_{P,for}$",
    "Reverse active energy[0.1pulse]": r"$E_{P,rev}$",
    "Absolute active energy[0.1pulse]": r"$E_{P,abs}$",
    "Forward reactive energy[0.1pulse]": r"$E_{Q,for}$",
    "Reverse reactive energy[0.1pulse]": r"$E_{Q,rev}$",
    "Absolute reactive energy[0.1pulse]": r"$E_{Q,abs}$",
}


def algorithm():
    df = pd.read_excel(DATA_PATH)

    # Get the pararmeter columns from the df
    df_parameters = df.iloc[:, 0:14]
    data = df_parameters.to_numpy()

    explained_variance, principal_components = pca(data)

    correlation_metric_matrix = compute_correlation_metric(
        data, principal_components, explained_variance
    )

    plot_number_grid(
        correlation_metric_matrix,
        list(ABBREVATIONS_COLUMN_NAMES.values()),
        title="TEST TITLE",
    )
    # plot_multiple_number_grids(
    #     correlation_metric_matrix, list(ABBREVATIONS_COLUMN_NAMES.values())
    # )


if __name__ == "__main__":
    algorithm()
