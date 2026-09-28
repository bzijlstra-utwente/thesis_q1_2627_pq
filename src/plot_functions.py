import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm


def scree_plot(eigenvalues: np.ndarray, threshold: float = 90):
    # Calculate variance explained from the eigenvalues
    explained_variance = eigenvalues / np.sum(eigenvalues) * 100
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    # PCs indices
    components = np.arange(1, len(eigenvalues) + 1)
    # Find number of components to reach the threshold
    n_components = int(np.argmax(cumulative_variance >= threshold) + 1)

    # ---------------------------------------------------------
    # Bar plot: explained variance
    # ---------------------------------------------------------

    # Create figure and axes
    _, ax1 = plt.subplots(figsize=(8, 5))

    # Make components before threshold blue, rest gray
    bar_colors = [
        "steelblue" if component <= n_components else "dimgray"
        for component in components
    ]

    ax1.bar(
        components, explained_variance, color=bar_colors, label="Explained Variance"
    )

    ax1.set_xlabel("Principal Component Index")
    ax1.set_ylabel("Variance Explained (%)")

    # ---------------------------------------------------------
    # Second y-axis: cumulative variance
    # ---------------------------------------------------------

    ax2 = ax1.twinx()

    ax2.plot(
        components,
        cumulative_variance,
        color="darkred",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Cumulative Variance",
    )

    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_ylim(0, 105)

    # ---------------------------------------------------------
    # 95% threshold
    # ---------------------------------------------------------

    ax2.axhline(threshold, color="red", linestyle="--", linewidth=1)

    ax1.axvline(n_components, color="red", linestyle="--", linewidth=1)

    # Add threshold text
    ax2.text(n_components + 1, threshold - 5, f"Threshold {threshold}%", color="red")

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    ax1.set_xticks(components)

    # Show every 5th component on the x-axis
    # ax1.set_xticks(components[::5])

    ax1.grid(axis="both", linestyle=":", alpha=0.5)

    ax1.set_title("Scree Plot")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.show()


def cov_matrix_plot(cov_matrix: np.ndarray, parameter_names: list[str]) -> None:
    # Plot
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cov_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        xticklabels=parameter_names,
        yticklabels=parameter_names,
        cbar_kws={"label": "Covariance"},
    )

    plt.title("Covariance Matrix")
    plt.xlabel("Variables")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()


def plot_signals_over_time(
    signals: np.ndarray,
    time: np.ndarray,
    signal_titles: list[str] | None = None,
    ncols: int = 2,
) -> None:
    """
    Plot signals over time.

    Parameters
    ----------
    signals : np.ndarray
        Array of shape (n, p), where p is the number of signals.
        A maximum of 14 signals is allowed.

    time : np.ndarray
        Array of shape (n,) containing the time values.

    signal_titles : list[str] | None
        List of strings which are the corresponding titles of the sub-plot for the plotted signals.
        If no list is provided, the default sub-plot title will be 'Signal [i]'.

    ncols : int
        Sets the number of columns for the sub-plots. The default number of columns is 2.
    """

    # Validate inputs
    if signals.ndim != 2:
        raise ValueError("signals must have shape (n, p)")

    if time.ndim != 1:
        raise ValueError("time must have shape (n,)")

    n, p = signals.shape

    if p > 14:
        raise ValueError("A maximum of 14 signals is allowed.")

    if time.shape[0] != n:
        raise ValueError(
            "The number of time points must match the number of parameter values."
        )

    if signal_titles and len(signal_titles) != p:
        raise ValueError(
            "The number of parameter names must match the number of signals"
        )

    nrows = int(np.ceil(p / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 3 * nrows),
        sharex=True,
    )

    # Make axes iterable even if there is only one subplot
    axes = np.atleast_1d(axes).ravel()

    for i in range(p):
        axes[i].plot(time, signals[:, i])

        if signal_titles is None:
            axes[i].set_title(f"Signal {i + 1}")
        else:
            axes[i].set_title(signal_titles[i])
        axes[i].set_ylabel("Value")
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(p, len(axes)):
        axes[i].set_visible(False)

    # Only the bottom row needs the x-label
    for ax in axes[-ncols:]:
        ax.set_xlabel("Time")

    fig.suptitle("signals over time")
    fig.tight_layout()

    plt.show()


def plot_distributions(data: np.ndarray) -> None:
    """
    Plot the probability density distributions of multiple parameters.

    A histogram normalized to probability density is plotted for each
    parameter. A normal distribution fitted to the parameter data is
    overlaid as a line.

    Parameters
    ----------
    data : np.ndarray
        Array with shape (p, n), where p is the number of parameters
        and n is the number of observations. A maximum of 14 parameters
        is allowed.

    Raises
    ------
    ValueError
        If data does not have shape (p, n) or contains more than
        14 parameters.
    """

    if data.ndim != 2:
        raise ValueError("data must have shape (p, n)")

    p, _ = data.shape

    if p > 14:
        raise ValueError("A maximum of 14 parameters is allowed.")

    ncols = 2
    nrows = int(np.ceil(p / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12, 3 * nrows),
    )

    axes = np.atleast_1d(axes).ravel()

    for i in range(p):
        # Plot probability density
        axes[i].hist(
            data[i],
            bins=30,
            density=True,
            edgecolor="black",
            alpha=0.7,
            label="Data",
        )

        # Fit normal distribution
        mu, sigma = norm.fit(data[i])

        # Create x-values for normal distribution
        x = np.linspace(
            data[i].min(),
            data[i].max(),
            200,
        )

        # Calculate probability density
        pdf = norm.pdf(x, mu, sigma)

        # Plot normal distribution
        axes[i].plot(
            x,
            pdf,
            linewidth=2,
            label=f"Normal ($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})",
        )

        axes[i].set_title(f"Parameter {i + 1}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Probability density")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    # Hide unused subplots
    for i in range(p, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Parameter distributions")
    fig.tight_layout()

    plt.show()


def plot_number_grid(
    data: np.ndarray,
    labels_y: list[str] | None = None,
    add_pc_labels_x: bool = True,
    title: str | None = None,
    numbers_in_cell: bool = False,
    ax=None,
):

    single_plot = None
    if ax is None:
        fig, ax = plt.subplots()
        single_plot = True
    else:
        fig = ax.figure
        single_plot = False

    image = ax.imshow(data, cmap="RdBu_r")

    # Add numbers to each cell
    if numbers_in_cell:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center")

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add labels to the y-axis
    if labels_y:
        ax.set_yticks(np.arange(len(labels_y)))
        ax.set_yticklabels(labels_y)

    # Add labels to the x-axis
    if add_pc_labels_x:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(
            [f"PC {i}" for i in range(1, data.shape[1] + 1)], rotation=45, ha="right"
        )

    # Add title
    if title:
        ax.set_title(title, x=-0.2, fontweight="bold")

    if single_plot:
        # Colorbar
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(r"$r \times \%var$")

        plt.show()

    return ax


def plot_multiple_number_grids(data: np.ndarray, labels_y: list[str]):
    _, axes = plt.subplots(3, 1, figsize=(8, 12))

    plot_number_grid(data, labels_y, ax=axes[0], title="TEST1")
    plot_number_grid(data, labels_y, ax=axes[1], title="TEST2")
    plot_number_grid(data, labels_y, ax=axes[2], title="TEST3")

    # Only show x-axis labels on bottom plot
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[1].tick_params(axis="x", labelbottom=False)
    axes[0].tick_params(axis="x", bottom=False)
    axes[1].tick_params(axis="x", bottom=False)

    plt.tight_layout()
    plt.show()
