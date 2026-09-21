import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "data/example_data.xlsx"
THRESHOLD = 0.9

def scree_plot(eigenvalues: np.ndarray, threshold: float = 90):
    # Calculate variance explained from the eigenvalues
    explained_variance = eigenvalues / np.sum(eigenvalues) * 100
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    # PCs indices
    components = np.arange(1, len(eigenvalues)+1)
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
    #ax1.set_xticks(components[::5])

    ax1.grid(axis="both", linestyle=":", alpha=0.5)

    ax1.set_title("Scree Plot")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    plt.show()


def standardize(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=0)

    std = np.std(data, axis=0, ddof=1)
    std[std == 0] = 1.0

    return (data - mean) / std


def main():
    df = pd.read_excel(DATA_PATH)

    # Drop not important columns
    df = df.iloc[:, 0:14]

    # Do the data processing in NumPy
    X = df.to_numpy()
    # Standardize the columns
    Z = standardize(X)
    # Get the covariance matrix
    C = np.cov(Z.T, ddof=1)

    # Get the eigenvectors and -values
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Swap to order from low->high to high->low
    eigenvectors = eigenvectors[::-1]
    eigenvalues = eigenvalues[:,::-1]


if __name__ == "__main__":
    main()
