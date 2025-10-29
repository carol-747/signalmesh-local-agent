"""
Sample data analysis script for demonstration purposes.
"""

import pandas as pd
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with loaded data
    """
    return pd.read_csv(file_path)


def calculate_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with statistical measures
    """
    return {
        'mean': df.mean(),
        'median': df.median(),
        'std': df.std()
    }


class DataAnalyzer:
    """Class for performing data analysis tasks."""

    def __init__(self, data: pd.DataFrame):
        """Initialize with a DataFrame."""
        self.data = data

    def filter_outliers(self, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """
        Filter outliers using z-score method.

        Args:
            column: Column name to filter
            threshold: Z-score threshold

        Returns:
            Filtered DataFrame
        """
        z_scores = np.abs((self.data[column] - self.data[column].mean()) / self.data[column].std())
        return self.data[z_scores < threshold]

    def generate_report(self) -> str:
        """Generate a summary report of the data."""
        return f"Data shape: {self.data.shape}\nColumns: {list(self.data.columns)}"


if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100)
    })

    analyzer = DataAnalyzer(data)
    print(analyzer.generate_report())
