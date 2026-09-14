import numpy as np
import pandas as pd

DATA_PATH = "data/example_data.xlsx"

def main():
    df = pd.read_excel(DATA_PATH)

    # Drop not important columns
    df = df.iloc[:, 0:14]

    # Standardize the columns of the df
    df_z = pd.DataFrame()
    for col_name in df.columns:
        df_z[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()

    


if __name__ == "__main__":
    main()
