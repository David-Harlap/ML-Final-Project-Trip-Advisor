import pandas as pd


def read_database_from_csv(csv_file):
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['Message', 'Rate']  # Set column names
    return df