import pandas as pd
from copy import deepcopy

import src.constants as constants


def read_database_from_csv(csv_file):
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['Review', 'Rate']  # Set column names
    return df.iloc[1:]


def read_featured_data_from_csv(csv_file: str):
    df = pd.read_csv(csv_file, header=None)
    df.columns = [constants.letter, constants.words, constants.sentences, constants.adjective, constants.comp_score,
                  constants.pos_score, constants.neg_score, constants.neu_score, constants.readability, constants.rate]
    df = shuffle(df.iloc[1:])
    return df


def shuffle(df: pd.DataFrame):
    shuffled_df = deepcopy(df)
    shuffled_df = shuffled_df.sample(frac=1)
    return shuffled_df


def divide(divide_precent: int, df: pd.DataFrame):
    shuffled_df = shuffle(df)
    if divide_precent < 1 or divide_precent > 100:
        divide_precent = 70
    limit = int(len(df) * float(divide_precent / 100))
    train = shuffled_df.iloc[:limit, :]
    test = shuffled_df.iloc[limit:, :]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_y = train["Rate"]
    train_x = train.drop(columns=["Rate"])
    test_y = test["Rate"]
    test_x = test.drop(columns=["Rate"])
    return train_x, train_y, test_x, test_y


def split_x_y(df: pd.DataFrame):
    df = df.reset_index(drop=True)
    y = df["Rate"]
    x = df.drop(columns=["Rate"])
    return x, y
