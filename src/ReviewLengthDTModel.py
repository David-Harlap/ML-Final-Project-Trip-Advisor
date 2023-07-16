import pandas as pd
from sklearn import tree
import numpy as np

import constants
from PreProcessing import create_review_length_features, under_sampling
from Utils import divide, read_database_from_csv, split_x_y
from constants import length

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import sklearn.metrics


class ReviewLengthDTModel:
    def __init__(self, df: pd.DataFrame, divide_precent: int, depth: int):
        length_df = create_review_length_features(df)
        train, test = under_sampling(length_df)
        y = length_df[constants.rate]
        X = length_df.drop(columns=[constants.rate])
        # self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, shuffle=True, random_state=0,
        #                                                                         test_size=0.3)
        self.train_x, self.train_y = split_x_y(train)
        self.test_x, self.test_y =  split_x_y(test)
        self.clf = tree.DecisionTreeClassifier(max_depth=depth)

    def fit_model(self):
        self.clf = self.clf.fit(self.train_x, self.train_y)
        # self.plot_tree()

    def predict(self, length):
        return self.clf.predict(length)

    def plot_tree(self):
        fig, ax = plt.subplots(figsize=(20, 20))  # whatever size you want
        tree.plot_tree(self.clf, ax=ax, fontsize=8)
        plt.show()

    def test_and_plot(self):
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        titles_options = [
            ("Review Length decision tree Confusion matrix, without normalization", None),
            ("Review Length decision tree Normalized confusion matrix", "true"),
        ]
        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay.from_estimator(
                self.clf,
                self.test_x,
                self.test_y,
                display_labels=self.clf.classes_,
                cmap=plt.cm.Blues,
                normalize=normalize,
            )
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

        plt.show()


df = read_database_from_csv(csv_file='Data/tripadvisor_hotel_reviews.csv')
precent = 80
depth = 8
model = ReviewLengthDTModel(df, precent, depth)
model.fit_model()
# data = pd.DataFrame([600], columns=["Length"])
# print(model.predict(data))
# model.plot_tree()
model.test_and_plot()
