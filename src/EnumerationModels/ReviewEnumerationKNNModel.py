import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import src.constants as constants
from src.PreProcessing import create_df_with_all_features, svm_under_sampling, under_sampling, parse_all_features
from src.Utils import divide, read_featured_data_from_csv, split_x_y

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def get_samples(df: pd.DataFrame):
    new_df = df.loc[:, [constants.letter, constants.words, constants.sentences, constants.rate]]
    return new_df


"""
This model is a Decision Tree based model that will get multiple enumerations of a review as a features:
1. The length of the review in terms of characters
2. The number of words used
3. The number of sentences used
"""


class ReviewEnumeratiohKNNModel:
    def __init__(self, df: pd.DataFrame, k: int, p: int, undersamepling_func):
        temp_df = get_samples(df)
        train, test = undersamepling_func(temp_df)
        self.train_x, self.train_y = split_x_y(train)
        self.test_x, self.test_y = split_x_y(test)
        self.clf = KNeighborsClassifier(n_neighbors=k, p=p, weights='distance')

    def fit_model(self):
        self.clf = self.clf.fit(self.train_x, self.train_y)
        # self.plot_tree()

    def predict(self, length):
        return self.clf.predict(length)

    def test_and_plot(self):
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        titles_options = [
            ("Review Enumeration KNN Confusion matrix, without normalization", None),
            ("Review Enumeration KNN Normalized confusion matrix", "true"),
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


# df = read_database_from_csv(csv_file='Data/tripadvisor_hotel_reviews.csv')
# temp_df = create_df_with_all_features(df)
# temp_df.to_csv(path_or_buf='Data/tripdavisor_featured_data.csv')

df = read_featured_data_from_csv(csv_file='../../Data/tripdavisor_featured_data.csv')
df = parse_all_features(df)

#
f = under_sampling
g = svm_under_sampling
k = 6
p = 2
#model = ReviewEnumeratiohKNNModel(df, k, p, f)
model = ReviewEnumeratiohKNNModel(df, k, p, g)
model.fit_model()
# # # data = pd.DataFrame([600], columns=["Length"])
# # # print(model.predict(data))
# model.plot_tree()
model.test_and_plot()
