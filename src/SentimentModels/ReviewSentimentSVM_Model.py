import pandas as pd
from sklearn.svm import SVC
import numpy as np

import src.constants as constants
from src.PreProcessing import create_df_with_all_features, svm_under_sampling, parse_all_features
from src.Utils import divide, read_featured_data_from_csv, split_x_y

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def get_samples(df: pd.DataFrame):
    new_df = df.loc[:, [constants.pos_score, constants.neg_score, constants.neu_score, constants.rate]]
    return new_df


"""
This model is a Decision Tree based model that will get multiple enumerations of a review as a features:
1. The length of the review in terms of characters
2. The number of words used
3. The number of sentences used
"""


class ReviewSentimentSVM_Model:
    def __init__(self, df: pd.DataFrame):
        temp_df = get_samples(df)
        train, test = svm_under_sampling(temp_df)
        self.train_x, self.train_y = split_x_y(train)
        self.test_x, self.test_y = split_x_y(test)
        self.clf = SVC(kernel='linear', gamma='scale', shrinking=True)

    def fit_model(self):
        self.clf = self.clf.fit(self.train_x, self.train_y)
        # self.plot_tree()

    def predict(self, length):
        return self.clf.predict(length)

    def plot_test_data(self):
        # Creating figure
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection="3d")

        # Add x, y gridlines
        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)
        plt.title("Model test data distribution")
        ax.set_xlabel('Positivity Score', fontweight='bold')
        ax.set_ylabel('Negativity Score', fontweight='bold')
        ax.set_zlabel('Neutrality Score', fontweight='bold')
        X = [[], [], [], [], []]
        Y = [[], [], [], [], []]
        Z = [[], [], [], [], []]
        color = ['black', 'gray', 'blue', 'green', 'red']
        shape = ['.', 'o', 'd', 'D', '*']
        for i in range(len(self.test_x)):
            rate = self.test_y.iloc[i]
            pos = self.test_x.iloc[i][constants.pos_score]
            neg = self.test_x.iloc[i][constants.neg_score]
            neu = self.test_x.iloc[i][constants.neu_score]
            X[rate-1].append(pos)
            Y[rate-1].append(neg)
            Z[rate-1].append(neu)

        for i in range(len(color)):
            ax.scatter3D(X[i], Y[i], Z[i], color=color[i], marker=shape[i])

        plt.legend(["Rate 1", "Rate 2", "Rate 3", "Rate 4", "Rate 5"])
        plt.show()

    def test_and_plot(self):
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        titles_options = [
            ("Review Sentiment SVM matrix, without normalization", None),
            ("Review Sentiment SVM Normalized confusion matrix", "true"),
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
model = ReviewSentimentSVM_Model(df)
model.fit_model()
# # # data = pd.DataFrame([600], columns=["Length"])
# # # print(model.predict(data))
# model.plot_tree()
#model.test_and_plot()
model.plot_test_data()
