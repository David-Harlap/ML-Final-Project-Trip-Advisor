import constants as constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PreProcessing import under_sampling, parse_all_features
from Utils import read_featured_data_from_csv, split_x_y
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay


def get_samples(df: pd.DataFrame):
    new_df = df.loc[:, [constants.adjective, constants.verb, constants.noun, constants.rate]]
    return new_df


"""
This model is a Decision Tree based model that will get multiple enumerations of a review as a features:
1. The length of the review in terms of characters
2. The number of words used
3. The number of sentences used
"""


class ReviewSentimentRF_Model:
    def __init__(self, df: pd.DataFrame, max_depth: int):
        temp_df = get_samples(df)
        train, test = under_sampling(temp_df)
        self.train_x, self.train_y = split_x_y(train)
        self.test_x, self.test_y = split_x_y(test)
        self.clf = RandomForestClassifier(max_depth=max_depth, random_state=0)
        self.max_depth = max_depth

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
        ax.set_xlabel('Adjective Score', fontweight='bold')
        ax.set_ylabel('Verb Score', fontweight='bold')
        ax.set_zlabel('Noun Score', fontweight='bold')
        X = [[], [], [], [], []]
        Y = [[], [], [], [], []]
        Z = [[], [], [], [], []]
        color = ['black', 'gray', 'blue', 'green', 'red']
        shape = ['.', 'o', 'd', 'D', '*']
        for i in range(len(self.test_x)):
            rate = self.test_y.iloc[i]
            adj = self.test_x.iloc[i][constants.pos_score]
            verb = self.test_x.iloc[i][constants.neg_score]
            noun = self.test_x.iloc[i][constants.neu_score]
            X[rate-1].append(adj)
            Y[rate-1].append(verb)
            Z[rate-1].append(noun)

        for i in range(len(color)):
            ax.scatter3D(X[i], Y[i], Z[i], color=color[i], marker=shape[i])

        plt.legend(["Rate 1", "Rate 2", "Rate 3", "Rate 4", "Rate 5"])
        plt.show()

    def test_and_plot(self):
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        titles_options = [
            (f"Review 'Point of Speech' Random Forest with max depth {self.max_depth} matrix, without normalization", None),
            (f"Review 'Point of Speech' Random Forest with max depth {self.max_depth} Normalized confusion matrix", "true"),
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

            total_samples = np.sum(disp.confusion_matrix)
            print(total_samples)
            misclassifications = total_samples - np.trace(disp.confusion_matrix)
            error_rate = misclassifications / total_samples
            print(error_rate)
            plt.text(x=0, y=5, s=f'Error Rate: {error_rate * 100:.2f}%', va='center_baseline', fontsize=12,
                     color='black', ha='center')

        plt.show()


# df = read_database_from_csv(csv_file='Data/tripadvisor_hotel_reviews.csv')
# temp_df = create_df_with_all_features(df)
# temp_df.to_csv(path_or_buf='Data/tripdavisor_featured_data.csv')

df = read_featured_data_from_csv(csv_file='../../Data/tripdavisor_featured_data.csv')
df = parse_all_features(df)
depth = 8
#
model = ReviewSentimentRF_Model(df, depth)
model.fit_model()
# # # data = pd.DataFrame([600], columns=["Length"])
# # # print(model.predict(data))
# model.plot_tree()
model.test_and_plot()
#model.plot_test_data()
