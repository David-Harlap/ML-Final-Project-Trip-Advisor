import copy

from Utils import read_database_from_csv, pd, divide
import constants

SENTIMENT_WORDS_POSITIVE = [
    'Excellent', 'Outstanding', 'Wonderful', 'Fantastic', 'Great', 'Amazing', 'Delightful', 'Terrific',
    'Impressive', 'Superb', 'Exceptional', 'Pleasant', 'Satisfying', 'Marvelous', 'Good', 'Love', 'Perfect',
    'Beautiful', 'Awesome', 'Brilliant', 'Happy', 'Enjoyable', 'Remarkable', 'Fabulous', 'Phenomenal',
    'Thrilling', 'Admirable', 'Splendid', 'Exquisite', 'Refreshing'
]

SENTIMENT_WORDS_NEGATIVE = [
    'Terrible', 'Disappointing', 'Awful', 'Horrible', 'Bad', 'Unpleasant', 'Mediocre', 'Frustrating',
    'Poor', 'Subpar', 'Displeasing', 'Unsatisfactory', 'Inferior', 'Unacceptable', 'Dissatisfying',
    'Hated', 'Abysmal', 'Disgusting', 'Unfortunate', 'Dreadful', 'Regrettable', 'Repulsive', 'Wretched',
    'Displeased', 'Horrendous', 'Miserable', 'Not good', 'Repugnant', 'Repellant', 'Gross'
]


def calculate_word_sum(string):
    words = string.split()
    word_sum = sum(len(word) for word in words)
    return word_sum


def count_words_in_string(string, word_list):
    words = string.split()
    count = sum(word in word_list for word in words)
    return count

def create_review_length_features(df: pd.DataFrame):
    data = []
    cols = [constants.length, constants.rate]
    for i, row in df.iterrows():
        length = len(row[constants.review])
        rate = row[constants.rate]
        data.append([length, rate])
    length_df = pd.DataFrame(data, columns=cols)
    return length_df

def create_review_number_of_words_features(df: pd.DataFrame):
    data = []
    cols = [constants.word_count, constants.rate]
    for i, row in df.iterrows():
        words = row[constants.review].split(' ')
        length = len(words)
        rate = row[constants.rate]
        data.append([length, rate])
    count_df = pd.DataFrame(data, columns=cols)
    return count_df

def under_sampling(df: pd.DataFrame):
    data = []
    cols = [constants.word_count, constants.rate]
    copy_df = copy.deepcopy(df)
    df_1 = copy_df[copy_df[constants.rate] == '1'].sample(frac=1).head(constants.min_class_samples)
    df_2 = copy_df[copy_df[constants.rate] == '2'].sample(frac=1).head(constants.min_class_samples)
    df_3 = copy_df[copy_df[constants.rate] == '3'].sample(frac=1).head(constants.min_class_samples)
    df_4 = copy_df[copy_df[constants.rate] == '4'].sample(frac=1).head(constants.min_class_samples)
    df_5 = copy_df[copy_df[constants.rate] == '5'].sample(frac=1).head(constants.min_class_samples)
    dfs = [df_1, df_2, df_3, df_4, df_5]
    train = []
    test = []
    for df_i in dfs:
        to_train = df_i.iloc[:constants.train_batch_size_per_class, :]
        to_test = df_i.iloc[constants.train_batch_size_per_class:, :]
        train.append(to_train)
        test.append(to_test)
    train_set = pd.concat(train, axis=0)
    test_set = pd.concat(test, axis=0)
    return train_set, test_set

# df = read_database_from_csv(csv_file='Data/tripadvisor_hotel_reviews.csv')
# print(create_review_length_features(df))
# output = create_review_number_of_words_features(df).sample(frac=1)
# print(output)
# a, b, c, d = divide(80, df)
# print(a.iloc[[1]])
# print(b.iloc[[1]])