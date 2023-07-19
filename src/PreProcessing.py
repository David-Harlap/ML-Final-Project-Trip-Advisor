import copy
import string
from src.Utils import read_database_from_csv, pd, divide
import src.constants as constants
import textstat

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
sia = SentimentIntensityAnalyzer()


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


def count_words(review):
    words = review.split()
    word_sum = len(words)
    return len(words)


def count_sentences(review):
    sentences = nltk.sent_tokenize(review)
    return len(sentences)


def count_letters(review):
    return len(review)


def count_punctuation(review):
    punctuation_count = sum(1 for char in review if char in string.punctuation)
    return punctuation_count


def polarity_scores(review):
    return sia.polarity_scores(review)


# Calculate readability scores using the Flesch-Kincaid formula
def readability_scores(review):
    return textstat.flesch_reading_ease(review)


def count_adjective(text):
    # Tokenize the text and tag parts of speech
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    # Count the number of adjectives
    adjective_count = sum([1 for word, pos in pos_tags if pos.startswith('JJ')])
    return adjective_count


def count_words_in_string(review, word_list):
    words = review.split()
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


def create_df_with_all_features(df: pd.DataFrame):
    data = []
    cols = [constants.letter, constants.words, constants.sentences, constants.adjective, constants.comp_score,
            constants.pos_score, constants.neg_score, constants.neu_score, constants.readability, constants.rate]

    for i, row in df.iterrows():
        letters_len = int(count_letters(row[constants.review]))
        words_len = int(count_words(row[constants.review]))
        sentences_len = int(count_sentences(row[constants.review]))
        adjective = int(count_adjective(row[constants.review]))
        scores = float(polarity_scores(row[constants.review]))
        readability = readability_scores(row[constants.review])
        rate = row[constants.rate]

        data.append([letters_len, words_len, sentences_len, adjective,
                     scores['compound'], scores['pos'], scores['neu'], scores['neg'] ,readability, rate])
    count_df = pd.DataFrame(data, columns=cols)
    print(df)
    return count_df


def under_sampling(df: pd.DataFrame):
    data = []
    cols = [constants.word_count, constants.rate]
    copy_df = copy.deepcopy(df)
    df_1 = copy_df[copy_df[constants.rate] == 1].sample(frac=1).head(constants.min_class_samples)
    df_2 = copy_df[copy_df[constants.rate] == 2].sample(frac=1).head(constants.min_class_samples)
    df_3 = copy_df[copy_df[constants.rate] == 3].sample(frac=1).head(constants.min_class_samples)
    df_4 = copy_df[copy_df[constants.rate] == 4].sample(frac=1).head(constants.min_class_samples)
    df_5 = copy_df[copy_df[constants.rate] == 5].sample(frac=1).head(constants.min_class_samples)
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

def svm_under_sampling(df: pd.DataFrame):
    data = []
    cols = [constants.word_count, constants.rate]
    copy_df = copy.deepcopy(df)
    df_1 = copy_df[copy_df[constants.rate] == 1].sample(frac=1).head(constants.svm_min_batch_size_per_class)
    df_2 = copy_df[copy_df[constants.rate] == 2].sample(frac=1).head(constants.svm_min_batch_size_per_class)
    df_3 = copy_df[copy_df[constants.rate] == 3].sample(frac=1).head(constants.svm_min_batch_size_per_class)
    df_4 = copy_df[copy_df[constants.rate] == 4].sample(frac=1).head(constants.svm_min_batch_size_per_class)
    df_5 = copy_df[copy_df[constants.rate] == 5].sample(frac=1).head(constants.svm_min_batch_size_per_class)
    dfs = [df_1, df_2, df_3, df_4, df_5]
    train = []
    test = []
    for df_i in dfs:
        to_train = df_i.iloc[:constants.svm_test_size_per_class, :]
        to_test = df_i.iloc[constants.svm_test_size_per_class:, :]
        train.append(to_train)
        test.append(to_test)
    train_set = pd.concat(train, axis=0)
    test_set = pd.concat(test, axis=0)
    return train_set, test_set


def set_vector(review):
    print("hi")


def parse_all_features(df: pd.DataFrame):
    # letter = pd.Series(df[constants.letter], dtype=int)
    # words = pd.Series(df[constants.words], dtype=int)
    # sen = pd.Series(df[constants.sentences], dtype=int)
    # rate = pd.Series(df[constants.rate], dtype=int)
    # adj = pd.Series(df[constants.adjective], dtype=int)
    # comp = pd.Series(df[constants.comp_score], dtype=float)
    # pos = pd.Series(df[constants.pos_score], dtype=float)
    # neg = pd.Series(df[constants.neg_score], dtype=float)
    # neu = pd.Series(df[constants.neu_score], dtype=float)
    # read = pd.Series(df[constants.readability].tolist(), dtype=float)
    # # df[constants.letter] = letter
    # new_df = pd.DataFrame(data=[letter, words, sen, adj, comp, pos, neg, neu, read, rate],
    #                       columns=[constants.letter, constants.words, constants.sentences, constants.adjective,
    #                                constants.comp_score, constants.pos_score, constants.neg_score, constants.neu_score,
    #                                constants.readability, constants.rate])
    # return new_df
    df = df.astype({constants.letter: 'int32', constants.words: 'int32', constants.sentences: 'int32',
                    constants.adjective: 'int32', constants.verb: 'int32', constants.noun: 'int32',
                    constants.comp_score: 'float', constants.pos_score: 'float', constants.neg_score: 'float',
                    constants.neu_score: 'float', constants.readability: 'float', constants.rate: 'int32'})
    print(df.dtypes)
    return df

# df = read_database_from_csv(csv_file='../data/tripadvisor_hotel_reviews.csv')
# print(create_review_length_features(df))
# output = create_df_with_all_features(df).sample(frac=1)
# print(output)
# train_x, train_y, test_x, test_y = divide(80, df)
# print(train_y.iloc[55][0])
# print(train_x.iloc[55][0])
# # print(train_x.shape)
# # print(test_x.shape)
# temp_review= train_x.iloc[55][0]
