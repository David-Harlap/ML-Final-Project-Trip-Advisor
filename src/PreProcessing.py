
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

