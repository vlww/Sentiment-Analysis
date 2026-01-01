import nltk
import re
import string
import itertools
import numpy as np
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt_tab')  #punkt_tab instead of punkt
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

# inspect how many positive and negative reviews there are
print("Number of reviews:", len(movie_reviews.fileids()))
print("Positive reviews:", len(movie_reviews.fileids('pos')))
print("Negative reviews:", len(movie_reviews.fileids('neg')))

# print the first few reviews with their labels
for fileid in movie_reviews.fileids('pos')[3:4]:  # just two examples per category
      words = movie_reviews.words(fileid)
      review_text = " ".join(words[:23])  # show first 30 words
      print(f"\nReview snippet: {review_text}")
      review_text = " ".join(words[23:42])
      print(f"                {review_text}...")
      print("Label: positive\n")

for fileid in movie_reviews.fileids('neg')[18:19]:  # just two examples per category
      words = movie_reviews.words(fileid)
      review_text = " ".join(words[:22])  # show first 30 words
      print(f"\nReview snippet: {review_text}")
      review_text = " ".join(words[22:45])
      print(f"                {review_text}...")
      print("Label: negative\n")

for fileid in movie_reviews.fileids('neg')[:1]:  # just two examples per category
      words = movie_reviews.words(fileid)
      review_text = " ".join(words[:22])  # show first 30 words
      print(f"\nReview snippet: {review_text}")
      review_text = " ".join(words[22:42])
      print(f"                {review_text}...")
      print("Label: negative\n")
for fileid in movie_reviews.fileids('pos')[1:2]:  # just two examples per category
      words = movie_reviews.words(fileid)
      review_text = " ".join(words[:23])  # show first 30 words
      print(f"\nReview snippet: {review_text}")
      review_text = " ".join(words[23:42])
      print(f"                {review_text}...")
      print("Label: positive\n")

# vader model and preprocessing stuff
vader = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
negation_words = {"no","not","nor","never","n't"}
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# dataset
texts, labels = [], []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category)[:200]:
        texts.append(movie_reviews.raw(fileid))
        labels.append(1 if category == 'pos' else 0)
labels = np.array(labels)

# functions
def remove_stop_words(text):
    tokens = word_tokenize(text)
    filtered_tokens = []
    for t in tokens:
        token_lower = t.lower()
        if token_lower not in stop_words or token_lower in negation_words:
            filtered_tokens.append(t)
    return ' '.join(tokens)

def stem(text):
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t.lower()) for t in tokens]
    return ' '.join(tokens)

def lemmatize(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokens]
    return ' '.join(tokens)

def lowercase_strip_punctuation_lemmatize(text):
    text_lower = text.lower()
    punct_pattern = r'[{}]'.format(re.escape(string.punctuation))
    text_clean = re.sub(punct_pattern, '', text_lower)
    tokens = word_tokenize(text_clean)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.33, random_state=42, stratify=labels)

# define selected preprocessing flows in fixed order
flows = [
    ('raw text (no preprocessing)                        |    ', lambda x: x),
    ('remove stop words                                  |    ', remove_stop_words),
    ('stem                                               |    ', stem),
    ('lemmatize                                          |    ', lemmatize),
    ('lowercase, strip punctuation, lemmatize            |    ', lowercase_strip_punctuation_lemmatize)
]

# run experiment
print("Flow                                               |     Accuracy")
print("-----------------------------------------------------------------")
for name, func in flows:
    X_test_proc = [func(x) for x in X_test]
    scores = np.array([vader.polarity_scores(t)['compound'] for t in X_test_proc])
    preds = (scores >= 0).astype(int)
    acc = accuracy_score(y_test, preds)
    print(f"{name:25s} {acc:.4f}")
