import pandas as pd
import matplotlib as plt
from os import path

# C-M-i auto-completion
# C-c C-c open shell
# C-c C-c/r/s send buffer/region
#     C-z switch
#     C-y e , C-RET send current statement
#     
# M-x describe-key

def load_data (relative_file_path):
    if '__file__' in locals():
        basic_path = path.split(path.abspath(__file__))[0]
    else: #fall back for ipython
        basic_path = path.abspath('c:/dev/applied-ml/poet-detection')
    rel_file_path = path.relpath(relative_file_path)
    source = pd.read_csv(path.join(basic_path, rel_file_path))
    return source

data = load_data("data/train.csv")

#print(data.describe())
#print(data.head)

print("Author frequency", data[['id', 'author']].groupby('author').agg('count'))


# ========================================
# NATURAL LANGUAGE PREPROCESSING
# ========================================
#execute only once to download resources
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

import nltk
from nltk.stem import WordNetLemmatizer
from functools import partial, reduce

def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)),
                            functions,
                            lambda x: x)

filter_fn = lambda coll: lambda word: word.lower() not in coll

stopwords = nltk.corpus.stopwords.words('english')
punctuation = ['.', '!', '?', ',', ';', '-']
lemm = WordNetLemmatizer()

cleaner = compose(list,
                  partial(filter, filter_fn(stopwords)),
                  partial(filter, filter_fn(punctuation)),
                  partial(map, lemm.lemmatize),
                  nltk.word_tokenize)

data['cleaned'] = data.text.apply(cleaner)


# ========================================
# VECTORIZATION
# ========================================

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

counts = CountVectorizer(stop_words='english').fit_transform(data.text)
X = TfidfTransformer(norm='l1').fit_transform(counts)


# ========================================
# Statistical Leaning
# ========================================

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB().fit(X, data.author)


# ========================================
# Evaluation
# ========================================

precision = lambda x, y: sum(x == y) / len(x)

print("In sample precision",
      precision(data.author, nb.predict(X)))

