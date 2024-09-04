# 5. **TF-IDF Vectorization**
#    - Main: Implement a TF-IDF vectorizer
#    - Extensions:
#      a) Handle out-of-vocabulary words
#      b) Implement n-gram features
#      c) Apply sublinear tf scaling
#      d) Discuss alternatives like word embeddings (Word2Vec, GloVe)

import numpy as np
from collections import Counter

class TfidfVectorizer:
    def __init__(self, ngram_range=(1, 1)):
        self.ngram_range = ngram_range
        self.vocabulary = {}
        self.idf = {}

    def fit(self, documents):
        """
        Fit the vectorizer to the documents.
        """
        
        self.vocabulary = {}
        self.idf = {}
        self.fit_iteration(documents)

    def fit_iteration(self, documents):
        doc_count = len(documents)
        word_doc_counts = Counter()

        for doc in documents:
            words = set(doc.lower().split())
            for word in words:
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)
                word_doc_counts[word] += 1

        self.idf = {word: np.log(doc_count / count) + 1 for word, count in word_doc_counts.items()}

    def transform(self, documents):
        tfidf_matrix = []
        for doc in documents:
            word_counts = Counter(doc.lower().split())
            tfidf_vector = np.zeros(len(self.vocabulary))
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    tf = count
                    idf = self.idf.get(word, 0)
                    tfidf_vector[self.vocabulary[word]] = tf * idf
            tfidf_matrix.append(tfidf_vector)
        return np.array(tfidf_matrix)

    def fit_transform(self, documents):
        """
        Fit the vectorizer to the documents and transform them to TF-IDF vectors.
        """
        self.fit(documents)
        return self.transform(documents)


documents = [
    "I love programming in Python",
    "Python is a great language",
    "I enjoy learning new things"
]

vectorizer = TfidfVectorizer(ngram_range=(1, 1))

print(vectorizer.fit_transform(documents))
