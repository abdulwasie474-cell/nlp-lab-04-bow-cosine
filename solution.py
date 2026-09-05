# Task 1
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]

vectorizer = CountVectorizer(stop_words='english')

X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())

df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print(df)

# task 2
documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]

query = ["Machine learning algorithma for data"]

doc_vectors = vectorizer.fit_transform(documents)

query_vector = vectorizer.transform(query)

scores = cosine_similarity(query_vector, doc_vectors)

print(scores)

ranking = scores[0].argsort()[::-1]

print("\nRanked Documents:")

for i in ranking:
    print(f"Document {i + 1}: {scores[0][i]:.4f}")
    print(documents[i])
    print()

    