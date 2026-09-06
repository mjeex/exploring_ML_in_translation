from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# loading data

dataset = load_dataset("stanfordnlp/sst2")

sentences = dataset["train"]["sentence"]
labels = dataset["train"]["label"]


# train/development split
X_train, X_dev, y_train, y_dev = train_test_split(
    sentences,
    labels,
    test_size=0.1,
    random_state=42,
    stratify=labels
)


# model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# train & predict
model.fit(X_train, y_train)

predictions = model.predict(X_dev)


# evaluate
print("Accuracy:", accuracy_score(y_dev, predictions))
print(classification_report(y_dev, predictions))
print(confusion_matrix(y_dev, predictions))