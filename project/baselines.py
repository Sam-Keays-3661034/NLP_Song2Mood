import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from cleaning import word_tokenize
from data_loader import EMOTIONS


def evaluate_metrics(true_labels: List[str], predicted_labels: List[str]) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="macro", zero_division=0)
    acc = accuracy_score(true_labels, predicted_labels)
    return {"precision": p, "recall": r, "f1": f, "accuracy": acc}


def bag_of_words_logreg(train_texts: List[str], train_labels: List[str], dev_texts: List[str], dev_labels: List[str]):
    vectorizer = CountVectorizer(analyzer=word_tokenize, max_features=30000)
    train_counts = vectorizer.fit_transform(train_texts)
    dev_counts = vectorizer.transform(dev_texts)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=0)
    model.fit(train_counts, train_labels)
    preds = model.predict(dev_counts)
    metrics = evaluate_metrics(dev_labels, preds)
    return preds, metrics, model, vectorizer


def tfidf_logreg(train_texts: List[str], train_labels: List[str], dev_texts: List[str], dev_labels: List[str]):
    vectorizer = TfidfVectorizer(analyzer=word_tokenize, max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
    train_tfidf = vectorizer.fit_transform(train_texts)
    dev_tfidf = vectorizer.transform(dev_texts)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=0)
    model.fit(train_tfidf, train_labels)
    preds = model.predict(dev_tfidf)
    metrics = evaluate_metrics(dev_labels, preds)
    return preds, metrics, model, vectorizer


def word2vec_logreg(train_texts: List[str], train_labels: List[str], dev_texts: List[str], dev_labels: List[str]):
    import gensim.downloader

    w2v_model = gensim.downloader.load("word2vec-google-news-300")
    vector_size = w2v_model.vector_size

    def vec_for_doc(tokenized_doc):
        vectors = [w2v_model[word] for word in tokenized_doc if word in w2v_model.key_to_index]
        if not vectors:
            return np.zeros(vector_size, dtype="float32")
        return np.mean(vectors, axis=0)

    train_vecs = [vec_for_doc(word_tokenize(text)) for text in train_texts]
    dev_vecs = [vec_for_doc(word_tokenize(text)) for text in dev_texts]

    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(train_vecs, train_labels)
    preds = model.predict(dev_vecs)
    metrics = evaluate_metrics(dev_labels, preds)
    return preds, metrics, model, w2v_model


def most_frequent_class(train_labels: List[int], dev_labels: List[int]):
    label_counts = Counter(train_labels)
    most_frequent_label = label_counts.most_common(1)[0][0]
    preds = [most_frequent_label] * len(dev_labels)
    metrics = evaluate_metrics(dev_labels, preds)
    return preds, metrics, most_frequent_label


def print_results(metrics: Dict[str, float]):
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("Accuracy:", metrics["accuracy"])


def print_per_class(true_labels: List[str], predicted_labels: List[str]):
    p_i, r_i, f_i, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None, zero_division=0)
    print("=== Per Emotion Metrics ===")
    for i, emotion in enumerate(EMOTIONS):
        print(f"{emotion}: Precision={p_i[i]} Recall={r_i[i]} F1={f_i[i]}")


__all__ = [
    "bag_of_words_logreg",
    "tfidf_logreg",
    "word2vec_logreg",
    "most_frequent_class",
    "evaluate_metrics",
    "print_results",
    "print_per_class",
]
