from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def position_tag(text: str) -> str:
    """Tag each token with START/MID/END based on position within the song."""
    tokens = text.split()
    n = len(tokens)
    tagged = []
    for i, tok in enumerate(tokens):
        ratio = i / n if n else 0
        if ratio < 0.2:
            tagged.append(f"{tok}_START")
        elif ratio > 0.8:
            tagged.append(f"{tok}_END")
        else:
            tagged.append(f"{tok}_MID")
    return " ".join(tagged)


def run_position_tfidf(train_texts: List[str], dev_texts: List[str], train_labels: List[int], dev_labels: List[int]):
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    train_tagged = [position_tag(t) for t in train_texts]
    dev_tagged = [position_tag(t) for t in dev_texts]

    x_train = vectorizer.fit_transform(train_tagged)
    x_dev = vectorizer.transform(dev_tagged)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(x_train, train_labels)
    preds = clf.predict(x_dev)

    p, r, f, _ = precision_recall_fscore_support(dev_labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(dev_labels, preds)
    metrics = {"precision": p, "recall": r, "f1": f, "accuracy": acc}
    return preds, metrics, clf, vectorizer


def get_segment(text: str, segment: str = "start", portion: float = 0.3) -> str:
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return ""

    cut = max(int(n * portion), 1)
    if segment == "start":
        return " ".join(tokens[:cut])
    if segment == "middle":
        start = int(n * 0.35)
        end = int(n * 0.65)
        return " ".join(tokens[start:end])
    if segment == "end":
        return " ".join(tokens[-cut:])
    return text


def train_segment_model(
    train_texts: List[str],
    dev_texts: List[str],
    train_labels: List[int],
    dev_labels: List[int],
    name: str,
) -> Tuple[List[int], dict, LogisticRegression, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_texts)
    x_dev = vectorizer.transform(dev_texts)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="multinomial")
    clf.fit(x_train, train_labels)
    preds = clf.predict(x_dev)

    p, r, f, _ = precision_recall_fscore_support(dev_labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(dev_labels, preds)
    metrics = {"precision": p, "recall": r, "f1": f, "accuracy": acc}
    return preds, metrics, clf, vectorizer


__all__ = ["position_tag", "run_position_tfidf", "get_segment", "train_segment_model"]
