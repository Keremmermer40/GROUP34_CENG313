from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_models():
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "tfidf_logistic": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "char_logistic": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      min_df=2, max_features=30000)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }
