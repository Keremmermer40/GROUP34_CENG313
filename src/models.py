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


def compare_models(development, seeds=(7, 21, 42)):
    """Model seçimini yalnızca geliştirme bölümünde yap."""
    import pandas as pd
    from .data import split_data, make_text
    from .utils import evaluate_model
    rows = []
    for seed in seeds:
        training, validation = split_data(development, seed)
        for name, model in build_models().items():
            model.fit(make_text(training), training['rule_violation'])
            metrics, _ = evaluate_model(validation, model.predict_proba(make_text(validation))[:, 1])
            rows.append(dict(model=name, seed=seed, **metrics))
    results = pd.DataFrame(rows)
    summary = results.groupby('model')[['roc_auc', 'mean_rule_auc', 'accuracy', 'f1']].mean()
    selected = summary.drop(index='dummy')['mean_rule_auc'].idxmax()
    return results, summary, selected
