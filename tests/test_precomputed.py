import numpy as np
import pytest
from gensim.models import KeyedVectors

from frameaxis.semaxis import SemAxis

DOCS = [
    "the food was tasty and the soup was tasty",
    "awful service and cold food",
    "warm soup warm bread friendly service",
    "the price was fair for the food",
    "cold soup awful price",
]


@pytest.fixture
def sa():
    rng = np.random.default_rng(0)
    words = ["good", "bad", "hot", "cold", "cheap", "expensive", "food", "tasty", "soup", "awful",
             "service", "warm", "bread", "friendly", "price", "fair"]
    kv = KeyedVectors(12)
    kv.add_vectors(words, rng.normal(size=(len(words), 12)))
    return SemAxis(kv, axes_str=[("bad", "good"), ("cold", "hot"), ("cheap", "expensive")])


@pytest.mark.parametrize("min_freq", [1, 2])
@pytest.mark.parametrize("to_filter", [set(), {"food", "foods"}])
def test_precomputed_matches_per_document_path(sa, min_freq, to_filter):
    sa._precompute_word_contribution_mat(DOCS, filter_stopword=True, min_freq=1)
    corpus_mean = np.array([0.01, -0.02, 0.03])
    for i, doc in enumerate(DOCS):
        pre_mean = sa.compute_document_mean_from_contribution_mat(i, min_freq=min_freq, to_filter=to_filter)
        pre_sm = sa.compute_document_second_moment_from_contribution_mat(i, corpus_mean, min_freq=min_freq, to_filter=to_filter)
        try:
            mean = np.ravel(sa.compute_document_mean_with_tf([doc], min_freq=min_freq, to_filter=to_filter)[0])
            sm = np.ravel(sa.compute_document_second_moment_with_tf([doc], corpus_mean, min_freq=min_freq, to_filter=to_filter)[0])
        except Exception:
            # the per-document path fails when no term survives; the precomputed path returns NaN
            assert np.all(np.isnan(pre_mean)) and np.all(np.isnan(pre_sm))
            continue
        np.testing.assert_allclose(pre_mean, mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(pre_sm, sm, rtol=1e-5, atol=1e-6)


def test_corpus_wide_min_freq_is_still_the_default(sa):
    # "bread" and "friendly" appear once in the corpus, so min_freq=2 at precompute time drops them everywhere.
    sa._precompute_word_contribution_mat(DOCS, filter_stopword=True, min_freq=2)
    assert "bread" not in sa.terms_words and "soup" in sa.terms_words
    assert sa.contributions.shape == (3, len(sa.terms_words))
