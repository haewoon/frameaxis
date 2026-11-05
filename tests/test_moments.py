"""The per-document SemAxis methods against direct computations on a toy embedding.

Each expected value is built from the kept words and their counts with plain
Python loops, independently of the matrix code in semaxis.py.
"""
import numpy as np
import pytest
from gensim.models import KeyedVectors
from scipy import stats

from frameaxis.semaxis import SemAxis

WORDS = ["good", "bad", "hot", "cold", "slow", "fast", "food", "foods", "tasty", "awful", "warm", "soup", "service"]
AXES = [("bad", "good"), ("cold", "hot"), ("slow", "fast")]
DOC = ["tasty soup tasty food and warm soup", "awful slow service with cold soup", "good food foods"]


@pytest.fixture
def sa():
    rng = np.random.default_rng(1)
    kv = KeyedVectors(12)
    kv.add_vectors(WORDS, rng.normal(size=(len(WORDS), 12)))
    return SemAxis(kv, axes_str=AXES)


def cosine(u, v):
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)  # KeyedVectors stores float32
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def kept_counts(sa, docs, to_filter=()):
    counts = {}
    for d in docs:
        for w in d.split():
            if w in sa.embedding and w not in ("and", "with") and w not in to_filter:
                counts[w] = counts.get(w, 0) + 1
    return dict(sorted(counts.items()))  # CountVectorizer orders terms alphabetically


def scores(sa, axis_vec, counts):
    return np.array([cosine(axis_vec, sa.embedding[w]) for w in counts]), np.array(list(counts.values()), dtype=float)


def repeated(s, f):
    return np.repeat(s, f.astype(int))


def masked(s, f, mask_percent):
    k = round(len(s) / 100 * mask_percent)
    if k == 0:
        return s[:0], f[:0]
    keep = (s >= np.sort(s)[-k]) | (s <= np.sort(s)[k - 1])
    return s[keep], f[keep]


def test_mean_and_second_moment(sa):
    counts = kept_counts(sa, DOC, {"food", "foods"})
    corpus_mean = np.array([0.1, -0.2, 0.05])
    mean = sa.compute_document_mean_with_tf(DOC, min_freq=1, to_filter={"food", "foods"})[0]
    sm = sa.compute_document_second_moment_with_tf(DOC, corpus_mean, min_freq=1, to_filter={"food", "foods"})[0]
    for i, axis in enumerate(sorted(sa.axes)):
        s, f = scores(sa, sa.axes[axis], counts)
        assert mean[i] == pytest.approx(np.mean(repeated(s, f)), rel=1e-9)
        assert sm[i] == pytest.approx(np.mean((repeated(s, f) - corpus_mean[i]) ** 2), rel=1e-9)


def test_kurtosis(sa):
    counts = kept_counts(sa, DOC)
    corpus_mean = np.array([0.1, -0.2, 0.05])
    keys, mean, kurt = sa.compute_document_mean_kurtosis_with_tf(DOC, min_freq=1)
    keys2, (kurt_corpus,) = sa.compute_document_kurtosis_with_tf(DOC, corpus_mean, min_freq=1)
    assert keys == keys2 == sorted(sa.axes)
    for i, axis in enumerate(keys):
        x = repeated(*scores(sa, sa.axes[axis], counts))
        assert mean[i] == pytest.approx(np.mean(x), rel=1e-9)
        assert kurt[i] == pytest.approx(stats.kurtosis(x, fisher=True, bias=True), rel=1e-9)
        d = x - corpus_mean[i]
        assert kurt_corpus[i] == pytest.approx(np.mean(d ** 4) / np.mean(d ** 2) ** 2 - 3, rel=1e-9)


def test_kurtosis_of_one_distinct_term_is_nan(sa):
    keys, mean, kurt = sa.compute_document_mean_kurtosis_with_tf(["good good good"], min_freq=1)
    assert np.isnan(kurt).all()
    np.testing.assert_allclose(mean, [cosine(sa.axes[a], sa.embedding["good"]) for a in keys], rtol=1e-9)


@pytest.mark.parametrize("mask_percent", [0, 20, 40])
def test_kurtosis_with_mask(sa, mask_percent):
    counts = kept_counts(sa, DOC)
    keys, mean, kurt = sa.compute_document_mean_kurtosis_with_tf_mask(DOC, min_freq=1, mask_percent=mask_percent)
    for i, axis in enumerate(keys):
        s, f = masked(*scores(sa, sa.axes[axis], counts), mask_percent)
        if len(s) == 0:
            assert np.isnan(mean[i]) and np.isnan(kurt[i])
            continue
        x = repeated(s, f)
        assert mean[i] == pytest.approx(np.mean(x), rel=1e-9)
        assert kurt[i] == pytest.approx(stats.kurtosis(x, fisher=True, bias=True), rel=1e-9, nan_ok=True)


def test_word_contributions(sa):
    axis = ("bad", "good")
    counts = kept_counts(sa, DOC, {"soup"})
    s, f = scores(sa, sa.axes[axis], counts)

    (contrib,), words = sa.word_contribution(DOC, axis, min_freq=1, to_filter={"soup"})
    assert words == list(counts)
    np.testing.assert_allclose(contrib[0], s * f / f.sum(), rtol=1e-9)
    assert contrib.sum() == pytest.approx(sa.compute_document_mean_with_tf(DOC, min_freq=1, to_filter={"soup"})[0][0])

    (contrib,), _ = sa.word_contribution_to_second_moment(DOC, axis, 0.1, min_freq=1, to_filter={"soup"})
    np.testing.assert_allclose(contrib[0], (s - 0.1) ** 2 * f / f.sum(), rtol=1e-9)

    counts = kept_counts(sa, DOC)
    s, f = scores(sa, sa.axes[axis], counts)
    (contrib,), words = sa.word_contribution_with_mask(DOC, axis, min_freq=1, mask_percent=40)
    k = round(len(s) / 100 * 40)
    keep = (s >= np.sort(s)[-k]) | (s <= np.sort(s)[k - 1])
    np.testing.assert_allclose(contrib[0], np.where(keep, s * f, 0.0) / f[keep].sum(), rtol=1e-9)


def test_document_without_terms_raises(sa):
    # Notebook 1 skips such a sentence by catching the exception.
    with pytest.raises(ValueError):
        sa.compute_document_mean_with_tf(["the and of"], min_freq=1)
    with pytest.raises(ValueError):
        sa.compute_document_second_moment_with_tf(["food foods"], np.zeros(3), min_freq=1, to_filter={"food", "foods"})
