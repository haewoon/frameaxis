"""Check both computation paths against the frozen SemEval-2014 reference (#12, #13).

The per-document methods (what notebook 1 runs) and the precomputed contribution
matrix must both reproduce reference/semeval14/<mode>/cell_means.tsv, for bias
(average) and intensity (second moment around the corpus mean).

Needs GloVe 840B at data/embeddings/ and reference/semeval14/ (scripts/make_reference.py);
skipped otherwise, so it does not run in CI. Loading GloVe takes a few minutes.
"""
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDING = os.path.join(ROOT, "data", "embeddings", "glove.840B.300d.w2vformat.txt")
REFERENCE = os.path.join(ROOT, "reference", "semeval14")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(EMBEDDING) and os.path.exists(os.path.join(REFERENCE, "average", "cell_means.tsv"))),
    reason="needs GloVe 840B and reference/semeval14",
)


@pytest.fixture(scope="module")
def setup():
    # pandas is in the examples group, which CI does not install; skip rather than fail without it.
    pd = pytest.importorskip("pandas")
    from frameaxis.semaxis import CoreUtil, SemAxis

    df = pd.read_csv(os.path.join(ROOT, "examples", "restaurant.tsv"), sep="\t").dropna()
    sa = SemAxis(CoreUtil.load_embedding(EMBEDDING, is_binary=False), axes_str=CoreUtil.load_wordnet_antonyms_axes())
    return pd, df, sa


def assert_matches_reference(pd, df, sa, rows, mode):
    axes = [str(c) for c in sorted(sa.axes.keys()) if len(c) == 2]
    table = pd.DataFrame(np.array(rows), columns=axes)
    table["aspect"], table["sentiment"] = df["aspect"].values, df["sentiment"].values
    observed = table.dropna().groupby(["aspect", "sentiment"])[axes].mean()

    reference = pd.read_csv(os.path.join(REFERENCE, mode, "cell_means.tsv"), sep="\t", index_col=["aspect", "sentiment"])
    np.testing.assert_allclose(observed.loc[reference.index, reference.columns].values, reference.values,
                               rtol=1e-4, atol=1e-6)


def per_document(fn):
    # Notebook 1 skips a sentence when no word survives filtering; keep it as a NaN row, dropped before averaging.
    try:
        return fn()
    except ValueError:
        return None


def test_per_document_path_reproduces_reference(setup):
    pd, df, sa = setup
    n_axes = len(sa.axes)
    nan = np.full(n_axes, np.nan)
    filters = [{a, a + "s"} for a in df["aspect"]]

    means = [per_document(lambda: sa.compute_document_mean_with_tf([t], min_freq=1, to_filter=f)[0])
             for t, f in zip(df["text"], filters)]
    means = np.array([nan if m is None else m for m in means])
    assert np.isnan(means).all(axis=1).sum() == len(df) - 3705  # rows the reference README reports
    assert_matches_reference(pd, df, sa, means, "average")

    corpus_mean = np.nanmean(means, axis=0)  # notebook 1: mean of the bias table
    moments = [per_document(lambda: sa.compute_document_second_moment_with_tf([t], corpus_mean, min_freq=1, to_filter=f)[0])
               for t, f in zip(df["text"], filters)]
    moments = np.array([nan if m is None else m for m in moments])
    assert_matches_reference(pd, df, sa, moments, "second_moment")


def test_precomputed_path_reproduces_reference(setup):
    pd, df, sa = setup
    sa._precompute_word_contribution_mat(list(df["text"]), filter_stopword=True, min_freq=1)
    filters = [{a, a + "s"} for a in df["aspect"]]

    means = np.array([sa.compute_document_mean_from_contribution_mat(i, min_freq=1, to_filter=f)
                      for i, f in enumerate(filters)])
    assert_matches_reference(pd, df, sa, means, "average")

    corpus_mean = np.nanmean(means, axis=0)
    moments = np.array([sa.compute_document_second_moment_from_contribution_mat(i, corpus_mean, min_freq=1, to_filter=f)
                        for i, f in enumerate(filters)])
    assert_matches_reference(pd, df, sa, moments, "second_moment")
