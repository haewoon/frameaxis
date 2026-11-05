"""Check the precomputed contribution-matrix path against the frozen reference (#12).

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


def test_precomputed_path_reproduces_reference_cell_means():
    # pandas is in the examples group, which CI does not install; skip rather than fail without it.
    pd = pytest.importorskip("pandas")

    from frameaxis.semaxis import CoreUtil, SemAxis

    df = pd.read_csv(os.path.join(ROOT, "examples", "restaurant.tsv"), sep="\t").dropna()
    sa = SemAxis(CoreUtil.load_embedding(EMBEDDING, is_binary=False), axes_str=CoreUtil.load_wordnet_antonyms_axes())
    sa._precompute_word_contribution_mat(list(df["text"]), filter_stopword=True, min_freq=1)

    means = np.array([
        sa.compute_document_mean_from_contribution_mat(i, min_freq=1, to_filter={a, a + "s"})
        for i, a in enumerate(df["aspect"])
    ])
    axes = [str(c) for c in sorted(sa.axes.keys()) if len(c) == 2]
    table = pd.DataFrame(means, columns=axes)
    table["aspect"], table["sentiment"] = df["aspect"].values, df["sentiment"].values
    observed = table.dropna().groupby(["aspect", "sentiment"])[axes].mean()

    reference = pd.read_csv(os.path.join(REFERENCE, "average", "cell_means.tsv"), sep="\t",
                            index_col=["aspect", "sentiment"])
    np.testing.assert_allclose(observed.loc[reference.index, reference.columns].values, reference.values,
                               rtol=1e-4, atol=1e-6)
