import os

import numpy as np
import pytest
from gensim.models import KeyedVectors

from frameaxis.semaxis import CoreUtil, SemAxis

GOOGLE_NEWS = os.path.join(os.path.dirname(__file__), "../../../data/embeddings/GoogleNews-vectors-negative300.bin")


@pytest.mark.skipif(not os.path.exists(GOOGLE_NEWS), reason="GoogleNews vectors not downloaded")
def test_load_google_embedding():
    sa = SemAxis(CoreUtil.load_embedding(GOOGLE_NEWS))
    assert sa.embedding_vector_size() == 300


@pytest.fixture
def toy_semaxis():
    rng = np.random.default_rng(0)
    words = ["good", "bad", "hot", "cold", "food", "tasty", "awful", "warm", "soup", "service"]
    kv = KeyedVectors(8)
    kv.add_vectors(words, rng.normal(size=(len(words), 8)))
    return SemAxis(kv, axes_str=[("bad", "good"), ("cold", "hot")])


def test_document_mean_matches_numpy(toy_semaxis):
    sa = toy_semaxis
    docs = ["tasty soup tasty food", "awful service", "warm soup and tasty food"]
    mean = np.ravel(sa.compute_document_mean_with_tf(docs, filter_stopword=True, min_freq=1)[0])

    counts = {}
    for d in docs:
        for w in d.split():
            if w in sa.embedding and w != "and":
                counts[w] = counts.get(w, 0) + 1
    expected = []
    for axis in sorted(sa.axes):
        a = sa.axes[axis] / np.linalg.norm(sa.axes[axis])
        num = sum(c * np.dot(a, sa.embedding[w] / np.linalg.norm(sa.embedding[w])) for w, c in counts.items())
        expected.append(num / sum(counts.values()))
    np.testing.assert_allclose(mean, expected, rtol=1e-5)
