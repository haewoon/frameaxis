import sys
sys.path.insert(1, '..')
from semaxis import CoreUtil
from semaxis import SemAxis

def test_load_google_embedding():
    sa = SemAxis(CoreUtil.load_embedding("../../../data/embeddings/GoogleNews-vectors-negative300.bin"))
    assert sa.embedding_vector_size() == 300
