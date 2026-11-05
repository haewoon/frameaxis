# frameaxis

a repository for "FrameAxis: characterizing microframe bias and intensity with word embedding" 
https://peerj.com/articles/cs-644/

<img src="https://github.com/haewoon/frameaxis/raw/main/semeval14.png">

**Install**

Requires Python 3.9 (the locked TensorFlow 2.6.2 and NumPy 1.19.5 only ship wheels up to 3.9; 3.8 is allowed but not tested in CI, since current Poetry can no longer create 3.8 environments) and [Poetry](https://python-poetry.org/).

```
poetry env use python3.9
poetry install --with examples,dev
poetry run python -c "import nltk; nltk.download('stopwords')"
```

Then prefix commands with `poetry run`, or activate the environment with `eval $(poetry env activate)`.

**Run tests**

```
cd tests
poetry run pytest
```

**Download embedding**

The example code works with [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings (glove.840B.300d, about 2 GB zipped and 5.6 GB unzipped).
The notebooks expect the file at `data/embeddings/` in the repository root, converted to word2vec text format:

```
mkdir -p data/embeddings && cd data/embeddings
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
poetry run python -m gensim.scripts.glove2word2vec -i glove.840B.300d.txt -o glove.840B.300d.w2vformat.txt
```

**Run sample code**

```
cd examples
poetry run jupyter notebook
```

Run notebooks from 1_ to 4_ in order.

**Significance test**

Notebook 2 builds the null model: for each aspect and sentiment, `N = 10,000` random subsets of the same size are drawn without replacement from the whole table, with a fixed seed (`SEED = 2021`). Notebook 3 tests each axis two-sided against that null, p = (1 + #{|null - mean(null)| >= |observed - mean(null)|}) / (N + 1), as the PeerJ paper describes, and applies Benjamini-Hochberg correction across all axes within each aspect and sentiment. Axes with q <= 0.05 are reported as significant. The helpers are `random_subset_null`, `two_sided_pvalue` and `fdr_adjust` in `frameaxis/core.py`.

Results produced before this change chose the tail from the sign of the score, used N = 1,000 and no correction (see #13); the `p_2021` column in notebook 3's output keeps that rule for comparison.
