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
