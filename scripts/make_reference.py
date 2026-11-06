"""Regenerate the SemEval-2014 restaurant reference result (#12).

Runs examples/1-3 in order with nbconvert, then copies the significance
tables into reference/semeval14/ and writes a README with every parameter
needed to reproduce them, plus CHANGES.md comparing the corrected test with
the 2021 rule.

Usage (from the repository root, with the `examples` extra installed and the
GloVe download described in the README):

    python scripts/make_reference.py

Takes roughly 10-20 minutes, most of it loading GloVe and the 10,000-sample
null in notebook 2. Pass --skip-run to rebuild reference/ from existing
notebook outputs.
"""
import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
EMBEDDING = os.path.join(ROOT, "data", "embeddings", "glove.840B.300d.w2vformat.txt")
OUT = os.path.join(ROOT, "reference", "semeval14")
NOTEBOOKS = ["1_Build big table.ipynb", "2_Bootstrapping.ipynb", "3_Significant axes.ipynb"]
MODES = ["average", "second_moment"]
ALPHA = 0.05
PACKAGES = ["numpy", "scipy", "scikit-learn", "gensim", "nltk", "pandas"]


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def run_notebooks():
    for nb in NOTEBOOKS:
        print("running", nb, flush=True)
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=-1", nb], cwd=EXAMPLES, check=True)


def notebook_param(nb_name, name):
    nb = json.load(open(os.path.join(EXAMPLES, nb_name)))
    for cell in nb["cells"]:
        for line in cell["source"]:
            if line.strip().startswith(name + " ="):
                return line.split("=", 1)[1].strip()
    return None


def copy_tables():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for mode in MODES:
        os.makedirs(os.path.join(OUT, mode))
        for f in sorted(glob.glob(os.path.join(EXAMPLES, mode, "*significant_axes_*.tsv"))):
            shutil.copy(f, os.path.join(OUT, mode))
    rows = {}
    for mode, name in [("average", "big_table_by_average.tsv"),
                       ("second_moment", "big_table_by_second_moment_with_corpus_mean.tsv")]:
        path = os.path.join(EXAMPLES, name)
        df = pd.read_csv(path, sep="\t")
        axes = [c for c in df.columns if "(" in c]
        # Per-cell observed values, small enough to commit, for regression tests on other code paths.
        df.dropna().groupby(["aspect", "sentiment"])[axes].mean().to_csv(
            os.path.join(OUT, mode, "cell_means.tsv"), sep="\t")
        rows[mode] = (len(df), len(df.dropna()), len(axes), sha256(path))
    return rows


def changes():
    lines = ["# Corrected test vs the 2021 rule", "",
             "Per aspect × sentiment cell: axes with p_2021 <= 0.05 (sign-chosen tail, N in the denominator), "
             "with two-sided p <= 0.05, and with BH q <= 0.05; then how many axes the 2021 rule kept that "
             "the corrected test drops, and the reverse.", ""]
    for mode in MODES:
        lines += ["## " + mode, "",
                  "| cell | 2021 rule | two-sided p | BH q | dropped | added | top axis (2021 rule) | top axis (BH) |",
                  "|---|---|---|---|---|---|---|---|"]
        for f in sorted(glob.glob(os.path.join(OUT, mode, "significant_axes_*.tsv"))):
            cell = os.path.basename(f)[len("significant_axes_"):-len(".tsv")]
            df = pd.read_csv(f, sep="\t")
            old = set(df.query("p_2021 <= @ALPHA")["axis"])
            new = set(df.query("q <= @ALPHA")["axis"])
            top_old = df.sort_values("diff_a_b", ascending=False).iloc[0]["axis"]
            top_new = df.query("q <= @ALPHA").sort_values("diff_a_b", ascending=False)
            top_new = top_new.iloc[0]["axis"] if len(top_new) else "none"
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                cell, len(old), int((df["p"] <= ALPHA).sum()), len(new), len(old - new), len(new - old),
                top_old, top_new))
        lines.append("")
    lines += ["The top-axis columns follow the first figure in notebook 4 (one axis per aspect): the largest "
              "positive `diff_a_b`. Before #13 it was taken over all axes; now only over q <= 0.05. The README "
              "figure uses the top two axes per aspect and is not compared here.", ""]
    open(os.path.join(OUT, "CHANGES.md"), "w").write("\n".join(lines))


def readme(rows):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    versions = []
    for p in PACKAGES:
        try:
            versions.append("{} {}".format(p, importlib.metadata.version(p)))
        except importlib.metadata.PackageNotFoundError:
            pass
    zip_path = os.path.join(ROOT, "data", "embeddings", "glove.840B.300d.zip")
    text = """# SemEval-2014 restaurant reference result

Generated by `scripts/make_reference.py` from notebooks 1-3 at commit `{commit}`.
Other code paths (the precomputed contribution matrix, a faster pipeline) should reproduce
`cell_means.tsv` within tolerance, and changes to the test should reproduce the significance tables.

## Inputs

- Data: `examples/restaurant.tsv` (SemEval-2014 restaurant sentences with aspect and sentiment), sha256 `{data_sha}`
- Embedding: GloVe 840B, 300d (https://nlp.stanford.edu/projects/glove/), converted to word2vec text format with
  `gensim.scripts.glove2word2vec` and the header set to `2196016 300` (see 'Download embedding' in the README)
  - `glove.840B.300d.w2vformat.txt` sha256 `{emb_sha}`
  - `glove.840B.300d.zip` sha256 `{zip_sha}`
- Axes: WordNet adjective antonyms (`frameaxis/axes/wordnet_antonyms.tsv`, 1,828 pairs); {n_axes} are present in the embedding.

## Parameters

- Unit: each sentence is a document; `min_freq = 1`; NLTK English stopwords removed; the aspect word and its
  plural (`aspect`, `aspect + "s"`) removed from each sentence (`to_filter`).
- Bias: frequency-weighted mean cosine score per axis for each sentence (`compute_document_mean_with_tf`).
- Intensity: second moment per sentence around the corpus mean, the mean bias over all sentences
  (`compute_document_second_moment_with_tf`).
- Cell values: the mean over the sentences of the cell, so every sentence carries the same weight
  ('Unit of analysis' in the README).
- Rows: {rows_avg} of 3,713 sentences produced a row in the bias table, {rows_sm} in the intensity table
  (notebook 1 skips a sentence when no word survives filtering).
- Null: N = {N} random subsets per aspect × sentiment cell, each the size of the cell, drawn without replacement
  from the whole table, seed {SEED}.
- Test: two-sided p = (1 + #{{|null - mean(null)| >= |observed - mean(null)|}}) / (N + 1); Benjamini-Hochberg
  across all axes within each cell; significant when q <= {ALPHA}.

## Files

- `<mode>/significant_axes_<aspect>_<sentiment>.tsv`: every axis with `diff_a_b` (observed minus null mean),
  `p`, `q`, and `p_2021` (the rule used before #13, for comparison only).
- `<mode>/effect_size_significant_axes_<aspect>_<sentiment>.tsv`: axes with q <= {ALPHA}, by `|diff_a_b|`.
- `<mode>/cell_means.tsv`: observed mean per axis for each aspect × sentiment cell.
- `CHANGES.md`: how the corrected test differs from the tables the notebooks produced before #13.

The full tables from notebook 1 are not committed (about 70 MB each). Their sha256: bias `{sha_avg}`,
intensity `{sha_sm}`.

## Environment

Python {py} on {os_arch}; {versions}
""".format(commit=commit, data_sha=sha256(os.path.join(EXAMPLES, "restaurant.tsv")), emb_sha=sha256(EMBEDDING),
           zip_sha=sha256(zip_path) if os.path.exists(zip_path) else "(zip not present when generated)",
           n_axes="{:,}".format(rows["average"][2]), rows_avg="{:,}".format(rows["average"][1]),
           rows_sm="{:,}".format(rows["second_moment"][1]), N=notebook_param("2_Bootstrapping.ipynb", "N"),
           SEED=notebook_param("2_Bootstrapping.ipynb", "SEED"), ALPHA=ALPHA,
           sha_avg=rows["average"][3], sha_sm=rows["second_moment"][3],
           py=platform.python_version(), os_arch="{} {}".format(platform.system(), platform.machine()),
           versions=", ".join(versions))
    open(os.path.join(OUT, "README.md"), "w").write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-run", action="store_true", help="reuse existing notebook outputs")
    args = parser.parse_args()
    if not os.path.exists(EMBEDDING):
        sys.exit("Missing {}; see 'Download embedding' in the README.".format(EMBEDDING))
    if not args.skip_run:
        run_notebooks()
    rows = copy_tables()
    changes()
    readme(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
