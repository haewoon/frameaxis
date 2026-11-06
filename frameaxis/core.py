from gensim.models.keyedvectors import KeyedVectors
from scipy.spatial import distance
import os
import numpy as np
from scipy import stats
from collections import defaultdict

AXES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axes")


class CoreUtil:
    @staticmethod
    def load_embedding(emb_file, is_binary=True):
        return KeyedVectors.load_word2vec_format(emb_file, binary=is_binary)

    @staticmethod
    def get_score(w_vec, axis_vec):
        return 1.0 - distance.cosine(w_vec, axis_vec)        
    
    @staticmethod
    def map_axis_to_vec(emb, axis):
        return (emb[axis[1]] - emb[axis[0]])

    @staticmethod
    def load_axes_file(filename):
        axes = []
        with open(filename) as fi:
            for line in fi:
                w1, w2 = [term.strip() for term in line.split("\t")]
                axes.append((w1,w2))                
        return axes 

    @staticmethod
    def load_conceptnet_antonyms_axes(version = "1.0"):
        if version == "1.0":
            return CoreUtil.load_axes_file(os.path.join(AXES_DIR, "732_semaxis_axes.tsv"))            
        else:
            raise Exception("Unknown version")

    @staticmethod
    def load_plutchik_wheel_of_emotions_axes():
        return CoreUtil.load_axes_file(os.path.join(AXES_DIR, "plutchik_wheel_of_emotions.tsv"))

    @staticmethod
    def load_wordnet_antonyms_axes():
        return CoreUtil.load_axes_file(os.path.join(AXES_DIR, "wordnet_antonyms.tsv"))

    @staticmethod
    def bootstrap_sampling(document, size):
        return np.random.choice(document, size, replace=True)

    @staticmethod
    def _average_dict(data):
        return sum([key*data[key] for key in data])/sum(data.values())
        
    @staticmethod
    def _moment_dict(data, k, data_mean=None):
        if not data_mean:
            data_mean = CoreUtil._average_dict(data)
        return sum(data[key]*(key-data_mean)**k for key in data)/sum(data.values())

    @staticmethod
    def compute_average(axis2score):
        results = []
        averages = {}
        for axis in sorted(axis2score):
            average = CoreUtil._average_dict(axis2score[axis])
            results.append([axis[0], axis[1], average])
            averages[axis] = average

        return results, averages

    @staticmethod
    def compute_kurtosis(axis2score, averages=None):
        results = []
        for axis in sorted(axis2score):
            data = axis2score[axis]
            if averages:
                results.append([axis[0], axis[1], 
                                CoreUtil._moment_dict(data,4,averages[axis])/(CoreUtil._moment_dict(data,2,averages[axis])**2)-3]) 
            else:
                results.append([axis[0], axis[1], 
                                CoreUtil._moment_dict(data,4)/(CoreUtil._moment_dict(data,2)**2)-3]) 
        return results   

    @staticmethod
    def compute_statistical_significance(result, samples):
        """Compare each axis value in the ``result`` file with the same axis in the ``samples`` files.

        Lines that do not parse, or whose value is NaN (kurtosis is NaN for a
        document with one distinct term), are skipped in both the result and the
        samples: two_sided_pvalue raises on NaN, and before #27 a NaN value got
        the smallest attainable p. ``percent_rank`` is over the sample values kept
        for that axis.
        """
        samples_dict = defaultdict(list)
        for sample in samples:
            with open(sample) as fi:
                for line in fi:
                    pole0, pole1, value = [term.strip() for term in line.split(",")]
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                    if not np.isnan(value):
                        samples_dict[(pole0, pole1)].append(value)

        axis_rank_table = []
        with open(result) as fi:
            for line in fi:
                pole0, pole1, value = [term.strip() for term in line.split(",")]
                sample_values = samples_dict[(pole0, pole1)]
                try:
                    value = float(value)
                except ValueError:
                    continue
                if np.isnan(value) or not sample_values:
                    continue
                rank = sum([x < value for x in sample_values])
                percent_rank = rank/float(len(sample_values))
                # percent_rank is one-sided; the last column is the two-sided p-value
                p_value = float(two_sided_pvalue(value, sample_values))
                axis_rank_table.append([pole0, pole1, percent_rank, value, np.mean(sample_values), p_value])

        return axis_rank_table

def random_subset_null(values, size, n_samples, seed, batch_size=256):
    """Null distribution of the column means of a random subset of rows.

    Draws ``n_samples`` subsets of ``size`` rows without replacement from
    ``values`` (rows x axes) and returns their column means as an
    ``n_samples x axes`` array. This is the null model of the PeerJ paper:
    samples from the whole corpus with the same size as the target corpus.
    The same ``seed`` always gives the same samples.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.isnan(values).any():
        # One NaN row would make every null mean NaN, and two_sided_pvalue would then flag every axis.
        raise ValueError("values contain NaN; drop those rows first (notebook 2 uses dropna)")
    rng = np.random.default_rng(seed)
    n_rows = values.shape[0]
    means = np.empty((n_samples, values.shape[1]))
    # Draw subsets in batches as 0/1 selection rows, so the means are one matrix product per batch.
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        select = np.zeros((stop - start, n_rows))
        for i in range(stop - start):
            select[i, rng.choice(n_rows, size, replace=False)] = 1.0
        means[start:stop] = select @ values / size
    return means


def two_sided_pvalue(observed, null_samples):
    """Two-sided p-value of ``observed`` against bootstrap ``null_samples``.

    Measures how far the observation is from the null mean, in either
    direction: p = (1 + #{|null - mean(null)| >= |observed - mean(null)|}) / (N + 1).
    ``observed`` may be a scalar or one value per axis, with ``null_samples``
    shaped ``N`` or ``N x axes``. The +1 keeps p above zero; with N samples
    the smallest attainable p is 1 / (N + 1).
    """
    null_samples = np.asarray(null_samples, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if np.isnan(observed).any() or np.isnan(null_samples).any():
        # NaN compares False, so it would count as more extreme than every null sample: p = 1 / (N + 1).
        raise ValueError("observed or null_samples contain NaN")
    center = null_samples.mean(axis=0)
    extreme = np.abs(null_samples - center) >= np.abs(observed - center)
    return (1.0 + extreme.sum(axis=0)) / (null_samples.shape[0] + 1.0)


def fdr_adjust(pvalues, method="bh"):
    """Benjamini-Hochberg ("bh") or Benjamini-Yekutieli ("by") adjusted p-values.

    Returns q-values in the input order; an axis is significant at FDR level
    alpha when its q-value is <= alpha.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if np.isnan(p).any():
        # np.minimum.accumulate would spread one NaN to every q-value.
        raise ValueError("pvalues contain NaN")
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    scale = m / np.arange(1, m + 1)
    if method == "by":
        scale = scale * np.sum(1.0 / np.arange(1, m + 1))
    elif method != "bh":
        raise ValueError("method must be 'bh' or 'by'")
    q_sorted = np.minimum.accumulate((p[order] * scale)[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q
