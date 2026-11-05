from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np
from nltk.corpus import stopwords
import math
import operator
try:
    from .core import CoreUtil
except ImportError:  # notebooks put frameaxis/ itself on sys.path
    from core import CoreUtil
import logging


class SemAxis:
    def __init__(self, embedding, axes_str=CoreUtil.load_conceptnet_antonyms_axes()):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filemode="w")
        self.logger = logging.getLogger(__name__)                
        self.embedding = getattr(embedding, "wv", embedding)  # KeyedVectors or a Word2Vec model
        self.axes = self._build_axes_on_embedding(axes_str)
        self.axes_mat = None

    
    def _build_axes_on_embedding(self, axes_str):
        mapped_axes = {}
        for axis in axes_str:
            try:
                mapped_axes[axis] = CoreUtil.map_axis_to_vec(self.embedding, axis)
            except KeyError:
                self.logger.error("{} axis is not included in embedding".format(axis), exc_info=True)

        return mapped_axes
    
    def embedding_vector_size(self):
        return self.embedding.vector_size

    def compute_word_score(self, w):
        results = {}
        w_vec = self.embedding[w]
        for axis_name in self.axes:
            results[axis_name] = CoreUtil.get_score(w_vec, self.axes[axis_name])
        return results

    def _compute_with_document(self, document, filter_stopword = True, min_freq = 10, power=1):
        vectorizer = CountVectorizer()
        transformed_data = vectorizer.fit_transform(document)
        tf = zip(vectorizer.get_feature_names_out(), np.ravel(transformed_data.sum(axis=0)))

        axis2score = defaultdict(float)
        word_count = 0
        for (term, frequency) in tf:
            if frequency < min_freq:
                continue
            if filter_stopword and (term in stopwords.words('english')):
                continue

            try:
                w_vec = self.embedding[term]
            except:
                continue

            for axis_name in self.axes:
                score = CoreUtil.get_score(w_vec, self.axes[axis_name])
                axis2score[axis_name] += math.pow(score,power) * frequency
            word_count += frequency
        return [[r[0][0], r[0][1], r[1]/word_count] 
                for r in sorted(axis2score.items(), key=operator.itemgetter(1), reverse=True)]

    def compute_document_score(self, document, filter_stopword = True, min_freq = 10):
        return self._compute_with_document(document, filter_stopword, min_freq, 1)

    def compute_document_strength(self, document, filter_stopword = True, min_freq = 10):
        return self._compute_with_document(document, filter_stopword, min_freq, 2)

    def compute_document_raw(self, document, filter_stopword = True, min_freq = 10):
        vectorizer = CountVectorizer()
        transformed_data = vectorizer.fit_transform(document)
        tf = zip(vectorizer.get_feature_names_out(), np.ravel(transformed_data.sum(axis=0)))

        axis2score = defaultdict(lambda: defaultdict(int))
        for (term, frequency) in tf:
            if frequency < min_freq:
                continue
            if filter_stopword and (term in stopwords.words('english')):
                continue

            try:
                w_vec = self.embedding[term]
            except:
                continue

            for axis_name in self.axes:
                score = CoreUtil.get_score(w_vec, self.axes[axis_name])
                axis2score[axis_name][score] += frequency

        return axis2score

    def _prepare_matrix_computation(self, document, filter_stopword, min_freq, to_filter):
        vectorizer = CountVectorizer()
        transformed_data = vectorizer.fit_transform(document)

        terms_filtered = []
        frequencies_filtered = []
        words = []
        frequencies = np.ravel(transformed_data.sum(axis=0))
        for w_index, w in enumerate(vectorizer.get_feature_names_out()):
            if w in self.embedding:
                if frequencies[w_index] < min_freq or (filter_stopword 
                    and (w in stopwords.words('english'))) or w in to_filter:
                    continue
                terms_filtered.append(self.embedding[w])
                frequencies_filtered.append(frequencies[w_index])
                words.append(w)

        if self.axes_mat is None:
            self.axes_mat = np.array([self.axes[k] for k in sorted(self.axes)])

        return terms_filtered, frequencies_filtered, words

    def compute_document_mean_kurtosis_with_tf(self, document, filter_stopword = True, min_freq = 10):
        terms_filtered, frequencies_filtered, _ =  self._prepare_matrix_computation(document, filter_stopword, min_freq)  
        import tensorflow as tf        
        tf.reset_default_graph()
        with tf.Session() as sess:
            axes_tfm_nn = tf.nn.l2_normalize(tf.constant(self.axes_mat), axis = 1)    
            frequencies_vec = tf.constant(frequencies_filtered, dtype=np.float32)
            term_filtered_tfm_nn = tf.nn.l2_normalize(tf.constant(np.array(terms_filtered)), axis = 1)

            result = tf.matmul(axes_tfm_nn, term_filtered_tfm_nn,
                             adjoint_b = True # transpose second matrix
                             )
            ws = tf.multiply(result, frequencies_vec)
            mean = tf.reduce_sum(ws, 1)/tf.reduce_sum(frequencies_vec)
            ss = tf.reduce_sum(tf.multiply(tf.square(result), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            ms = tf.square(mean)
            # std = tf.sqrt(ss-ms)
            diff = result-tf.reshape(mean, [-1,1])
            moment4 = tf.reduce_sum(tf.multiply(tf.pow(diff, 4), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            moment2 = tf.reduce_sum(tf.multiply(tf.pow(diff, 2), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            kurtosis = moment4 / tf.square(moment2) - 3    
            mean, kurtosis = sess.run([mean, kurtosis])

        return sorted(self.axes.keys()), mean, kurtosis


    def _get_top_mask(self, t, k):
        import tensorflow as tf
        top_k = tf.math.top_k(t, k)
        tpivots = tf.reduce_min(top_k.values, axis=1)
        tpivots_ft = tf.reshape(tpivots, [-1,1])
        return tf.greater_equal(t, tpivots_ft) 

    def _apply_mask(self, test, frequencies_vec, mask_percent):
        import tensorflow as tf
        k = round(int(test.shape[1])/100*mask_percent)

        top_mask = self._get_top_mask(test, k)
        bot_mask = self._get_top_mask(-1*test, k)
        mask = tf.math.logical_or(top_mask, bot_mask)
        zeros = tf.zeros(test.shape)

        multiply = [test.shape[0]]
        r_matrix = tf.reshape(tf.tile(frequencies_vec, multiply), [ multiply[0], tf.shape(frequencies_vec)[0]])
        masked_frequencies_tsr = tf.where(mask, r_matrix, zeros)
        
        return tf.where(mask, test, zeros), masked_frequencies_tsr      

    def compute_document_mean_kurtosis_with_tf_mask(self, document, filter_stopword = True, min_freq = 10, mask_percent = 5):
        terms_filtered, frequencies_filtered, _ =  self._prepare_matrix_computation(document, filter_stopword, min_freq)  
        import tensorflow as tf        
        tf.reset_default_graph()
        with tf.Session() as sess:
            axes_tfm_nn = tf.nn.l2_normalize(tf.constant(self.axes_mat), axis = 1)    
            frequencies_vec = tf.constant(frequencies_filtered, dtype=np.float32)
            term_filtered_tfm_nn = tf.nn.l2_normalize(tf.constant(np.array(terms_filtered)), axis = 1)

            test = tf.matmul(axes_tfm_nn, term_filtered_tfm_nn,
                             adjoint_b = True # transpose second matrix
                             )
            result, masked_frequencies_tsr = self._apply_mask(test, frequencies_vec, mask_percent)
            
            ws = tf.multiply(result, masked_frequencies_tsr)
            mean = tf.reduce_sum(ws, 1)/tf.reduce_sum(masked_frequencies_tsr, 1)
            # ss = tf.reduce_sum(tf.multiply(tf.square(result), masked_frequencies_tsr), 1)/tf.reduce_sum(masked_frequencies_tsr, 1)
            # ms = tf.square(mean)
            # std = tf.sqrt(ss-ms)
            diff = result-tf.reshape(mean, [-1,1])
            moment4 = tf.reduce_sum(tf.multiply(tf.pow(diff, 4), masked_frequencies_tsr), 1)/tf.reduce_sum(masked_frequencies_tsr, 1)
            moment2 = tf.reduce_sum(tf.multiply(tf.pow(diff, 2), masked_frequencies_tsr), 1)/tf.reduce_sum(masked_frequencies_tsr, 1)
            kurtosis = moment4 / tf.square(moment2) - 3    
            mean, kurtosis = sess.run([mean, kurtosis])

        return sorted(self.axes.keys()), mean, kurtosis


    def compute_document_mean_with_tf(self, document, filter_stopword = True, min_freq = 10, to_filter = set([])):
        terms_filtered, frequencies_filtered, _ =  self._prepare_matrix_computation(document, filter_stopword, min_freq, to_filter)  
               
        # self.logger.debug(terms_filtered)
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

        tf.reset_default_graph()
        with tf.Session() as sess:
            axes_tfm_nn = tf.nn.l2_normalize(tf.constant(self.axes_mat), axis = 1)    
            frequencies_vec = tf.constant(frequencies_filtered, dtype=np.float32)
            term_filtered_tfm_nn = tf.nn.l2_normalize(tf.constant(np.array(terms_filtered)), axis = 1)

            result = tf.matmul(axes_tfm_nn, term_filtered_tfm_nn,
                             adjoint_b = True # transpose second matrix
                             )
            ws = tf.multiply(result, frequencies_vec)
            mean = tf.reduce_sum(ws, 1)/tf.reduce_sum(frequencies_vec)
            mean = sess.run([mean])

        return mean

    def compute_document_kurtosis_with_tf(self, document, corpus_mean, filter_stopword = True, min_freq = 10):
        terms_filtered, frequencies_filtered, _ =  self._prepare_matrix_computation(document, filter_stopword, min_freq)  
        import tensorflow as tf        
        tf.reset_default_graph()
        with tf.Session() as sess:
            axes_tfm_nn = tf.nn.l2_normalize(tf.constant(self.axes_mat), axis = 1)    
            frequencies_vec = tf.constant(frequencies_filtered, dtype=np.float32)
            term_filtered_tfm_nn = tf.nn.l2_normalize(tf.constant(np.array(terms_filtered)), axis = 1)

            result = tf.matmul(axes_tfm_nn, term_filtered_tfm_nn,
                             adjoint_b = True # transpose second matrix
                             )
            ws = tf.multiply(result, frequencies_vec)
            # mean = tf.reduce_sum(ws, 1)/tf.reduce_sum(frequencies_vec)
            mean = tf.constant(corpus_mean, dtype=np.float32)
            # ss = tf.reduce_sum(tf.multiply(tf.square(result), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            # ms = tf.square(mean)
            # std = tf.sqrt(ss-ms)
            diff = result-tf.reshape(mean, [-1,1])
            moment4 = tf.reduce_sum(tf.multiply(tf.pow(diff, 4), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            moment2 = tf.reduce_sum(tf.multiply(tf.pow(diff, 2), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            kurtosis = moment4 / tf.square(moment2) - 3    
            kurtosis = sess.run([kurtosis])

        return sorted(self.axes.keys()), kurtosis

    def compute_document_second_moment_with_tf(self, document, corpus_mean, filter_stopword = True, min_freq = 10, to_filter = set([])):
        terms_filtered, frequencies_filtered, _ =  self._prepare_matrix_computation(document, filter_stopword, min_freq, to_filter)  
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

        tf.reset_default_graph()
        with tf.Session() as sess:
            axes_tfm_nn = tf.nn.l2_normalize(tf.constant(self.axes_mat), axis = 1)    
            frequencies_vec = tf.constant(frequencies_filtered, dtype=np.float32)
            term_filtered_tfm_nn = tf.nn.l2_normalize(tf.constant(np.array(terms_filtered)), axis = 1)

            result = tf.matmul(axes_tfm_nn, term_filtered_tfm_nn,
                             adjoint_b = True # transpose second matrix
                             )
            ws = tf.multiply(result, frequencies_vec)
            # mean = tf.reduce_sum(ws, 1)/tf.reduce_sum(frequencies_vec)
            # if corpus_mean:
            mean = tf.constant(corpus_mean, dtype=np.float32)
            # else:
            #     mean = tf.reduce_sum(ws, 1)/tf.reduce_sum(frequencies_vec)
                
            # ss = tf.reduce_sum(tf.multiply(tf.square(result), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            # ms = tf.square(mean)
            # std = tf.sqrt(ss-ms)
            diff = result-tf.reshape(mean, [-1,1])
            # moment4 = tf.reduce_sum(tf.multiply(tf.pow(diff, 4), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            moment2 = tf.reduce_sum(tf.multiply(tf.pow(diff, 2), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            # kurtosis = moment4 / tf.square(moment2) - 3    
            moment2 = sess.run([moment2])

        return moment2
    

    def word_contribution(self, document, axis, filter_stopword = True, min_freq = 10, to_filter = set([])):
        terms_filtered, frequencies_filtered, words =  self._prepare_matrix_computation(document, filter_stopword, min_freq, to_filter)  
        axis_mat = np.array([CoreUtil.map_axis_to_vec(self.embedding, axis)])
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        with tf.Session() as sess:
            axis_tfm_nn = tf.nn.l2_normalize(tf.constant(axis_mat), axis = 1)
            frequencies_vec = tf.constant(frequencies_filtered, dtype=np.float32)
            term_freq_tfm_nn = tf.nn.l2_normalize(tf.constant(np.array(terms_filtered)), axis = 1)

            result = tf.matmul(axis_tfm_nn, term_freq_tfm_nn,
                             adjoint_b = True # transpose second matrix
                             )
            ws = tf.multiply(result, frequencies_vec)
            contrib = ws/tf.reduce_sum(frequencies_vec)
            contrib = sess.run([contrib])
        return contrib, words


    def word_contribution_with_mask(self, document, axis, filter_stopword = True, min_freq = 10, mask_percent = 5):
        terms_filtered, frequencies_filtered, words =  self._prepare_matrix_computation(document, filter_stopword, min_freq)  
        axis_mat = np.array([self.axes[axis]])
        import tensorflow as tf        
        tf.reset_default_graph()
        with tf.Session() as sess:
            axis_tfm_nn = tf.nn.l2_normalize(tf.constant(axis_mat), axis = 1)
            frequencies_vec = tf.constant(frequencies_filtered, dtype=np.float32)
            term_freq_tfm_nn = tf.nn.l2_normalize(tf.constant(np.array(terms_filtered)), axis = 1)

            test = tf.matmul(axis_tfm_nn, term_freq_tfm_nn,
                             adjoint_b = True # transpose second matrix
                             )
            result, masked_frequencies_tsr = self._apply_mask(test, frequencies_vec, mask_percent)

            ws = tf.multiply(result, masked_frequencies_tsr)
            contrib = ws/tf.reduce_sum(masked_frequencies_tsr, 1)
            contrib = sess.run([contrib])
        return contrib, words        

    def word_contribution_to_second_moment(self, document, axis, corpus_mean, filter_stopword = True, min_freq = 10, to_filter = set([])):
        terms_filtered, frequencies_filtered, words =  self._prepare_matrix_computation(document, filter_stopword, min_freq, to_filter)  
        axis_mat = np.array([self.axes[axis]])
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        tf.reset_default_graph()
        with tf.Session() as sess:
            axis_tfm_nn = tf.nn.l2_normalize(tf.constant(axis_mat), axis = 1)
            frequencies_vec = tf.constant(frequencies_filtered, dtype=np.float32)
            term_freq_tfm_nn = tf.nn.l2_normalize(tf.constant(np.array(terms_filtered)), axis = 1)

            result = tf.matmul(axis_tfm_nn, term_freq_tfm_nn,
                             adjoint_b = True # transpose second matrix
                             )
            ws = tf.multiply(result, frequencies_vec)
            mean = tf.constant(corpus_mean, dtype=np.float32)
            # else:
            #     mean = tf.reduce_sum(ws, 1)/tf.reduce_sum(frequencies_vec)
                
            # ss = tf.reduce_sum(tf.multiply(tf.square(result), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            # ms = tf.square(mean)
            # std = tf.sqrt(ss-ms)
            diff = result-tf.reshape(mean, [-1,1])
            # moment4 = tf.reduce_sum(tf.multiply(tf.pow(diff, 4), frequencies_vec), 1)/tf.reduce_sum(frequencies_vec)
            contrib = tf.multiply(tf.pow(diff, 2), frequencies_vec)/tf.reduce_sum(frequencies_vec)
            # kurtosis = moment4 / tf.square(moment2) - 3    
            contrib = sess.run([contrib])
        return contrib, words     

    def _prepare_contribution_mat_and_tf(self, documents, filter_stopword, min_freq):
            vectorizer = CountVectorizer()
            transformed_data = vectorizer.fit_transform(documents).toarray() #return dense matrix
            frequencies = np.ravel(transformed_data.sum(axis=0))
            english_stopwords = set(stopwords.words('english')) if filter_stopword else set()

            less_freq_idx = np.where(frequencies < min_freq)[0]
            no_emb_idx = [idx for w, idx in vectorizer.vocabulary_.items() if w not in self.embedding]
            stopwords_idx = [idx for w, idx in vectorizer.vocabulary_.items() if w in english_stopwords]
            to_filter_out = set(less_freq_idx) | set(no_emb_idx) | set(stopwords_idx)
            freq_filtered = np.delete(transformed_data, sorted(to_filter_out), axis=1) # delete terms appearing less than min_freq

            self.doc_idx_to_filter = np.where(np.sum(freq_filtered,axis=1)<1)[0] # delete docs where all terms appear less than min_freq
            # vocabulary_ iterates in first-seen order; sort by column index so rows match freq_filtered columns
            kept = [w for w, idx in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]) if idx not in to_filter_out]
            self.terms_words = kept
            terms_filtered = [self.embedding[w] for w in kept]
            if self.axes_mat is None:
                self.axes_mat = np.array([self.axes[k] for k in sorted(self.axes)])

            return terms_filtered, freq_filtered

    def _precompute_word_contribution_mat(self, documents, filter_stopword = True, min_freq = 10):
        """Precompute every term's score on every axis for a corpus of documents.

        Terms are dropped here if they appear fewer than ``min_freq`` times across
        ALL documents. The per-document methods instead apply ``min_freq`` within
        each document; pass ``min_freq`` (and ``to_filter``) to the
        ``*_from_contribution_mat`` methods to get the same result.
        """
        self.terms_filtered, self.doc_freqs = self._prepare_contribution_mat_and_tf(documents, filter_stopword, min_freq)
        axes_nn = self.axes_mat / np.linalg.norm(self.axes_mat, axis=1, keepdims=True)
        terms = np.array(self.terms_filtered, dtype=np.float64).reshape(len(self.terms_filtered), -1)
        terms_nn = terms / np.linalg.norm(terms, axis=1, keepdims=True)
        self.contributions = axes_nn @ terms_nn.T  # axes x terms

    def _document_freqs(self, doc_idx, min_freq, to_filter):
        freqs = np.array(self.doc_freqs[doc_idx, :], dtype=np.float64)
        if min_freq is not None:
            freqs[freqs < min_freq] = 0.0
        if to_filter:
            to_filter = set(to_filter)
            freqs[[i for i, w in enumerate(self.terms_words) if w in to_filter]] = 0.0
        return freqs

    def compute_document_mean_from_contribution_mat(self, doc_idx, min_freq = None, to_filter = set([])):
        """Frequency-weighted mean score per axis for document ``doc_idx``.

        ``min_freq`` and ``to_filter`` drop terms within this document, as the
        per-document ``compute_document_mean_with_tf`` does. Returns NaN on every
        axis when no term is left.
        """
        freqs = self._document_freqs(doc_idx, min_freq, to_filter)
        with np.errstate(invalid='ignore', divide='ignore'):
            return (self.contributions @ freqs) / freqs.sum()

    def compute_document_second_moment_from_contribution_mat(self, doc_idx, corpus_mean, min_freq = None, to_filter = set([])):
        """Second moment around ``corpus_mean`` per axis for document ``doc_idx``.

        Same filtering as ``compute_document_mean_from_contribution_mat``; matches
        ``compute_document_second_moment_with_tf``.
        """
        freqs = self._document_freqs(doc_idx, min_freq, to_filter)
        diff = self.contributions - np.reshape(np.asarray(corpus_mean, dtype=np.float64), (-1, 1))
        with np.errstate(invalid='ignore', divide='ignore'):
            return ((diff ** 2) @ freqs) / freqs.sum()
