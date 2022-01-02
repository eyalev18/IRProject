import math

from pyspark.shell import spark
import pyspark
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
import pandas as pd
from google.cloud import storage

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')


from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql import SQLContext
from graphframes import *





english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124
def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

def word_count(text, id):
  ''' Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  '''
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  # YOUR CODE HERE
  ret_lst=[]
  no_dups= Counter(tokens)
  for w in no_dups:
    if(w not in all_stopwords):
     tup= (w,(id,no_dups[w]))
     ret_lst.append(tup)
  return ret_lst

def reduce_word_counts(unsorted_pl):
  ''' Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples
  Returns:
  --------
    list of tuples
      A sorted posting list.
  '''
  # YOUR CODE HERE
  return list(unsorted_pl)

def calculate_df(postings):
  ''' Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, df) pair.
  '''
  # YOUR CODE HERE
  return postings.map(lambda w:(w[0],len(w[1])))


def partition_postings_and_write(postings):
    ''' A function that partitions the posting lists into buckets, writes out
    all posting lists in a bucket to disk, and returns the posting locations for
    each bucket. Partitioning should be done through the use of `token2bucket`
    above. Writing to disk should use the function  `write_a_posting_list`, a
    static method implemented in inverted_index_colab.py under the InvertedIndex
    class.
    Parameters:
    -----------
      postings: RDD
        An RDD where each item is a (w, posting_list) pair.
    Returns:
    --------
      RDD
        An RDD where each item is a posting locations dictionary for a bucket. The
        posting locations maintain a list for each word of file locations and
        offsets its posting list was written to. See `write_a_posting_list` for
        more details.
    '''
    # YOUR CODE HERE
    with open(r'C:\Users\Owner\PycharmProjects\IRPtoject\IndexTry.pkl', 'rb') as inp:
        InvertedIndex = pickle.load(inp)
    #print(inverted.df)
    rdd = postings.map(lambda s: (token2bucket_id(s[0]), [(s[0], s[1])]))
    rdd2 = rdd.reduceByKey(lambda x, y: x + y)
    rdd3 = rdd2.map(lambda s: InvertedIndex.write_a_posting_list(s))
    return rdd3

#
full_path = "gs://wikidata_preprocessed/*"
parquetFile = spark.read.parquet(full_path)
doc_text_pairs = parquetFile.select("text", "id").rdd



word_counts = doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))

postings = word_counts.groupByKey().mapValues(reduce_word_counts)
postings_filtered = postings.filter(lambda x: len(x[1])>50)
w2df = calculate_df(postings_filtered)
w2df_dict = w2df.collectAsMap()
posting_locs_list = partition_postings_and_write(postings_filtered).collect()
super_posting_locs = defaultdict(list)
for posting_loc in posting_locs_list:
  for k, v in posting_loc.items():
    super_posting_locs[k].extend(v)


# Create inverted index instance








from nltk.stem.porter import *
from nltk.corpus import stopwords

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
def tokenize(text):
  return [token.group() for token in RE_WORD.finditer(text.lower())]


stemmer = PorterStemmer()
english_stopwords = frozenset(stopwords.words('english'))


# Getting tokens from the text while removing punctuations.
def filter_tokens(tokens, tokens2remove=english_stopwords, use_stemming=True):
    ''' The function takes a list of tokens, filters out `tokens2remove` and
        stem the tokens using `stemmer`.
    Parameters:
    -----------
    tokens: list of str.
      Input tokens.
    tokens2remove: frozenset.
      Tokens to remove (before stemming).
    use_stemming: bool.
      If true, apply stemmer.stem on tokens.
    Returns:
    --------
    list of tokens from the text.
    '''
    # YOUR CODE HERE

    if (tokens2remove is not None):
        # tokens= list(filter(lambda word:word not in list(tokens2remove),tokens))
        future_tokens = []
        for tok in tokens:
            if (tok not in tokens2remove):
                future_tokens.append(tok)
        tokens = future_tokens
    if (use_stemming != False):
        ret = []
        for w in range(len(tokens)):
            ret.append(stemmer.stem(tokens[w]))
        tokens = ret

    return tokens

import numpy as np
DL = {}

def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(DL)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq / DL[str(doc_id)]) * math.log(N / index.df[term], 10)) for doc_id, freq in
                               list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(index.term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    # YOUR CODE HERE
    ret_dic = {}
    for num in range(len(D)):
        co = sum([o * t for o, t in zip(Q, D.iloc[num])])

        Qsum = sum([o * o for o in Q])

        Dsum = sum([o * o for o in D.iloc[num]])

        QanDmul = Qsum * Dsum
        QandD = QanDmul ** 0.5
        final = co / QandD
        place = int(D.iloc[num].name)
        ret_dic[place] = round(final, 5)
    return ret_dic


def get_top_n(sim_dict, N=100):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


def get_posting_gen(index):
    """
    This function returning the generator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter())
    return words, pls
