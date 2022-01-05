import pickle
import numpy as np
from collections import Counter
import math
import pandas as pd
from operator import itemgetter
import re
from nltk.stem.porter import *
from nltk.corpus import stopwords
import nltk
from time import time
from google.cloud import storage

# bucket_name = '315302083'
# index_dst = f'gs://{bucket_name}/postings_gcp/index.pkl'
# print(index_dst)
# client = storage.Client()
# blobs = client.list_blobs(bucket_name)
# for b in blobs:
#     print(b.name)

#nltk.download('stopwords')


#
#
with open(f'postings_gcp\index.pkl', 'rb') as inp:
    inverted = pickle.load(inp)

# with open(f'gs://{bucket_name}/postings_gcp/index.pkl', 'rb') as inp:
#     inverted = pickle.load(inp)


#print(len(inverted.df))

#words, pls = zip(*inverted.posting_lists_iter())

# with open('BigIndex.pkl', 'wb') as outp:
#     pickle.dump(inverted, outp, pickle.HIGHEST_PROTOCOL)


# super_posting_locs = defaultdict(list)
# for blob in client.list_blobs(bucket_name, prefix='postings_gcp'):
#     if not blob.name.endswith("pickle"):
#         continue
#     with blob.open("rb") as f:
#         posting_locs = pickle.load(f)
#         for k, v in posting_locs.items():
#               super_posting_locs[k].extend(v)


def get_posting_gen(index):
    """
    This function returning the generator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter())
    return words, pls
# words, pls= get_posting_gen(inverted)
BLOCK_SIZE = 1999998

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
from contextlib import closing


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        #locs = [locs]
        for f_name, offset in locs:
            # for f_name in locs:

            if f_name not in self._open_files:
                self._open_files[f_name] = open('postings_gcp/'+f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def read_posting_list(inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list


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
    total_vocab_size = len(index.df)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.df.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            #idf = math.log((len(DL)) / (df + epsilon), 10)  # smoothing
            idf = math.log((6348910) / (df + epsilon), 10) # smoothing
            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf

            except:
                pass
    return Q




#==================================================

def get_candidate_documents_and_scores(query_to_search, index):
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
    N = 6348910
    for term in np.unique(query_to_search):
        list_of_doc = read_posting_list(index, term)
        if len(list_of_doc)>0:
            normlized_tfidf = [(doc_id, freq * math.log(N / index.df[term], 10)) for doc_id, freq in
                               list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf
    #candidates= take(20,000, candidates.items())
    candidates= dict(sorted(candidates.items(), key=itemgetter(1), reverse=True)[:18000])
    #candidates = dict(sorted(candidates.items(), key=itemgetter(1), reverse=True))
    return candidates

#++++++++++++++++++++++++++++++++++++++++++++++++++++

def generate_document_tfidf_matrix(query_to_search, index):
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

    total_vocab_size = len(index.df)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    #print(len(unique_candidates))
    #D = np.zeros((len(unique_candidates), total_vocab_size))
    D = np.zeros((len(unique_candidates), len(query_to_search)))

    D = pd.DataFrame(D)

    D.index = unique_candidates
    #D.columns = index.df.keys()\
    D.columns = query_to_search

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        if(tfidf!=0):
            D.loc[doc_id][term] = tfidf

    return D

#=====================================

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
    dic = {}
    #left = sum([a ** 2 for a in Q])
    for i in range(len(D)):
        x=0
        # ser=pd.Series(D.iloc[i])
        # min=ser.min()
        # max=ser.max()
        # diff=max-min
        # if(diff==0):
        #     diff=1
        # v=ser.var()

        # v=v/100
        #print(v)
        # ser = pd.Series(D.iloc[i])
        # var = ser.var()
        for j in D.iloc[i]:
            if(j!=0):
                x+=1
        #if(D.iloc[i].name in [3951433,62637003,103325,19492975,2009711]):

            #print((i,max-min))

        up = sum([a * b for a, b in zip(Q, D.iloc[i])])
        up=up*x
        # if D.iloc[i].name in [35458904, 27051151, 2155752]:
        #     print(v)
        #     print(up)
        # if (x > 1):
        #  up=up/var

        #up=up/diff
        # right = sum([a ** 2 for a in D.iloc[i]])
        # sum_down = (left * right) ** 0.5
        # total = up / sum_down
        #dic[int(D.iloc[i].name)] = round(total, 5)
        dic[int(D.iloc[i].name)] = round(up, 5)
    return dic


def get_top_n(sim_dict, N=3):
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


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
def tokenize(text):
  return [token.group() for token in RE_WORD.finditer(text.lower())]

stemmer = PorterStemmer()
# Getting tokens from the text while removing punctuations.
def filter_tokens(tokens, tokens2remove=None, use_stemming=False):
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
  if (tokens2remove != None):
    lst2 = []
    for i in tokens:
      if (i not in tokens2remove):
        lst2.append(i)
    tokens = lst2
  if (use_stemming == True):
    lst = []
    for i in tokens:
      lst.append(stemmer.stem(i))
    tokens = lst
  return tokens

english_stopwords = frozenset(stopwords.words('english'))
#stop = frozenset(corpus_stopwords).union(english_stopwords)
#stop_rare = stop.union(corpus_rarewords)

def tokenize_no_stop(text):
  return filter_tokens(tokenize(text), stop)
def tokenize_no_stop_rare(text):
  return filter_tokens(tokenize(text), stop_rare)
def tokenize_no_stop_stem(text):
  return filter_tokens(tokenize(text), stop_rare, True)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)



# pl = read_posting_list(inverted, 'science')
# print(pl)
# for tup in pl:
#     if tup[0]== 35458904:
#         print(tup)



word= "best marvel"
token=tokenize(word)
w=filter_tokens(token,english_stopwords)
print(w)

Q=(generate_query_tfidf_vector(w,inverted))
print(1)
D=generate_document_tfidf_matrix(w,inverted)
print(2)
#
shortQ=[]
for x in Q:
    if (x!=0):
        shortQ.append(x)
# print(shortQ)

cs=cosine_similarity(D, shortQ)
topn=get_top_n(cs, N=100)
print(topn)


'''
def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i,doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions)+1) / (i+1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions)/len(precisions),3)

import json

with open('queries_train.json', 'rt') as f:
  queries = json.load(f)

qs_res = []
a=0
for q, true_wids in queries.items():
    duration, ap = None, None
    t_start = time()
    token=tokenize(q)
    w=filter_tokens(token,english_stopwords)
    Q = (generate_query_tfidf_vector(w, inverted))
    D=generate_document_tfidf_matrix(w,inverted)
    shortQ=[]
    for x in Q:
        if (x!=0):
            shortQ.append(x)
    cs=cosine_similarity(D, shortQ)
    topn=get_top_n(cs, N=100)
    duration = time() - t_start
    #print(topn)
    pred_wids, _ = zip(*topn)
    ap = average_precision(true_wids, pred_wids)
    qs_res.append((q, duration, ap))
    print((q, duration, ap))
    a+=ap
print(qs_res)
print(a/len(qs_res))
where_does_vanilla_flavoring_come_from=['where','does','vanilla','flavoring','come','from']

'''
#3951433,62637003,103325

# rubber_duck = ['rubber', 'duck']
# data_science = ['data','science']
# python=['python']
# best_marvel_movie=['best','marvel','movie']
# migraine='migraine'
#

#
#
#
#
#
def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i,doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions)+1) / (i+1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions)/len(precisions),3)
#
#
true_list=[57069491, 65967176, 42163310, 878659, 27306717, 41677925, 1074657, 44240443, 17296107, 60952488, 43603241, 22114132, 46208997, 36450985, 41974555, 56289672, 60616450, 59502488, 33038861, 61699239, 61651800, 39368416, 29129051, 55935213, 54537218, 62372638, 60774345, 63090183, 37497391, 51430647, 67229718, 9110929, 61329320, 44254295, 41974496, 200563, 58481694, 48530084, 56289572, 22144990, 612052, 59162931, 55511148, 55511147, 61073786, 59892, 36484005, 36484254, 66423851, 62482816, 1275470, 5676692, 7927053, 60754840, 26999426, 60744481, 56289553, 60463979, 701741, 60283633, 1129847, 36439749, 4451883, 55511155, 22144721, 45359871, 723126, 43655965, 57275457, 12673434, 43867095, 26763420, 39293265, 15003874, 41668588, 61592102, 67063919, 11891433, 64057670, 61513780, 39345917, 67063906, 1221476, 41008758, 60587000, 7729, 2152196, 5027882, 509738, 403585, 26866372, 1339248, 3473503, 4148655]

res=[(18307001, 1220.62918), (18636995, 1071.44117), (21720983, 927.18499), (24572192, 801.4232), (16142831, 728.67863), (14314727, 673.19549), (3306354, 602.91684), (4036172, 573.32583), (28174239, 546.20073), (14551473, 541.2689), (30172694, 517.84268), (7317283, 504.28014), (57977922, 495.64943), (4224440, 488.25167), (14938664, 485.78576), (3663849, 479.62096), (9425157, 474.68913), (24604354, 472.22321), (7706340, 470.99025), (47207631, 456.19474), (303981, 420.43894), (3362779, 406.87639), (24837269, 399.47864), (60983259, 399.47864), (46736596, 398.24568), (64071066, 397.01272), (2794268, 392.08089), (16831542, 383.45018), (1241849, 376.05242), (11304669, 374.81947), (67175796, 374.81947), (625413, 353.85917), (13215306, 348.92733), (24146034, 345.22846), (7081866, 343.9955), (42773108, 343.9955), (21475702, 337.8307), (20842611, 336.59774), (36048519, 336.59774), (4151001, 331.66591), (41375240, 330.43295), (52583447, 315.63744), (53070348, 314.40449), (60727309, 314.40449), (35232529, 309.47265), (11605778, 307.00673), (42773116, 307.00673), (9149389, 300.84194), (13020084, 294.67715), (52583855, 294.67715), (3627682, 287.27939), (3382456, 286.04643), (33967871, 286.04643), (901742, 284.81348), (3017764, 283.58052), (32398891, 283.58052), (56818610, 278.64868), (1964969, 277.41572), (7893660, 277.41572), (41579194, 276.18276), (30839552, 274.94981), (35787446, 273.71685), (44936694, 271.25093), (56025646, 270.01797), (7369530, 268.78501), (14030366, 268.78501), (21834676, 268.78501), (59204030, 265.08614), (2845957, 262.62022), (52592185, 262.62022), (12839459, 261.38726), (38740754, 261.38726), (1488937, 260.1543), (36890459, 260.1543), (12523627, 258.92134), (6236356, 257.68838), (21281430, 256.45542), (23199340, 256.45542), (55997551, 256.45542), (39467037, 253.98951), (47441413, 252.75655), (1856911, 251.52359), (2830666, 249.05767), (22292298, 247.82471), (33740155, 246.59175), (53654218, 246.59175), (66570306, 246.59175), (6027734, 244.12584), (25645175, 244.12584), (31868355, 244.12584), (33790441, 242.89288), (62501055, 242.89288), (1070235, 241.65992), (3483096, 236.72808), (11088709, 235.49512), (57461112, 234.26217), (22421210, 231.79625), (151343, 230.56329), (58823153, 229.33033), (59241382, 228.09737)]

pred_wids, _ = zip(*res)

print(average_precision(true_list,pred_wids))