import pickle
import numpy as np
from collections import Counter
import math
import pandas as pd

#
#
with open(f'index.pkl', 'rb') as inp:
    inverted = pickle.load(inp)
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
                self._open_files[f_name] = open(f_name, 'rb')
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
        list_of_doc = read_posting_list(inverted, term)
        if len(list_of_doc)>0:
            normlized_tfidf = [(doc_id, freq * math.log(N / index.df[term], 10)) for doc_id, freq in
                               list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

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
    left = sum([a ** 2 for a in Q])
    for i in range(len(D)):
        x=0
        # ser=pd.Series(D.iloc[i])
        # min=ser.min()
        # max=ser.max()
        # diff=max-min
        # if(diff==0):
        #     diff=1
        for j in D.iloc[i]:
            if(j!=0):
                x+=1
        up = sum([a * b for a, b in zip(Q, D.iloc[i])])
        up=up*x

        #up=up/diff
        right = sum([a ** 2 for a in D.iloc[i]])
        sum_down = (left * right) ** 0.5
        total = up / sum_down
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


#print(inverted.df['constructing'])

#print(list(inverted.posting_locs.values())[100000])
#
# pl = read_posting_list(inverted, 'python')
#
# for tup in pl:
#     if tup[0]== 23862:
#         print(tup)


Q=(generate_query_tfidf_vector(['rubber', 'duck'],inverted))


D=generate_document_tfidf_matrix(['rubber', 'duck'],inverted)

shortQ=[]
for x in Q:
    if (x!=0):
        shortQ.append(x)
# #print(shortQ)

cs=cosine_similarity(D, shortQ)
topn=get_top_n(cs, N=100)
print(topn)


# for i in cs:
#     if cs[i]!=0:
#      print(i)



# print(words)
# print(pls)

# pl = read_posting_list(inverted, 'science')
#
# print(pl)
# for tup in pl:
#     if tup[0]== 35458904:
#         print(tup)


#with open('IndexTry.pkl', 'wb') as outp:
    #pickle.dump(inverted, outp, pickle.HIGHEST_PROTOCOL)


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


# true_list=[599738, 21905962, 39322520, 2608233, 12234103, 6264793, 6497307, 39303262, 54655786, 3378227, 64516, 41300601, 27898480, 672302, 1793236, 1765938, 1179787, 22503790, 43399069, 6263016, 31468446, 23030371, 1540241, 38904159, 11882165, 43707, 54406331, 3275717, 3781845, 57228339, 312903, 16308445, 8277225, 864249, 3115732, 15190087, 2541491, 1923274, 37400147, 47987665, 978408, 36835927, 2589070, 559356, 19696772, 17069844, 13315204, 34829413, 23454546, 2045055, 311935, 37674, 582735, 52077192, 4463369, 11760348]
#
# res=[(11027248, 1005.19427), (2018532, 938.18132), (18792945, 883.35254), (462670, 840.70794), (1179787, 834.61585), (599738, 560.47196), (36206195, 530.01152), (38393, 520.8734), (77548, 505.64318), (19837004, 493.45901), (66632680, 472.1367), (19179331, 469.09066), (39322520, 426.44605), (5835638, 395.98562), (40275772, 392.93958), (17069844, 389.89354), (18792948, 386.84749), (12863003, 380.75541), (37674, 353.34102), (5798464, 347.24893), (3417508, 328.97267), (13091964, 319.83454), (74961, 304.60432), (286855, 301.55828), (93996, 289.37411), (29704635, 268.05181), (512470, 265.00576), (188408, 255.86763), (1765938, 249.77555), (14055573, 246.7295), (57113470, 246.7295), (29378, 237.59137), (7623862, 237.59137), (24975724, 231.49929), (272599, 225.4072), (23451635, 216.26907), (16085319, 210.17698), (64473465, 210.17698), (43707, 201.03885), (230149, 201.03885), (1738731, 201.03885), (54655786, 201.03885), (282854, 197.99281), (39956275, 197.99281), (100039, 194.94677), (2554026, 185.80864), (4100225, 182.76259), (19385859, 176.67051), (33698386, 176.67051), (63477845, 173.62447), (2579000, 167.53238), (10432306, 164.48634), (291791, 161.44029), (3654631, 158.39425), (62408906, 158.39425), (51847, 155.34821), (507836, 155.34821), (2576325, 155.34821), (9020, 152.30216), (364625, 152.30216), (1488795, 152.30216), (3467679, 149.25612), (18792950, 149.25612), (44748, 146.21008), (2608233, 146.21008), (30524712, 146.21008), (103367, 143.16403), (17350757, 143.16403), (65037, 140.11799), (1073628, 140.11799), (6497307, 140.11799), (190287, 137.07195), (3825974, 137.07195), (10768534, 137.07195), (39286085, 137.07195), (57161669, 137.07195), (76029, 134.0259), (2397993, 134.0259), (3192040, 134.0259), (18053246, 134.0259), (20286795, 134.0259), (7345, 130.97986), (73440, 130.97986), (28285446, 130.97986), (35397113, 130.97986), (57911638, 130.97986), (61717911, 130.97986), (184918, 127.93382), (1241652, 127.93382), (3439992, 127.93382), (23433315, 127.93382), (204174, 121.84173), (3904119, 121.84173), (7197997, 121.84173), (8993, 118.79569), (31411, 118.79569), (122374, 118.79569), (2133681, 118.79569), (3995, 115.74964), (183340, 115.74964)]
#
# pred_wids, _ = zip(*res)
#
# print(average_precision(true_list,pred_wids))