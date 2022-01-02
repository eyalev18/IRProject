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
        list_of_doc = read_posting_list(inverted, 'python')
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

    raise NotImplementedError()





#print(list(inverted.posting_locs.values())[100000])
#
# pl = read_posting_list(inverted, 'python')
#
# print(pl)

Q=(generate_query_tfidf_vector(['data','science'],inverted))
#l=get_candidate_documents_and_scores(['data','science'],inverted)
D=generate_document_tfidf_matrix(['data','science'],inverted)
cs=cosine_similarity(D, Q)
print(cs)



# print(words)
# print(pls)

# pl = read_posting_list(inverted, 'python')
#
# print((pl))


#with open('IndexTry.pkl', 'wb') as outp:
    #pickle.dump(inverted, outp, pickle.HIGHEST_PROTOCOL)

