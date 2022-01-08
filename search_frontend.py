import pickle
import numpy as np
from collections import Counter
import math
import pandas as pd
from operator import itemgetter
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import *
from nltk.corpus import stopwords

from time import time
from google.cloud import storage
import os

from flask import Flask, request, jsonify

#============================================
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
                #self._open_files[f_name] = open(f_name, 'rb')
                self._open_files[f_name] = open('postings_gcp/'+f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def title_read(self, locs, n_bytes):
        b = []
        #locs = [locs]
        for f_name, offset in locs:
            # for f_name in locs:

            if f_name not in self._open_files:
                self._open_files[f_name] = open('postings_title/'+f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def anchor_read(self, locs, n_bytes):
        b = []
        #locs = [locs]
        for f_name, offset in locs:
            # for f_name in locs:

            if f_name not in self._open_files:
                self._open_files[f_name] = open('postings_anchor/'+f_name, 'rb')
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

def read_posting_list_title(inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.title_read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list


def read_posting_list_anchor(inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.anchor_read(locs, inverted.df[w] * TUPLE_SIZE)
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





os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='myfirstproject-329911-1e93f669b9fa.json'


# with open(f'postings_gcp/index.pkl', 'rb') as inp:
#         inverted = pickle.load(inp)
#
with open(f'postings_title\TitleIndex.pkl', 'rb') as inp:
    title_index = pickle.load(inp)
#
# df = pd.read_csv('PageRank.csv', header=None)
# df.columns = ['0', '1']

with open(f'postings_anchor/AnchorIndex.pkl', 'rb') as inp:
        anchor_index = pickle.load(inp)



with open(f'pageviews.pkl', 'rb') as f:
  wid2pv = pickle.loads(f.read())

#==================

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False



@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION

    # with open(f'postings_gcp\index.pkl', 'rb') as inp:
    #     inverted = pickle.load(inp)

    token = tokenize(query)
    w = filter_tokens(token, english_stopwords)
    Q = (generate_query_tfidf_vector(w, inverted))
    D = generate_document_tfidf_matrix(w, inverted)
    shortQ = []
    for x in Q:
        if (x != 0):
            shortQ.append(x)
    cs = cosine_similarity(D, shortQ)
    res = get_top_n(cs, N=100)
    for num in range(len(res)):
        tup=(res[num][0],title_index.titles[res[num][0]])
        res[num]=tup

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    token = tokenize(query)
    w = filter_tokens(token, english_stopwords)
    dic={}
    for word in w:
        x=read_posting_list_title(title_index,word)
        for post in x:
            if post[0] not in dic:
                dic[post[0]]=1
            else:
                dic[post[0]]= dic[post[0]]+1
    #res= [(doc_id, score) for doc_id, score in dic.items()]
    res= sorted([(doc_id, score) for doc_id, score in dic.items()], key=lambda x: x[1], reverse=True)
    for num in range(len(res)):
        tup=(res[num][0],title_index.titles[res[num][0]])
        res[num]=tup



    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION


    token = tokenize(query)
    w = filter_tokens(token, english_stopwords)
    dic={}
    for word in w:
        x=read_posting_list_anchor(anchor_index,word)
        for post in x:
            if post[0] not in dic:
                dic[post[0]]=1
            else:
                dic[post[0]]= dic[post[0]]+1
    #res= [(doc_id, score) for doc_id, score in dic.items()]
    res= sorted([(doc_id, score) for doc_id, score in dic.items()], key=lambda x: x[1], reverse=True)
    for num in range(len(res)):
        tup=(res[num][0],title_index.titles[res[num][0]])
        res[num]=tup




    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION


    # print(df)



    for i in wiki_ids:
        res.append(df.loc[df['0'] == i, '1'].iloc[0])
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLU for i in wiki_ids:
    #         res.append(wid2pv[i])TION

    for i in wiki_ids:
        res.append(wid2pv[i])

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)




#==================================================================
