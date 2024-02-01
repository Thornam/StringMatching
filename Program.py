#  Importing libraries and module and some setting for notebookimport pandas as pd 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct 
from fuzzywuzzy import fuzz
import pandas as pd


def pre_process(string):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    string = fuzz.utils.full_process(string)
    return string

def ngrams(string, n=3):
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)
    return csr_matrix((data,indices,indptr),shape=(M,N))


# unpacks the resulting sparse matrix
def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similairity': similairity})








orbis_temp = orbis[ orbis['Country_'] == c_]
preqin_temp = preqin[ preqin['PORTFOLIO COMPANY COUNTRY'] == c_]

preqin_company = preqin_temp['Company'].unique()
orbis_company = orbis_temp.iloc[:, 1].unique()

del preqin_temp, orbis_temp

collected = list(preqin_company) + list(orbis_company)

del orbis_company

collected = [fuzz.utils.full_process(com) for com in collected]

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix_c = vectorizer.fit_transform(collected)

#  Top 4 with similarity above 0.8
matches = awesome_cossim_top(tf_idf_matrix_c, tf_idf_matrix_c.transpose(), 3, 0.8) # Muligvis opdateres til 4

matches_df = get_matches_df(matches, collected, top = False)
matches_df = matches_df[matches_df['similairity'] < 0.99999] # For removing all exact matches

del matches, collected

for m in range(0, len(matches_df)):
    #if (matches_df.iloc[m, 0] == matches_df.iloc[m, 1]) & (matches_df.iloc[m, 0] in preqin_company):
    #    in_.append(matches_df.iloc[m, 0])
    if (matches_df.iloc[m, 0] in preqin_company) & (matches_df.iloc[m, 1] not in preqin_company):
        in_.append(matches_df.iloc[m, 0])
    elif (matches_df.iloc[m, 0] not in preqin_company) & (matches_df.iloc[m, 1] in preqin_company):
        in_.append(matches_df.iloc[m, 1])