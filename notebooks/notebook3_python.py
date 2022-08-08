# %% [markdown]
# # TO-DOs
# 
# + Init lda to NMF. Sufficient to manage to install this modified version of lda package, needed for init lda to NMF:
# ```!pip install --upgrade git+git://github.com/llaurabat91/lda.git``` (currently not working) (C)

# %%
# # Install necessary packages
# !pip install gdown
# !pip install gensim==3.8.3

# %%
# !pip install lda

# %%
# Restart RUNTIME after installing packages!

# %%
import pandas as pd
import numpy as np
import gdown
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product, combinations_with_replacement
import lda
import pickle

# %%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import linalg
from sklearn.decomposition import NMF
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.downloader as api
import itertools

import sys
sys.path.append('../pymodules')

# %%
# run if interested in the Additional section

# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 118)
# pd.set_option('display.max_colwidth', 1000)
# pd.set_option('display.width', 10000)

# %%
# define paths and seed
SEED = 92
data_path = "./"

# %%
# define dictionary with paths to data in Google Drive
data_url_dict = {"10K_vocab_2019_min25": ("https://drive.google.com/uc?id=1va-ob3C7UD0pYV4Z6WEhL2WMP7ynQJCH", "csv"),
                "10K_dtmatrix_2019_min25": ("https://drive.google.com/uc?id=1c-lNSgaj3tKjnkpn_12ddCL3OP6IsXLT", "txt"),
                "10K_raw_text_2019_min25": ("https://drive.google.com/uc?id=1T97btZK10417MNXGlx_ASFY0mzqpNtu9", "csv"),
                "w2v_sg_min25": ("https://drive.google.com/uc?id=1JpP3oxpZ1Dr5vTyutwIKIUoknsa418YF", "sav"),
                "glove_min25": ("https://drive.google.com/uc?id=1hmXKSrmRV2P6wr7C20s7fAhKUy3gp7SO", "sav"),
                "tic_cik_crosswalk": ("https://drive.google.com/uc?id=1YsbeWapKf_hvfP3qMo9-Xj-PFuH5R96o", "csv")}

# %%
# download each file in the dictionary (it takes a while)
for file_name, attributes in data_url_dict.items():
    url = attributes[0]
    extension = attributes[1]
    gdown.download(url, f"{file_name}.{extension}", quiet=False) 

# %%
# if you can't run any of the models, you can load results from here
# data_url_dict = {"X_tfidf": ("https://drive.google.com/uc?id=1uYYDcL_1P9dB6HKqm726DZ3q02V21XA_", "npy"),
#                 "X_tfidf_w2v_sg": ("https://drive.google.com/uc?id=1jqmSngcW6JA4ESyrjvQk9lheNspLZ6Sl", "npy"),
#                 "X_tfidf_glove": ("https://drive.google.com/uc?id=14FO6USCeRLrJ1_Nfjgs23a3BBgSj_lZw", "npy"),
#                 "X_NMF": ("https://drive.google.com/uc?id=1sROI2YFlGQU9CjHlUGek6zMwDLZxVDJv", "npy"),
#                 "X_LSA": ("https://drive.google.com/uc?id=12tEYC8D1YdEMHh_U3UC3J9YlbUreqiRg", "npy"),
#                 "X_lda": ("https://drive.google.com/uc?id=11AB4gSjmPD8z069TsczTYbEZ7t5pXHy-", "npy"),
#                 "X_d2v": ("https://drive.google.com/uc?id=1inbsTzUlXo56zSR-LZdvodIWGZaDiNv0", "npy"),
#                 "X_countvec": ("https://drive.google.com/uc?id=1M0O2GqEjPubQc906xRrG3NrF3ow2tN3X", "npy"),
#                 "X_avgw2v": ("https://drive.google.com/uc?id=1Tz1JTccsWcOxW5HMwtfZ0Ii7JKMOe3-o", "npy"),
#                 "X_avglove": ("https://drive.google.com/uc?id=1lv9qSazyLMpwyKAb-rXTknF4K0MSnMqr", "npy"),
#                 "X_avglove_pretrained": ("https://drive.google.com/uc?id=1daFs4bRcchxk0Q54w4mcGUYh49VpmyHb", "npy"),
#                 }

# for file_name, attributes in data_url_dict.items():
#     url = attributes[0]
#     extension = attributes[1]
#     gdown.download(url, f"{file_name}.{extension}", quiet=False) 

# %% [markdown]
# # 1. Load data

# %% [markdown]
# This tutorial uses text data from the **10-K reports** filed by most publicly-traded firms in the U.S. in 2019. 10-K reports are a very rich source of data since firms include information regarding their organizational structure, financial performance and risk factors. We will use a version of the data where the risk factors section of each report has been splitted into sentences and each sentence has been assigned an ID that combines the firm identifier (i.e. **CIK**) and a sentence number. The data we use has a total of 1,077,416 sentences for 2,500 firms.
# 
# More on the 10-K reports [here](https://www.investor.gov/introduction-investing/getting-started/researching-investments/how-read-10-k).

# %%
raw_data = pd.read_csv(data_path + '10K_raw_text_2019_min25.csv', index_col='Unnamed: 0')

# %%
dt_mat = pd.read_csv(data_path + '10K_dtmatrix_2019_min25.txt', index_col='Unnamed: 0')
dt_vals = dt_mat.values

# %%
dt_mat.shape

# %%
vocab_df = pd.read_csv(data_path + '10K_vocab_2019_min25.csv', index_col='Unnamed: 0', keep_default_na=False)

# %%
vocab = vocab_df.to_dict()['0']

# %%
tics = pd.read_csv(data_path+'tic_cik_crosswalk.csv') #3196
raw_data.cik.isin(tics.cik).sum()

# %%
# alternative ticker datasets (even fewer matches)

# tickers = pd.read_csv('../../cik_ticker.csv', sep='|') #2684
# tickers2 = pd.read_csv('../../ticker_2.txt', sep='\t', header=None, names=['ticker', 'cik']) #3183

# %%
df = raw_data.merge(tics, on='cik', how='left').copy()

# %% [markdown]
# # Documents as frequency vectors: an illustration

# %%
names_cosine =  ['AAPL', 'GOOGL', 'KO', 'PEP', 'DAL', 'LUV', 'CALM', 'BALL', 'BUKS']
idxs_cosine = np.array(df[df.tic.isin(names_cosine)].index)
cosine_dict = {k:v for k,v in zip(names_cosine, idxs_cosine)}

# %%
cosine_similarities = {f'{k1}_{k2}':cosine_similarity(dt_vals[cosine_dict[k1]].reshape(1, -1), 
                                                      dt_vals[cosine_dict[k2]].reshape(1, -1))[0][0] 
                       for k1, k2 in list(combinations_with_replacement(names_cosine, 2))
                      if k1!=k2}
cosine_similarities = {k: v for k, v in sorted(cosine_similarities.items(), key=lambda item: item[1])}

# %%
fig, ax = plt.subplots(figsize=(22,5))

k = len(cosine_similarities)
sims = list(cosine_similarities.values())
names = list(cosine_similarities.keys())

ax.scatter(sims, np.zeros((k,)) )
for i in range(k):
    ax.annotate(names[i], (sims[i], 0+0.005), rotation=45)
ax.set_ylim(bottom=-0.005, top=0.03)
ax.set_xlabel('Cosine similarity', fontsize=16)
ax.set_yticks([])    
plt.show()

# %%
companies =  ['AAPL', 'GOOGL', 'KO', 'PEP', 'DAL', 'LUV']
temp = df[df.tic.isin(companies)]
idxs_comps = np.array(temp.index).astype(int)
names_comps = temp.tic.values

# %%
target_dict = {'customer':4394, 'airline':1172, 'environment':5876, 'price':12951, 'product':13154,
               'outsourcing':11949}
terms = list(target_dict.keys())
ixs_terms = np.array(list(target_dict.values())).astype(int)

# %%
heatmap_df = pd.DataFrame(dt_vals[idxs_comps][:,ixs_terms], columns=terms, index=names_comps)
heatmap_df

# %%
heatmap_df_rel = heatmap_df/heatmap_df.sum(0)

# %%
heatmap_df_rel

# %%
sns.heatmap(heatmap_df_rel, cmap='Blues')#, annot=heatmap_df)
plt.show()

# %%
# alternative visualization (not very insightful)
# fig, ax = plt.subplots()

# ax.scatter(x=heatmap_df['price'], y=heatmap_df['product'])
# ax.set_xlabel('price', fontsize=12)
# ax.set_ylabel('product', fontsize=12)

# names = ['AAPL', 'GOOGL', 'KO', 'PEP', 'DAL', 'LUV']
# for i in range(len(heatmap_df['price'])):
#     plt.annotate(names[i], (heatmap_df['price'][i]+0.2, heatmap_df['product'][i] + 0.6))

# plt.show()

# %% [markdown]
# # Estimate models

# %%
# %%
# Get BoW and set K
X = dt_vals
K = 20

# %%
%%time
# %%
# few seconds
# Count Vectorizer
vectorizer =CountVectorizer()
X_countvec =vectorizer.fit_transform(df.tokens_25)

X_countvec = np.asarray(X_countvec.todense())

# %%
np.save(data_path + 'X_countvec.npy', X_countvec)

# %%
# %%
# few seconds
# TFIDF
tfidf =TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
X_tfidf =tfidf.fit_transform(X)

X_tfidf = np.asarray(X_tfidf.todense())

# %%
# %%
np.save(data_path + 'X_tfidf.npy', X_tfidf)

# %%
%%time
# %%
# 8 min
# LSA
K2 = 200
U, D, Vt = linalg.svd(X)
print(U.shape,  D.shape, Vt.shape)

Vt_r = Vt[:K2]
print(Vt_r.shape)

X_LSA = X@Vt_r.T
print(X_LSA.shape)
# %%

# %%
np.save(data_path + 'X_LSA_K200.npy', X_LSA)

# %%
%%time
# %%
# 10min
# NMF

NMFmodel = NMF(n_components=K, init='random', random_state=0,
max_iter=800)
X_NMF = NMFmodel.fit_transform(X)

# %%
# %%
np.save(data_path + 'X_NMF.npy', X_NMF)

# %%
%%time
# 30min
# LDA
X = X.astype('int64')
lda_model = lda.LDA(n_topics=K, n_iter=1500, random_state=1)
X_lda =  lda_model.fit_transform(X) 

# %%
#%%
np.save(data_path + 'X_lda.npy', X_lda)
#%%
# X_lda = np.load(save_path + 'X_lda.npy')
#%%

# %%
%%time
# doc-to-vec
# 9 min
# tag documents
tagged_data = [TaggedDocument(words=doc, 
tags=[str(i)]) for i, doc in enumerate(df.tokens_25.values)]

#%%
d2v_model = Doc2Vec(tagged_data, vector_size=200, epochs=30, dm=0,
window=7, min_count=1, workers=4, negative=20, seed=SEED)
#%%
d2v_model.save(data_path + "d2v_size200_epochs30.model")

# %%
df.shape

# %%
%%time
#%%
d2v_model= Doc2Vec.load(data_path + "d2v_size200_epochs30.model")

#%%
# list of sentences
# each sentence is a list of 700-dim words
X_d2v = np.array([d2v_model.dv[i]
                        for i in range(df.shape[0])])

# %%
#%%
np.save(data_path + 'X_d2v_size200_epochs30.npy', X_d2v)
# %%

# %%
# load w2v SG
from gensim.models import Word2Vec
w2v_sg = Word2Vec.load('w2v_sg_min25_size200.sav')

# %%
%%time
# tfidf-weighted w2v SG

X_tfidf = np.load(data_path + 'X_tfidf.npy')

vocab_rev = {v:k for k,v in vocab.items()}

weighted_docs = []
for d_ix, doc in enumerate(X_tfidf):
    weighted_doc = []
    for w_ix, w_tfidf in enumerate(doc):
        weighted_emb = w_tfidf*w2v_sg.wv[vocab_rev[w_ix]] 
        weighted_doc += [weighted_emb]
    weighted_doc = np.array(weighted_doc).sum(0)
#     weighted_doc = np.array(weighted_doc).sum(0)/doc.sum()
    weighted_docs.append(weighted_doc)
X_tfidf_w2v_sg = np.array(weighted_docs)  
np.save(data_path + 'X_tfidf_w2v_sg_size200.npy', X_tfidf_w2v_sg)

# %%
%%time
# Averaging word-to-vec
#48 sec

#prepare sentences
sentences = [doc.split(' ') for doc in df['final_text_25']]

sentvecs = []
for sentence in sentences:
    vecs = [w2v_sg.wv[w] for w in sentence if w in w2v_sg.wv]
    if len(vecs)== 0:
        sentvecs.append(np.nan)
        continue
    sentvec = np.mean(vecs,axis=0)
    sentvecs.append(sentvec.reshape(1,-1))
#%%
# list of sentences
# each sentence is a list of (embedding size)-dim words

X_avgw2v = [doc[0] for doc in sentvecs]
np.save(data_path + 'X_avgw2v_size200.npy', X_avgw2v)

# %%
# load GloVe

with open(data_path + f'glove_min25_size200.sav', 'rb') as fr:
    glove = pickle.load(fr)

# %%
%%time
# tfidf-weighted GloVe
vocab_rev = {v:k for k,v in vocab.items()}

weighted_docs = []
for d_ix, doc in enumerate(X_tfidf):
    weighted_doc = []
    for w_ix, w_tfidf in enumerate(doc):
        weighted_emb = w_tfidf*glove.word_vectors[glove.dictionary[vocab_rev[w_ix]]]
        weighted_doc += [weighted_emb]
    weighted_doc = np.array(weighted_doc).sum(0)
#     weighted_doc = np.array(weighted_doc).sum()/doc.sum()
    weighted_docs.append(weighted_doc)
X_tfidf_glove = np.array(weighted_docs) 
np.save(data_path + 'X_tfidf_glove_size200.npy', X_tfidf_glove)

# %%
%%time
# Averaging Glove
#50 sec
 
#prepare sentences
sentences = [doc.split(' ') for doc in df['final_text_25']]

sentvecs = []
for sentence in sentences:
    vecs = [glove.word_vectors[glove.dictionary[w]] for w in sentence if w in glove.dictionary]
    if len(vecs)== 0:
        sentvecs.append(np.nan)
        continue
    sentvec = np.mean(vecs,axis=0)
    sentvecs.append(sentvec.reshape(1,-1))
#%%
# list of sentences
# each sentence is a list of 700-dim words

X_avglove = [doc[0] for doc in sentvecs]
np.save(data_path + 'X_avglove_size200.npy', X_avglove)

# %%
# load GloVe pretrained on Wikipedia

glove_wiki = api.load("glove-wiki-gigaword-300")

# %%
%%time
# Averaging Pre-Trained Glove
#few seconds
 
#prepare sentences
sentences = [doc.split(' ') for doc in df['final_text_25']]

sentvecs = []
for sentence in sentences:
    vecs = [glove_wiki[w] for w in sentence if w in glove_wiki]
    if len(vecs)== 0:
        sentvecs.append(np.nan)
        continue
    sentvec = np.mean(vecs,axis=0)
    sentvecs.append(sentvec.reshape(1,-1))
#%%
# list of sentences
# each sentence is a list of 300(?)-dim words

X_avglove_pretrained = [doc[0] for doc in sentvecs]
np.save(data_path + 'X_avglove_pretrained.npy', X_avglove_pretrained)

# %% [markdown]
# # Cosine similarity comparison

# %%
#%%
X_countvec = np.load(data_path + 'X_countvec.npy')
X_tfidf = np.load(data_path + 'X_tfidf.npy')
X_LSA = np.load(data_path + 'X_LSA.npy')
X_NMF = np.load(data_path + 'X_NMF.npy')
X_lda = np.load(data_path + 'X_lda.npy')
X_d2v = np.load(data_path + 'X_d2v_size200.npy')
X_tfidf_w2v_sg = np.load(data_path + 'X_tfidf_w2v_sg_size200.npy')
X_avgw2v = np.load(data_path + 'X_avgw2v_size200.npy')
X_tfidf_glove = np.load(data_path + 'X_tfidf_glove_size200.npy')
X_avglove = np.load(data_path + 'X_avglove_size200.npy')
X_avglove_pretrained = np.load(data_path + 'X_avglove_pretrained.npy')

# %%
####### COSINE SIMILARITIES
# %%
sims_countvec = cosine_similarity(X_countvec)
sims_tfidf = cosine_similarity(X_tfidf)
sims_lsa = cosine_similarity(X_LSA)
sims_nmf = cosine_similarity(X_NMF)
sims_lda = cosine_similarity(X_lda)
sims_d2v = cosine_similarity(X_d2v)
sims_tfidf_w2v_sg = cosine_similarity(X_tfidf_w2v_sg)
sims_avgw2v = cosine_similarity(X_avgw2v)
sims_tfidf_glove = cosine_similarity(X_tfidf_glove)
sims_avglove = cosine_similarity(X_avglove)
sims_avglove_pretrained = cosine_similarity(X_avglove_pretrained)
#%%

# %%
flat_sims_countvec = np.array(sims_countvec[np.tril_indices(sims_countvec.shape[0])])
flat_sims_tfidf = np.array(sims_tfidf[np.tril_indices(sims_tfidf.shape[0])])
flat_sims_lsa = np.array(sims_lsa[np.tril_indices(sims_lsa.shape[0])])
flat_sims_nmf = np.array(sims_nmf[np.tril_indices(sims_nmf.shape[0])])
flat_sims_lda = np.array(sims_lda[np.tril_indices(sims_lda.shape[0])])
flat_sims_d2v = np.array(sims_d2v[np.tril_indices(sims_d2v.shape[0])])
flat_sims_tfidf_w2v_sg = np.array(sims_tfidf_w2v_sg[np.tril_indices(sims_tfidf_w2v_sg.shape[0])])
flat_sims_avgw2v = np.array(sims_avgw2v[np.tril_indices(sims_avgw2v.shape[0])])
flat_sims_tfidf_glove = np.array(sims_tfidf_glove[np.tril_indices(sims_tfidf_glove.shape[0])])
flat_sims_avglove = np.array(sims_avglove[np.tril_indices(sims_avglove.shape[0])])
flat_sims_avglove_pretrained = np.array(sims_avglove_pretrained[np.tril_indices(sims_avglove_pretrained.shape[0])])
#%%

# %%
# run to have plots in LaTeX format

params = {
            'font.family': 'serif',
#           'font.serif': 'cmr10',
          'text.usetex': True,
        #   'text.latex.unicode': True,
          'axes.titlesize': 15,
          'axes.labelsize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'font.weight': 'bold'}
plt.rcParams.update(params)

# %%
# l_range = 0
# u_range = 5000

# ticks = np.arange(l_range, u_range, 300)
# labs = np.array(np.tril_indices(sims_lsa.shape[0])).T
# labs = labs[ticks]

sample_range = np.linspace(0, flat_sims_tfidf.shape[0]-1,10000).astype(int)

#%%
fig, ax = plt.subplots(11,11, figsize=(22,22))
fig.suptitle('Cosine similarity for 10,000 pairs of firms \n comparison across models',
size=24)
sims_all = [flat_sims_countvec, flat_sims_tfidf, flat_sims_lsa, flat_sims_nmf, flat_sims_lda, 
            flat_sims_d2v, flat_sims_tfidf_w2v_sg, flat_sims_avgw2v,
            flat_sims_tfidf_glove, flat_sims_avglove, flat_sims_avglove_pretrained]
names = ['countvectorizer', 'tfidf', 'lsa', 'nmf', 'lda', 'd2v', 'tfidf-w2v', 'avgw2v',
         'tfidf-glove','avglove', 'glove-pretrained']

# scaler = StandardScaler()

for i in np.arange(11):
    for j in np.arange(11):
        x = sims_all[i][sample_range]
        y = sims_all[j][sample_range]

        # x = scaler.fit_transform(x.reshape(-1, 1) )
        # x = x.squeeze()
        # y = scaler.fit_transform(y.reshape(-1, 1) )
        # y = y.squeeze()

        ax[i][j].scatter(x,y, alpha=0.3, s=1)
        ax[i][j].set_ylim(bottom=0)
        ax[i][j].set_xlim(left=0)
        ax[i][j].set_xlabel(names[i])#, size=12)
        ax[i][j].set_ylabel(names[j])#, size=12)

        m, b = np.polyfit(x, y, 1)
        ax[i][j].plot(x, m*x + b, color='red', linewidth=1, linestyle='dashed')

        ax[i][j].plot(np.arange(0,3.05,0.05), np.arange(0,3.05,0.05), color='black', linewidth=1, linestyle='dashed')


plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.91, wspace=0.5, hspace=0.5)
plt.savefig('cosine_similarity_10000pairs.png')

plt.show()

# %% [markdown]
# # Pearson correlations

# %%
#%%
X_countvec = np.load(data_path + 'X_countvec.npy')
X_tfidf = np.load(data_path + 'X_tfidf.npy')
X_LSA = np.load(data_path + 'X_LSA.npy')
X_NMF = np.load(data_path + 'X_NMF.npy')
X_lda = np.load(data_path + 'X_lda.npy')
X_d2v = np.load(data_path + 'X_d2v_size200_epochs30.npy')
X_tfidf_w2v_sg = np.load(data_path + 'X_tfidf_w2v_sg_size200.npy')
X_avgw2v = np.load(data_path + 'X_avgw2v_size200.npy')
X_tfidf_glove = np.load(data_path + 'X_tfidf_glove_size200.npy')
X_avglove = np.load(data_path + 'X_avglove_size200.npy')
X_avglove_pretrained = np.load(data_path + 'X_avglove_pretrained.npy')

# %%
####### COSINE SIMILARITIES
# %%
sims_countvec = cosine_similarity(X_countvec)
sims_tfidf = cosine_similarity(X_tfidf)
sims_lsa = cosine_similarity(X_LSA)
sims_nmf = cosine_similarity(X_NMF)
sims_lda = cosine_similarity(X_lda)
sims_d2v = cosine_similarity(X_d2v)
sims_tfidf_w2v_sg = cosine_similarity(X_tfidf_w2v_sg)
sims_avgw2v = cosine_similarity(X_avgw2v)
sims_tfidf_glove = cosine_similarity(X_tfidf_glove)
sims_avglove = cosine_similarity(X_avglove)
sims_avglove_pretrained = cosine_similarity(X_avglove_pretrained)
#%%

# %%
flat_sims_countvec = np.array(sims_countvec[np.tril_indices(sims_countvec.shape[0])])
flat_sims_tfidf = np.array(sims_tfidf[np.tril_indices(sims_tfidf.shape[0])])
flat_sims_lsa = np.array(sims_lsa[np.tril_indices(sims_lsa.shape[0])])
flat_sims_nmf = np.array(sims_nmf[np.tril_indices(sims_nmf.shape[0])])
flat_sims_lda = np.array(sims_lda[np.tril_indices(sims_lda.shape[0])])
flat_sims_d2v = np.array(sims_d2v[np.tril_indices(sims_d2v.shape[0])])
flat_sims_tfidf_w2v_sg = np.array(sims_tfidf_w2v_sg[np.tril_indices(sims_tfidf_w2v_sg.shape[0])])
flat_sims_avgw2v = np.array(sims_avgw2v[np.tril_indices(sims_avgw2v.shape[0])])
flat_sims_tfidf_glove = np.array(sims_tfidf_glove[np.tril_indices(sims_tfidf_glove.shape[0])])
flat_sims_avglove = np.array(sims_avglove[np.tril_indices(sims_avglove.shape[0])])
flat_sims_avglove_pretrained = np.array(sims_avglove_pretrained[np.tril_indices(sims_avglove_pretrained.shape[0])])

# %%
sims_all = [flat_sims_countvec, flat_sims_tfidf, flat_sims_avglove_pretrained,
            flat_sims_avglove, flat_sims_tfidf_glove, flat_sims_avgw2v,
            flat_sims_tfidf_w2v_sg, flat_sims_d2v, flat_sims_lsa,
            flat_sims_nmf, flat_sims_lda]

names = ['Term Counts (Raw)', 'Term Counts (TF-IDF)', 'GloVe (Pre-Trained, Avg)',
         'GloVe (Trained, Avg)', 'GloVe (Trained, TF-IDF)',
         'Word2Vec (Trained, Avg)', 'Word2Vec (Trained, TF-IDF)',
         'Doc2Vec', 'LSA', 'NMF', 'LDA']

# %%
%%time
pears_mat = []
for n_ix, n in enumerate(sims_all):
    row = []
    for m_ix, m in enumerate(sims_all):
        pears = np.corrcoef(n,m)[1,0]
        row.append(pears)
    pears_mat.append(np.array(row)) 
pears_mat = np.array(pears_mat)

# %%
pears_df = pd.DataFrame(pears_mat, index=names)
pears_df.columns = names

# %%
# run to have plots in LaTeX format

params = {'font.family': 'serif',
          'text.usetex': True,
          'axes.titlesize': 15,
          'axes.labelsize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'font.weight': 'bold'}
plt.rcParams.update(params)

# %%
# fig = plt.subplots(figsize=(8,8))
# sns.set(font_scale=1.1)
# sns.heatmap(pears_df, cmap='Blues', annot=True, mask = np.triu(pears_mat))
# plt.tight_layout()
# plt.savefig('pears_heatmap.png')
# plt.show()

fig = plt.subplots(figsize=(8,8))
sns.set(font_scale=1.1, rc={'text.usetex': True})
ax = sns.heatmap(pears_df, cmap='Blues', annot=True, mask = np.triu(pears_mat))
ax.set_aspect("equal")
plt.xticks(rotation=45, ha="right", position=(0, 0.03), rotation_mode="anchor")
plt.tight_layout()
plt.savefig('pears_heatmap.png')

# %% [markdown]
# # Ranking comparison

# %%
#%%
# extract 10,000 unique triplets of firms
n_samples = 10000
indices = np.arange(sims_lsa.shape[0])

# %%
triplet_idx = np.array([np.random.choice(indices, 3, replace=False) for i in range(n_samples)])
triplet_idx = np.sort(triplet_idx, axis=1)
triplet_idx = np.unique(triplet_idx, axis=0)
triplets = np.array([np.array([tr_idx[np.array([0,1])],
tr_idx[np.array([0,2])]]) for tr_idx in triplet_idx])

# %%
#%%
triplet_sims_countvec = np.array([[sims_countvec[tuple(tr[0])],
sims_countvec[tuple(tr[1])]] for tr in triplets])
rankings_countvec = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_countvec])

# %%
#%%
triplet_sims_tfidf = np.array([[sims_tfidf[tuple(tr[0])],
sims_tfidf[tuple(tr[1])]] for tr in triplets])
rankings_tfidf = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_tfidf])

# %%
#%%
triplet_sims_lsa = np.array([[sims_lsa[tuple(tr[0])],
sims_lsa[tuple(tr[1])]] for tr in triplets])
rankings_lsa = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_lsa])

# %%
#%%
triplet_sims_nmf = np.array([[sims_nmf[tuple(tr[0])],
sims_nmf[tuple(tr[1])]] for tr in triplets])
rankings_nmf = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_nmf])

# %%
#%%
triplet_sims_lda = np.array([[sims_lda[tuple(tr[0])],
sims_lda[tuple(tr[1])]] for tr in triplets])
rankings_lda = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_lda])

# %%
#%%
triplet_sims_d2v = np.array([[sims_d2v[tuple(tr[0])],
sims_d2v[tuple(tr[1])]] for tr in triplets])
rankings_d2v = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_d2v])

# %%
#%%
triplet_sims_tfidf_w2v_sg = np.array([[sims_tfidf_w2v_sg[tuple(tr[0])],
sims_tfidf_w2v_sg[tuple(tr[1])]] for tr in triplets])
rankings_tfidf_w2v_sg = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_tfidf_w2v_sg])

# %%
#%%
triplet_sims_avgw2v = np.array([[sims_avgw2v[tuple(tr[0])],
sims_avgw2v[tuple(tr[1])]] for tr in triplets])
rankings_avgw2v = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_avgw2v])

# %%
#%%
triplet_sims_tfidf_glove = np.array([[sims_tfidf_glove[tuple(tr[0])],
sims_tfidf_glove[tuple(tr[1])]] for tr in triplets])
rankings_tfidf_glove = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_tfidf_glove])

# %%
#%%
triplet_sims_avglove = np.array([[sims_avglove[tuple(tr[0])],
sims_avglove[tuple(tr[1])]] for tr in triplets])
rankings_avglove = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_avglove])

# %%
#%%
triplet_sims_glove_pretrained = np.array([[sims_avglove_pretrained[tuple(tr[0])],
sims_avglove_pretrained[tuple(tr[1])]] for tr in triplets])
rankings_glove_pretrained = np.array([1 if tr[0]>tr[1] else 0 
for tr in triplet_sims_glove_pretrained])

# %%
#%%
rankings = {'countvectorizer': rankings_countvec,'tfidf':rankings_tfidf,
            'glove-pretrained': rankings_glove_pretrained, 'avglove': rankings_avglove,
            'tfidf-glove': rankings_tfidf_glove, 'avgw2v': rankings_avgw2v,
            'tfidf-w2v': rankings_tfidf_w2v_sg, 'd2v':rankings_d2v,
            'lsa':rankings_lsa, 'nmf':rankings_nmf, 'lda':rankings_lda}

names = ['countvectorizer', 'tfidf', 'glove-pretrained', 'avglove', 'tfidf-glove',
         'avgw2v', 'tfidf-w2v', 'd2v', 'lsa', 'nmf', 'lda']

names_clean = ['Term Counts (Raw)', 'Term Counts (TF-IDF)', 'GloVe (Pre-Trained, Avg)',
               'GloVe (Trained, Avg)', 'GloVe (Trained, TF-IDF)',
               'Word2Vec (Trained, Avg)', 'Word2Vec (Trained, TF-IDF)',
               'Doc2Vec', 'LSA', 'NMF', 'LDA']

#%%
fractions = {f'{a}_{b}': (rankings[a]==rankings[b]).sum()/n_samples
for a,b in list(itertools.combinations(names, 2))}

# %%
fractions

# %%
fraction_mat = []
for n_ix, n in enumerate(names):
    row = []
    for m_ix, m in enumerate(names):
        frac = (rankings[n]==rankings[m]).sum()/n_samples
        row.append(frac)
    fraction_mat.append(np.array(row)) 
fraction_mat = np.array(fraction_mat)

# %%
fraction_df = pd.DataFrame(fraction_mat, index=names_clean)
fraction_df.columns = names_clean

# %%
# run to have plots in LaTeX format

params = {'font.family': 'serif',
          'text.usetex': True,
          'axes.titlesize': 15,
          'axes.labelsize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'font.weight': 'bold'}
plt.rcParams.update(params)

# %%
fig = plt.subplots(figsize=(8,8))
sns.set(font_scale=1.1, rc={'text.usetex': True})
ax = sns.heatmap(fraction_df, cmap='Blues', annot=True, mask = np.triu(fraction_mat))
ax.set_aspect("equal")
plt.xticks(rotation=45, ha="right", position=(0, 0.03), rotation_mode="anchor")
plt.tight_layout()
plt.savefig('fraction_heatmap.png')
plt.show()

# %% [markdown]
# # Additional: KNN analysis

# %%
# %%
def get_ranking(sims, target_ix, cols, similar=True, k=6):
    ranked_sents = df.iloc[(np.argsort(sims[target_ix])[::-1]),:]
    if similar:
        return ranked_sents.iloc[1:k+1][cols]
    else:
        return ranked_sents.iloc[-k:][cols]

#%%
def show_ranking_cik(sims_tfidf, sims_lsa, sims_nmf,sims_lda,
sims_d2v, target_ix, k=6, cols=['cik']):
    cols2 = ['index']

    # print(f'{k} most SIMILAR documents to "{np.array(df.cik.values)[target_ix]}"')
    print(f'{k} most SIMILAR documents to cik at index "{target_ix}"')

    tfidf_similar_sents = get_ranking(sims_tfidf, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)
    lsa_similar_sents = get_ranking(sims_lsa, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)
    nmf_similar_sents = get_ranking(sims_nmf, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)
    lda_similar_sents = get_ranking(sims_lda, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)   
    d2v_similar_sents = get_ranking(sims_d2v, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)

    sims_df = pd.concat([tfidf_similar_sents[cols2],lsa_similar_sents[cols2],
    nmf_similar_sents[cols2], lda_similar_sents[cols2], d2v_similar_sents[cols2]], axis=1)
    header = pd.MultiIndex.from_product([['-- TFIDF --','-- LSA --','-- NMF --','-- LDA --','-- d2v --'],
                                     ['index']])
    sims_df.columns = header
    display(sims_df)

    print(f'\n \n {k} most DISSIMILAR documents to cik at index "{target_ix}"')
    # print(f'\n \n {k} most DISSIMILAR documents to "{np.array(df.cik.values)[target_ix]}"')
    tfidf_diff_sents = get_ranking(sims_tfidf, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)
    lsa_diff_sents = get_ranking(sims_lsa, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)
    nmf_diff_sents = get_ranking(sims_nmf, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)
    lda_diff_sents = get_ranking(sims_lda, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)    
    d2v_diff_sents = get_ranking(sims_d2v, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)
    diff_df = pd.concat([tfidf_diff_sents[cols2],lsa_diff_sents[cols2],
    nmf_diff_sents[cols2], lda_diff_sents[cols2], d2v_diff_sents[cols2]], axis=1)
    diff_df.columns = header
    display(diff_df)

# %%
def show_similar_text(sims_tfidf, sims_lsa, sims_nmf,sims_lda,
sims_d2v, target_ix, k=2, cols=['clean_text_rest']):
    print(f'{k} most SIMILAR documents to "{np.array(df.clean_text_rest.values)[target_ix][:1000]}"')
    tfidf_similar_sents = get_ranking(sims_tfidf, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)
    lsa_similar_sents = get_ranking(sims_lsa, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)
    nmf_similar_sents = get_ranking(sims_nmf, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)
    lda_similar_sents = get_ranking(sims_lda, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)
    d2v_similar_sents = get_ranking(sims_d2v, target_ix, similar=True, k=k, cols=cols).reset_index(drop=False)

    sims_df_1 = pd.concat([tfidf_similar_sents,lsa_similar_sents,
    nmf_similar_sents], axis=1)
    header = pd.MultiIndex.from_product([['-- TFIDF --','-- LSA --','-- NMF --'],
                                     ['index','clean_text_rest']])
    sims_df_1.columns = header

    sims_df_2 = pd.concat([lda_similar_sents, d2v_similar_sents], axis=1)
    header = pd.MultiIndex.from_product([['-- LDA --','-- d2v --'],
                                     ['index', 'clean_text_rest']])
    sims_df_2.columns = header
    display(sims_df_1)
    display(sims_df_2)
# %%
def show_dissimilar_text(sims_tfidf, sims_lsa, sims_nmf,sims_lda,
sims_d2v, target_ix, k=2, cols=['clean_text_rest']):

    print(f'\n \n {k} most DISSIMILAR documents to "{np.array(df.clean_text_rest.values)[target_ix][:1000]}"')
    tfidf_diff_sents = get_ranking(sims_tfidf, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)
    lsa_diff_sents = get_ranking(sims_lsa, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)
    nmf_diff_sents = get_ranking(sims_nmf, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)
    lda_diff_sents = get_ranking(sims_lda, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)
    d2v_diff_sents = get_ranking(sims_d2v, target_ix, similar=False, k=k, cols=cols).reset_index(drop=False)

    diff_df_1 = pd.concat([tfidf_diff_sents,lsa_diff_sents,
    nmf_diff_sents], axis=1)
    header = pd.MultiIndex.from_product([['-- TFIDF --','-- LSA --','-- NMF --'],
                                    ['index', 'clean_text_rest']])
    diff_df_1.columns = header

    diff_df_2 = pd.concat([lda_diff_sents, d2v_diff_sents], axis=1)

    header = pd.MultiIndex.from_product([['-- LDA --','-- d2v --'],
                                    ['index', 'clean_text_rest']])
    diff_df_2.columns = header

    display(diff_df_1)
    display(diff_df_2)
# %%
#%%

# %%
# explore some examples
targets = [34, 48, 4]
k_neighbors = 5
#%%
for target in targets:
    print('=========================================================================================')
    show_ranking_cik(sims_tfidf=sims_tfidf, sims_lsa=sims_lsa, 
    sims_nmf=sims_nmf, sims_lda=sims_lda, 
                     sims_d2v=sims_d2v, 
    target_ix=target, k=k_neighbors, cols=['cik'])

# %%
#%%
k_neighbors = 2
for target in targets:
    print('=========================================================================================')
    show_similar_text(sims_tfidf=sims_tfidf, sims_lsa=sims_lsa, 
    sims_nmf=sims_nmf, sims_lda=sims_lda, 
                      sims_d2v=sims_d2v, 
    target_ix=target, k=k_neighbors, cols=['clean_text_rest'])

# %%
#%%
k_neighbors = 2
for target in targets:
    print('=========================================================================================')
    show_dissimilar_text(sims_tfidf=sims_tfidf, sims_lsa=sims_lsa, 
    sims_nmf=sims_nmf, sims_lda=sims_lda, 
                         sims_d2v=sims_d2v, 
    target_ix=target, k=k_neighbors, cols=['clean_text_rest'])


