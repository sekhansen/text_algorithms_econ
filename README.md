# ARE_text_algorithms_economics

## To-dos
1. Add AR notebooks to inventory (LAURA)
2. Discuss with Stephen if and how we want to introduce the data used in each notebook (BOTH)
3. Ask Stephen whether we want Jax/Numpyro notebooks on Bayesian regression in Appendix or not at all
4. Ask Stephen if the repository is going to be completely public
5. Consolidate a single data folder in Google Drive with a publicly accessible link (STEPHEN)
6. Make all notebooks colab-friendly, including the code to work with Google Drive (BOTH)

## Pymodules

1. Preprocessing class


## Custom implementations

- Word2Vec Jax implementation
- LDA and STM implementations
- Preprocessing class

## Notebooks content outline


### Notebook 1: Simple Dictionary Example (LAURA)
- Show how a simple dictionary of positive and negative words can track the movement of the economy
- Very simple preprocessing using standard packages
- [Dictionary example taken from](https://github.com/sekhansen/course_unstructured_data/blob/main/notebooks/preprocessing_notebook.ipynb) 
- Data: Bank of England MPC Minutes
- Highlight the potential for more complete processing

### Notebook 2: Preprocessing (LAURA)
- [Building on this notebook](https://github.com/sekhansen/mres_methods_course/blob/main/notebooks/preprocessing_notebook.ipynb)
- Update preprocessing class
- Show comparison with "simple" preprocessing from Notebook 1
- Data: Bank of England MPC Minutes
- End up producing clean tokens as well as doc-term frequency matrix

### Notebook 2 Appendix: Complex processing (YABRA)
- [Mainly building on this notebook](https://github.com/yabramuvdi/imperial-workshop/blob/master/notebooks/preprocessing_notebook.ipynb)
- Data: 10K 2019 sentences

### Notebook 3: Document embeddings (LAURA)
- [Mainly building on this notebook](https://github.com/llaurabat91/annual_review_project/blob/main/similarity_results.ipynb)
- Introduce 10K data (a row is a company in 2019)
- Simple example of how a vector-representation of documents is meaningful: e.g. 2 terms and different companies on a plane
- Cosine similarity on the document-term matrix (tfidf)
- Motivation to jump to dimensionality reduction techniques e.g. with LSA
- Introduce LSA, d2v, NMF, lda

### Notebook 4: Word embeddings (YABRA)
- [Building on this notebook](https://github.com/yabramuvdi/imperial-workshop/blob/master/notebooks/word2vec_notebook.ipynb)
- Data: Bank of England Inflation Reports and MPC minutes

### Notebook 5: Comparison between word embeddings models (LAURA)
- [Building on this notebook](https://github.com/llaurabat91/annual_review_project/blob/main/word_embeddings_last.ipynb)
- Data: 10 data (a row is a company in 2019)

### Notebook 6: Topic models (STEPHEN)
- [Building on lda notebook](https://github.com/sekhansen/mres_methods_course/blob/main/notebooks/lda_notebook.ipynb)
- [Building on stm notebook](https://github.com/llaurabat91/text-mining-lessons/blob/main/stm_notebook.ipynb)

### Notebook 7: Contextual embedding models - BERT and friends (YABRA)
- [Building on this notebook](https://github.com/sekhansen/mres_methods_course/blob/main/notebooks/bert_introduction.ipynb)
- Are these representations better than averaging word2vec vectors?
