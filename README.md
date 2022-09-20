# Text Algorithms in Economics

## Notebooks content outline

All notebooks contain a button that allows the user to execute the notebook in Google Colab:  ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) 

### Notebook 1: Simple dictionary examples with regular expressions
- **Summary**: This notebook illustrates how a simple count of negative and positive terms can generate a sentiment index that correlates with GDP growth.
- **Data**: Minutes from the Monetary Policy Committee at the Bank of England.


### Notebook 2: Preprocessing and document-term matrix creation
- **Summary**: This notebook illustrates how to apply multiple preprocessing steps to clean text data and build a document-term matrix.
- **Data**: Minutes from the Monetary Policy Committee at the Bank of England.

### Notebook 2 Appendix: Advanced preprocessing (ELLIOTT)
- TODO (?)
- Part of speech tagging
- Syntactic dependency parsing
- Named entity recognition

### Notebook 3: Dimensionality reduction with LDA
- **Summary**: This notebook illustrates how to reduce the dimension of the document-term matrix with one particular method; Latent Dirichlet Allocation (LDA).
- **Data**: USA State of the Union Addresses.

### Notebook 4: Word2Vec
- **Summary**: This notebook illustrates how to estimate word embeddings using the word2vec algorithm.
- **Data**: Bank of England Inflation Reports and Minutes from the Monetary Policy Committee at the Bank of England.

### Notebook 5: BERT 
- **Summary**: This notebook illustrates multiple strategies to generate embedded representations of text sequences using BERT. It then compares the quality of these representations by using them for a regression task.
- **Data**: 10-K reports for selected firms.

### Notebook 5 Appendix: Extensions of BERT
- **Summary**: This notebook illustrates how to use BERT models that have been finetuned for particular tasks. Concretely, the notebook will explore a BERT model for sequence similarity ([*Sentence BERT*](https://www.sbert.net/index.html)) and a BERT model for sentiment analysis ([*Twitter roBERTa*](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)).
- **Data**: 10-K reports for selected firms.
