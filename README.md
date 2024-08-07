# Text Algorithms in Economics

Companion python notebooks to the ['Text Algorithms in Economics' article](https://www.annualreviews.org/content/journals/10.1146/annurev-economics-082222-074352) by Elliott Ash and Stephen Hansen. Notebooks developed by [Yabra Muvdi](https://github.com/yabramuvdi). If you reuse in teaching or research, please cite the published article in the *Annual Review of Economics* (2023).

## Notebooks content outline

All notebooks contain a button that allows the user to execute the notebook in Google Colab:  ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) 

### Notebook 1: Simple dictionary examples with regular expressions ([here](./notebooks/1_regex_dictionary.ipynb))
- **Summary**: This notebook illustrates how a simple count of negative and positive terms can generate a sentiment index that correlates with GDP growth.
- **Data**: Minutes from the Monetary Policy Committee at the Bank of England.

### Notebook 2: Preprocessing and document-term matrix creation ([here](./notebooks/2_preprocessing.ipynb))
- **Summary**: This notebook illustrates how to apply multiple preprocessing steps to clean text data and build a document-term matrix.
- **Data**: Minutes from the Monetary Policy Committee at the Bank of England.

### Notebook 3: Dimensionality reduction with LDA ([here](./notebooks/3_LDA.ipynb))
- **Summary**: This notebook illustrates how to reduce the dimension of the document-term matrix with one particular method; Latent Dirichlet Allocation (LDA).
- **Data**: USA State of the Union Addresses and Minutes from the Monetary Policy Committee at the Bank of England.

### Notebook 4: Word2Vec ([here](./notebooks/4_word2vec.ipynb))
- **Summary**: This notebook illustrates how to estimate word embeddings using the word2vec algorithm.
- **Data**: Bank of England Inflation Reports and Minutes from the Monetary Policy Committee at the Bank of England.

### Notebook 5: Large language models for feature generation ([here](./notebooks/5_llm_features.ipynb))
- **Summary**: This notebook illustrates multiple strategies to generate embedded representations of text sequences using BERT. It then compares the quality of these representations by using them for a regression task.
- **Data**: 10-K reports for selected firms.

### Notebook 6: Finetuning a large language model ([here](./notebooks/6_llm_finetuning.ipynb))
- **Summary**: This notebook illustrates how to finetune a large language model for a particular classification task.
- **Data**: 10-K reports for selected firms.

### Notebook 7: GPT demonstration ([here](./notebooks/7_gpt_demonstration.ipynb))
- **Summary**: This notebook shows how to interact with GPT using OpenAI's API.
