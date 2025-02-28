# Semantic Search 
## Embedding-Augmented Retrieval System

(https://docs.google.com/document/d/1dxJrHSRRuKj5rnlWklquTwP6U1giKueaaBx4spn-ayU/edit?usp=sharing)

The goal is to create a framework that:
1) Create a list of all the **embeddings** of the txt files of the database
2) At inference time, the **embedding of the query** is used to quickly retrieve the **most relevant files**

**DB**: The database is a directory containing *txt* files.

**Emb_DB**: JSON file with all the embeddings of the pages "name.txt": [x1, ..., xn]

**q**: Query [string]. The embedding of *q* will be used to find the most similar pages.

**f**: Embedding model [string -> array of floar]. The Embedding vector encodes the meaning of the pages/query

**r**: Recommendation [list of str]. Contains 'k' names of the files that are semantically the most similar to the query


## Pre-processing

1.a) load the model *f* for the encodings

1.b) compute *Emb_DB*


## Inference Time

2.a) use the model to compute the embedding *y** of the query *s*

2.b) use the *Emb_DB* and K-NN to find the *k* pages most similar to *s*

2.c) retrieve the pages from the *DB*
