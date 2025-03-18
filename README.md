# SEMANTIC SEARCH


>## Embedding-Augmented Retrieval System

This project aims to develop a **semantic retrieval framework** for 
efficiently searching documents. During pre-processing, a pre-trained 
model generates embeddings for all text files in the database, creating 
a structured representation of their content.

At inference time, retrieval is performed by computing the 
**embedding of the query** and comparing it against the stored embeddings. 
The most relevant documents are identified based on cosine similarity, 
ensuring that semantically similar files are recommended.

We introduce a learned endomorphism that **aligns the query embedding** 
with the page embedding to enhance retrieval accuracy. This 
transformation is trained to maximize the **cosine similarity** between 
the query and its corresponding target document. The mapping function 
is trained on a dataset of query-document pairs, enabling it to adapt 
to the specific characteristics of the database and improve retrieval 
performance.

>## Dataset Ideas:
Our priority is to create a very diverse dataset of **query-document pairs**, 
so that we can reliably obtain a mapping that reliably improves the system. 
To achieve this, we considered the following datasets:
1) **WikiPassageQA**, a Wikipedia-based retrieval dataset that can be repurposed for retrieval.
2) **BEIR Benchmark** (General Information Retrieval), a collection of 18 datasets covering various retrieval 
tasks, including document and passage retrieval.
3) **MS MARCO** (Web and Passage Retrieval), a large-scale dataset for passage ranking and retrieval, commonly 
used to benchmark dense retrieval models.
4) **TREC** Datasets (Web & Enterprise Retrieval), A series of datasets designed for various retrieval tasks, 
including ad-hoc search, enterprise search, and web retrieval


>## Performance
The goal is to maximize the **cosine similarity** between the query and 
the document. A secondary evaluation metric for the recommendation 
system is the **ranking** of the target pages. To maximize the performance 
of the system, we can try various models, with or without embedding 
normalization, etc...


>## Key Elements

*DB* – The database is a directory containing text files.

*DB_emb* – JSON file with all the embeddings of the pages "name.txt": [*x<sub>1</sub>, ..., x<sub>n</sub>*]

*q* – the query [string]. The embedding of q will be used to find the most similar pages.

*f* – the embedding model [string -> array of float]. The embedding vector encodes the meaning of the pages/query

*𝔼* – embedding space to better align query embedding to page embedding

*𝒯: 𝔼 → 𝔼* – endomorphism in embedding space

*r* – names of recommended documents [list of str]. Contains *k* names of the files that are semantically 
the most similar to the query


>## The Pipeline

###  Pre-processing
* Load the model for the encodings 
* Compute the embedding dataset

### Training 
The mapping is optimized so that it maximizes the cosine similarity between the query and the target document ,

### Inference Time 
* Use the model to compute the embedding *y* of the query *s* 
* Use the *DB_emb* and K-NN to find the *k* pages most similar to *s* 
* Retrieve the pages from the DB


>## Notes
* Some queries might be associated with more than one document.
* Google Docs: https://docs.google.com/document/d/1dxJrHSRRuKj5rnlWklquTwP6U1giKueaaBx4spn-ayU/edit?usp=sharing
