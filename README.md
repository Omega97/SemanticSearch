# WikiRecommendation

Embedding-Augmented Retrieval for Wikipedia pages

The goal is to create a framework that:
1) pre-processes the database, and creates a list of all the embeddings of the pages
2) at inference time, the embedding of the query is used to quickly retrieve the most relevant pages

**DB**: Database of pages [page_ID, content, ...] (if there are other features, everything will be converted to string and joined). The DB may be very big, so it's divided into smaller files, so that each individual page is easier to access.

**Loc_DB**: Database of the location of all files in the DB [list of files of [page_ID, file_ID]]

**Emb_DB**: Dataset of all the embeddings of the pages [page_ID, y1, y2, ..., yn]

**q**: Query [string]. The embedding of q will be used to find the most similar pages.

**f**: Embedding model [string -> array of floar]. The Embedding vector encodes the meaning of the pages/query

**r**: Recommendation [list of int]. Contains 'k' IDs of the pages most semantically similar to the query


## Pre-processing

1.a) compute Loc_DB

1.b) load model and compute Emb_DB


## Inference Time

2.a) use the model to compute the embedding 'y*' of the query 's'

2.b) use the Emb_DB and K-NN to find the 'k' pages most similar to 's'

2.c) use Loc_DB to retrieve the pages from the DB


## Notes
For a large database divided into smaller files, I recommend using individual embedding files for each smaller dataset file rather than having one large file for all embeddings.

