from semanticsearch.src.semantic_retrieval import SemanticRetrieval
from semanticsearch.src.semantic_retrieval import main as sr_main
from semanticsearch.src.embedding import Embeddings


# Main
sr_main
# Load Embedding model ✅
# Load Reranking model ✅
# Semantic Retrieval Initialization
SemanticRetrieval.__init__
SemanticRetrieval._preprocessing
# 	Load Database ✅
# 	Load Embeddings ✅
Embeddings._load
#       Ensure the directory exists ✅
#       Ensure the data file exists ✅
# 		Load file paths from Database
# 		Load existing embeddings (only of files in directory)
# 		Find out what embeddings are missing
# 		Compute missing embeddings
# 		Save all embeddings

# Inference
SemanticRetrieval.recommend
# 	Compute query embedding
# 	Run KNN -> permutation on all the elements
# 	Run re-ranking -> permutation on top elements
# 	return top result -> permutation, document

# Testing
# 	Get list of query-document pairs
# 	Build dataset
# 	Run inference on each query
# 	Note the rank of the correct document
