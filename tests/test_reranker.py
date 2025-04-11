from semanticsearch.src.reranking import Reranker

query = 'what is panda'
docs = ['hi', 'panda is turtle', 'galapagos', 'panda panda panda panda panda', 'a panda is a panda', 'pandas is a python library to manipulate databases', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']

r = Reranker()
result = r.rerank(query, docs)
for doc, score in zip(result[0], result[1]):
    print(f"score {score:.2f} -> {doc}")