from semanticsearch.src.reranking import Reranker

query = 'what is panda'
docs = [
    'hi', 
    'panda is turtle', 
    'galapagos', 
    'panda panda panda panda panda bla bla bla il cielo è blu come l\'uranio impoverito suscita reazioni esilaranti. COme state voi? Me lo chiedo? Vittorio Emanuele secondo è mio figlio', 
    'a panda is a panda', 
    'pandas is a python library to manipulate databases', 
    'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.',
    'The word panda was borrowed into English from French, but no conclusive explanation of the origin of the French word panda has been found.']

r = Reranker(chunking_enabled=True)
docs, scores, chunks = r.doc_rerank(query, docs)

for i in range(len(docs)):
    print(f'---POSITION {i+1}:---')
    print(f'score = {scores[i]}')
    print(f'doc = {docs[i]}')
    print(f'chunk = {chunks[i]}')