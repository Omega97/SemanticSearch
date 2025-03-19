from semanticsearch.src.inference import PageRecommender


def test(data_path='..\\data\\raw',
         emb_file='..\\data\\embeddings.json',
         k=3):
    recommender = PageRecommender(data_path=data_path, emb_file=emb_file, k=k)
    query = 'Diamonds are found mostly in Africa.'  # Example query
    print(recommender.recommend(query))


if __name__ == '__main__':
    test()
