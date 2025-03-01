from semanticsearch.src.inference import PageRecommender


def main():
    # Load the recommendation system
    pr = PageRecommender(k=3)

    # Write here the query
    # query = 'Dinosaur bones discovered in the Sahara Desert.'
    # query = 'Science was invented in the 17th century.'
    query = 'Diamonds are found mostly in Africa.'

    # Get recommendations
    recommendations = pr.recommend(query)

    # Show recommendations
    print(f'\n\nQuery: {query}\n')
    for i in range(len(recommendations)):
        file_name = recommendations[i]
        doc = pr.get_document(file_name)
        print(f'\n\n{doc[:100]}...')


if __name__ == "__main__":
    main()
