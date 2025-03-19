from semanticsearch.src.inference import PageRecommender


def main(width=50):
    """
    Example of how to use the system
    """
    # Load the recommendation system
    pr = PageRecommender(k=3)

    while True:
        # Get the query
        query = input("\n\nWrite your query: ")
        if query == '':
            break

        # Get recommendations
        recommendations = pr.recommend(query)

        # Show recommendations
        file_name = recommendations[0]
        doc = pr.get_document(file_name)
        print()
        for i in range(5):
            print(f'{doc[i*width:(i+1)*width]}')
        print(f'\nSee also: {recommendations[1:]}')


if __name__ == "__main__":
    main()
