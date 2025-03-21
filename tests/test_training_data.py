from semanticsearch.src.training_data import TrainingData


def test(path):
    td = TrainingData(path)
    for doc_name in td.data:
        print()
        print(f'document: {doc_name}')
        question, answer = td.data[doc_name][0]
        print(f'question:\n {question}')
        print(f'answer:\n {answer}')
        print(f'shape: {td.data[doc_name].shape}')


if __name__ == '__main__':
    test('..\\data\\training_dataset')
    # test('..\\data\\wikiqa')
