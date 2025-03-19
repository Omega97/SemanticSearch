from semanticsearch.src.training_data import TrainingData


def test(path='..\\data\\training_dataset'):
    td = TrainingData(path)
    for key in td.data:
        print()
        print(f'document: {key}')
        print(f'first line: {td.data[key][0]}')
        print(f'shape: {td.data[key].shape}')


if __name__ == '__main__':
    test()
