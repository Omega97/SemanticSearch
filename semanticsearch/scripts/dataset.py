from misc import read_jsonl


def main(file_path="..\\..\\data\\training_golden_answers.jsonl",
         output_path_1="..\\..\\data\\training_dataset\\data_1.tsv",
         output_path_2="..\\..\\data\\training_dataset\\data_2.tsv"):

    # keys = ['question', 'golden_answers', 'metadata', 'answer']
    data = read_jsonl(file_path)
    for i in range(10):
        print(data[i])

    with open(output_path_1, 'w') as f:
        for dct in data:
            query = dct['question']
            query = query.replace('\t', ' ').replace('\n', ' ')
            for answer in dct['golden_answers']:
                answer = answer.replace('\t', ' ').replace('\n', ' ')
                f.write(f"{query}\t{answer}\n")

    with open(output_path_2, 'w') as f:
        for dct in data:
            query = dct['metadata']['document_name']
            query = query.replace('\t', ' ').replace('\n', ' ')
            query = query.replace('.html', '')
            for answer in dct['golden_answers']:
                answer = answer.replace('\t', ' ').replace('\n', ' ')
                f.write(f"{query}\t{answer}\n")


if __name__ == "__main__":
    main()
