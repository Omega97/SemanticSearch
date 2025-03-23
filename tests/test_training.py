import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time
from semanticsearch.src.training import Trainer, EmbeddingTrainer
from semanticsearch.src.embedding import EmbeddingModel, EmbeddingModelWithCorrection
from semanticsearch.src.training_data import TrainingData
from semanticsearch.src.ranking import compute_recall_at_k


def test(N=2000, M=32, L=8, lambda_reg=1e-3):
    """
    Evaluates the trained refinement.
    """
    # For reproducibility
    torch.manual_seed(0)

    # Example data
    Y = torch.randn(N, M)
    target_mat = torch.eye(M) + 0.1 * torch.randn(M, M)
    Y_target = Y @ target_mat

    trainer = Trainer(Y, Y_target, L=L)
    trainer.train(lambda_reg=lambda_reg)

    # Get result
    refinement_matrix = trainer.get_refinement_matrix()

    # Apply to Y
    Y_refined = trainer.Y @ refinement_matrix

    # print difference between Y_target and Y_refined
    print(f"\nDifference between Y_target and Y_refined:")
    print(f"{torch.norm(Y_target - Y_refined).item():.6f}")

    # print difference between target_mat and refinement_matrix
    print(f"Difference between target_mat and refinement_matrix:")
    print(f"{torch.norm(target_mat - refinement_matrix).item():.6f}")

    # Check the final loss
    loss = nn.MSELoss()(Y_refined, trainer.Y_target)
    print(f"\nFinal loss: {loss.item():.6f}")

    # norm of the refinement matrix
    print(f"Norm of correction matrix: {torch.norm(trainer.A @ trainer.B).item():.6f}")

    # plots
    fig, ax = plt.subplots(1, 2)

    # Plot the refinement matrix
    plt.sca(ax[0])
    plt.imshow(refinement_matrix.detach().numpy(), cmap='bwr', vmin=-1.1, vmax=1.1)
    plt.title("Refinement matrix")
    plt.colorbar()

    # Plot the target matrix
    plt.sca(ax[1])
    plt.imshow(target_mat.detach().numpy(), cmap='bwr', vmin=-1.1, vmax=1.1)
    plt.title("Target matrix")
    plt.colorbar()

    plt.show()


def test_training(train_dir_path='..\\data\\training_dataset', test_name='golden_answer',
                  hidden_dims=8, lambda_reg=1e-3, max_size=1000):
    """
    Tests the training of the RefinementTrainer.
    """
    fig, ax = plt.subplots(ncols=2)

    # load the data
    print('\nLoading the data...')
    data = TrainingData(train_dir_path)
    train_names = [name for name in data.get_names() if name != test_name]
    train_queries, train_docs = data.get_queries_and_docs(*train_names, max_size=max_size)
    test_queries, test_docs = data.get_queries_and_docs(test_name, max_size=max_size)

    # generate the embeddings
    print('\nInitializing the model...')
    model = EmbeddingModel()
    X_train = torch.tensor(model.encode(train_queries))
    Y_train = torch.tensor(model.encode(train_docs))
    X_test = torch.tensor(model.encode(test_queries))
    Y_test = torch.tensor(model.encode(test_docs))

    # evaluate pre-training performance
    print('\nEvaluating pre-training performance...')
    score_pre = compute_recall_at_k(X_test, Y_test, k=1)
    print(f'Performance before training: {score_pre:.2%}')

    # train the model
    print('\nTraining...')
    trainer = Trainer(X_train, Y_train, L=hidden_dims, lr=1e-2, epsilon_0=0.1)
    plt.sca(ax[0])
    _mat = trainer.get_refinement_matrix().detach().numpy()
    plt.imshow(_mat[:20, :20], cmap='bwr', vmin=-1.1, vmax=1.1)
    plt.title("Initial refinement matrix")
    correction_matrix = trainer.train(lambda_reg=lambda_reg, num_epochs=3000)
    correction_matrix = correction_matrix.detach().numpy()

    # evaluate post-training performance
    # new_model = EmbeddingModelWithCorrection(model.model_name, correction_matrix)
    # X_test_2 = torch.tensor(new_model.encode(test_queries))
    # Y_test_2 = torch.tensor(new_model.encode(test_docs))
    X_test_2 = X_test @ torch.tensor(correction_matrix)
    Y_test_2 = Y_test @ torch.tensor(correction_matrix)
    score_post = compute_recall_at_k(X_test_2, Y_test_2, k=1)
    print(f'Performance after training: {score_post:.2%}')

    # plot the results
    plt.sca(ax[1])
    _mat = trainer.get_refinement_matrix().detach().numpy()
    plt.imshow(_mat[:20, :20], cmap='bwr', vmin=-1.1, vmax=1.1)
    plt.title("Refinement matrix")
    plt.show()


def test_training_2(train_dir_path='..\\data\\training_dataset',
                    test_name='golden_answer',
                    batch_size=500, max_size=500):
    """
    Tests the training
    Y = X @ (I + mat)
    """
    # load the data
    print('\nLoading the data...')
    data = TrainingData(train_dir_path)
    train_names = [name for name in data.get_names() if name != test_name]
    train_queries, train_docs = data.get_queries_and_docs(*train_names)
    test_queries, test_docs = data.get_queries_and_docs(test_name, shuffle=True, max_size=max_size)

    # initialize the embedding model
    print('\nInitializing the model...')
    model = EmbeddingModel()

    # generate the test embeddings
    print('\nGenerating the test embeddings...')
    X_test = torch.tensor(model.encode(test_queries))
    Y_test = torch.tensor(model.encode(test_docs))

    # initialize the training parameters
    print('\nInitializing the training parameters...')
    epsilon_0 = 0.001
    M = 384
    mat = nn.Parameter(torch.randn(M, M) * epsilon_0)
    optimizer = optim.Adam([mat], lr=1e-3)
    criterion = nn.MSELoss()
    n_cycles = len(train_queries) // batch_size

    # evaluate pre-training performance
    score_pre = compute_recall_at_k(X_test, Y_test, k=1)
    print(f'\nPerformance before training: {score_pre:.2%}')

    # train the model
    print('\nTraining...')
    for i_cycle in range(2):
        print(f'\nCycle {i_cycle + 1}/{n_cycles}')

        # get the batch data
        print('\nGetting the data...')
        i_start = i_cycle * batch_size
        i_end = (i_cycle + 1) * batch_size
        i_end = min(i_end, len(train_queries))
        indices = range(i_start, i_end)
        queries = train_queries[indices]
        docs = train_docs[indices]

        # generate the embeddings
        print(f'\nGenerating the embeddings for {i_end-i_start} samples...')
        t = time()
        X_train = torch.tensor(model.encode(queries))
        Y_train = torch.tensor(model.encode(docs))
        t = time() - t
        print(f'Elapsed time: {t:.1f} s')

        # train the model
        optimizer.zero_grad()

        # Calculate Y_diff
        Y_diff = Y_train - X_train @ mat

        # Compute main loss
        loss = criterion(Y_diff, torch.zeros(Y_diff.shape))
        loss.backward()
        optimizer.step()

    # normalize the correction matrix
    mat = mat.detach().numpy()
    mat = mat / np.linalg.norm(mat)

    # evaluate post-training performance by applying the correction matrix at various intensity levels
    print('\nEvaluating post-training performance...')
    _x = np.linspace(0., 1., 31) ** 2
    _y = []
    X_test = X_test.detach().numpy()
    Y_test = Y_test.detach().numpy()
    for i in range(len(_x)):
        print(f'\r{i+1}/{len(_x)}')
        X_test_2 = X_test @ (np.eye(M) + mat * float(_x[i]))
        score_post = compute_recall_at_k(X_test_2, Y_test, k=1)
        _y.append(score_post)
    print()
    plt.plot(_x, _y)
    plt.xlabel('Correction matrix intensity')
    plt.ylabel('Recall@1')
    plt.title('Performance after training')
    plt.show()


def run_training():
    np.random.seed(0)
    trainer = EmbeddingTrainer(model_name="all-MiniLM-L6-v2",
                               embedding_size=384,
                               train_dir_path='..\\data\\training_dataset',
                               test_name='wikiqa_2224', # yahoo_train_2001, wikiqa_2224
                               batch_size=500,
                               max_size=600)
    trainer.run_training()


if __name__ == "__main__":
    # test()
    # test_training()
    # test_training_2()
    run_training()
