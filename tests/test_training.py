import torch
import numpy as np
import matplotlib.pyplot as plt
from semanticsearch.src.training import EmbeddingTrainer
from semanticsearch.src.embedding import EmbeddingModel


def test_training(train_data_dir_path='..\\data\\training_dataset',
                  train_embedding_path='..\\data\\train_embedding.pt',
                  test_data_dir_path='..\\data\\test_dataset',
                  test_embedding_path='..\\data\\test_embedding.pt',
                  matrix_path='..\\data\\train_matrix.pt'):
    """
    Evaluates the trained refinement.
    """
    # For reproducibility
    torch.manual_seed(0)

    # Load the model
    model = EmbeddingModel()

    # Training
    trainer = EmbeddingTrainer(model,
                               train_data_dir_path, train_embedding_path, matrix_path,
                               lr=0.01, epsilon_0=1e-1, reg=1e-3,
                               max_lines=None,
                               do_force_recompute_embeddings=False)
    trainer.train(n_cycles=100, reg_lambda=1e-3)

    # Get result
    A = trainer.get_transformation_matrix().detach().numpy()
    y = trainer.get_loss_values()
    del trainer

    # Tester
    tester = EmbeddingTrainer(model,
                              test_data_dir_path, test_embedding_path, matrix_path,
                              max_lines=500,
                              do_force_recompute_embeddings=False)
    print('Computing performance...')
    print(tester.compute_model_performance())

    fig, ax = plt.subplots(ncols=2)

    plt.sca(ax[0])
    plt.title('Transformation Matrix')
    plt.imshow(A, cmap='bwr')
    clim = np.std(y)
    plt.clim(-clim, clim)
    # plt.colorbar()

    plt.sca(ax[1])
    plt.title('Loss')
    plt.plot(y)
    plt.ylim(0, None)

    plt.show()


if __name__ == "__main__":
    test_training()
