import torch
import numpy as np
import matplotlib.pyplot as plt
from semanticsearch.src.training import EmbeddingTrainer
from semanticsearch.src.embedding import EmbeddingModel


def test_training(data_dir_path='..\\data\\training_dataset',
                  embedding_path='..\\data\\train_embedding.pt',
                  matrix_path='..\\data\\train_matrix.pt'):
    """
    Evaluates the trained refinement.
    """
    # For reproducibility
    torch.manual_seed(0)

    # load the model
    model = EmbeddingModel()

    # training
    trainer = EmbeddingTrainer(model,
                               data_dir_path, embedding_path, matrix_path,
                               lr=0.01, epsilon_0=1e-1, reg=1e-3,
                               max_lines=None,
                               do_force_recompute_embeddings=False)

    trainer.train(n_cycles=1000, reg_lambda=1e-3)

    A = trainer.get_transformation_matrix().detach().numpy()

    fig, ax = plt.subplots(ncols=2)
    y = trainer.get_loss_values()

    plt.sca(ax[0])
    plt.title('Transformation Matrix')
    plt.imshow(A, cmap='bwr')
    clim = np.std(y)
    plt.clim(-clim, clim)
    # plt.colorbar()

    plt.sca(ax[1])
    plt.title('Loss')
    plt.plot(y)

    plt.show()


if __name__ == "__main__":
    test_training()
