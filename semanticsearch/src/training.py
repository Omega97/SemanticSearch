import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.signal import savgol_filter
from typing import List
from semanticsearch.src.embedding import EmbeddingModel
from semanticsearch.src.training_data import TrainingData, load_tsv
from semanticsearch.src.ranking import compute_recall_at_k


class Trainer:
    def __init__(self, Y, Y_target, L, lr=1e-3, epsilon_0=0.1):
        """
        Initializes random Y and Y_target for demonstration,
        plus trainable parameters A and B.
        """
        self.Y = Y
        self.Y_target = Y_target
        self.N = Y.shape[0]
        self.M = Y.shape[1]
        self.L = L

        # Trainable low-rank update parameters
        self.A = nn.Parameter(torch.randn(self.M, L) * epsilon_0)
        self.B = nn.Parameter(torch.randn(L, self.M) * epsilon_0)

        # Optimizer and loss function
        self.optimizer = optim.Adam([self.A, self.B], lr=lr)
        self.criterion = nn.MSELoss()

        # Identity matrix used repeatedly
        self.I = torch.eye(self.M)

    def get_refinement_matrix(self):
        """
        Returns the refinement matrix I + AB.
        """
        return self.I + self.A @ self.B

    def train(self, num_epochs=1000, lambda_reg=1e-3):
        """
        Training loop with simple MSE + regularization to keep AB small.
        """
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            # Construct I + AB
            refinement_matrix = self.get_refinement_matrix()

            # Apply to Y
            Y_refined = self.Y @ refinement_matrix

            # Compute main loss
            loss = self.criterion(Y_refined, self.Y_target)

            # Optional regularization to keep AB small
            reg = (self.A @ self.B).pow(2).sum()
            loss += lambda_reg * reg

            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"{epoch + 1:5.0f}/{num_epochs}, RMSE: {(loss.item())**.5:.6f}")

        return self.get_refinement_matrix()

    def save_refinement_matrix(self, path):
        """
        Saves the refinement matrix to the given path.
        """
        torch.save(self.get_refinement_matrix(), path)


class EmbeddingTrainer:
    """
    To train the perturbation of the map between the query embedding and document embedding
    we sweep once through the entire dataset, and we accumulate the gradient.
    """
    def __init__(self, model_name, embedding_size, train_dir_path, test_name,
                 batch_size=500, max_size=1000, learning_rate=1e-2, max_n_cycles=None):
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.train_dir_path = train_dir_path
        self.test_name = test_name
        self.batch_size = batch_size
        self.max_size = max_size
        self.learning_rate = learning_rate
        self.max_n_cycles = max_n_cycles

        self.model = None
        self.data = None
        self.X_test = None
        self.Y_test = None
        self.optimizer = None
        self.criterion = None
        self.train_queries = None
        self.train_docs = None
        self.test_queries = None
        self.test_docs = None
        self.n_cycles = None
        self.perturbation = None

        self._load_model()

    def _load_model(self):
        print('Loading model...')
        self.model = EmbeddingModel(self.model_name)

    def prepare_data(self):
        print('\nLoading the data...')
        self.data = TrainingData(self.train_dir_path)
        train_names = [name for name in self.data.get_names() if name != self.test_name]
        self.train_queries, self.train_docs = self.data.get_queries_and_docs(*train_names)
        self.test_queries, self.test_docs = self.data.get_queries_and_docs(self.test_name, max_size=self.max_size)

        print('\nGenerating the test embeddings...')
        self.X_test = torch.tensor(self.model.encode(self.test_queries))
        self.Y_test = torch.tensor(self.model.encode(self.test_docs))

    def get_refinement_matrix(self, epsilon):
        """This matrix converts an embedding vector in query space to document space"""
        return np.eye(len(self.perturbation_np)) + epsilon * self.perturbation_np

    def save_refinement_matrix(self, path, epsilon):
        """
        Saves the refinement matrix to the given path.
        """
        torch.save(self.get_refinement_matrix(epsilon), path)

    def initialize_training(self, epsilon_0=0.001):
        print('\nInitializing the training parameters...')
        self.perturbation = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size) * epsilon_0)
        self.optimizer = optim.SGD([self.perturbation], lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        if self.max_n_cycles is None:
            self.n_cycles = len(self.train_queries) // self.batch_size
        else:
            self.n_cycles = self.max_n_cycles

    def evaluate_pre_training(self):
        score_pre = compute_recall_at_k(self.X_test, self.Y_test, k=1)
        print(f'\nPerformance before training: {score_pre:.2%}')

    def train(self):
        print('\nTraining...')

        self.optimizer.zero_grad()  # Initialize gradients to zero outside the loop

        for i_cycle in range(self.n_cycles):
            print(f'\nCycle {i_cycle + 1}/{self.n_cycles}')

            print('\nGetting the data...')
            i_start = i_cycle * self.batch_size
            i_end = (i_cycle + 1) * self.batch_size
            i_end = min(i_end, len(self.train_queries))
            indices = range(i_start, i_end)
            queries = [self.train_queries[i] for i in indices]
            docs = [self.train_docs[i] for i in indices]

            print(f'\nGenerating the embeddings for {i_end - i_start} samples...')
            t = time()
            X = torch.tensor(self.model.encode(queries))
            Y_true = torch.tensor(self.model.encode(docs))
            t = time() - t
            print(f'Elapsed time: {t:.1f} s')

            Y = X + X @ self.perturbation
            loss = self.criterion(Y, Y_true)

            loss.backward()  # Accumulate gradients

        self.optimizer.step()  # Perform a single optimization step after all batches

        # normalize perturbation
        self.perturbation_np = self.perturbation.detach().numpy()
        self.perturbation_np /= np.linalg.norm(self.perturbation_np)

    def evaluate_post_training(self, n_points=31):
        print('\nEvaluating post-training performance...')
        _x = np.linspace(-1., 1., n_points+1)
        _x = (_x**3 + _x)/2
        _x *= 5
        _y = []
        X_test_np = self.X_test.detach().numpy()
        Y_test_np = self.Y_test.detach().numpy()

        for i in range(len(_x)):
            print(f'\r{i+1}/{len(_x)}', end='')
            mat = self.get_refinement_matrix(epsilon=float(_x[i]))
            X_test_2 = X_test_np @ mat
            score_post = compute_recall_at_k(X_test_2, Y_test_np, k=1)
            _y.append(score_post)
        print()
        _y_smooth = savgol_filter(_y, window_length=9, polyorder=2)
        best_x = _x[np.argmax(_y_smooth)]
        print(f'\nBest epsilon = {best_x:.2f}')

        plt.plot(_x, _y)
        plt.plot(_x, _y_smooth, '--', c='k')
        plt.xlabel('Correction matrix intensity')
        plt.ylabel('Recall@1')
        plt.title('Performance after training')
        plt.show()

    def run_training(self):
        self.prepare_data()
        self.initialize_training()
        self.evaluate_pre_training()
        self.train()
        self.evaluate_post_training()


class EmbeddingTrainer_v2:
    def __init__(self, model, data_dir_path, embedding_path, matrix_path,
                 lr=0.1, epsilon_0=1e-3, reg=1e-3):
        """
        Initializes random Y and Y_target for demonstration,
        plus trainable parameters A and B.
        """
        self.model = model
        self.data_dir_path = data_dir_path
        self.embedding_path = embedding_path
        self.matrix_path = matrix_path
        self.lr = lr
        self.epsilon_0 = epsilon_0
        self.reg = reg

        # load embeddings
        self.query_embeddings = None
        self.doc_embeddings = None
        self._load_embeddings()

        # Trainable low-rank update parameters
        self.A = None
        self._load_matrix()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.get_matrices(), lr=lr)
        self.criterion = nn.MSELoss()

    def get_matrices(self) -> List[torch.Tensor]:
        """Return the components of the refinement matrix."""
        return [self.A]

    def get_refinement_matrix(self) -> torch.Tensor:
        """Returns the refinement matrix."""
        return self.A

    def train(self, n_cycles=1000, reg_lambda=1e-3):
        """
        Training loop with MSE + regularization to keep A small.
        """
        for i in range(n_cycles):
            self.optimizer.zero_grad()

            # Construct model
            refinement_matrix = self.get_refinement_matrix()

            # Compute output
            new_doc_embeddings = self.query_embeddings @ refinement_matrix

            # Compute main loss
            mse_loss = self.criterion(new_doc_embeddings, self.doc_embeddings)
            reg_loss = reg_lambda * torch.norm(self.A, p=2)
            total_loss = mse_loss + reg_loss
            total_loss.backward()

            # Update self.A
            self.optimizer.step()

            # Optional: Print loss for monitoring
            if (i + 1) % 10 == 0:
                print(f"Cycle {i + 1}/{n_cycles}, "
                      f"MSE Loss: {mse_loss.item()}, "
                      f"Reg Loss: {reg_loss.item()}, "
                      f"Total Loss: {total_loss.item()}")

        # save
        self.save_matrix()
        return self.get_refinement_matrix()

    def compute_embeddings(self, batch_size=300):
        """Compute embeddings"""
        print('Computing embeddings...')
        files = [name for name in os.listdir(self.data_dir_path) if name.endswith('tsv')]
        files = ['papers_dataset_924.tsv']

        text_queries = []
        text_docs = []
        for file in files:
            data = load_tsv(file)
            queries = data[:, 0]
            docs = data[:, 1]


        rows = len(text_queries)
        results = []
        for i in range(0, rows, batch_size):
            print(f'{i:4}/{rows}')
            batch = self.query_embeddings[i:i + batch_size]
            result = self.model.encode(batch)
            results.append(result)
        return torch.cat(results, dim=0)  # Concatenates the results along the 0th dimension.

    def save_embeddings(self):
        assert self.embedding_path.endswith('.pt')
        data = {'query_embeddings': self.query_embeddings,
                'doc_embeddings': self.doc_embeddings}
        torch.save(data, self.embedding_path)

    def _load_embeddings(self):
        assert self.embedding_path.endswith('.pt')
        if os.path.exists(self.embedding_path):
            data = torch.load(self.embedding_path)
            self.query_embeddings = data['query_embeddings']
            self.doc_embeddings = data['doc_embeddings']
        else:
            self.compute_embeddings()
            self.save_embeddings()

    def save_matrix(self):
        """ Saves the refinement matrix to the given path. """
        assert self.matrix_path.endswith('.pt')
        data = {'matrix_A': self.A}
        torch.save(data, self.matrix_path)

    def get_embedding_dim(self):
        assert self.query_embeddings is not None
        return self.query_embeddings.shape[1]

    def reset_matrix(self, mat: torch.Tensor):
        self.A = mat

    def _load_matrix(self):
        assert self.matrix_path.endswith('.pt')
        if os.path.exists(self.matrix_path):
            data = torch.load(self.matrix_path)
            self.A = data['matrix_A']
        else:
            m = self.get_embedding_dim()
            mat = nn.Parameter(torch.randn(m, m) * self.epsilon_0)
            self.reset_matrix(mat)
