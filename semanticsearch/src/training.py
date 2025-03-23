import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.signal import savgol_filter
from semanticsearch.src.embedding import EmbeddingModel
from semanticsearch.src.training_data import TrainingData
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
                 batch_size=500, max_size=1000, learning_rate=1e-2):
        self.embedding_size = embedding_size
        self.train_dir_path = train_dir_path
        self.test_name = test_name
        self.batch_size = batch_size
        self.max_size = max_size
        self.learning_rate = learning_rate
        self.model = EmbeddingModel(model_name)
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

    def prepare_data(self):
        print('\nLoading the data...')
        self.data = TrainingData(self.train_dir_path)
        train_names = [name for name in self.data.get_names() if name != self.test_name]
        self.train_queries, self.train_docs = self.data.get_queries_and_docs(*train_names)
        self.test_queries, self.test_docs = self.data.get_queries_and_docs(self.test_name, max_size=self.max_size)

        print('\nGenerating the test embeddings...')
        self.X_test = torch.tensor(self.model.encode(self.test_queries))
        self.Y_test = torch.tensor(self.model.encode(self.test_docs))

    def initialize_training(self, epsilon_0=0.001):
        print('\nInitializing the training parameters...')
        self.perturbation = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size) * epsilon_0)
        self.optimizer = optim.Adam([self.perturbation], lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.n_cycles = len(self.train_queries) // self.batch_size

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
            queries = self.train_queries[indices]
            docs = self.train_docs[indices]

            print(f'\nGenerating the embeddings for {i_end - i_start} samples...')
            t = time()
            query_emb = torch.tensor(self.model.encode(queries))
            doc_emb = torch.tensor(self.model.encode(docs))
            t = time() - t
            print(f'Elapsed time: {t:.1f} s')

            loss = self.criterion(doc_emb, query_emb @ self.perturbation)
            loss.backward()  # Accumulate gradients

        self.optimizer.step()  # Perform a single optimization step after all batches

    def evaluate_post_training(self, n_points=30):
        print('\nEvaluating post-training performance...')
        _x = np.linspace(-1., 1., n_points+1)
        _x = (_x**3 + _x)/2
        _x *= 5
        _y = []
        X_test_np = self.X_test.detach().numpy()
        Y_test_np = self.Y_test.detach().numpy()
        mat_np = self.perturbation.detach().numpy()
        mat_np = mat_np / np.linalg.norm(mat_np)

        I = np.eye(self.embedding_size)
        for i in range(len(_x)):
            print(f'\r{i+1}/{len(_x)}', end='')
            X_test_2 = X_test_np @ (I - mat_np * float(_x[i]))
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
