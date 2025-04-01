"""
Todo train test split
"""
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
from semanticsearch.src.training_data import load_tsv
from semanticsearch.src.ranking import compute_recall_at_k


class EmbeddingTrainer:
    def __init__(self, model, data_dir_path, embedding_path, matrix_path,
                 lr=0.1, reg=1e-3, epsilon_0=1e-3, batch_size=300, max_lines=None,
                 do_force_recompute_embeddings=False):
        """
        :param model:
        :param data_dir_path: directory with the text documents
        :param embedding_path: path for the query and document embeddings
        :param matrix_path: path for the stored matrix
        :param lr: learning rate
        :param reg: regularization
        :param epsilon_0: initial magnitude of the matrix
        :param batch_size: how many strings does the model process at the same time
        :param max_lines: max number of lines to convert to embedding
        """
        self.model = model
        self.data_dir_path = data_dir_path
        self.embedding_path = embedding_path
        self.matrix_path = matrix_path
        self.lr = lr
        self.reg = reg
        self.epsilon_0 = epsilon_0
        self.batch_size = batch_size
        self.max_lines = max_lines
        self.loss_values = None

        # Load embeddings
        self.query_embeddings = None
        self.doc_embeddings = None
        if do_force_recompute_embeddings:
            self.compute_embeddings()
            self.save_embeddings()
        else:
            self._load_embeddings()

        # Trainable low-rank update parameters
        self.matrix = None
        self.reset_matrix()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.get_matrices(), lr=lr)
        self.criterion = nn.MSELoss()

    def get_matrices(self) -> List[torch.Tensor]:
        """Return the components of the refinement matrix."""
        return [self.matrix]

    def get_transformation_matrix(self) -> torch.Tensor:
        """Returns the refinement matrix as a Tensor."""
        return self.matrix

    def get_transformation_matrix_numpy(self) -> np.array:
        """Returns the refinement matrix as an array."""
        return self.matrix.detach().numpy()

    def train(self, n_cycles=1000, reg_lambda=1e-3):
        """ Training loop with MSE + regularization to keep A small. """
        self.loss_values = []

        for i in range(n_cycles):
            self.optimizer.zero_grad()

            # Construct model
            refinement_matrix = self.get_transformation_matrix()

            # Compute output
            new_doc_embeddings = self.query_embeddings @ refinement_matrix

            # Compute main loss
            mse_loss = self.criterion(new_doc_embeddings, self.doc_embeddings)
            reg_loss = reg_lambda * torch.norm(self.matrix, p=2)
            total_loss = mse_loss + reg_loss
            total_loss.backward()

            # Update self.A
            self.optimizer.step()

            # Optional: Print loss for monitoring
            loss = total_loss.item()
            self.loss_values.append(loss)
            if (i + 1) % 10 == 0:
                print(f"\r{(i+1)/n_cycles:6.2%}  Loss={loss:.6f}", end='')
        print()

        # save
        self.save_matrix()

        return self.get_transformation_matrix()

    def _load_text(self):
        """Load all the text at once into a list"""
        names = [name for name in os.listdir(self.data_dir_path) if name.endswith('.tsv')]
        text_queries = []
        text_docs = []

        for name in names:
            print(f' Loading {name}')
            path = os.path.join(self.data_dir_path, name)
            queries, docs = load_tsv(path)
            text_queries += queries
            text_docs += docs

        return text_queries, text_docs

    def shuffle_embeddings(self):
        """Shuffle the order of the query-doc embeddings"""
        n = len(self.query_embeddings)
        indices = np.random.permutation(np.arange(n))
        self.query_embeddings = self.query_embeddings[indices]
        self.doc_embeddings = self.doc_embeddings[indices]

    def compute_embeddings(self):
        """Compute embeddings"""
        text_queries, text_docs = self._load_text()

        print('Computing embeddings...')
        rows = len(text_queries)
        for i in range(0, rows, self.batch_size):

            # compute
            x = self.model.encode(text_queries[i:i + self.batch_size])
            y = self.model.encode(text_docs[i:i + self.batch_size])

            # append
            if i == 0:
                self.query_embeddings = x
                self.doc_embeddings = y
            else:
                self.query_embeddings = np.concatenate((self.query_embeddings, x))
                self.doc_embeddings = np.concatenate((self.doc_embeddings, y))

            print(f'\r{i / rows:6.2%}', end='')
            if self.max_lines is not None:
                if len(self.query_embeddings) >= self.max_lines:
                    break
        print()

        self.shuffle_embeddings()
        self.query_embeddings = torch.Tensor(self.query_embeddings)
        self.doc_embeddings = torch.Tensor(self.doc_embeddings)

        return self.query_embeddings, self.doc_embeddings

    def save_embeddings(self):
        """Save dictionary of query-document embeddings to embedding_path"""
        print('Saving embeddings...')
        assert self.embedding_path.endswith('.pt')
        data = {'query_embeddings': self.query_embeddings,
                'doc_embeddings': self.doc_embeddings}
        torch.save(data, self.embedding_path)

    def _load_embeddings(self):
        """load dictionary of query-document embeddings from embedding_path"""
        print('Loading embeddings...')
        assert self.embedding_path.endswith('.pt')
        if os.path.exists(self.embedding_path):
            print(f' Loading from {self.embedding_path}')
            data = torch.load(self.embedding_path, weights_only=False)
            self.query_embeddings = data['query_embeddings']
            self.doc_embeddings = data['doc_embeddings']
        else:
            self.compute_embeddings()
            self.save_embeddings()

    def save_matrix(self):
        """ Saves the refinement matrix to the given path. """
        print('Saving matrix...')
        assert self.matrix_path.endswith('.pt')
        data = {'matrix_A': self.matrix}
        torch.save(data, self.matrix_path)

    def get_embedding_dim(self):
        """Get the number of dimensions of the embedding space"""
        assert self.query_embeddings is not None
        return self.query_embeddings.shape[1]

    def set_matrix(self, mat: torch.Tensor):
        """overwrite matrix A"""
        self.matrix = mat

    def reset_matrix(self):
        """overwrite matrix A"""
        m = self.get_embedding_dim()
        mat = nn.Parameter(torch.randn(m, m) * self.epsilon_0)
        self.set_matrix(mat)

    def _load_matrix(self):
        """Load matrix A from file matrix_path"""
        print('Loading matrix...')
        assert self.matrix_path.endswith('.pt')
        if os.path.exists(self.matrix_path):
            data = torch.load(self.matrix_path, weights_only=False)
            self.matrix = data['matrix_A']
        else:
            self.reset_matrix()

    def get_loss_values(self) -> np.array:
        assert self.loss_values is not None
        return np.array(self.loss_values)

    def compute_model_performance(self, k=3):
        A = self.query_embeddings
        B = self.doc_embeddings
        if self.max_lines is not None:
            A = A[:self.max_lines]
            B = B[:self.max_lines]
        mat = self.get_transformation_matrix_numpy()
        before = compute_recall_at_k(A, B, k=k)
        after = compute_recall_at_k(A @ mat, B, k=k)
        return {"performance_before_correction": before,
                "performance_after_correction": after}
