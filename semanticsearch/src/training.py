import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class RefinementTrainer:
    def __init__(self, Y, Y_target, L, lr=1e-3, epsilon_0=0.01):
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

        # Optimizer and loss
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
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

    def save_refinement_matrix(self, path):
        """
        Saves the refinement matrix to the given path.
        """
        torch.save(self.get_refinement_matrix(), path)


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

    trainer = RefinementTrainer(Y, Y_target, L=L)
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


if __name__ == "__main__":
    test()
