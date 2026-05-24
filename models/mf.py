import numpy as np


class MF:
    def __init__(self, n_users, n_items, k=20, lr=0.01, reg=0.01):
        self.U = np.random.normal(0, 0.1, (n_users, k))
        self.V = np.random.normal(0, 0.1, (n_items, k))

        self.lr = lr
        self.reg = reg

    def train(self, R, epochs=10):
        users, items = R.nonzero()

        for epoch in range(epochs):
            for u, i in zip(users, items):
                error = R[u, i] - np.dot(self.U[u], self.V[i])

                self.U[u] += self.lr * (error * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (error * self.U[u] - self.reg * self.V[i])

            print(f"MF Epoch {epoch+1}")